# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Shared source selection and audits; profiling hooks load only for captures."""
import atexit
import functools
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import sys
import weakref

ROOT = Path(os.environ['GDN_REPLAY_PREPARED'])
DIRECTORY = Path(os.environ.get('STUDY_RUN_DIR', '/nonexistent'))
VARIANT = os.environ.get('STUDY_VARIANT', 'disabled')
PROFILE = os.environ.get('STUDY_PROFILE') == '1'
COLLECT_STATS = os.environ.get('STUDY_COLLECT_STATS') == '1'
PLANS, MODULES, STATS_HANDLES = [], {}, {}
FI_STATS = None

if PROFILE:
    import profile_hooks


def write_event(row):
    if DIRECTORY.is_dir():
        with (DIRECTORY / f'timing_audit.{os.getpid()}.jsonl').open('a') as out:
            out.write(json.dumps(row, default=str) + '\n')


def snapshot(reason):
    if PROFILE:
        profile_hooks.flush()
    live = []
    for ref in PLANS:
        plan = ref()
        if plan is not None:
            live.append({'replays': plan.replays, 'fallbacks': plan.fallbacks,
                         'owner_alive': plan._owner() is not None,
                         'counts': list(plan._signature.counts), 'layer_id': plan._signature.layer_id})
    write_event({'event': 'snapshot', 'reason': reason, 'pid': os.getpid(), 'variant': VARIANT,
                 'live_plans': live, 'module_paths': MODULES,
                 'FI_control': dict(FI_STATS) if FI_STATS is not None else None})


class StatsBuffer:
    def __init__(self, capacity):
        self.capacity, self.rows = capacity, []
        self.overflow, self.error = False, None

    def add(self, row):
        inflight = row.get('inflightBatchingStats')
        if inflight is None:
            return
        if len(self.rows) >= self.capacity:
            self.overflow = True
            return
        self.rows.append({'timestamp': row['timestamp'], 'iter': row.get('iter'),
                          'context': inflight.get('numContextRequests', 0),
                          'generation': inflight.get('numGenRequests', 0)})


def start_collector(llm):
    import threading
    stop = threading.Event()
    capacity = int(os.environ.get('GDN_STATS_CAPACITY', '32768')) if PROFILE else 16384
    buffer = StatsBuffer(capacity)

    def collect():
        try:
            while not stop.is_set():
                for row in llm.get_stats(timeout=.1):
                    buffer.add(row)
        except (RuntimeError, ValueError, KeyError, TypeError, AttributeError) as exc:
            buffer.error = type(exc).__name__

    thread = threading.Thread(target=collect, daemon=True)
    STATS_HANDLES[id(llm)] = (stop, thread, buffer)
    thread.start()
    write_event({'event': 'control_collector_started'})


def patch(module):
    global FI_STATS
    name = module.__name__
    MODULES[name] = module.__file__
    if name == 'flashinfer.gdn_prefill':
        import fi_launch_control
        _, _, FI_STATS = fi_launch_control.install()
        write_event({'event': 'FI_control_installed', 'all_aliases_asserted': True})
    elif name.endswith('.gdn_prefill_graph'):
        cls, original = module._GuardedReplay, module._GuardedReplay.__init__

        @functools.wraps(original)
        def init(self, *args, **kwargs):
            original(self, *args, **kwargs)
            PLANS.append(weakref.ref(self))
        cls.__init__ = init
    elif name.endswith('.model_engine'):
        for cls in list(vars(module).values()):
            if isinstance(cls, type) and cls.__module__ == name and 'release_executor_resources' in cls.__dict__:
                original = cls.release_executor_resources

                @functools.wraps(original)
                def release(self, *args, __original=original, **kwargs):
                    snapshot('before_resource_release')
                    result = __original(self, *args, **kwargs)
                    snapshot('after_resource_release')
                    return result
                cls.release_executor_resources = release
    elif name.endswith(('.kv_cache_manager_v2', '.mamba_cache_manager')):
        hybrid = name.endswith('.mamba_cache_manager')
        cls = getattr(module, 'MambaHybridCacheManagerV2' if hybrid else 'KVCacheManagerV2')
        original = cls.__init__

        @functools.wraps(original)
        def init(self, *args, **kwargs):
            original(self, *args, **kwargs)
            row = {'event': 'allocation', 'class': type(self).__name__, 'dtype': str(self.dtype)}
            assert 'BF16' in row['dtype'].upper()
            if hybrid:
                row.update(ssm_state_dtype=str(self.ssm_state_dtype), conv_state_dtype=str(self.conv_state_dtype))
                assert row['ssm_state_dtype'] == row['conv_state_dtype'] == 'torch.bfloat16'
            write_event(row)
        cls.__init__ = init
    elif name == 'tensorrt_llm.llmapi.llm':
        cls, original = module.LLM, module.LLM.__init__

        @functools.wraps(original)
        def init(self, *args, **kwargs):
            if COLLECT_STATS:
                write_event({'event': 'control_init_enter'})
            ready = False
            try:
                original(self, *args, **kwargs)
                cache = self.args.kv_cache_config
                assert cache.max_tokens == 24576 and cache.dtype == 'auto' and not cache.enable_block_reuse and cache.host_cache_size == 0 and not cache.use_uvm
                write_event({'event': 'capacity', 'logical_max_tokens': cache.max_tokens,
                             'physical_capacity_diagnostic': self.get_kv_cache_capacity(),
                             'resolved_config': self.args.model_dump(mode='json')})
                ready = True
            finally:
                if COLLECT_STATS:
                    write_event({'event': 'control_init_exit', 'returned': ready})
            if COLLECT_STATS:
                start_collector(self)
        cls.__init__ = init
        if COLLECT_STATS:
            shutdown_original = cls.shutdown

            @functools.wraps(shutdown_original)
            def shutdown(self, *args, **kwargs):
                handle = STATS_HANDLES.pop(id(self), None)
                if handle is not None:
                    stop, thread, buffer = handle
                    stop.set()
                    thread.join(timeout=5)
                    write_event({'event': 'control_iteration_rows', 'rows': buffer.rows,
                                 'overflow': buffer.overflow, 'collector_error': buffer.error,
                                 'thread_stopped': not thread.is_alive()})
                return shutdown_original(self, *args, **kwargs)
            cls.shutdown = shutdown
    if PROFILE:
        profile_hooks.patch(module)


TARGETS = {'flashinfer.gdn_prefill', 'tensorrt_llm._torch.modules.mamba.gdn_prefill_graph',
           'tensorrt_llm._torch.pyexecutor.model_engine',
           'tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2',
           'tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager', 'tensorrt_llm.llmapi.llm'}
if PROFILE:
    TARGETS.update(profile_hooks.TARGETS)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        overlay = ROOT / 'overlay' / (fullname.replace('.', '/') + '.py')
        custom = VARIANT == 'candidate' and overlay.is_file()
        if not custom and fullname not in TARGETS:
            return None
        spec = importlib.util.spec_from_file_location(fullname, overlay) if custom else importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        original = spec.loader.exec_module

        def execute(module):
            original(module)
            if fullname in TARGETS:
                patch(module)
        spec.loader.exec_module = execute
        return spec


if VARIANT in ['baseline', 'candidate']:
    sys.meta_path.insert(0, Finder())
    atexit.register(snapshot, 'atexit')
