# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Known FI4699-style launcher caching, used identically in both arms."""
import functools
import inspect
import importlib
import textwrap


def install():
    from flashinfer.gdn_kernels.delta_rule_dsl import delta_rule_sm120 as mod
    from flashinfer.gdn_kernels.delta_rule_dsl.custom_compile_cache import get_cached_compile
    original = mod.delta_rule_prefill_dsl
    source = textwrap.dedent(inspect.getsource(original))
    construct = """    delta_rule_kernel = _FullyFusedDeltaRuleSm120(
        needs_alpha, needs_beta, needs_init_state, needs_checkpointing, kernel_dtype
    )
"""
    if source.count(construct) != 1:
        raise RuntimeError('Pinned FI constructor changed')
    source = source.replace(construct, '')
    marker = '    # Keep head counts and varlen extents runtime values across cached compiles.\n'
    fast = """    _launch_control_launch_stats['calls'] += 1
    delta_rule_kernel = _launch_control_get_kernel(
        needs_alpha, needs_beta, needs_init_state, needs_checkpointing, kernel_dtype
    )
    cached = _launch_control_get_cached(delta_rule_kernel, sm12x_compile_options(q.device))
    if cached is not None:
        _launch_control_launch_stats['hits'] += 1
        cached(q_tma, k_tma, v_tma, o_tma,
               alpha.reshape(-1) if needs_alpha else None,
               beta.reshape(-1) if needs_beta else None,
               state.reshape(-1), init_state.reshape(-1) if needs_init_state else None,
               state_checkpoints.reshape(-1) if needs_checkpointing else None,
               checkpoint_cu_starts if needs_checkpointing else None,
               tensormaps_t, cu_seqlens,
               cutlass.Float32(scale), cutlass.Int32(num_q_heads),
               cutlass.Int32(num_k_heads), cutlass.Int32(num_v_heads),
               cutlass.Int32(num_sab_heads), cutlass.Int32(num_seqs),
               cutlass.Int32(total_checkpoints), cutlass.Int32(checkpoint_every_n_tokens),
               num_seqs * num_sab_heads, stream)
        return
    _launch_control_launch_stats['misses'] += 1

"""
    if source.count(marker) != 1:
        raise RuntimeError('Pinned FI marker changed')
    source = source.replace(marker, fast + marker)
    mod._launch_control_get_kernel = functools.lru_cache(None)(mod._FullyFusedDeltaRuleSm120)
    mod._launch_control_get_cached = get_cached_compile
    mod.sm12x_compile_options = functools.lru_cache(None)(mod.sm12x_compile_options)
    mod._launch_control_launch_stats = dict(calls=0, hits=0, misses=0)
    exec(compile(source, '<equal-FI4699-control>', 'exec'), mod.__dict__)
    control = mod.delta_rule_prefill_dsl
    aliases = [importlib.import_module(name) for name in (
        'flashinfer.gdn_kernels.delta_rule_dsl',
        'flashinfer.gdn_kernels',
        'flashinfer.gdn_prefill',
    )]
    for alias in aliases:
        if alias.chunk_gated_delta_rule_sm120 is not original:
            raise RuntimeError('Unexpected pre-existing FI alias replacement')
        alias.chunk_gated_delta_rule_sm120 = control
    if not all(alias.chunk_gated_delta_rule_sm120 is control for alias in aliases):
        raise RuntimeError('FI control did not reach every live alias')
    return original, control, mod._launch_control_launch_stats
