# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Capture-only request and GDN markers; numerical functions remain unchanged."""
import functools
import json
import diag_probe as probe

flush = probe.flush
TARGETS = {'starlette.applications', 'tensorrt_llm._torch.pyexecutor.py_executor',
           'tensorrt_llm._torch.modules.mamba.gdn_mixer'}


def patch(module):
    name = module.__name__
    if name == 'starlette.applications':
        cls, original = module.Starlette, module.Starlette.__call__

        @functools.wraps(original)
        async def call(self, scope, receive, send):
            if scope.get('type') != 'http' or scope.get('path') != '/v1/completions':
                return await original(self, scope, receive, send)
            tag = dict(scope.get('headers', [])).get(b'x-study-request-id', b'').decode()
            token = probe.TAG.set(tag)
            probe.emit('http_arrival', tag=tag)

            async def observed_send(message):
                if message.get('type') == 'http.response.body':
                    for line in message.get('body', b'').splitlines():
                        if line.startswith(b'data: ') and line[6:] != b'[DONE]':
                            payload = json.loads(line[6:])
                            if payload.get('choices'):
                                probe.emit('sse_send', tag=tag, seq=probe.next_sequence('server_sse', tag),
                                           response_id=str(payload.get('id', '')))
                return await send(message)
            try:
                return await original(self, scope, receive, observed_send)
            finally:
                probe.emit('http_done', tag=tag)
                probe.TAG.reset(token)
        cls.__call__ = call
    elif name == 'tensorrt_llm.llmapi.llm':
        original = module.LLM.generate_async

        @functools.wraps(original)
        def generate(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            tag = probe.TAG.get()
            if tag:
                probe.emit('llm_submit', tag=tag, req=str(getattr(result, 'request_id', 'unavailable')))
            return result
        module.LLM.generate_async = generate
        shutdown_original = module.LLM.shutdown

        @functools.wraps(shutdown_original)
        def shutdown(self, *args, **kwargs):
            flush()
            try:
                return shutdown_original(self, *args, **kwargs)
            finally:
                flush()
        module.LLM.shutdown = shutdown
    elif name == 'tensorrt_llm._torch.pyexecutor.py_executor':
        cls, original = module.PyExecutor, module.PyExecutor._enqueue_responses

        @functools.wraps(original)
        def enqueue(self, responses):
            identifiers = []
            # Do not consume an unknown iterable before the original method does.
            if isinstance(responses, (list, tuple)):
                for item in responses:
                    identifier = str(item[0])
                    identifiers.append(identifier)
                    probe.emit('enqueue_prepare', req=identifier, seq=probe.next_sequence('enqueue', identifier), iteration=self.iter_counter)
            else:
                probe.emit('enqueue_unmapped_iterable', kind=type(responses).__name__, iteration=self.iter_counter)
            result = original(self, responses)
            if identifiers:
                probe.emit('enqueue_complete', requests=identifiers, iteration=self.iter_counter)
            return result
        cls._enqueue_responses = enqueue
        original_handle = cls._handle_responses

        @functools.wraps(original_handle)
        def handle(self, *args, **kwargs):
            probe.emit('handle_responses_begin', iteration=self.iter_counter)
            try:
                return original_handle(self, *args, **kwargs)
            finally:
                probe.emit('handle_responses_end', iteration=self.iter_counter)
        cls._handle_responses = handle
    elif name == 'tensorrt_llm._torch.modules.mamba.gdn_mixer':
        cls, original = module.Qwen3NextGatedDeltaNet, module.Qwen3NextGatedDeltaNet.forward_core

        @functools.wraps(original)
        def forward_core(self, mixed_qkv, a, b, attn_metadata, *args, **kwargs):
            attributes = {'layer': self.layer_idx, 'layer_object': id(self), 'contexts': attn_metadata.num_contexts,
                          'context_tokens': attn_metadata.num_ctx_tokens, 'tokens': attn_metadata.num_tokens}
            probe.begin_span('gdn_eager_core', **attributes)
            try:
                return original(self, mixed_qkv, a, b, attn_metadata, *args, **kwargs)
            finally:
                probe.end_span('gdn_eager_core', **attributes)
        cls.forward_core = forward_core
    elif name == 'tensorrt_llm._torch.modules.mamba.gdn_prefill_graph':
        cls, original = module._GuardedReplay, module._GuardedReplay.__call__

        @functools.wraps(original)
        def replay(self):
            previous = (self.replays, self.fallbacks)
            attributes = {'plan': id(self), 'layer': self._signature.layer_id, 'contexts': self._signature.counts[0],
                          'context_tokens': self._signature.counts[1], 'tokens': self._signature.counts[2]}
            probe.begin_span('gdn_prepared_call', **attributes)
            try:
                return original(self)
            finally:
                decision = 'replay' if self.replays > previous[0] else 'eager_fallback' if self.fallbacks > previous[1] else 'exception'
                probe.end_span('gdn_prepared_call', decision=decision, **attributes)
        cls.__call__ = replay
