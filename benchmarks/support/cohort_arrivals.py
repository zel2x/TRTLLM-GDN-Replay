# SPDX-License-Identifier: Apache-2.0
"""Release eight ready request tasks together, then drain the entire group."""
import asyncio
from collections import deque
import json
from pathlib import Path
import time

class CohortError(RuntimeError):
    pass

def validate_group(group, *, size=8, max_spread_ms=5):
    rows = group['requests']
    if len(rows) != size or not all(row.get('success') and row.get('done_ns', 0) >= row.get('started_ns', 1) > 0 for row in rows):
        raise CohortError('Incomplete or failed synchronized group')
    spread = (max(row['started_ns'] for row in rows) - min(row['started_ns'] for row in rows)) / 1e6
    if spread > max_spread_ms:
        raise CohortError(f'Client-start spread {spread:.3f} ms exceeds {max_spread_ms} ms')
    return {'client_start_spread_ms': spread, 'first_start_ns': min(row['started_ns'] for row in rows), 'last_done_ns': max(row['done_ns'] for row in rows)}

class Controller:
    def __init__(self, directory, size=8, max_spread_ms=5, expected_counts=(8, 200)):
        self.directory = Path(directory)
        self.size = size
        self.expected_counts = expected_counts
        self.max_spread_ms = max_spread_ms
        self.phases = []
        self.pending = deque()
        self.original = None

    def save(self):
        path = self.directory / 'cohort_arrivals.json'
        self.directory.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps({'group_size': self.size, 'max_client_start_spread_ms': self.max_spread_ms, 'phases': self.phases, 'pending': len(self.pending)}, indent=2) + '\n')
        temp.replace(path)

    async def arrivals(self, input_requests, request_rate, burstiness):
        if request_rate != float('inf') or len(input_requests) % self.size:
            raise CohortError('Synchronized preset requires an integral number of eight-request groups and infinite offered rate')
        phase = {'phase': 'warmup' if not self.phases else 'measured', 'cohorts': []}
        self.phases.append(phase)
        for offset in range(0, len(input_requests), self.size):
            record = {'index': offset // self.size, 'requests': []}
            phase['cohorts'].append(record)
            state = {'record': record, 'gate': asyncio.Event(), 'drained': asyncio.Event(), 'ready': 0, 'finished': 0, 'tasks': [], 'error': None}
            try:
                for index in range(offset, offset + self.size):
                    row = {'index': index}
                    record['requests'].append(row)
                    self.pending.append((state, row))
                    yield input_requests[index]
                await state['drained'].wait()
                if state['error']:
                    raise CohortError(state['error'])
                record['validation'] = validate_group(record, size=self.size, max_spread_ms=self.max_spread_ms)
                if len(phase['cohorts']) > 1:
                    previous = phase['cohorts'][-2]['validation']
                    if record['validation']['first_start_ns'] < previous['last_done_ns']:
                        raise CohortError('Next group started before the previous group drained')
                self.save()
            except BaseException as exc:
                record['error'] = type(exc).__name__ + ': ' + str(exc)
                state['gate'].set()
                for task in state['tasks']:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*state['tasks'], return_exceptions=True)
                self.save()
                raise

    async def request(self, *args, **kwargs):
        if not self.pending:
            raise CohortError('Unexpected request outside a synchronized group')
        state, row = self.pending.popleft()
        state['tasks'].append(asyncio.current_task())
        state['ready'] += 1
        if state['ready'] == self.size:
            state['record']['released_ns'] = time.perf_counter_ns()
            state['record']['ready_at_release'] = state['ready']
            state['gate'].set()
        try:
            await state['gate'].wait()
            row['started_ns'] = time.perf_counter_ns()
            result = await self.original(*args, **kwargs)
            row['success'] = bool(result.success)
            if not row['success']:
                state['error'] = 'Official HTTP request failed'
                state['drained'].set()
            return result
        except BaseException as exc:
            row['success'] = False
            state['error'] = type(exc).__name__
            state['drained'].set()
            raise
        finally:
            row['done_ns'] = time.perf_counter_ns()
            state['finished'] += 1
            if state['finished'] == self.size:
                state['drained'].set()

    def finish(self):
        self.save()
        if len(self.phases) != 2 or self.pending:
            raise CohortError('Expected complete warmup and measured phases')
        summary = {}
        for phase, expected in zip(self.phases, self.expected_counts):
            if sum(len(c['requests']) for c in phase['cohorts']) != expected:
                raise CohortError('Synchronized request count differs')
            validations = [validate_group(c, size=self.size, max_spread_ms=self.max_spread_ms) for c in phase['cohorts']]
            summary[phase['phase']] = {'requests': expected, 'groups': len(validations), 'max_client_start_spread_ms': max(c['client_start_spread_ms'] for c in validations)}
        return summary

def install(official, directory, size=8, max_spread_ms=5, expected_counts=(8, 200)):
    controller = Controller(directory, size=size, max_spread_ms=max_spread_ms, expected_counts=expected_counts)
    controller.original = official.ASYNC_REQUEST_FUNCS['openai']
    official.get_request = controller.arrivals
    official.ASYNC_REQUEST_FUNCS['openai'] = controller.request
    return controller.finish
