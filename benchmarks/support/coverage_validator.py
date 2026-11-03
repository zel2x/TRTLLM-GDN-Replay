# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Read raw NVTX markers; JSON sidecars need not contain parent-server records."""
import collections
import sqlite3

def read_nvtx(path):
    db=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True)
    try:
        rows=db.execute("select coalesce(n.text,s.value) from NVTX_EVENTS n left join StringIds s on s.id=n.textId where coalesce(n.text,s.value) like 'study|%'").fetchall()
        result=[]
        for (label,) in rows:
            parts=label.split('|')
            result.append({'event':parts[1],'attrs':dict(part.split('=',1)for part in parts[2:]if '='in part)})
        return result
    finally:db.close()

def validate_server(markers,count):
    tags={'diag-'+str(i)for i in range(9,9+count)}
    selected=[row for row in markers if row['attrs'].get('tag')in tags]
    totals=collections.Counter((row['attrs'].get('tag'),row['event'])for row in selected)
    for tag in tags:
        assert totals[tag,'http_arrival']==1,('HTTP arrival missing/duplicate',tag)
        assert totals[tag,'llm_submit']==1,('LLM submission missing/duplicate',tag)
        assert totals[tag,'sse_send']==128,('Server send count mismatch',tag,totals[tag,'sse_send'])
    submissions=[row for row in selected if row['event']=='llm_submit']
    request_ids={row['attrs'].get('req')for row in submissions}
    assert len(request_ids)==count and not request_ids.intersection({None,'','unavailable'})
    return {'measured_requests':count,'HTTP_arrivals':count,'LLM_submissions':count,'server_send_events':count*128,'distinct_parent_request_ids':len(request_ids)}

def validate_coverage(markers,sidecars,count,variant,*,require_replay=True,required_replay_tokens=None):
    result=validate_server(markers,count)
    kinds={row['event']for row in markers}|{row['event']for row in sidecars}
    assert {'http_arrival','llm_submit','enqueue_prepare','sse_send','client_receive','gdn_eager_core_begin'}<=kinds
    expected_tags={'diag-'+str(i)for i in range(9,9+count)}
    clients=collections.Counter(row.get('tag')for row in sidecars if row['event']=='client_receive'and row.get('tag')in expected_tags)
    assert len(clients)==count and all(clients[tag]==128 for tag in expected_tags)
    if variant=='candidate':
        captured_monos={int(row['attrs']['mono'])for row in markers if row['event']=='gdn_prepared_call_end'}
        observed=[row for row in sidecars if row['event']=='gdn_prepared_call_end'and row.get('decision')=='replay'and row['mono_ns']in captured_monos]
        if require_replay:assert observed,'No GDN replay observed during raw capture interval'
        if required_replay_tokens is not None:
            required = [x for x in observed if x.get('context_tokens') == required_replay_tokens and x.get('tokens') == required_replay_tokens]
            assert required, 'No replay of required GDN capture size'
            result['required_size_GDN_layer_replays'] = len(required)
        result['measured_GDN_layer_replays']=len(observed)
    result.update(event_kinds=sorted(kinds),source='raw NVTX for server markers; existing worker/client JSON for additional checks',performance_use_prohibited=True)
    return result
