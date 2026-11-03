# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Bounded diagnostic-only event buffer. Never records prompts or token text."""
import atexit
from contextvars import ContextVar
import json
import os
from pathlib import Path
import sys
import threading
import time

TAG=ContextVar('study_request_tag',default=None)
DIRECTORY=Path(os.environ.get('STUDY_RUN_DIR','/nonexistent'))
ROWS=[]
OVERFLOW=False
LIMIT=200000
COUNTS={}


def emit(event,**attributes):
    global OVERFLOW
    if len(ROWS)>=LIMIT:
        OVERFLOW=True
        return
    now=time.perf_counter_ns()
    row={'event':event,'mono_ns':now,'unix_ns':time.time_ns(),'pid':os.getpid(),
         'tid':threading.get_native_id(),**attributes}
    module=sys.modules.get('torch')
    if module is not None:
        label='study|'+event+'|mono='+str(now)+'|tag='+str(attributes.get('tag',''))+'|req='+str(attributes.get('req',''))+'|seq='+str(attributes.get('seq',''))+'|layer='+str(attributes.get('layer',''))+'|ctx='+str(attributes.get('contexts',''))+'|tok='+str(attributes.get('tokens',''))+'|iter='+str(attributes.get('iteration',''))
        try:module.cuda.nvtx.mark(label)
        except RuntimeError as exc:row['nvtx_error']=str(type(exc).__name__)
    ROWS.append(row)


def next_sequence(namespace,key):
    pair=(namespace,str(key));COUNTS[pair]=COUNTS.get(pair,0)+1
    return COUNTS[pair]


def flush():
    if not DIRECTORY.is_dir():return
    path=DIRECTORY/('pipeline.'+str(os.getpid())+'.json')
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps({'pid':os.getpid(),'overflow':OVERFLOW,'events':list(ROWS)},separators=(',',':'))+'\n')
    temporary.replace(path)

atexit.register(flush)


def begin_span(name,**attributes):
    emit(name+'_begin',**attributes)
    module=sys.modules.get('torch')
    if module is not None:
        module.cuda.nvtx.range_push('study_span|'+name+'|layer='+str(attributes.get('layer',''))+'|ctx='+str(attributes.get('contexts',''))+'|tok='+str(attributes.get('tokens','')))

def end_span(name,**attributes):
    module=sys.modules.get('torch')
    if module is not None:module.cuda.nvtx.range_pop()
    emit(name+'_end',**attributes)
