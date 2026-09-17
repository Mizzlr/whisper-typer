#!/usr/bin/python3
"""Compare synthetic UI workloads on Xvfb; never opens desktop test windows."""
import argparse
from datetime import datetime,timedelta
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import tkinter as tk


def median_run(action,runs):
    values=[]
    for _ in range(runs):
        started=time.perf_counter();action();values.append((time.perf_counter()-started)*1000)
    return round(statistics.median(values),3)


def measure(module,records,runs,reviewed,now):
    gc.collect()
    with tempfile.TemporaryDirectory() as directory:
        base=Path(directory);root=tk.Tk();root.geometry('680x650')
        store=module.DictationStore(base,base/'reviews.jsonl')
        for index in range(records):
            stamp=(now-timedelta(seconds=index*2)).isoformat()
            store.records[stamp]={'timestamp':stamp,'whisper_text':str(index),'final_text':f'The meeting note for item {index} is ready.'}
            if reviewed and index%5==0:
                store.corrections[stamp]={'pasted':store.records[stamp]['final_text'],'corrected':f'The meeting notes for item {index} are ready.','status':'changed','grammar_latency_ms':200}
        app=module.DictationWindow(root,store,base/'settings.json')
        try:
            app.window_end=100;app.render();root.update()
            canonical=app.collect_rows()
            digest=hashlib.sha256(json.dumps(canonical,sort_keys=True).encode()).hexdigest()
            for _ in range(3):app.collect_rows()
            collect=median_run(app.collect_rows,runs)
            newest=now.isoformat();revision=0
            def update():
                nonlocal revision
                revision+=1
                store.corrections[newest]={'pasted':store.records[newest]['final_text'],'corrected':f'The meeting notes for revision {revision} are ready.','status':'changed'}
                app.render();root.update()
            push=median_run(update,runs)
            idle=median_run(lambda:(app.poll(),root.update()),5)
            def resize():
                root.geometry('1080x1920');root.update()
                root.geometry('680x650');root.update()
            geometry=median_run(resize,5)
            assert len(app.row_blocks)<=app.MAX_RENDERED
            return {'collect_median_ms':collect,'changed_card_median_ms':push,'idle_poll_median_ms':idle,
                    'fullscreen_restore_median_ms':geometry,'rows_sha256':digest,'mounted_cards':len(app.row_blocks)}
        finally:app.close()
    gc.collect()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-ref',default='7869b33')
    parser.add_argument('--records',type=int,default=5000)
    parser.add_argument('--runs',type=int,default=20)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.records<100 or args.runs<1:parser.error('Use at least 100 records and one run')
    if os.environ.get('WHISPER_BENCH_HEADLESS')!='1':
        env=dict(os.environ,WHISPER_BENCH_HEADLESS='1')
        return subprocess.run(['xvfb-run','-a',sys.executable,__file__,*sys.argv[1:]],env=env).returncode
    source=Path(__file__).with_name('dictation-window.py')
    report={'records':args.records,'runs':args.runs,'baseline_ref':args.baseline_ref,'workloads':{}}
    now=datetime.now().astimezone()
    with tempfile.TemporaryDirectory() as directory:
        previous=Path(directory)/'baseline.py'
        previous.write_bytes(subprocess.check_output(['git','show',args.baseline_ref+':infra/dictation-window.py'],cwd=source.parent.parent))
        for label,path in (('before',previous),('after',source)):
            spec=importlib.util.spec_from_file_location('benchmark_'+label,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            for reviewed in (False,True):
                name='reviewed' if reviewed else 'plain'
                report['workloads'].setdefault(name,{})[label]=measure(module,args.records,args.runs,reviewed,now)
    for workload in report['workloads'].values():
        assert workload['before']['rows_sha256']==workload['after']['rows_sha256'],'UI row semantics changed'
    if args.output:
        args.output.write_text(json.dumps(report,indent=2)+'\n');args.output.chmod(0o600)
    print(json.dumps(report,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
