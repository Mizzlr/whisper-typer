"""Hands-free sessions: one native journal capture process, one aggregate card."""
from datetime import datetime
import json
import os
import queue
from pathlib import Path
import subprocess
import threading
import uuid


class RecordingSessions:
    def __init__(self, directory, binary, config, enqueue):
        self.directory, self.binary, self.config = Path(directory), Path(binary), Path(config)
        self.enqueue = enqueue
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.directory, 0o700)
        self.sessions = {}
        self.process = None
        self.active_id = None
        self.state = 'idle'
        self.closed = False
        self.title_jobs=queue.Queue(maxsize=20)
        self.title_stop=threading.Event()
        self.title_thread=None
        self.load()

    def load(self):
        for path in sorted(self.directory.glob('*.jsonl'), reverse=True)[:50]:
            try:
                for line in path.read_text().splitlines():
                    try:
                        event = json.loads(line)
                        if event.get('type') == 'recording': self.receive(event['payload'])
                    except (ValueError, KeyError, TypeError): continue
            except OSError: continue
        for session in self.sessions.values():
            if session['status'] in ('started','chunk','finishing'):
                session['status'] = 'interrupted'
            try:
                summary=json.loads((self.directory/'summaries'/(session['id']+'.json')).read_text())
                session.update(summary=summary['text'],summary_seq=summary['through_seq'],summary_title=summary.get('title',''))
            except (OSError,ValueError,KeyError): pass
            if not session.get('summary_title'):
                try:
                    saved=json.loads((self.directory/'titles'/(session['id']+'.json')).read_text())
                    session['summary_title']=saved['title']
                except (OSError,ValueError,KeyError):pass
        if self.sessions and self.binary.is_file():
            try:
                data=[{'id':session['id'],'chunks':list(session['chunks'].values())} for session in self.sessions.values()]
                result=subprocess.run([str(self.binary),'--filter-recordings'],input=json.dumps(data),
                                      capture_output=True,text=True,timeout=5,check=True)
                for identity,seqs in json.loads(result.stdout).items():
                    for seq in seqs:self.sessions[identity]['chunks'].pop(seq,None)
            except (OSError,ValueError,KeyError,AttributeError,subprocess.SubprocessError): pass

    def receive(self, payload):
        identity = payload['session_id']
        session = self.sessions.setdefault(identity, {'id':identity, 'chunks':{}, 'timestamp':payload['timestamp'],
                                                     'status':'starting', 'error':None})
        session.setdefault('started_at', session['timestamp'])
        status = payload['status']
        if status in ('title_ready','title_error'):
            session['title_busy']=False
            if status=='title_ready' and not session.get('summary_title'):session['summary_title']=payload['title']
            return
        if status == 'activity':
            session.setdefault('activity',{}).update(payload.get('activity',{}))
            return
        if status == 'filtered':
            session['chunks'].pop(payload.get('seq'),None)
            session['filtered_count']=session.get('filtered_count',0)+1
            return
        if status.startswith('summary_'):
            session['summary_busy']=False
            if status=='summary_ready':
                session.update(summary=payload['text'],summary_seq=payload['through_seq'],summary_title=payload.get('title') or session.get('summary_title',''),summary_error=None)
            else: session['summary_error']='Summary unavailable; transcript retained'
            return
        session['timestamp'] = payload['timestamp']
        if payload.get('journal_saved') is False:
            session['error'] = 'Daily Voice Journal unavailable; session text retained'
        if status == 'chunk':
            seq, text = payload.get('seq'), payload.get('text')
            if isinstance(seq, int) and isinstance(text, str):
                session['chunks'][seq] = {'seq':seq, 'timestamp':payload.get('speech_ended_at',payload['timestamp']),
                                         'text':text.strip(), 'end_reason':payload.get('end_reason')}
            # Chunks may complete after Stop; keep the finishing state.
            if session['status'] != 'finishing': session['status'] = 'chunk'
        elif status == 'error': session['error'] = payload.get('message','Recording failed')
        else: session['status'] = status
        if identity == self.active_id:
            if status == 'stopped': self.state = 'idle'; self.active_id = None
            elif status == 'finishing': self.state = 'finishing'
            elif status == 'started' and self.state != 'finishing': self.state = 'recording'

    def start(self):
        if self.state != 'idle' or (self.process and self.process.poll() is None): return
        identity = uuid.uuid4().hex
        stamp = datetime.now().astimezone().isoformat()
        self.sessions[identity] = {'id':identity,'chunks':{},'timestamp':stamp,'status':'starting','error':None}
        self.active_id, self.state = identity, 'starting'
        output = self.directory / (datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+identity+'.jsonl')
        try:
            process = subprocess.Popen([str(self.binary),'--ui-session',identity,'--config',str(self.config),
                                        '--output',str(output)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.DEVNULL, text=True, encoding='utf-8', bufsize=1)
        except OSError:
            self.sessions[identity]['error'] = 'Recorder could not start'
            self.sessions[identity]['status'] = 'stopped'
            self.state, self.active_id = 'idle', None
            return
        self.process = process
        enqueue = self.enqueue  # Reader thread never holds or calls a Tk widget.
        def read():
            try:
                for line in process.stdout:
                    try:
                        data = json.loads(line)
                        if data.get('type') == 'recording': enqueue(data)
                    except (ValueError, TypeError): continue
            finally:
                code = process.wait()
                process.stdout.close()
                if process.stdin and not process.stdin.closed: process.stdin.close()
                if code:
                    enqueue({'type':'recording','payload':{'session_id':identity,'timestamp':datetime.now().astimezone().isoformat(),
                                                          'status':'error','message':'Recorder exited unexpectedly'}})
                # Also restores controls if startup failed before writing an event.
                enqueue({'type':'recording','payload':{'session_id':identity,'timestamp':datetime.now().astimezone().isoformat(),
                                                      'status':'stopped'}})
        threading.Thread(target=read, name='journal-session-events', daemon=True).start()

    def stop(self):
        if self.state not in ('starting','recording'): return
        self.state = 'finishing'
        if self.active_id: self.sessions[self.active_id]['status'] = 'finishing'
        if self.process and self.process.stdin and not self.process.stdin.closed:
            try: self.process.stdin.write('stop\n'); self.process.stdin.flush()
            except (BrokenPipeError, OSError): pass

    def rows(self):
        rows = []
        for session in self.sessions.values():
            segments = [session['chunks'][seq].copy() for seq in sorted(session['chunks']) if session['chunks'][seq]['text'].strip()]
            text = ' '.join(segment['text'] for segment in segments)
            status = {'starting':'Starting…','started':'Recording','chunk':'Recording','finishing':'Finishing…',
                      'stopped':'Recorded','interrupted':'Interrupted'}.get(session['status'],'Recorded')
            if session['error']: status = session['error']
            activity=session.get('activity',{})
            if session['status'] in ('started','chunk'):
                state=activity.get('state')
                status = f"Silence {activity.get('silence_ms',0):.0f} ms" if state=='silence' else 'Speaking' if state=='speaking' else 'Listening'
                if activity.get('transcribing'): status += ' · Transcribing'
            empty = not text.strip() and session['status'] not in ('starting','started','chunk','finishing')
            if empty:status='Empty recording'+(' · '+session['error'] if session['error'] else '')
            rows.append({'key':'recording:'+session['id'],'kind':'recording',
                         'timestamp':session.get('started_at',session['timestamp']), 'sort_timestamp':session['timestamp'],
                         'segments':segments, 'recording_active':session['status'] in ('starting','started','chunk','finishing'),
                         'summary':session.get('summary',''), 'summary_seq':session.get('summary_seq'),
                         'summary_title':session.get('summary_title',''),
                         'title_busy':session.get('title_busy',False),
                         'summary_busy':session.get('summary_busy',False), 'summary_error':session.get('summary_error'),
                         'activity':activity.copy(),
                         'original':text,'corrected':text,'status':status,'recording_empty':empty,'timings':'','paste_ms':None,'grammar_ms':None,
                         'reason':None,'recording_state':session['status'],'recording_error':session['error']})
        return sorted(rows, key=lambda row:row['timestamp'], reverse=True)

    def request_title(self,identity):
        session=self.sessions[identity]
        if self.closed or session.get('summary_title') or session.get('title_attempted') or session.get('summary_busy') or not self.config.is_file():return False
        if session['status'] in ('starting','started','chunk','finishing'):return False
        text=session.get('summary') or '\n\n'.join(session['chunks'][seq]['text'] for seq in sorted(session['chunks']))
        if not text.strip():return False
        if not identity or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in identity):return False
        try:self.title_jobs.put_nowait((identity,text))
        except queue.Full:return False
        session.update(title_busy=True,title_attempted=True)
        if self.title_thread is None:
            self.title_thread=threading.Thread(target=self.title_worker,args=(self.title_jobs,self.title_stop,self.config,self.directory,self.enqueue),name='recording-titles',daemon=True)
            self.title_thread.start()
        return True

    @staticmethod
    def title_worker(jobs,stop,config,directory,enqueue):
        from recording_summary import model_settings,generate_title
        while not stop.is_set():
            try:identity,text=jobs.get(timeout=.2)
            except queue.Empty:continue
            payload={'session_id':identity,'timestamp':datetime.now().astimezone().isoformat()}
            try:
                host,model=model_settings(config);title=generate_title(text,host,model)
                folder=directory/'titles';folder.mkdir(mode=0o700,exist_ok=True)
                target=folder/(identity+'.json');temp=target.with_suffix('.new')
                fd=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
                with os.fdopen(fd,'w') as out:json.dump({'title':title,'model':model},out)
                os.replace(temp,target)
                payload.update(status='title_ready',title=title)
            except Exception:payload['status']='title_error'
            if not stop.is_set():enqueue({'type':'recording','payload':payload})

    def summarize(self, identity):
        from recording_summary import model_settings, summarize, split_summary
        session=self.sessions[identity]
        if session.get('summary_busy') or not session['chunks']: return
        segments=[session['chunks'][seq].copy() for seq in sorted(session['chunks'])]
        text='\n\n'.join(segment['text'] for segment in segments)
        through_seq=segments[-1]['seq']
        session.update(summary_busy=True,summary_error=None)
        enqueue=self.enqueue;config=self.config;directory=self.directory
        def worker():
            payload={'session_id':identity,'timestamp':datetime.now().astimezone().isoformat()}
            try:
                host,model=model_settings(config)
                title,result=split_summary(summarize(text,host,model))
                folder=directory/'summaries';folder.mkdir(mode=0o700,exist_ok=True)
                # IDs originate in session filenames; refuse path components.
                if not identity or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in identity):
                    raise ValueError('Invalid session identity')
                target=folder/(identity+'.json');temp=target.with_suffix('.new')
                fd=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
                with os.fdopen(fd,'w') as out: json.dump({'text':result,'title':title,'through_seq':through_seq,'model':model},out)
                os.replace(temp,target)
                payload.update(status='summary_ready',text=result,title=title,through_seq=through_seq)
            except Exception:
                payload['status']='summary_error'
            enqueue({'type':'recording','payload':payload})
        threading.Thread(target=worker,name='meeting-summary',daemon=True).start()

    def close(self):
        self.stop()
        self.closed = True
        self.title_stop.set()
        # EOF also stops native capture. Its worker finishes and persists pending chunks.
        if self.process and self.process.stdin and not self.process.stdin.closed:
            try: self.process.stdin.close()
            except OSError: pass
