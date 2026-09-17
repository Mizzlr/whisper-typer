#!/usr/bin/python3
"""Replay through the native recorder with deterministic local ASR/punctuation."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import threading
import unittest
import wave


class RecorderTests(unittest.TestCase):
    def test_restored_transcripts_share_filters_without_rewriting_raw_history(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);(base/'voice-journal').mkdir()
            (base/'voice-journal/hallucinations.txt').write_text(r'\bthank you\b'+'\n')
            data=[{'id':'stored','chunks':[{'seq':1,'text':'Okay.'},{'seq':2,'text':'Thank you.'},
                  {'seq':3,'text':'Okay, we can review the release tomorrow.'}]}]
            binary=Path(os.environ.get('WHISPER_TEST_RECORDER','target/release/voice-journal')).resolve()
            result=subprocess.run([str(binary),'--filter-recordings'],input=json.dumps(data),capture_output=True,
                                  text=True,timeout=5,check=True,env=dict(os.environ,HOME=str(base)))
            self.assertEqual(json.loads(result.stdout),{'stored':[1,2]})
            self.assertEqual(list(base.rglob('*.jsonl')),[])

    def test_chunks_join_in_order_and_successful_audio_is_removed(self):
        self.replay(False)

    def test_failed_transcriptions_keep_private_audio_and_report_failure(self):
        self.replay(True)

    def test_recording_uses_shared_journal_filters(self):
        self.replay(False,True)

    def test_unknown_markers_are_cleaned_before_punctuation_and_percentages_survive(self):
        self.replay(False,markers=True)

    def replay(self,fail,filter_texts=False,markers=False):
        count=[];punctuation_inputs=[]
        class Handler(BaseHTTPRequestHandler):
            protocol_version='HTTP/1.1'
            def log_message(self,*_): pass
            def do_POST(self):
                body=self.rfile.read(int(self.headers['Content-Length']))
                if self.path=='/transcribe':
                    data=body[body.index(b'RIFF'):]
                    with wave.open(io.BytesIO(data)) as wav:
                        assert wav.getframerate()==16000 and wav.getnchannels()==1
                        count.append(wav.getnframes())
                    code=500 if fail else 200
                    payload={'text':'First sentence' if len(count)==1 else 'Final tail'}
                    if filter_texts:payload={'text':'Thank you' if len(count)==1 else 'Okay'}
                    if markers:payload={'text':'First <Unk>sentence 100%' if len(count)==1 else 'Final<UNK>tail 12.5%'}
                else:
                    code=200;payload={'text':json.loads(body)['text']+'.'}
                    punctuation_inputs.append(json.loads(body)['text'])
                    if markers:payload={'text':'First <Unk>sentence 100%.' if len(count)==1 else 'Final tail 12.5.'}
                encoded=json.dumps(payload).encode()
                self.send_response(code);self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(encoded)));self.end_headers();self.wfile.write(encoded)
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);server.daemon_threads=True
        threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.05},daemon=True).start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                p=Path(directory);audio=p/'speech.wav';output=p/'session.jsonl';config=p/'config.yaml'
                (p/'voice-journal').mkdir()
                (p/'voice-journal/hallucinations.txt').write_text(r'\bthank you\b'+'\n')
                with wave.open(str(audio),'wb') as wav:
                    wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(16000)
                    samples=[0]*16000+[3200]*16000+[0]*16000+[3200]*(16000+123)
                    wav.writeframes(struct.pack('<'+'h'*len(samples),*samples))
                config.write_text(json.dumps({'remote_asr':{'enabled':True,'url':f'http://127.0.0.1:{server.server_port}/transcribe',
                                                           'timeout_seconds':2,'fallback_local':False},
                    'punctuation':{'enabled':True,'url':f'http://127.0.0.1:{server.server_port}/punctuate'},
                    'corrections':{'enabled':False},'spelling':{'enabled':False},'ollama':{'enabled':False}}))
                binary=Path(os.environ.get('WHISPER_TEST_RECORDER','target/release/voice-journal')).resolve()
                result=subprocess.run([str(binary),'--ui-session','native-test','--config',str(config),
                                       '--output',str(output),'--input-wav',str(audio),'--journal-dir',str(p/'journal')],
                                      capture_output=True,text=True,timeout=10,env=dict(os.environ,HOME=str(p),WHISPER_VOICE_JOURNAL_VAD='rms'))
                self.assertEqual(result.returncode,0)
                events=[json.loads(line) for line in result.stdout.splitlines()]
                persisted=[e for e in events if e['payload']['status']!='activity']
                self.assertEqual(persisted,[json.loads(line) for line in output.read_text().splitlines()])
                self.assertEqual(events[0]['payload']['status'],'started')
                self.assertEqual(events[-1]['payload']['status'],'stopped')
                self.assertEqual(len(count),2);self.assertEqual(output.stat().st_mode&0o777,0o600)
                if fail:
                    errors=[e for e in events if e['payload']['status']=='error'];self.assertEqual(len(errors),2)
                    saved=list(p.glob('session.*.wav'));self.assertEqual(len(saved),2)
                    self.assertTrue(all(w.stat().st_mode&0o777==0o600 for w in saved))
                elif filter_texts:
                    self.assertEqual([e['payload']['status'] for e in persisted].count('filtered'),2)
                    self.assertFalse(any(e['payload']['status']=='chunk' for e in persisted))
                    journal=next((p/'journal').glob('journal_????-??-??.md')).read_text()
                    self.assertNotIn('Thank you',journal);self.assertNotIn('Okay',journal)
                    raw=next((p/'journal').glob('*.unfiltered.md')).read_text()
                    self.assertIn('Thank you',raw);self.assertIn('Okay',raw)
                elif markers:
                    chunks=[e['payload'] for e in events if e['payload']['status']=='chunk']
                    self.assertEqual(punctuation_inputs,['First sentence 100%','Final tail 12.5%'])
                    self.assertEqual([c['text'] for c in chunks],punctuation_inputs)
                    self.assertEqual([c['raw_text'] for c in chunks],['First <Unk>sentence 100%','Final<UNK>tail 12.5%'])
                    self.assertTrue(all(c['journal_saved'] for c in chunks))
                else:
                    chunks=[e['payload'] for e in events if e['payload']['status']=='chunk']
                    self.assertEqual([e['seq'] for e in chunks],[1,2])
                    self.assertEqual(' '.join(e['text'] for e in chunks),'First sentence. Final tail.')
                    self.assertEqual(list(p.glob('session.*.wav')),[])
                    journal=next((p/'journal').glob('journal_????-??-??.md')).read_text()
                    self.assertTrue(all(e['payload']['journal_saved'] for e in persisted))
                    self.assertEqual([e['end_reason'] for e in chunks],['pause','stop'])
                    self.assertTrue(all(f.stat().st_mode&0o777==0o600 for f in (p/'journal').glob('*.md')))
                    self.assertIn('[recording native-test #1] First sentence.',journal)
                    self.assertIn('[recording native-test #2] Final tail.',journal)
                    self.assertEqual(journal.count('First sentence.'),1)
        finally: server.shutdown();server.server_close()


if __name__=='__main__': unittest.main()
