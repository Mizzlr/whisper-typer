#!/usr/bin/python3
"""Push ordering, persistent HTTP, and aggregate sessions on a virtual display."""
import http.client
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import tkinter as tk
import unittest
from datetime import datetime
from ui_events import UiEventBridge
from recording_session import RecordingSessions

spec=importlib.util.spec_from_file_location('window',Path(__file__).with_name('dictation-window.py'))
window=importlib.util.module_from_spec(spec); spec.loader.exec_module(window)


class PushTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.path=Path(self.temp.name)
        self.root=tk.Tk(); self.root.withdraw()
        self.app=window.DictationWindow(self.root,window.DictationStore(self.path,self.path/'reviews.jsonl'),self.path/'settings.json')
        self.bridge=UiEventBridge(self.root,self.app.receive_events,0)
        self.app.event_bridge=self.bridge
        self.client=http.client.HTTPConnection('127.0.0.1',self.bridge.server.server_port,timeout=2)

    def tearDown(self):
        self.client.close(); self.app.close(); self.temp.cleanup()
        # Reclaim destroyed Tk interpreters on their owning thread before the
        # next test starts workers that can trigger cyclic collection.
        self.client=self.bridge=self.app=self.root=None
        __import__('gc').collect()

    def send(self,event,origin=None):
        headers={'Content-Type':'application/json'}
        if origin: headers['Origin']=origin
        self.client.request('POST','/events',json.dumps(event),headers)
        response=self.client.getresponse(); response.read(); return response.status

    def test_push_merges_review_even_when_it_arrives_before_record(self):
        stamp=datetime.now().astimezone().isoformat()
        review={'type':'grammar_review','payload':{'dictation_timestamp':stamp,'pasted':'She have files.',
                  'corrected':'She has files.','status':'changed','grammar_latency_ms':306}}
        record={'type':'dictation','payload':{'timestamp':stamp,'whisper_text':'she have files',
                  'final_text':'She have files.','total_latency_ms':67,'background_review':True}}
        self.assertEqual(self.send(review),202); sock=self.client.sock
        self.assertEqual(self.send(record),202); self.assertIs(sock,self.client.sock)
        self.root.update()
        row=self.app.collect_rows()[0]
        self.assertEqual((row['original'],row['corrected']),('She have files.','She has files.'))
        self.assertEqual((row['number'],row['paste_ms'],row['grammar_ms']),(1,67,306))
        self.assertIn('− She have files.',self.app.text.get('1.0','end'))
        self.assertFalse(list(self.path.glob('*.jsonl'))) # no polling/file input
        self.client.request('GET','/health'); response=self.client.getresponse(); health=json.loads(response.read())
        self.assertEqual(health['counts']['grammar_review'],1)
        self.assertLess(health['last_dispatch_ms'],200)

    def test_bad_payload_and_browser_origin_are_rejected(self):
        self.assertEqual(self.send({'type':'anything','payload':{}}),400)
        self.client.close()
        self.assertEqual(self.send({},'http://example.com'),403)
        self.assertEqual(self.app.store.records,{})

    def test_overview_window_declines_workspace_focus_request_and_copy_still_works(self):
        from Xlib import X,display,protocol
        d=display.Display()
        self.root.deiconify();self.root.update()
        wrapper=d.create_resource_object('window',self.root.winfo_id()).query_tree().parent
        self.assertEqual(wrapper.get_wm_hints().input,0)
        self.assertEqual(self.root.wm_attributes('-type'),('normal',))
        other=d.screen().root.create_window(0,0,200,100,0,d.screen().root_depth,X.InputOutput,X.CopyFromParent)
        other.map();other.set_input_focus(X.RevertToParent,X.CurrentTime);d.sync()
        event=protocol.event.ClientMessage(window=wrapper,client_type=d.intern_atom('WM_PROTOCOLS'),
                                          data=(32,[d.intern_atom('WM_TAKE_FOCUS'),X.CurrentTime,0,0,0]))
        wrapper.send_event(event);d.sync();self.root.update()
        self.assertEqual(d.get_input_focus().focus.id,other.id)
        stamp=datetime.now().astimezone().isoformat()
        self.app.store.records[stamp]={'timestamp':stamp,'whisper_text':'Copy works.','final_text':'Copy works.'}
        self.app.render();self.root.update()
        self.app.copy_buttons[stamp].invoke()
        self.assertEqual(self.root.clipboard_get(),'Copy works.')
        self.assertEqual(self.app.topmost_button.winfo_rootx()+self.app.topmost_button.winfo_width(),
                         self.app.copy_buttons[stamp].winfo_rootx()+self.app.copy_buttons[stamp].winfo_width())
        self.assertEqual(d.get_input_focus().focus.id,other.id)
        other.destroy();d.close()

    def test_show_endpoint_reveals_existing_window_without_creating_another(self):
        self.assertEqual(self.root.state(),'withdrawn')
        self.client.request('POST','/show','{}',{'Content-Type':'application/json'})
        response=self.client.getresponse();self.assertEqual(response.status,202);response.read()
        self.root.update()
        self.assertEqual(self.root.state(),'normal')
        self.assertEqual(self.root.wm_attributes('-type'),('normal',))

    def test_only_changed_card_is_rebuilt_and_new_rows_keep_existing_widgets(self):
        from datetime import timedelta
        now=datetime.now().astimezone()
        stamps=[(now-timedelta(seconds=i*3)).isoformat() for i in range(20)]
        for i,stamp in enumerate(stamps):
            self.app.store.records[stamp]={'timestamp':stamp,'whisper_text':f'Sentence {i}.','final_text':f'Sentence {i}.'}
        self.root.deiconify(); self.root.geometry('660x300'); self.app.render();self.root.update()
        existing=dict(self.app.copy_buttons)
        self.app.text.yview_moveto(.5); self.root.update_idletasks()
        before=self.app.text.get('@0,0','@0,0 lineend')
        self.app.store.corrections[stamps[0]]={'dictation_timestamp':stamps[0],'pasted':'Sentence 0.',
                                             'corrected':'New sentence 0.','status':'changed'}
        self.app.render();self.root.update_idletasks()
        self.assertIs(self.app.copy_buttons[stamps[1]],existing[stamps[1]])
        self.assertIsNot(self.app.copy_buttons[stamps[0]],existing[stamps[0]])
        self.assertEqual(self.app.text.get('@0,0','@0,0 lineend'),before)
        newer=(now+timedelta(seconds=1)).isoformat()
        self.app.store.records[newer]={'timestamp':newer,'whisper_text':'Latest.','final_text':'Latest.'}
        self.app.render();self.root.update_idletasks()
        self.assertIs(self.app.copy_buttons[stamps[1]],existing[stamps[1]])
        self.assertEqual(len(self.app.row_headers),42)
        self.assertEqual(self.app.text.get('1.0','end').count('Latest.'),1)
        self.assertEqual(self.app.text.get('@0,0','@0,0 lineend'),before)
        del self.app.store.records[stamps[-1]];self.app.render()
        self.assertEqual(len(self.app.row_headers),40)

    def test_recording_chunks_share_one_card_and_copy_button(self):
        self.app.recordings=RecordingSessions(self.path/'recordings',self.path/'unused',self.path/'config',self.bridge.enqueue)
        stamp=datetime.now().astimezone().isoformat()
        for status,details in [('started',{}),('chunk',{'seq':1,'text':'First sentence.'}),
                               ('chunk',{'seq':2,'text':'Second sentence.'}),('chunk',{'seq':2,'text':'Second sentence.'}),('stopped',{})]:
            self.assertTrue(self.bridge.enqueue({'type':'recording','payload':{'session_id':'test',
                                   'timestamp':stamp,'status':status,**details}}))
        self.root.update()
        self.assertEqual(len(self.app.rows),1); self.assertEqual(len(self.app.copy_buttons),1)
        row=self.app.rows[0]; self.assertEqual(row['corrected'],'First sentence. Second sentence.')
        self.app.copy_buttons[row['key']].invoke()
        self.assertEqual(self.root.clipboard_get(),'First sentence. Second sentence.')

    def test_recording_timestamps_append_without_rebuilding_earlier_segments(self):
        from datetime import timedelta
        sessions=RecordingSessions(self.path/'recordings',self.path/'unused',self.path/'config',self.bridge.enqueue)
        self.app.recordings=sessions
        start=datetime.now().astimezone()-timedelta(minutes=4)
        def send(status,offset=0,**details):
            self.bridge.enqueue({'type':'recording','payload':{'session_id':'meeting',
                'timestamp':(start+timedelta(seconds=offset)).isoformat(),'status':status,**details}})
            self.root.update()
        send('started')
        key='recording:meeting';button=self.app.copy_buttons[key]
        send('chunk',10,seq=1,text='First sentence.')
        first=self.app.row_blocks[key]['view']
        self.assertIn('First sentence.',first.text.get('1.0','end'))
        self.assertEqual(len(self.app.row_blocks[key]['segments']),1)
        send('chunk',130,seq=2,text='Second sentence.')
        self.assertIs(self.app.copy_buttons[key],button)
        self.assertIn(first,self.app.row_blocks[key]['headers'])
        self.assertEqual([segment['timestamp'] for segment in self.app.rows[0]['segments']],
                         [(start+timedelta(seconds=n)).isoformat() for n in (10,130)])
        self.assertEqual(self.app.rows[0]['timestamp'],start.isoformat())
        self.assertEqual(len(self.app.copy_buttons),1)
        button.invoke();self.assertEqual(self.root.clipboard_get(),'First sentence. Second sentence.')
        send('chunk',130,seq=2,text='Second sentence.')
        self.assertEqual(len(self.app.row_blocks[key]['segments']),2)
        send('stopped',140)
        self.assertIs(self.app.copy_buttons[key],button)
        self.assertFalse(self.app.rows[0]['recording_active'])
        self.assertIn('Recorded',self.app.row_blocks[key]['metadata'].cget('text'))

    def test_recording_pane_fits_short_text_grows_and_shrinks_for_summary(self):
        sessions=RecordingSessions(self.path/'recordings',self.path/'unused',self.path/'config',self.bridge.enqueue)
        self.app.recordings=sessions;self.root.deiconify();self.root.geometry('680x650')
        stamp=datetime.now().astimezone().isoformat()
        sessions.receive({'session_id':'fit','timestamp':stamp,'status':'started'})
        sessions.receive({'session_id':'fit','timestamp':stamp,'status':'chunk','seq':1,'text':'Hello, hello, hello.'})
        self.app.render();self.root.update();view=self.app.row_blocks['recording:fit']['view']
        short=view.winfo_height();self.assertLess(short,130)
        self.assertEqual(view.text.yview(),(0.0,1.0))
        sessions.receive({'session_id':'fit','timestamp':stamp,'status':'chunk','seq':2,'text':'Meeting discussion. '*300})
        self.app.render();self.root.update();self.assertGreater(view.winfo_height(),short)
        self.assertLessEqual(view.winfo_height(),self.root.winfo_height()//2)
        sessions.receive({'session_id':'fit','timestamp':stamp,'status':'summary_ready','text':'Review the release tomorrow.','title':'Release Review Planning','through_seq':2})
        self.app.render();self.root.update();self.assertLess(view.winfo_height(),130)
        view.transcript_button.invoke();self.root.update();self.assertGreater(view.winfo_height(),short)

    def test_recording_and_summary_wheel_hands_off_at_both_edges(self):
        from datetime import timedelta
        now=datetime.now().astimezone()
        for index in range(80):
            stamp=(now-timedelta(seconds=index+1)).isoformat()
            self.app.store.records[stamp]={'timestamp':stamp,'whisper_text':str(index),'final_text':f'Other dictation {index}.'}
        sessions=RecordingSessions(self.path/'recordings',self.path/'unused',self.path/'config',self.bridge.enqueue)
        self.app.recordings=sessions;stamp=now.isoformat()
        for status,details in [('started',{}),('chunk',{'seq':1,'text':'Meeting notes. '*300}),
                               ('summary_ready',{'text':'Summary discussion. '*300,'title':'Meeting Notes Review','through_seq':1})]:
            sessions.receive({'session_id':'handoff','timestamp':stamp,'status':status,**details})
        self.root.deiconify();self.root.geometry('680x650');self.app.render();self.root.update()
        view=self.app.row_blocks['recording:handoff']['view']
        for mode in ('transcript','summary'):
            with self.subTest(mode=mode):
                view.show(mode);self.root.update()
                self.app.text.yview_moveto(0);view.scroll('moveto',1);self.root.update()
                before=self.app.text.yview()[0];view.text.event_generate('<Button-5>');self.root.update()
                self.assertGreater(self.app.text.yview()[0],before)
                self.assertAlmostEqual(view.text.yview()[1],1)
                self.app.text.yview_moveto(.02);view.scroll('moveto',0);self.root.update()
                before=self.app.text.yview()[0];view.topic.event_generate('<Button-4>');self.root.update()
                self.assertLess(self.app.text.yview()[0],before)
                self.assertAlmostEqual(view.text.yview()[0],0)
                view.scroll('moveto',.5);self.root.update();before=self.app.text.yview()
                view.text.event_generate('<Button-5>');self.root.update()
                self.assertEqual(self.app.text.yview(),before)

    def test_recording_pane_is_bounded_and_scrolls_without_moving_outer_history(self):
        sessions=RecordingSessions(self.path/'recordings',self.path/'unused',self.path/'config',self.bridge.enqueue)
        self.app.recordings=sessions;self.root.deiconify();self.root.geometry('680x470')
        stamp=datetime.now().astimezone().isoformat()
        sessions.receive({'session_id':'scroll','timestamp':stamp,'status':'started'})
        for seq in range(1,401):sessions.receive({'session_id':'scroll','timestamp':stamp,'status':'chunk','seq':seq,'text':f'Segment {seq}. '+('Meeting notes. '*10)})
        self.app.render();self.root.update()
        view=self.app.row_blocks['recording:scroll']['view']
        self.assertLessEqual(view.winfo_height(),self.root.winfo_height()//2)
        self.assertGreater(view.text.yview()[0],.5)
        # A single upward movement near the tail is still deliberate reading;
        # activity updates and new chunks must not snap it back to the bottom.
        tail_position=view.text.index('@0,0')
        view.text.event_generate('<Button-4>');self.root.update()
        near_tail=view.text.index('@0,0')
        self.assertTrue(view.text.compare('@0,0','<',tail_position))
        self.assertGreaterEqual(view.text.yview()[1],.98)
        sessions.receive({'session_id':'scroll','timestamp':stamp,'status':'activity','activity':{'state':'speaking'}})
        self.app.render();self.root.update()
        self.assertEqual(view.text.index('@0,0'),near_tail)
        sessions.receive({'session_id':'scroll','timestamp':stamp,'status':'chunk','seq':401,'text':'New tail segment.'})
        self.app.render();self.root.update()
        self.assertEqual(view.text.index('@0,0'),near_tail)
        # Real X11 wheel input must reach a nested, disabled transcript widget
        # even while another application has keyboard focus.
        from Xlib import X,display
        from Xlib.ext import xtest
        connection=display.Display()
        other=connection.screen().root.create_window(800,600,100,100,0,connection.screen().root_depth,X.InputOutput,X.CopyFromParent)
        other.map();other.set_input_focus(X.RevertToParent,X.CurrentTime);connection.sync()
        view.scroll('moveto',1);self.root.update()
        position=view.text.index('@0,0');outer=self.app.text.yview()
        xtest.fake_input(connection,X.MotionNotify,x=view.text.winfo_rootx()+20,y=view.text.winfo_rooty()+20)
        xtest.fake_input(connection,X.ButtonPress,4);xtest.fake_input(connection,X.ButtonRelease,4)
        connection.sync();self.root.update()
        self.assertTrue(view.text.compare('@0,0','<',position))
        self.assertEqual(self.app.text.yview(),outer)
        other.destroy();connection.close()
        view.text.yview_moveto(.5);self.root.update()
        before=self.app.text.yview();inner_before=view.text.index('@0,0')
        view.text.event_generate('<Button-4>');self.root.update()
        self.assertTrue(view.text.compare('@0,0','<',inner_before))
        inner_before=view.text.index('@0,0')
        view.text.event_generate('<Button-5>');self.root.update()
        self.assertTrue(view.text.compare('@0,0','>',inner_before))
        self.assertEqual(self.app.text.yview(),before)
        view.text.yview_moveto(.2);self.root.update();position=view.text.index('@0,0')
        sessions.receive({'session_id':'scroll','timestamp':stamp,'status':'chunk','seq':402,'text':'Latest segment.'})
        self.app.render();self.root.update()
        self.assertIs(self.app.row_blocks['recording:scroll']['view'],view)
        self.assertEqual(view.text.index('@0,0'),position)

        view.scroll('moveto',1);self.root.update()
        self.assertTrue(view.follow_tail)
        sessions.receive({'session_id':'scroll','timestamp':stamp,'status':'chunk','seq':403,'text':'Follow this latest segment.'})
        self.app.render();self.root.update()
        self.assertGreaterEqual(view.text.yview()[1],1-1e-9)

        sessions.receive({'session_id':'scroll','timestamp':stamp,'status':'stopped'})
        self.app.render();self.root.update()
        self.assertFalse(self.app.rows[0]['recording_active'])
        self.assertIs(self.app.row_blocks['recording:scroll']['view'],view)
        view.text.yview_moveto(.5);self.root.update()
        for event,comparison in [('<Button-4>','<'),('<Button-5>','>')]:
            position=view.text.index('@0,0');outer=self.app.text.yview()
            view.text.event_generate(event);self.root.update()
            self.assertTrue(view.text.compare('@0,0',comparison,position))
            self.assertEqual(self.app.text.yview(),outer)

        def check_card_wheel():
            block=self.app.row_blocks['recording:scroll'];pane=block['view']
            targets=[pane.text,block['metadata'],self.app.copy_buttons['recording:scroll'],
                     *block['actions'].values(),pane.transcript_button]
            for target in targets:
                for event,details,comparison in [('<Button-4>',{},'<'),('<Button-5>',{},'>'),
                                                  ('<MouseWheel>',{'delta':120},'<'),
                                                  ('<MouseWheel>',{'delta':-120},'>')]:
                    with self.subTest(state=self.app.rows[0]['recording_state'],target=target,event=event,details=details):
                        pane.text.yview_moveto(.5);self.root.update()
                        position=pane.text.index('@0,0');outer=self.app.text.yview()
                        target.event_generate(event,**details);self.root.update()
                        self.assertTrue(pane.text.compare('@0,0',comparison,position))
                        self.assertEqual(self.app.text.yview(),outer)
        check_card_wheel()
        sessions.receive({'session_id':'scroll','timestamp':stamp,'status':'started'})
        self.app.render();self.root.update()
        self.assertTrue(self.app.rows[0]['recording_active'])
        check_card_wheel()
        sessions.receive({'session_id':'scroll','timestamp':stamp,'status':'stopped'})
        events=[{'session_id':'scroll','timestamp':stamp,'status':'started'}]
        events.extend({'session_id':'scroll','timestamp':stamp,'status':'chunk',**segment}
                      for segment in sessions.sessions['scroll']['chunks'].values())
        events.append({'session_id':'scroll','timestamp':stamp,'status':'stopped'})
        (sessions.directory/'saved.jsonl').write_text(''.join(json.dumps({'type':'recording','payload':event})+'\n' for event in events))
        self.app.recordings=RecordingSessions(sessions.directory,self.path/'unused',self.path/'config',self.bridge.enqueue)
        self.app.remove_row('recording:scroll');self.app.rendered=None
        self.app.render();self.root.update()
        self.assertIsNot(self.app.row_blocks['recording:scroll']['view'],view)
        self.assertFalse(self.app.rows[0]['recording_active'])
        check_card_wheel()

    def test_theme_switch_preserves_recording_widgets_and_reading_position(self):
        sessions=RecordingSessions(self.path/'recordings',self.path/'unused',self.path/'config',self.bridge.enqueue)
        self.app.recordings=sessions;stamp=datetime.now().astimezone().isoformat()
        sessions.receive({'session_id':'theme','timestamp':stamp,'status':'started'})
        for seq in range(1,41):sessions.receive({'session_id':'theme','timestamp':stamp,'status':'chunk','seq':seq,'text':f'Segment {seq}. '+('Meeting notes. '*10)})
        self.root.deiconify();self.root.geometry('680x470');self.app.render();self.root.update()
        key='recording:theme';pane=self.app.row_blocks[key]['view'];button=self.app.copy_buttons[key]
        pane.scroll('moveto',.2);self.root.update();position=pane.text.index('@0,0');outer=self.app.text.yview()
        self.assertEqual(pane.text.cget('background'),'#fdfbf5')
        self.assertEqual(pane.text.cget('foreground'),'#3a3326')
        for theme in ['dark','light','dark','light']:
            self.app.theme_button.invoke();self.root.update()
            self.assertEqual(self.app.theme,theme)
            self.assertIs(self.app.row_blocks[key]['view'],pane)
            self.assertIs(self.app.copy_buttons[key],button)
            self.assertEqual(pane.text.index('@0,0'),position)
            self.assertEqual(self.app.text.yview(),outer)
            self.assertFalse(pane.follow_tail)
            self.assertTrue(self.app.rows[0]['recording_active'])
            self.assertEqual(json.loads(self.app.settings_path.read_text())['theme'],theme)
        self.assertEqual(pane.text.cget('background'),'#fdfbf5')
        self.assertEqual(pane.text.cget('foreground'),'#3a3326')
        button.invoke();self.assertIn('Segment 40.',self.root.clipboard_get())

    def test_completed_recording_gets_cached_topic_without_resummarizing(self):
        from unittest.mock import patch
        folder=self.path/'recordings';config=self.path/'title-config';config.write_text('{}')
        sessions=RecordingSessions(folder,self.path/'unused',config,self.bridge.enqueue)
        self.app.recordings=sessions;stamp=datetime.now().astimezone().isoformat()
        events=[{'session_id':'old','timestamp':stamp,'status':'started'},
                {'session_id':'old','timestamp':stamp,'status':'chunk','seq':1,'text':'Review the release tomorrow.'},
                {'session_id':'old','timestamp':stamp,'status':'stopped'}]
        history=folder/'old.jsonl';history.write_text('\n'.join(json.dumps({'type':'recording','payload':event}) for event in events))
        for event in events:sessions.receive(event)
        summaries=folder/'summaries';summaries.mkdir();saved=summaries/'old.json'
        saved.write_text(json.dumps({'text':'Action: review the release tomorrow.','through_seq':1}))
        sessions.load();before=(history.read_bytes(),saved.read_bytes())
        with patch('recording_summary.model_settings',return_value=('http://localhost:1','fixture')),patch('recording_summary.generate_title',return_value='Release Review Planning') as generate:
            self.app.render();self.app.render()
            deadline=time.monotonic()+2
            while not sessions.sessions['old'].get('summary_title') and time.monotonic()<deadline:
                self.root.update();time.sleep(.005)
            self.assertEqual(sessions.rows()[0]['summary_title'],'Release Review Planning')
            generate.assert_called_once_with('Action: review the release tomorrow.','http://localhost:1','fixture')
            self.app.render();generate.assert_called_once()
        view=self.app.row_blocks['recording:old']['view']
        self.assertEqual(view.topic.cget('text'),'Release Review Planning')
        self.assertEqual(view.note.cget('text'),'')
        self.assertEqual((history.read_bytes(),saved.read_bytes()),before)
        title=folder/'titles/old.json';self.assertEqual(title.stat().st_mode&0o777,0o600)
        restored=RecordingSessions(folder,self.path/'unused',config,self.bridge.enqueue)
        self.assertEqual(restored.rows()[0]['summary_title'],'Release Review Planning')
        self.assertFalse(restored.request_title('old'));restored.close()
        sessions.close();sessions.title_thread.join(1);self.assertFalse(sessions.title_thread.is_alive())

    def test_summary_keeps_recording_active_and_is_saved_without_changing_transcript(self):
        from unittest.mock import patch
        sessions=RecordingSessions(self.path/'recordings',self.path/'unused',self.path/'config',self.bridge.enqueue)
        self.app.recordings=sessions;stamp=datetime.now().astimezone().isoformat()
        for status,details in [('started',{}),('chunk',{'seq':1,'text':'Review the release tomorrow.'})]:
            sessions.receive({'session_id':'summary','timestamp':stamp,'status':status,**details})
        self.app.render();button=self.app.copy_buttons['recording:summary']
        with patch('recording_summary.model_settings',return_value=('http://localhost:1','fixture')),patch('recording_summary.summarize',return_value='Title: Release Review Planning\n\nAction: review the release tomorrow.'):
            self.app.row_blocks['recording:summary']['actions']['Summarize'].invoke()
            deadline=time.monotonic()+2
            while sessions.sessions['summary'].get('summary_busy') and time.monotonic()<deadline:
                self.root.update();time.sleep(.005)
        row=sessions.rows()[0]
        self.assertTrue(row['recording_active']);self.assertEqual(row['corrected'],'Review the release tomorrow.')
        self.assertEqual(row['summary'],'Action: review the release tomorrow.')
        self.assertEqual(row['summary_seq'],1)
        self.assertEqual(row['summary_title'],'Release Review Planning')
        self.assertEqual(self.app.row_blocks['recording:summary']['view'].topic.cget('text'),'Release Review Planning')
        path=self.path/'recordings/summaries/summary.json';self.assertEqual(path.stat().st_mode&0o777,0o600)
        self.assertEqual(json.loads(path.read_text())['title'],'Release Review Planning')
        restored=RecordingSessions(self.path/'recordings',self.path/'unused',self.path/'config',self.bridge.enqueue)
        restored.receive({'session_id':'summary','timestamp':stamp,'status':'started'})
        restored.load();self.assertEqual(restored.rows()[0]['summary_title'],'Release Review Planning')
        self.assertIs(button,self.app.copy_buttons['recording:summary'])
        button.invoke();self.assertEqual(self.root.clipboard_get(),'Review the release tomorrow.')
        view=self.app.row_blocks['recording:summary']['view'];self.assertEqual(view.note.cget('text'),'')
        sessions.receive({'session_id':'summary','timestamp':stamp,'status':'chunk','seq':2,'text':'Also review the tests.'})
        self.app.render();self.assertEqual(view.note.cget('text'),'Summary is behind')

    def test_download_saves_whole_transcript_and_summary_failure_keeps_capture_active(self):
        from unittest.mock import patch
        from recording_summary import transcript_export
        sessions=RecordingSessions(self.path/'recordings',self.path/'unused',self.path/'config',self.bridge.enqueue)
        self.app.recordings=sessions;stamp=datetime.now().astimezone().isoformat()
        sessions.receive({'session_id':'download','timestamp':stamp,'status':'started'})
        sessions.receive({'session_id':'download','timestamp':stamp,'status':'chunk','seq':1,'text':'Original  words.'})
        self.app.render();row=self.app.rows[0];target=self.path/'transcript.txt'
        with patch('tkinter.filedialog.asksaveasfilename',return_value=str(target)):
            self.app.row_blocks[row['key']]['actions']['Download'].invoke()
        self.assertEqual(target.read_text(),transcript_export(row));self.assertEqual(target.stat().st_mode&0o777,0o600)
        with patch('recording_summary.model_settings',side_effect=OSError('unavailable')):
            self.app.row_blocks[row['key']]['actions']['Summarize'].invoke()
            deadline=time.monotonic()+2
            while sessions.sessions['download'].get('summary_busy') and time.monotonic()<deadline:
                self.root.update();time.sleep(.005)
        row=sessions.rows()[0]
        self.assertTrue(row['recording_active']);self.assertEqual(row['corrected'],'Original  words.')
        self.assertTrue(row['summary_error']);self.assertFalse(row['summary_busy'])

    def test_active_meeting_remains_visible_through_long_silence(self):
        from datetime import timedelta
        sessions=RecordingSessions(self.path/'recordings',self.path/'unused',self.path/'config',self.bridge.enqueue)
        self.app.recordings=sessions
        old=(datetime.now().astimezone()-timedelta(hours=2)).isoformat()
        sessions.receive({'session_id':'long','timestamp':old,'status':'started'})
        self.app.render();self.root.update()
        self.assertIn('recording:long',self.app.copy_buttons)
        self.assertEqual(len(self.app.row_blocks),1)
        sessions.receive({'session_id':'long','timestamp':datetime.now().astimezone().isoformat(),
                          'status':'stopped'})
        self.app.render();self.root.update()
        self.assertIn('recording:long',self.app.copy_buttons)

    def test_empty_recent_view_populates_recording_history_and_marks_empty_cards(self):
        from datetime import timedelta
        sessions=RecordingSessions(self.path/'recordings',self.path/'unused',self.path/'config',self.bridge.enqueue)
        self.app.recordings=sessions;self.app.show_dictations.set(False)
        now=datetime.now().astimezone()
        for index in range(28):
            stamp=(now-timedelta(hours=2+index)).isoformat();identity=f'fallback-{index}'
            sessions.receive({'session_id':identity,'timestamp':stamp,'status':'started'})
            if index%2:sessions.receive({'session_id':identity,'timestamp':stamp,'status':'chunk','seq':1,'text':f'Meeting {index}.'})
            sessions.receive({'session_id':identity,'timestamp':stamp,'status':'stopped'})
        self.root.deiconify();self.app.render();self.root.update()
        self.assertEqual(len(self.app.row_blocks),20)
        empty=self.app.row_blocks['recording:fallback-0']
        self.assertNotIn('view',empty)
        self.assertTrue(any('Empty recording' in widget.cget('text') for widget in empty['headers'][0].winfo_children() if isinstance(widget,tk.Label)))
        self.assertEqual(self.app.copy_buttons['recording:fallback-0'].cget('state'),'disabled')
        self.app.load_older();self.root.update()
        self.assertEqual(len(self.app.row_blocks),28)
        self.assertIn('recording:fallback-27',self.app.row_blocks)
        self.app.show_recordings.set(False);self.app.change_view();self.root.update()
        self.assertEqual(len(self.app.row_blocks),0)
        self.assertIn('No categories selected.',self.app.text.get('1.0','end'))

    def test_dictation_fallback_collects_twenty_across_days_and_continues_scrolling(self):
        from datetime import timedelta
        now=datetime.now().astimezone()
        for age in range(1,5):
            day=(now-timedelta(days=age)).replace(hour=12,minute=0,second=0,microsecond=0)
            records=[{'timestamp':(day+timedelta(minutes=index)).isoformat(),'whisper_text':str(index),'final_text':f'Older dictation {age}-{index}.'} for index in range(7)]
            (self.path/(day.strftime('%Y-%m-%d')+'.jsonl')).write_text(''.join(json.dumps(record)+'\n' for record in records))
        self.app.show_recordings.set(False);self.app.store.refresh()
        self.root.deiconify();self.app.render();self.root.update()
        self.assertEqual(len(self.app.row_blocks),20)
        self.assertEqual(len(self.app.store.loaded_paths),3)
        self.app.text.yview_moveto(0);self.app.trim_history();self.root.update()
        self.assertEqual(len(self.app.row_blocks),20)
        self.app.load_older();self.root.update();self.app.load_older();self.root.update()
        self.assertEqual(len(self.app.row_blocks),28)
        self.assertEqual(len(self.app.store.loaded_paths),4)
        self.assertLessEqual(len(self.app.row_blocks),100)

    def test_finished_empty_recording_collapses_live_pane_without_hiding_errors(self):
        sessions=RecordingSessions(self.path/'recordings',self.path/'unused',self.path/'config',self.bridge.enqueue)
        self.app.recordings=sessions;stamp=datetime.now().astimezone().isoformat()
        sessions.receive({'session_id':'empty','timestamp':stamp,'status':'started'})
        self.root.deiconify();self.app.render();self.root.update()
        self.assertIn('view',self.app.row_blocks['recording:empty'])
        self.assertFalse(sessions.rows()[0]['recording_empty'])
        sessions.receive({'session_id':'empty','timestamp':stamp,'status':'stopped'})
        self.app.render();self.root.update()
        self.assertNotIn('view',self.app.row_blocks['recording:empty'])
        self.assertEqual(sessions.rows()[0]['status'],'Empty recording')
        sessions.receive({'session_id':'empty','timestamp':stamp,'status':'error','message':'Capture failed'})
        self.assertIn('Capture failed',sessions.rows()[0]['status'])
        self.assertTrue(sessions.rows()[0]['recording_empty'])

    def test_all_category_combinations_include_older_categories_and_deduplicate_selected_origins(self):
        from datetime import timedelta
        from itertools import product
        from clipboard_history import ClipboardStore
        now=datetime.now().astimezone();stamp=(now-timedelta(minutes=40)).isoformat()
        self.app.store.records[stamp]={'timestamp':stamp,'whisper_text':'old dictation','final_text':'Older dictated text.'}
        sessions=RecordingSessions(self.path/'recordings',self.path/'unused',self.path/'config',self.bridge.enqueue)
        old=(now-timedelta(hours=26)).isoformat()
        for status,details in [('started',{}),('chunk',{'seq':1,'text':'Older meeting transcript.'}),('stopped',{})]:
            sessions.receive({'session_id':'older','timestamp':old,'status':status,**details})
        clips=ClipboardStore(self.path/'clips');clips.add('text','Independent clipboard text',now.isoformat())
        clips.add('text','Older dictated text.',now.isoformat())
        clips.add('text','Older meeting transcript.',now.isoformat())
        self.app.recordings=sessions;self.app.clipboard_store=clips;self.app.clipboard_items=clips.rows()
        self.root.deiconify();self.root.geometry('680x600')
        independent='clip:'+next(item['id'] for item in clips.rows() if item['text']=='Independent clipboard text')
        for clipboard,dictations,recordings in product((False,True),repeat=3):
            with self.subTest(clipboard=clipboard,dictations=dictations,recordings=recordings):
                for variable,button,wanted in ((self.app.show_clipboard,self.app.clipboard_button,clipboard),
                                               (self.app.show_dictations,self.app.dictations_button,dictations),
                                               (self.app.show_recordings,self.app.recordings_button,recordings)):
                    if variable.get()!=wanted:button.invoke()
                self.root.update()
                expected=set()
                if clipboard:
                    expected.add(independent)
                if dictations:expected.add(stamp)
                if recordings:expected.add('recording:older')
                self.assertEqual(set(self.app.row_blocks),expected)
                self.assertEqual({row.get('key',row['timestamp']) for row in self.app.collect_rows()},expected)
                saved=json.loads(self.app.settings_path.read_text())
                self.assertEqual((saved['show_clipboard'],saved['show_dictations'],saved['show_recordings']),(clipboard,dictations,recordings))
        all_rows=self.app.collect_rows();self.app.show_clipboard.set(False)
        self.app.render(all_rows);self.root.update()
        self.assertFalse(any(key.startswith('clip:') for key in self.app.row_blocks))
        self.assertEqual(sessions.rows()[0]['corrected'],'Older meeting transcript.')

    def test_clipboard_hides_archived_voice_text_without_loading_dictation_cards(self):
        from datetime import timedelta
        from clipboard_history import ClipboardStore
        now=datetime.now().astimezone();old=(now-timedelta(hours=26)).isoformat()
        archive=self.path/(datetime.fromisoformat(old).strftime('%Y-%m-%d')+'.jsonl')
        archive.write_text(json.dumps({'timestamp':old,'whisper_text':'archived','final_text':'Archived dictated text.'})+'\n')
        clips=ClipboardStore(self.path/'clips');clips.add('text','Archived dictated text.',old)
        clips.add('text','Unrelated clipboard text',now.isoformat())
        self.app.clipboard_store=clips;self.app.clipboard_items=clips.rows()
        self.app.show_clipboard.set(True);self.app.show_dictations.set(False);self.app.show_recordings.set(False)
        self.app.change_view();self.root.update()
        self.assertEqual(len(self.app.row_blocks),1)
        self.assertEqual(self.app.collect_rows()[0]['original'],'Unrelated clipboard text')
        self.assertEqual(self.app.store.records,{})
        self.assertNotIn(archive,self.app.store.loaded_paths)
        self.assertIn(archive,self.app.clipboard_archive_reader.tails)
        with archive.open('a') as file:file.write(json.dumps({'timestamp':old,'whisper_text':'another','final_text':'Another dictated text.'})+'\n')
        clips.add('text','Another dictated text.',old);self.app.clipboard_changed();self.root.update()
        self.assertEqual(len(self.app.row_blocks),1)

    def test_recent_clipboard_does_not_block_dictation_archive_fallback(self):
        from datetime import timedelta
        from clipboard_history import ClipboardStore
        now=datetime.now().astimezone();old=(now-timedelta(days=3)).isoformat()
        (self.path/(now-timedelta(days=3)).strftime('%Y-%m-%d')).with_suffix('.jsonl').write_text(json.dumps({'timestamp':old,'whisper_text':'archive','final_text':'Archived dictation.'})+'\n')
        clips=ClipboardStore(self.path/'clips');clips.add('text','Recent clipboard',now.isoformat())
        self.app.clipboard_store=clips;self.app.clipboard_items=clips.rows()
        self.app.show_clipboard.set(True);self.app.show_recordings.set(False)
        self.app.change_view();self.root.update()
        self.assertIn(old,self.app.row_blocks)
        self.assertEqual(len(self.app.row_blocks),2)
        self.app.dictations_button.invoke();self.root.update()
        self.assertEqual(len(self.app.row_blocks),1)
        self.assertTrue(all(key.startswith('clip:') for key in self.app.row_blocks))

    def test_category_buttons_filter_independently_and_persist_without_stopping_capture(self):
        from datetime import timedelta
        from clipboard_history import ClipboardStore
        from PIL import Image
        import io
        now=datetime.now().astimezone()
        for index in range(2):
            stamp=(now-timedelta(seconds=index)).isoformat()
            self.app.store.records[stamp]={'timestamp':stamp,'whisper_text':str(index),'final_text':f'Dictation {index}.'}
        clips=ClipboardStore(self.path/'clips');clips.add('text','Independent clipboard text')
        png=io.BytesIO();Image.new('RGB',(200,100),'red').save(png,format='PNG');clips.add('image',png.getvalue())
        sessions=RecordingSessions(self.path/'recordings',self.path/'unused',self.path/'config',self.bridge.enqueue)
        sessions.receive({'session_id':'filter','timestamp':now.isoformat(),'status':'started'})
        sessions.receive({'session_id':'filter','timestamp':now.isoformat(),'status':'chunk','seq':1,'text':'Meeting transcript.'})
        self.app.recordings=sessions;self.app.clipboard_store=clips;self.app.clipboard_items=clips.rows();self.app.show_clipboard.set(True)
        self.root.deiconify();self.app.render();self.root.update()
        self.assertEqual(len(self.app.rows),5)
        self.app.dictations_button.invoke();self.root.update();self.assertEqual(len(self.app.rows),3)
        self.app.recordings_button.invoke();self.root.update();self.assertEqual(len(self.app.rows),2)
        self.app.clipboard_button.invoke();self.root.update();self.assertEqual(len(self.app.rows),0)
        self.assertEqual(len(sessions.rows()),1);self.assertTrue(sessions.rows()[0]['recording_active'])
        self.app.dictations_button.invoke();self.root.update();self.assertEqual(len(self.app.rows),2)
        settings=json.loads(self.app.settings_path.read_text())
        self.assertTrue(settings['show_dictations']);self.assertFalse(settings['show_recordings']);self.assertFalse(settings['show_clipboard'])
        other=tk.Tk()
        other.withdraw()
        restored=window.DictationWindow(other,window.DictationStore(self.path/'unused-history',self.path/'unused-reviews'),self.app.settings_path)
        self.assertTrue(restored.show_dictations.get());self.assertFalse(restored.show_recordings.get());restored.close()

    def test_fullscreen_reflows_cards_and_keeps_controls_visible_on_restore(self):
        from datetime import timedelta
        now=datetime.now().astimezone()
        for index in range(150):
            stamp=(now-timedelta(minutes=index)).isoformat()
            self.app.store.records[stamp]={'timestamp':stamp,'whisper_text':str(index),'final_text':('Fullscreen meeting notes '+str(index)+' ')*20}
        self.app.window_end=100;self.root.deiconify();self.app.render();self.root.update()
        buttons=dict(self.app.copy_buttons)
        for width,height in [(1080,1920),(540,430),(1920,1080),(540,430)]:
            self.root.geometry(f'{width}x{height}');self.root.update()
            self.assertFalse(self.app.text.tk.getboolean(self.app.text.tk.call(self.app.text._w,'pendingsync')))
            for key,button in buttons.items():self.assertIs(self.app.copy_buttons[key],button)
            for button in (self.app.record_button,self.app.clipboard_button,self.app.dictations_button,self.app.recordings_button,self.app.topmost_button):
                self.assertTrue(button.winfo_ismapped())
                self.assertLessEqual(button.winfo_rootx()+button.winfo_width(),self.root.winfo_rootx()+width)
            self.assertEqual(self.app.compact_toolbar,width<800)
            self.app.wheel(1);self.root.update()
            self.assertFalse(self.app.scroll_sync_pending)
            self.assertLessEqual(len(self.app.row_blocks),100)

    def test_wheel_burst_pages_once_and_direction_changes_cancel_stale_work(self):
        from datetime import timedelta
        now=datetime.now().astimezone()
        for i in range(800):
            stamp=(now-timedelta(minutes=i)).isoformat()
            self.app.store.records[stamp]={'timestamp':stamp,'whisper_text':str(i),'final_text':('Meeting transcript '+str(i)+' ')*15}
        self.root.deiconify();self.root.geometry('550x430')
        self.app.window_end=100;self.app.render();self.root.update()
        def settle():
            ready=tk.BooleanVar(value=False)
            self.root.after(35,lambda:ready.set(True));self.root.wait_variable(ready)
            self.root.update()
        # A gesture dispatches a burst before any queued page load can run.
        for direction in (1,-1):
            self.app.text.yview_moveto(1 if direction>0 else 0);self.root.update()
            before=self.app.window_end if direction>0 else self.app.window_start
            for _ in range(32):self.app.text.event_generate('<Button-5>' if direction>0 else '<Button-4>')
            settle()
            after=self.app.window_end if direction>0 else self.app.window_start
            self.assertEqual(abs(after-before),self.app.PAGE_SIZE)
            self.assertLessEqual(len(self.app.row_blocks),100)
        self.app.text.yview_moveto(1);self.root.update();before=self.app.window_end
        for _ in range(32):self.app.wheel(1)
        self.app.wheel(-1);settle()
        self.assertEqual(self.app.window_end,before)
        # Scrollbar dragging should also coalesce, including repeated end stops.
        self.app.text.yview_moveto(1);self.root.update()
        for _ in range(32):self.app.scroll('moveto',1)
        settle()
        self.assertEqual(self.app.window_end,before+self.app.PAGE_SIZE)

    def test_returning_to_top_releases_older_cards_without_rebuilding_live_card(self):
        from datetime import timedelta
        now=datetime.now().astimezone()
        for i in range(100):
            stamp=(now-timedelta(minutes=i)).isoformat()
            self.app.store.records[stamp]={'timestamp':stamp,'whisper_text':str(i),'final_text':str(i)}
        self.app.window_end=75
        self.root.deiconify();self.app.render();self.root.update()
        key=now.isoformat();button=self.app.copy_buttons[key]
        self.assertGreater(len(self.app.row_blocks),60)
        self.app.scroll('moveto',0);self.root.update()
        settled=tk.BooleanVar(value=False)
        self.root.after(30,lambda:settled.set(True));self.root.wait_variable(settled)
        self.assertEqual(self.app.older_count,0)
        self.assertLessEqual(len(self.app.row_blocks),16)
        self.assertIs(self.app.copy_buttons[key],button)

    def test_start_stop_and_restore_partial_recording(self):
        script=self.path/'recorder.py'
        script.write_text('''#!/usr/bin/python3
import json,sys
from datetime import datetime
args=sys.argv; identity=args[args.index('--ui-session')+1]; output=args[args.index('--output')+1]
def emit(status,**details):
    data={'type':'recording','payload':{'session_id':identity,'timestamp':datetime.now().astimezone().isoformat(),'status':status,**details}}
    with open(output,'a') as f: f.write(json.dumps(data)+'\\n')
    print(json.dumps(data),flush=True)
emit('started');emit('chunk',seq=1,text='First sentence.')
sys.stdin.readline();emit('finishing');emit('chunk',seq=2,text='Final tail.');emit('stopped')
''')
        script.chmod(0o700)
        sessions=RecordingSessions(self.path/'recordings',script,self.path/'config',self.bridge.enqueue)
        self.app.recordings=sessions
        self.app.toggle_recording()
        deadline=time.monotonic()+2
        while sessions.state!='recording' and time.monotonic()<deadline: self.root.update();time.sleep(.005)
        self.assertEqual(sessions.state,'recording');self.assertEqual(self.app.record_button.cget('text'),'■ Stop')
        self.app.toggle_recording();self.assertEqual(sessions.state,'finishing')
        while sessions.state!='idle' and time.monotonic()<deadline: self.root.update();time.sleep(.005)
        self.assertEqual(sessions.state,'idle'); self.assertEqual(sessions.rows()[0]['corrected'],'First sentence. Final tail.')
        sessions.process.wait(timeout=2)
        restored=RecordingSessions(self.path/'recordings',script,self.path/'config',self.bridge.enqueue)
        self.assertEqual(restored.rows()[0]['corrected'],'First sentence. Final tail.')
        self.assertEqual(len(restored.rows()[0]['segments']),2)
        self.assertEqual([s['seq'] for s in restored.rows()[0]['segments']],[1,2])
        self.assertTrue(all(s['timestamp'] for s in restored.rows()[0]['segments']))
        self.assertEqual(restored.rows()[0]['recording_state'],'stopped')


if __name__=='__main__':
    if os.environ.get('WHISPER_GUI_HEADLESS')!='1':
        env=dict(os.environ,WHISPER_GUI_HEADLESS='1')
        raise SystemExit(subprocess.run(['xvfb-run','-a',sys.executable,__file__],env=env).returncode)
    unittest.main()
