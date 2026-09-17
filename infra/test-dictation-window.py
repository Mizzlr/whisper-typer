#!/usr/bin/python3
"""Tests for word diffs, incremental history joining, and explicit-only copying."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import tkinter as tk

spec = importlib.util.spec_from_file_location('dictation_window', Path(__file__).with_name('dictation-window.py'))
window = importlib.util.module_from_spec(spec)
spec.loader.exec_module(window)


class DictationTests(unittest.TestCase):
    def test_word_diff_preserves_text_and_marks_only_changed_words(self):
        before, after = 'She have two files.', 'She has two files.'
        left, right = window.diff_segments(before, after)
        self.assertEqual(''.join(s for s, _ in left), before)
        self.assertEqual(''.join(s for s, _ in right), after)
        self.assertEqual([s for s, tag in left if tag == 'removed'], ['have'])
        self.assertEqual([s for s, tag in right if tag == 'added'], ['has'])

    def test_unicode_and_punctuation_diff(self):
        before, after = '你好 🙂. For now.', '你好 🙂 for now.'
        left, right = window.diff_segments(before, after)
        self.assertEqual(''.join(s for s, _ in left), before)
        self.assertEqual(''.join(s for s, _ in right), after)
        self.assertFalse(any(tag != 'normal' for _, tag in left + right))

    def test_partial_utf8_and_background_result_join(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);history=root/'2026-09-17.jsonl';review=root/'reviews.jsonl'
            store=window.DictationStore(root,review)
            row={'timestamp':__import__('datetime').datetime.now().isoformat(),'whisper_text':'she have two files 🙂','final_text':'She have two files 🙂. ','background_review':True}
            data=(json.dumps(row,ensure_ascii=False)+'\n').encode();cut=data.index('🙂'.encode())+1
            history.write_bytes(data[:cut]);store.refresh();self.assertEqual(store.visible(),[])
            with history.open('ab') as f:f.write(data[cut:])
            store.refresh();self.assertEqual(len(store.visible()),1);self.assertEqual(store.visible()[0]['status'],'Checking grammar…')
            review.write_text(json.dumps({'dictation_timestamp':row['timestamp'],'status':'changed','corrected':'She has two files 🙂.'})+'\n')
            store.refresh();self.assertEqual(store.visible()[0]['corrected'],'She has two files 🙂.')
            self.assertFalse(store.refresh())

    def test_original_and_historical_corrected_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=root/'2026-09-17.jsonl';p.write_text(json.dumps({'timestamp':'2026-09-17T20:00:00','whisper_text':'hello','ollama_text':'Hello.','final_text':'Hello. '})+'\n')
            store=window.DictationStore(root,root/'review.jsonl');store.refresh();r=store.visible()[0]
            self.assertEqual((r['original'],r['corrected']),('Hello.','Hello.'))
            self.assertEqual(len(store.visible('hello','2026-09-17')),1)
            self.assertEqual(store.visible('no-match'),[])

    def test_pasted_baseline_and_cosmetic_only_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);store=window.DictationStore(root,root/'reviews.jsonl')
            stamp='2026-09-17T21:00:00'
            store.records[stamp]={'timestamp':stamp,'whisper_text':'she have files','final_text':'She have files.'}
            store.corrections[stamp]={'pasted':'She have files. ','corrected':'she have files!','status':'changed','accepted':True}
            row=store.visible()[0]
            self.assertEqual(row['original'],'She have files.')
            self.assertEqual(row['corrected'],row['original'])
            self.assertEqual(row['status'],'No grammar change')
            store.corrections[stamp]['corrected']='She has files.'
            row=store.visible()[0]
            self.assertEqual(row['corrected'],'She has files.')
            self.assertEqual(row['status'],'Grammar changed')

    def test_round_trip_stats_update_with_review(self):
        with tempfile.TemporaryDirectory() as directory:
            store=window.DictationStore(directory,Path(directory)/'review.jsonl')
            stamp=__import__('datetime').datetime.now().isoformat()
            store.records[stamp]={'timestamp':stamp,'whisper_text':'hello','final_text':'Hello.',
                                  'background_review':True,'total_latency_ms':81,'grammar_gate_latency_ms':12.4}
            self.assertIn('Until paste: 81 ms',store.visible()[0]['timings'])
            self.assertIn('Grammar: pending',store.visible()[0]['timings'])
            store.corrections[stamp]={'pasted':'Hello.','corrected':'Hello.','status':'unchanged','grammar_latency_ms':183.2}
            self.assertEqual(store.visible()[0]['timings'],'Until paste: 81 ms · Grammar round trip: 183 ms · Judge: 12 ms')

    def test_archive_days_load_only_on_request(self):
        from datetime import datetime,timedelta
        from unittest.mock import patch
        fixed=datetime.now().replace(hour=22,minute=0,second=0,microsecond=0)
        class Clock(datetime):
            @classmethod
            def now(cls,tz=None):return fixed.astimezone(tz) if tz else fixed
        with tempfile.TemporaryDirectory() as directory, patch.object(window,'datetime',Clock), patch.object(window.time,'time',return_value=fixed.timestamp()):
            root=Path(directory)
            for age in (0,1,2):
                stamp=(fixed-timedelta(days=age)).replace(hour=23)
                (root/(stamp.strftime('%Y-%m-%d')+'.jsonl')).write_text(json.dumps({'timestamp':stamp.isoformat(),'whisper_text':'hello','final_text':'Hello.'})+'\n')
            store=window.DictationStore(root,root/'reviews.jsonl');store.refresh()
            self.assertEqual(len(store.records),1)
            self.assertTrue(store.load_older_day())
            self.assertEqual(len(store.records),2)
            self.assertFalse(store.load_older_day())

    def test_daily_numbers_and_rejected_suggestion(self):
        with tempfile.TemporaryDirectory() as directory:
            store=window.DictationStore(directory,Path(directory)/'review.jsonl')
            for stamp in ['2026-09-16T21:00:00','2026-09-17T21:00:00','2026-09-17T21:01:00']:
                store.records[stamp]={'timestamp':stamp,'whisper_text':'hello','final_text':'Hello, hello, hello, hello, hello.'}
            latest='2026-09-17T21:01:00'
            store.corrections[latest]={'pasted':'Hello, hello, hello, hello, hello.','corrected':'Hello, hello, hello, hello, hello.',
                                       'status':'unchanged','accepted':False,'fallback_reason':'large_length_change'}
            rows=store.visible()
            self.assertEqual([row['number'] for row in rows],[2,1,1])
            self.assertEqual(rows[0]['status'],'Suggestion rejected')
            self.assertEqual(rows[0]['original'],rows[0]['corrected'])


class GuiChecks(unittest.TestCase):
    @unittest.skipUnless('--gui' in sys.argv,'requires Xvfb')
    def test_sliding_render_window_loads_both_directions_and_excludes_old_rows(self):
        from datetime import datetime,timedelta
        with tempfile.TemporaryDirectory() as directory:
            store=window.DictationStore(directory,Path(directory)/'reviews.jsonl')
            now=datetime.now()
            stamps=[]
            for index in range(800):
                stamp=(now-timedelta(minutes=index)).isoformat();stamps.append(stamp)
                store.records[stamp]={'timestamp':stamp,'whisper_text':f'Entry {index}','final_text':f'Entry {index}'}
            old=(now-timedelta(hours=25)).isoformat()
            store.records[old]={'timestamp':old,'whisper_text':'Too old','final_text':'Too old'}
            root=tk.Tk();app=window.DictationWindow(root,store,Path(directory)/'settings.json')
            self.addCleanup(app.close)
            root.geometry('550x430');app.render();root.update()
            self.assertNotIn(old,[r['timestamp'] for r in app.rows])
            for _ in range(18):
                app.text.yview_moveto(1);root.update()
                anchor=None
                for key,mark in app.row_starts.items():
                    if app.text.compare(mark,'<=',app.text.index('@0,0')) and (anchor is None or app.text.compare(mark,'>',app.row_starts[anchor])):anchor=key
                app.load_older();root.update()
                self.assertLessEqual(len(app.row_blocks),app.MAX_RENDERED)
                self.assertLessEqual(len(app.copy_buttons),app.MAX_RENDERED)
                if anchor:self.assertTrue(anchor in app.row_blocks,'Visible anchor was pruned')
            self.assertGreater(app.window_start,0)
            first=app.rendered[0][0];header=app.row_blocks[first]['headers'][0]
            new=(now+timedelta(seconds=1)).isoformat()
            store.records[new]={'timestamp':new,'whisper_text':'New entry','final_text':'New entry'}
            app.render();root.update()
            self.assertEqual(app.rendered[0][0],first)
            self.assertIs(app.row_blocks[first]['headers'][0],header)
            while app.window_start:
                app.text.yview_moveto(0);root.update();app.load_newer();root.update()
                self.assertLessEqual(len(app.row_blocks),app.MAX_RENDERED)
            self.assertIn(new,app.row_blocks)
            self.assertNotIn(old,app.row_blocks)

    @unittest.skipUnless('--gui' in sys.argv,'requires desktop')
    def test_render_and_copy_only_on_click(self):
        with tempfile.TemporaryDirectory() as directory:
            root_path=Path(directory);store=window.DictationStore(root_path,root_path/'reviews.jsonl')
            from datetime import datetime,timedelta
            stamp=datetime.now().isoformat();store.records[stamp]={'timestamp':stamp,'whisper_text':'She have two files.','final_text':'She have two files.'}
            other=(datetime.now()-timedelta(minutes=1)).isoformat();store.records[other]={'timestamp':other,'whisper_text':'short','final_text':'A much longer sentence that wraps across multiple lines. '*8}
            older=(datetime.now()-timedelta(minutes=20)).isoformat();store.records[older]={'timestamp':older,'whisper_text':'older','final_text':'Older dictation.'}
            root=tk.Tk()
            try: old_clipboard=root.clipboard_get()
            except tk.TclError: old_clipboard=None
            def cleanup():
                try: root.destroy()
                except tk.TclError: pass
                if old_clipboard is not None:
                    import subprocess
                    subprocess.run(['xsel','--clipboard','--input'],input=old_clipboard.encode(),check=True)
            self.addCleanup(cleanup)
            root.clipboard_clear();root.clipboard_append('clipboard untouched')
            app=window.DictationWindow(root,store,root_path/'settings.json');app.render();root.update()
            self.assertEqual(root.clipboard_get(),'clipboard untouched')
            self.assertNotIn(older,app.copy_buttons)
            store.corrections[stamp]={'status':'changed','corrected':'She has two files.'};app.render();root.update()
            self.assertEqual(root.clipboard_get(),'clipboard untouched')
            self.assertTrue(app.text.tag_ranges('removed'));self.assertTrue(app.text.tag_ranges('added'))
            self.assertEqual(app.text.cget('state'),'disabled')
            def right_edges():
                return [b.winfo_rootx()+b.winfo_width() for b in app.copy_buttons.values()]
            self.assertEqual(len(set(right_edges())),1)
            root.geometry('480x430');root.update()
            self.assertEqual(len(set(right_edges())),1)
            app.copy_buttons[stamp].invoke();root.update();self.assertEqual(root.clipboard_get(),'She has two files.')
            self.assertEqual(len(set(right_edges())),1)
            header=app.row_headers[0]
            app.text.yview_moveto(.3);root.update()
            position=app.text.index('@0,0')
            app.load_older();root.update()
            self.assertIn(older,app.copy_buttons)
            self.assertIs(app.row_headers[0],header)
            self.assertEqual(app.text.index('@0,0'),position)
            self.assertFalse(hasattr(app,'query'))
            self.assertFalse(hasattr(app,'more_button'))
            self.assertTrue(app.topmost.get())
            headless=os.environ.get('WHISPER_GUI_HEADLESS') == '1'
            if not headless: self.assertTrue(root.attributes('-topmost'))
            app.topmost.set(False);app.set_topmost();root.update()
            if not headless:
                for _ in range(50):
                    if not root.attributes('-topmost'): break
                    settled=tk.BooleanVar(value=False)
                    root.after(20,lambda:settled.set(True));root.wait_variable(settled)
                self.assertFalse(root.attributes('-topmost'))
            self.assertFalse(app.topmost.get())
            app.close();self.assertTrue((root_path/'settings.json').exists())
            self.assertFalse(json.loads((root_path/'settings.json').read_text())['topmost'])


if __name__=='__main__':
    if '--gui' in sys.argv and os.environ.get('WHISPER_GUI_HEADLESS') != '1':
        import shutil,subprocess
        runner=shutil.which('xvfb-run')
        if not runner:
            raise SystemExit('GUI checks require xvfb-run to avoid opening windows on your desktop.')
        env=dict(os.environ,WHISPER_GUI_HEADLESS='1')
        raise SystemExit(subprocess.run([runner,'-a',sys.executable,__file__,'--gui'],env=env).returncode)
    argv=[v for v in sys.argv if v!='--gui'];unittest.main(argv=argv)
