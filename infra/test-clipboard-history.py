#!/usr/bin/python3
"""Clipboard tests always run on an isolated Xvfb display."""
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import tkinter as tk
import unittest

from PIL import Image
from clipboard_history import ClipboardMonitor, ClipboardStore

spec = importlib.util.spec_from_file_location('window', Path(__file__).with_name('dictation-window.py'))
window = importlib.util.module_from_spec(spec); spec.loader.exec_module(window)


def png():
    out=io.BytesIO(); Image.new('RGBA',(240,120),(12,34,56,170)).save(out,format='PNG'); return out.getvalue()


class HistoryTests(unittest.TestCase):
    def test_text_exact_persistent_and_deduplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            store=ClipboardStore(directory)
            text='  first\tsecond\n你好 🙂\n'
            store.add('text',text,'2026-09-17T21:00:00')
            store.add('text',text,'2026-09-17T21:01:00')
            rows=ClipboardStore(directory).rows()
            self.assertEqual(len(rows),1); self.assertEqual(rows[0]['text'],text)
            self.assertEqual(rows[0]['timestamp'],'2026-09-17T21:00:00')
            self.assertEqual(store.database.stat().st_mode & 0o777,0o600)
            self.assertEqual(store.directory.stat().st_mode & 0o777,0o700)

    def test_image_retention_and_thumbnail(self):
        with tempfile.TemporaryDirectory() as directory:
            store=ClipboardStore(directory,limit=2)
            store.add('image',png(),'2026-09-17T21:00:00'); item=store.rows()[0]
            with Image.open(store.images/item['thumbnail']) as image:
                self.assertLessEqual(image.width,144); self.assertLessEqual(image.height,88)
            self.assertEqual((store.images/item['image']).stat().st_mode & 0o777,0o600)
            store.add('text','next','2026-09-17T21:01:00'); store.add('text','last','2026-09-17T21:02:00')
            self.assertEqual(len(store.rows()),2); self.assertEqual(list(store.images.iterdir()),[])
            self.assertFalse(store.add('image',b'invalid image'))


class ClipboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=tk.Tk();cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.path=Path(self.temp.name)
        self.root.withdraw(); self.store=ClipboardStore(self.path/'clips')
        self.monitor=ClipboardMonitor(self.root,self.store,lambda:None,screenshot_dirs=[])
        self.providers=[]

    def tearDown(self):
        self.monitor.close()
        self.monitor.executor.shutdown(wait=True)
        for child in self.providers:
            child.terminate(); child.communicate(timeout=2)
        for timer in self.root.tk.call('after','info'): self.root.after_cancel(timer)
        for child in self.root.winfo_children(): child.destroy()
        self.temp.cleanup()

    def wait(self, predicate):
        deadline=time.monotonic()+3
        while not predicate():
            self.root.update()
            if time.monotonic()>deadline: self.fail('asynchronous clipboard action timed out')
            time.sleep(.005)
        self.root.update()

    def provide(self, kind, payload):
        file=self.path/('provider.png' if kind=='image' else 'provider.txt')
        file.write_bytes(payload if isinstance(payload,bytes) else payload.encode())
        code="""import sys,gi
gi.require_version('Gtk','3.0')
from gi.repository import Gtk,Gdk,GdkPixbuf,GLib
from pathlib import Path
clipboard=Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD)
if sys.argv[1]=='image':clipboard.set_image(GdkPixbuf.Pixbuf.new_from_file(sys.argv[2]))
else:clipboard.set_text(Path(sys.argv[2]).read_text(),-1)
print('ready',flush=True)
GLib.MainLoop().run()
"""
        child=subprocess.Popen([sys.executable,'-c',code,kind,str(file)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        self.providers.append(child); self.assertEqual(child.stdout.readline(),b'ready\n')

    def read(self, target='UTF8_STRING'):
        reader=subprocess.Popen(['xclip','-selection','clipboard','-out','-target',target],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        self.wait(lambda:reader.poll() is not None)
        out,err=reader.communicate(timeout=1); self.assertEqual(reader.returncode,0,err)
        return out

    def test_external_text_capture_and_recopy(self):
        text='  clipboard\ttext\n你好 🙂\n'
        self.provide('text',text); self.wait(lambda:any(row['text']==text for row in self.store.rows()))
        item=self.store.rows()[0]; self.assertEqual(item['text'],text)
        self.monitor.copy(item); self.assertEqual(self.read().decode(),text)

    def test_external_image_capture_thumbnail_zoom_and_explicit_copy(self):
        self.provide('image',png()); self.wait(lambda:bool(self.store.rows()))
        item=self.store.rows()[0]; self.assertEqual(item['kind'],'image')
        dictations=window.DictationStore(self.path/'dictations',self.path/'review.jsonl')
        app=window.DictationWindow(self.root,dictations,self.path/'settings.json',self.store)
        app.clipboard_monitor=self.monitor; app.render(); self.root.deiconify(); self.root.update()
        self.assertEqual(app.photos,{})
        app.show_clipboard.set(True); app.change_view(); self.root.update()
        self.assertEqual(len(app.photos),1)
        app.content_widgets[0].event_generate('<Button-1>'); self.root.update()
        self.assertIsNotNone(app.zoom_view)
        viewer=app.zoom_view
        self.wait(lambda:viewer.scale is not None)
        viewer.zoom(4);self.wait(lambda:viewer.scale==viewer.target_scale)
        self.assertGreater(viewer.photo.width(),240)
        viewer.copy_button.invoke();self.root.update()
        self.assertIs(app.zoom_view,viewer)
        self.assertEqual(viewer.copy_button.cget('text'),'Copied')
        zoom_copy=Image.open(io.BytesIO(self.read('image/png'))).convert('RGBA')
        original=Image.open(io.BytesIO(png())).convert('RGBA')
        self.assertEqual(zoom_copy.size,original.size);self.assertEqual(zoom_copy.tobytes(),original.tobytes())
        self.assertEqual(self.store.rows()[0]['timestamp'],item['timestamp'])
        viewer.close_button.invoke();self.root.update();viewer.worker.join(2)
        self.assertFalse(viewer.worker.is_alive())
        self.assertIsNone(app.zoom_view)
        app.copy_buttons['clip:'+item['id']].invoke();self.root.update()
        copied=Image.open(io.BytesIO(self.read('image/png'))).convert('RGBA')
        expected=Image.open(io.BytesIO(png())).convert('RGBA')
        self.assertEqual(copied.size,expected.size); self.assertEqual(copied.tobytes(),expected.tobytes())
        self.assertTrue(json.loads((self.path/'settings.json').read_text())['show_clipboard'])

    def test_screenshot_import_does_not_replace_clipboard(self):
        self.provide('text','still on clipboard'); self.wait(lambda:any(row['text']=='still on clipboard' for row in self.store.rows()))
        directory=self.path/'screenshots'; directory.mkdir(); (directory/'Screenshot.png').write_bytes(png())
        self.monitor.screenshot_dirs=[directory]; self.monitor.scan_screenshots()
        self.wait(lambda:any(row['kind']=='image' for row in self.store.rows()))
        self.assertEqual(self.read().decode(),'still on clipboard')
        file=directory/'Screenshot.png';stat=file.stat()
        self.assertIn((file,stat.st_mtime_ns,stat.st_size),ClipboardStore(self.store.directory).file_signatures())

    def test_mixed_view_deduplicates_dictation_paste(self):
        from datetime import datetime,timedelta
        stamp=datetime.now().isoformat(); dictations=window.DictationStore(self.path/'dictations',self.path/'review.jsonl')
        dictations.records[stamp]={'timestamp':stamp,'whisper_text':'hello','final_text':'Hello.'}
        view_store=ClipboardStore(self.path/'view-clips')
        view_store.add('text','Hello.',stamp)
        app=window.DictationWindow(self.root,dictations,self.path/'settings.json',view_store)
        app.show_clipboard.set(True); self.assertEqual(len(app.collect_rows()),1)
        newer=(datetime.now()+timedelta(seconds=5)).isoformat(); view_store.add('text','Hello.',newer)
        app.clipboard_changed()
        rows=app.collect_rows(); self.assertEqual(len(rows),1)
        self.assertNotIn('kind', rows[0]); self.assertEqual(rows[0]['number'],1)
        self.assertEqual(rows[0]['timestamp'],stamp)
        self.assertNotIn('sort_timestamp',rows[0])
        app.show_dictations.set(False)
        self.assertEqual(app.collect_rows(),[])
        app.show_dictations.set(True)
        self.assertEqual(len(app.collect_rows()),1)

    def test_copying_corrected_text_preserves_original_diff_and_stats(self):
        from datetime import datetime,timedelta
        stamp=datetime.now().isoformat()
        store=window.DictationStore(self.path/'dictations',self.path/'review.jsonl')
        store.records[stamp]={'timestamp':stamp,'whisper_text':'she have files','final_text':'She have files.',
                              'total_latency_ms':67,'background_review':True}
        store.corrections[stamp]={'dictation_timestamp':stamp,'pasted':'She have files.',
                                 'corrected':'She has files.','status':'changed','grammar_latency_ms':306}
        clips=ClipboardStore(self.path/'view-clips')
        clips.add('text','She has files.',(datetime.now()+timedelta(seconds=5)).isoformat())
        app=window.DictationWindow(self.root,store,self.path/'settings.json',clips)
        app.show_clipboard.set(True); app.render()
        row=app.collect_rows()[0]
        self.assertEqual(len(app.collect_rows()),1)
        self.assertEqual((row['original'],row['corrected']),('She have files.','She has files.'))
        self.assertEqual((row['number'],row['paste_ms'],row['grammar_ms']),(1,67,306))
        displayed=app.text.get('1.0','end')
        self.assertIn('− She have files.',displayed); self.assertIn('+ She has files.',displayed)


if __name__=='__main__':
    if os.environ.get('WHISPER_GUI_HEADLESS')!='1':
        env=dict(os.environ,WHISPER_GUI_HEADLESS='1')
        raise SystemExit(subprocess.run(['xvfb-run','-a',sys.executable,__file__],env=env).returncode)
    unittest.main()
