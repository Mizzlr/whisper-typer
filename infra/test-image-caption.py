#!/usr/bin/python3
"""Caption transport, cached numbering, and responsive UI on isolated Xvfb."""
import base64
import gc
from datetime import datetime,timedelta
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
import unittest

from PIL import Image
from clipboard_history import ClipboardStore
from image_caption import ImageCaptioner,VisionClient

spec=importlib.util.spec_from_file_location('window',Path(__file__).with_name('dictation-window.py'))
window=importlib.util.module_from_spec(spec);spec.loader.exec_module(window)


def png(color='red',size=(240,120)):
    out=io.BytesIO();Image.new('RGB',size,color).save(out,format='PNG');return out.getvalue()


class ModelServer:
    def __init__(self):
        self.requests=[];self.ports=[];self.entered=threading.Event();self.release=threading.Event();self.release.set();self.fail=False
        fixture=self
        class Handler(BaseHTTPRequestHandler):
            protocol_version='HTTP/1.1'
            def log_message(self,*_):pass
            def do_POST(self):
                fixture.requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                fixture.ports.append(self.client_address[1]);fixture.entered.set();fixture.release.wait(3)
                result={'done':True,'message':{'content':json.dumps({'title':'Project review dashboard','description':'A dashboard shows the project review status. It includes recent notes and controls.'})}}
                data=json.dumps(result).encode();self.send_response(503 if fixture.fail else 200)
                self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.host=f'http://127.0.0.1:{self.server.server_port}'

    def close(self):
        self.release.set();self.server.shutdown();self.server.server_close();self.thread.join()


class CacheTests(unittest.TestCase):
    def test_numbering_and_caption_survive_recopy_restart_retention_and_new_day(self):
        with tempfile.TemporaryDirectory() as directory:
            store=ClipboardStore(directory,limit=2);store.add('image',png(),'2026-09-17T10:00:00')
            first=store.rows()[0];pixels=(store.images/first['image']).read_bytes()
            store.save_caption(first['id'],'Review','A project review dashboard.','fixture',123)
            store.add('image',png(),'2026-09-17T10:01:00')
            row=ClipboardStore(directory,limit=2).rows()[0]
            self.assertEqual((row['image_number'],row['title'],row['description']),(1,'Review','A project review dashboard.'))
            self.assertEqual((store.images/row['image']).read_bytes(),pixels)
            store.add('image',png('blue'),'2026-09-17T10:02:00')
            self.assertEqual(store.rows()[0]['image_number'],2)
            store.add('text','Keep the clipboard exact.','2026-09-17T10:03:00')
            self.assertFalse(store.save_caption(first['id'],'Late','Late result.','fixture',1))
            store.add('image',png('green'),'2026-09-17T10:04:00')
            self.assertEqual(store.rows()[0]['image_number'],3)
            store.add('image',png('green'),'2026-09-18T10:00:00')
            self.assertEqual(store.rows()[0]['image_number'],3)
            self.assertEqual(store.rows()[0]['timestamp'],'2026-09-17T10:04:00')
            store.add('image',png('purple'),'2026-09-18T10:00:01')
            self.assertEqual(store.rows()[0]['image_number'],1)
            self.assertEqual(store.database.stat().st_mode&0o777,0o600)

    def test_existing_history_and_title_schema_migrate_without_losing_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            store=ClipboardStore(directory);store.add('image',png(),'2026-09-17T10:00:00');row=store.rows()[0]
            pixels=(store.images/row['image']).read_bytes()
            with store.connect() as db:
                db.execute('DROP TABLE image_titles');db.execute('DROP TABLE image_numbers');db.execute('DROP TABLE image_counters')
                db.execute('CREATE TABLE image_titles (id TEXT PRIMARY KEY,title TEXT NOT NULL,model TEXT NOT NULL,latency_ms REAL NOT NULL)')
                db.execute('INSERT INTO image_titles VALUES (?,?,?,?)',(row['id'],'Existing title','fixture',123))
            restored=ClipboardStore(directory);new=restored.rows()[0]
            self.assertEqual((new['title'],new['description'],new['image_number']),('Existing title','',1))
            self.assertEqual((restored.images/new['image']).read_bytes(),pixels)

    def test_vision_transport_reuses_connection_and_keeps_original_pixels(self):
        server=ModelServer();client=VisionClient(server.host)
        try:
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'source.png';path.write_bytes(png(size=(3000,1500)));before=path.read_bytes()
                for _ in range(2):
                    title,description,latency=client.caption(path)
                    self.assertEqual(title,'Project review dashboard');self.assertIn('recent notes',description);self.assertGreater(latency,0)
                self.assertEqual(path.read_bytes(),before)
                self.assertEqual(len(set(server.ports)),1)
                payload=server.requests[0]
                self.assertEqual(payload['model'],'qwen3-vl:2b-instruct');self.assertEqual(payload['keep_alive'],-1)
                self.assertFalse(payload['stream']);self.assertFalse(payload['think'])
                with Image.open(io.BytesIO(base64.b64decode(payload['messages'][0]['images'][0]))) as image:
                    self.assertLessEqual(max(image.size),1280)
        finally:client.close();server.close()


class UiTests(unittest.TestCase):
    def setUp(self):
        gc.collect()
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name);self.store=ClipboardStore(self.path/'clips')
        self.server=ModelServer();self.root=tk.Tk();self.root.withdraw()
        self.app=window.DictationWindow(self.root,window.DictationStore(self.path,self.path/'reviews.jsonl'),self.path/'settings.json',self.store)
        self.app.show_clipboard.set(True);self.root.deiconify();self.root.geometry('550x430')

    def start(self):
        self.app.clipboard_changed()
        self.app.image_captioner=ImageCaptioner(self.root,self.store,self.app.clipboard_changed,self.server.host)

    def tearDown(self):
        zoom_worker=self.app.zoom_view.worker if self.app.zoom_view else None
        worker=self.app.image_captioner.worker if self.app.image_captioner else None
        self.server.release.set();self.app.close()
        if worker:worker.join(2);self.assertFalse(worker.is_alive())
        if zoom_worker:zoom_worker.join(2);self.assertFalse(zoom_worker.is_alive())
        self.server.close();self.temp.cleanup()
        self.app=self.root=None
        gc.collect()

    def wait(self,predicate,seconds=3):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            self.root.update()
            if predicate():return
            time.sleep(.005)
        self.fail('Caption action did not finish')

    def test_historical_description_is_on_demand_responsive_and_updates_in_place(self):
        self.store.add('image',png(),(datetime.now()-timedelta(hours=1)).isoformat());item=self.store.rows()[0];key='clip:'+item['id']
        self.start();self.root.update()
        self.app.load_older();self.root.update()
        self.assertEqual(len(self.server.requests),0)
        block=self.app.row_blocks[key];view=block['image_view'];copy=self.app.copy_buttons[key];photo=self.app.photos[key]
        self.assertIn('#1',block['headers'][0].winfo_children()[-1].cget('text'))
        self.root.clipboard_clear();self.root.clipboard_append('Do not replace this clipboard text.')
        self.server.release.clear();block['image_action'].invoke()
        self.wait(self.server.entered.is_set)
        self.assertFalse(self.app.image_captioner.request(item['id']))
        tick=[];self.root.after(1,lambda:tick.append(True));self.wait(lambda:bool(tick))
        self.assertEqual(self.root.clipboard_get(),'Do not replace this clipboard text.')
        self.assertEqual(block['image_action'].cget('state'),'disabled')
        self.server.release.set();self.wait(lambda:bool(self.store.rows()[0]['description']))
        self.wait(lambda:not self.app.image_captioner.inflight)
        self.assertEqual(block['image_action'].cget('state'),'disabled')
        self.assertEqual(block['image_action'].cget('text'),'Described')
        self.assertFalse(self.app.image_captioner.request(item['id']))
        block['image_action'].invoke();self.root.update()
        self.assertEqual(len(self.server.requests),1)
        self.assertIs(self.app.row_blocks[key]['image_view'],view);self.assertIs(self.app.copy_buttons[key],copy);self.assertIs(self.app.photos[key],photo)
        self.assertEqual(view.caption_title.cget('text'),'Project review dashboard')
        self.assertIn('recent notes',view.description.cget('text'))
        self.assertGreater(view.caption_title.winfo_x(),view.thumbnail.winfo_x()+view.thumbnail.winfo_width())
        self.assertEqual(self.root.clipboard_get(),'Do not replace this clipboard text.')

    def test_zoom_fits_large_image_and_preserves_clipboard_and_position(self):
        self.store.add('image',png(size=(4000,2000)))
        item=self.store.rows()[0]
        self.store.save_caption(item['id'],'A very long screenshot title that needs to leave room for the zoom controls','Example description.','fixture',1)
        self.app.clipboard_changed();self.root.update()
        item=self.store.rows()[0];key='clip:'+item['id'];view=self.app.row_blocks[key]['image_view']
        self.root.clipboard_clear();self.root.clipboard_append('Keep exact clipboard')
        position=self.app.text.index('@0,0');stamp=item['timestamp']
        view.thumbnail.event_generate('<Button-1>');self.root.update()
        overlay=self.app.zoom_view;self.assertIsNotNone(overlay)
        self.wait(lambda:overlay.photo.width()>144)
        self.assertLessEqual(overlay.photo.width(),overlay.winfo_width())
        self.assertLessEqual(overlay.photo.height(),overlay.winfo_height())
        for control in (overlay.minus_button,overlay.plus_button,overlay.fit_button,overlay.close_button):
            self.assertTrue(control.winfo_ismapped())
            self.assertLessEqual(control.winfo_rootx()+control.winfo_width(),overlay.winfo_rootx()+overlay.winfo_width())
        self.assertEqual(self.root.clipboard_get(),'Keep exact clipboard')
        overlay.close_button.invoke();self.root.update()
        self.assertIsNone(self.app.zoom_view)
        self.assertEqual(self.app.text.index('@0,0'),position)
        self.assertEqual(self.store.rows()[0]['timestamp'],stamp)

    def test_zoom_pans_beyond_window_and_wheel_keeps_pointer_anchor(self):
        self.store.add('image',png(size=(2000,1200)));self.app.clipboard_changed();self.root.update()
        item=self.store.rows()[0];key='clip:'+item['id'];preview=self.app.row_blocks[key]['image_view']
        self.root.clipboard_clear();self.root.clipboard_append('Clipboard stays exact')
        position=self.app.text.index('@0,0');outer=self.app.text.yview()
        preview.thumbnail.event_generate('<Button-1>');self.root.update();viewer=self.app.zoom_view
        self.wait(lambda:viewer.scale is not None)
        initial=viewer.scale;viewer.plus_button.invoke()
        self.wait(lambda:viewer.scale>initial)
        viewer.zoom(4);self.wait(lambda:viewer.photo.width()>viewer.canvas.winfo_width() and viewer.photo.height()>viewer.canvas.winfo_height())
        canvas=viewer.canvas;before=(canvas.xview(),canvas.yview())
        canvas.event_generate('<ButtonPress-1>',x=230,y=140)
        canvas.event_generate('<B1-Motion>',x=100,y=60)
        canvas.event_generate('<ButtonRelease-1>',x=100,y=60);self.root.update()
        self.assertNotEqual(canvas.xview(),before[0]);self.assertNotEqual(canvas.yview(),before[1])
        x,y=150,100
        anchor=((canvas.canvasx(x)-viewer.offset[0])/viewer.photo.width(),(canvas.canvasy(y)-viewer.offset[1])/viewer.photo.height())
        scale=viewer.scale;canvas.event_generate('<Button-4>',x=x,y=y)
        self.wait(lambda:viewer.scale>scale)
        after=((canvas.canvasx(x)-viewer.offset[0])/viewer.photo.width(),(canvas.canvasy(y)-viewer.offset[1])/viewer.photo.height())
        self.assertAlmostEqual(anchor[0],after[0],delta=.005);self.assertAlmostEqual(anchor[1],after[1],delta=.005)
        scale=viewer.scale;viewer.minus_button.invoke();self.wait(lambda:viewer.scale<scale)
        viewer.fit_button.invoke();self.wait(lambda:viewer.scale==initial)
        self.assertLessEqual(viewer.photo.width(),canvas.winfo_width());self.assertLessEqual(viewer.photo.height(),canvas.winfo_height())
        self.assertEqual(self.root.clipboard_get(),'Clipboard stays exact')
        self.assertEqual(self.app.text.yview(),outer)
        viewer.close_button.invoke();self.root.update();viewer.worker.join(2)
        self.assertFalse(viewer.worker.is_alive());self.assertIsNone(self.app.zoom_view)
        self.assertEqual(self.app.text.index('@0,0'),position)
        self.assertEqual(self.store.rows()[0]['timestamp'],item['timestamp'])

    def test_horizontal_pan_controls_keep_zoom_and_outer_history_stable(self):
        self.store.add('image',png(size=(2000,1200)));self.app.clipboard_changed();self.root.update()
        item=self.store.rows()[0];preview=self.app.row_blocks['clip:'+item['id']]['image_view']
        preview.thumbnail.event_generate('<Button-1>');self.root.update();viewer=self.app.zoom_view
        self.wait(lambda:viewer.scale is not None)
        viewer.zoom(4);self.wait(lambda:viewer.photo.width()>viewer.canvas.winfo_width())
        canvas=viewer.canvas;scale=viewer.scale;outer=self.app.text.yview()
        for widget,event,direction in ((canvas,'<Shift-Button-5>',1),(canvas,'<Shift-Button-4>',-1),
                                       (canvas,'<Right>',1),(canvas,'<Left>',-1),
                                       (viewer.horizontal,'<Button-5>',1),(viewer.horizontal,'<Button-4>',-1)):
            canvas.xview_moveto(.3);self.root.update();before=canvas.xview()[0]
            widget.event_generate(event);self.root.update()
            self.assertGreater((canvas.xview()[0]-before)*direction,0,event)
            self.assertEqual(viewer.scale,scale,event)
        canvas.xview_moveto(.3);self.root.update();before=canvas.xview()[0]
        viewer.horizontal.event_generate('<MouseWheel>',delta=-120);self.root.update()
        self.assertGreater(canvas.xview()[0],before)
        self.assertFalse(canvas.bind('<Double-Button-1>'))
        for _ in range(2):
            before=canvas.xview()[0]
            canvas.event_generate('<ButtonPress-1>',x=230,y=140)
            canvas.event_generate('<B1-Motion>',x=180,y=140)
            canvas.event_generate('<ButtonRelease-1>',x=180,y=140);self.root.update()
            self.assertGreater(canvas.xview()[0],before)
            self.assertEqual(viewer.scale,scale)
        self.assertEqual(self.app.text.yview(),outer)

    def test_escape_exits_viewer_and_rapid_requests_keep_latest_zoom(self):
        self.store.add('image',png(size=(2000,1200)));self.app.clipboard_changed();self.root.update()
        item=self.store.rows()[0];key='clip:'+item['id'];preview=self.app.row_blocks[key]['image_view']
        existing=self.root.bind('<Escape>')
        for _ in range(2):
            preview.thumbnail.event_generate('<Button-1>');self.root.update();viewer=self.app.zoom_view
            self.wait(lambda:viewer.scale is not None)
            for _ in range(12):viewer.zoom(1.25)
            for _ in range(6):viewer.zoom(1/1.25)
            requested=viewer.target_scale;tick=[]
            self.root.after(1,lambda:tick.append(True));self.wait(lambda:bool(tick))
            self.wait(lambda:viewer.scale==requested)
            self.assertLessEqual(viewer.tasks.qsize(),1);self.assertLessEqual(viewer.results.qsize(),1)
            self.assertIs(self.root.focus_get(),viewer.canvas)
            viewer.canvas.event_generate('<Escape>');self.root.update()
            self.assertIsNone(self.app.zoom_view)
            self.assertEqual(self.root.bind('<Escape>'),existing)
            viewer.worker.join(2);self.assertFalse(viewer.worker.is_alive())

    def test_escape_receives_real_key_and_restores_previous_application_focus(self):
        from Xlib import X,display
        from Xlib.ext import xtest
        connection=display.Display()
        other=connection.screen().root.create_window(700,500,100,100,0,connection.screen().root_depth,X.InputOutput,X.CopyFromParent)
        other.map();other.set_input_focus(X.RevertToParent,X.CurrentTime);connection.sync()
        try:
            self.store.add('image',png());self.app.clipboard_changed();self.root.update()
            item=self.store.rows()[0];key='clip:'+item['id'];self.app.row_blocks[key]['image_view'].thumbnail.event_generate('<Button-1>');self.root.update()
            viewer=self.app.zoom_view;self.assertIsNotNone(viewer)
            keycode=connection.keysym_to_keycode(0xff1b)
            xtest.fake_input(connection,X.KeyPress,keycode);xtest.fake_input(connection,X.KeyRelease,keycode);connection.sync()
            self.wait(lambda:self.app.zoom_view is None)
            self.assertEqual(connection.get_input_focus().focus.id,other.id)
            viewer.worker.join(2);self.assertFalse(viewer.worker.is_alive())
        finally:other.destroy();connection.close()

    def test_new_images_auto_describe_and_same_image_recopy_uses_cache(self):
        self.store.add('image',png(),(datetime.now()-timedelta(hours=1)).isoformat());self.start()
        self.store.add('image',png('blue'));self.app.clipboard_changed()
        self.wait(lambda:any(row.get('description') for row in self.store.rows()))
        self.wait(lambda:not self.app.image_captioner.inflight)
        blue=self.store.rows()[0];number=blue['image_number']
        self.assertEqual(len(self.server.requests),1)
        self.assertEqual(sum(bool(row.get('description')) for row in self.store.rows()),1)
        self.store.add('image',png('blue'));self.app.clipboard_changed();self.root.update()
        self.assertEqual(self.store.rows()[0]['image_number'],number)
        self.assertEqual(len(self.server.requests),1)

    def test_failure_keeps_image_and_previous_caption_and_allows_manual_retry(self):
        self.store.add('image',png());item=self.store.rows()[0]
        self.start();key='clip:'+item['id'];block=self.app.row_blocks[key]
        pixels=(self.store.images/item['image']).read_bytes();self.server.fail=True
        block['image_action'].invoke();self.wait(lambda:bool(self.app.image_captioner.retry_after))
        self.assertEqual(block['image_action'].cget('state'),'normal')
        self.assertFalse(self.store.rows()[0]['title'])
        self.assertEqual((self.store.images/item['image']).read_bytes(),pixels)
        self.server.fail=False;block['image_action'].invoke()
        self.wait(lambda:self.store.rows()[0]['title']=='Project review dashboard')


if __name__=='__main__':
    if os.environ.get('WHISPER_GUI_HEADLESS')!='1':
        env=dict(os.environ,WHISPER_GUI_HEADLESS='1')
        raise SystemExit(subprocess.run(['xvfb-run','-a',sys.executable,__file__],env=env).returncode)
    unittest.main()
