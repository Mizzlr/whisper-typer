#!/usr/bin/env python3
"""Folio: a quiet, native Tk reading window with private paste history."""
import base64
import csv
import io
import json
import mimetypes
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from tkinter import ttk
from urllib.parse import urlparse, unquote
from PIL import Image, ImageTk
from tkinterweb import HtmlFrame
from files import Document, PathResolver, load_document, pasted_paths, image_text, unzip_contents, pretty_json, latest_history_file
from rendering import CSS, rendered_body, theme_colors
from clipboard import read_clipboard
from history import History
from tk_widgets import Table, LIGHT, DARK
from syntax import syntax_safe, source_style
from image_zoom import ImageZoom, ZoomStyle


class TableParser(HTMLParser):
    def __init__(self):
        super().__init__();self.rows=[];self.row=None;self.cell=None
    def handle_starttag(self,tag,attrs):
        if tag=='tr':self.row=[]
        elif tag in ('td','th'):self.cell=''
        elif tag=='br' and self.cell is not None:self.cell+='\n'
    def handle_data(self,data):
        if self.cell is not None:self.cell+=data
    def handle_endtag(self,tag):
        if tag in ('td','th') and self.cell is not None:
            self.row.append(self.cell.strip());self.cell=None
        elif tag=='tr' and self.row is not None:self.rows.append(self.row);self.row=None


def local_resource(url,data=None,method='GET',encoding=None):
    """Embedded content may use local files/data only; no network fetching."""
    parsed=urlparse(url)
    if parsed.scheme=='data':
        header,content=url.split(',',1)
        raw=base64.b64decode(content) if ';base64' in header else unquote(content).encode()
        return url,raw,header[5:].split(';')[0],200
    if parsed.scheme=='file' and parsed.netloc in ('','localhost'):
        path=Path(unquote(parsed.path))
        return url,path.read_bytes(),mimetypes.guess_type(str(path))[0] or 'application/octet-stream',200
    raise ValueError('External embedded resources are disabled.')


_MATH_OR_DIAGRAM_PATTERN = re.compile(
    r'\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|(?<![\\$])\$(?!\$)(?:[^$\n]|\\\$)+?\$|```mermaid|^\s*(?:flowchart|graph|sequenceDiagram|classDiagram|stateDiagram|erDiagram|gantt|pie|mindmap|timeline|journey)\b',
    re.M
)


def typeset(source, dark=False):
    if not _MATH_OR_DIAGRAM_PATTERN.search(source):
        return rendered_body(source)
    worker = Path(__file__).with_name('render_worker.py')
    env = dict(os.environ, QTWEBENGINE_CHROMIUM_FLAGS='--disable-gpu', QT_QUICK_BACKEND='software')
    out_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as out_tmp:
            out_path = Path(out_tmp.name)
        cmd = ['xvfb-run', '-a', '/usr/bin/python3', str(worker)] + (['--dark'] if dark else []) + ['--output', str(out_path)]
        result = subprocess.run(cmd, input=source, text=True, capture_output=True, timeout=15, env=env)
        if result.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
            content = out_path.read_text(encoding='utf-8')
            return json.loads(content)
        if result.stdout and 'FOLIO_JSON_START' in result.stdout:
            raw = result.stdout.split('FOLIO_JSON_START', 1)[1].split('FOLIO_JSON_END', 1)[0]
            return json.loads(raw)
    except Exception:
        pass
    finally:
        if out_path and out_path.exists():
            try:
                out_path.unlink()
            except OSError:
                pass
    # Always gracefully fall back to local offline markdown rendering!
    return rendered_body(source)


AVAILABLE_FONTS=[
    'JetBrains Mono',
    'Google Sans Mono',
    'DejaVu Sans Mono',
    'Ubuntu Sans Mono',
    'Liberation Mono',
]


class Folio:
    def __init__(self,root,history_path=None):
        self.root=root;root.title('Folio');root.wm_class('Folio','Folio') if hasattr(root,'wm_class') else None
        self.history=History(history_path)
        self.settings_path=Path(history_path).with_suffix('.json') if history_path else Path.home()/'.config/folio/settings.json'
        try:self.settings=json.loads(self.settings_path.read_text())
        except (OSError,ValueError):self.settings={}
        self.dark=bool(self.settings.get('dark',False));self.top=bool(self.settings.get('top',True))
        self.font_family=self.settings.get('font_family','JetBrains Mono')
        self.font_size=int(self.settings.get('font_size',10))
        self.font_scale=float(self.settings.get('font_scale',1.0))
        self.current_folder=None
        root.geometry(self.settings.get('geometry','1180x820'))
        root.attributes('-topmost',self.top)
        self.resolver=PathResolver();self.batches=self.history.recent();self.history_complete=False
        self.current=None;self.documents={};self.rendered={};self.callbacks={}
        self.context_entries=[];self.context_stack=[];self.context_label='Paths';self.active_batch=None
        self.view='list';self.root_tab='Paths';self.navigation=0;self.paste_generation=0
        self.closing=False;self.busy=0;self.image_scale=1.;self.image=None;self.image_photo=None
        self.zoom_view=None;self.zoom_images={}
        self.temp=tempfile.TemporaryDirectory(prefix='folio-');self.zip_roots={}
        self.executor=ThreadPoolExecutor(max_workers=3,thread_name_prefix='folio')
        self.results=queue.Queue();self.paste_timer=None
        self.frame=tk.Frame(root);self.frame.pack(fill='both',expand=True,padx=16,pady=10)
        self.nav=tk.Frame(self.frame);self.nav.pack(fill='x',pady=(0,8))
        self.date=tk.Label(self.nav,font=(self.font_family,9),anchor='w');self.date.pack(side='left')
        self.folder_button=self.button(self.nav,'📁 Folders ▾',self.open_folder_dropdown)
        self.folder_button.pack(side='left',padx=(8,0))
        self.top_button=self.button(self.nav,'Top',self.toggle_top);self.top_button.configure(width=5);self.top_button.pack(side='right')
        self.back_button=self.button(self.nav,'← Back',self.go_back);self.back_button.configure(width=7);self.back_button.pack(side='right',padx=5)
        self.copy_button=self.button(self.nav,'Copy',self.copy_visible);self.copy_button.configure(width=5);self.copy_button.pack(side='right',padx=(0,5))
        self.theme_button=self.button(self.nav,'◐',self.toggle_theme);self.theme_button.configure(width=2);self.theme_button.pack(side='right',padx=(0,6))
        self.font_up_button=self.button(self.nav,'A+',lambda:self.zoom(1));self.font_up_button.configure(width=3);self.font_up_button.pack(side='right',padx=(0,2))
        self.font_down_button=self.button(self.nav,'A-',lambda:self.zoom(-1));self.font_down_button.configure(width=3);self.font_down_button.pack(side='right',padx=(0,2))
        self.font_select_button=self.button(self.nav,'Aa ▾',self.show_font_menu);self.font_select_button.configure(width=4);self.font_select_button.pack(side='right',padx=(0,6))
        self.font_up_button.bind('<Button-3>',lambda e:self.zoom(0))
        self.font_down_button.bind('<Button-3>',lambda e:self.zoom(0))
        self.prettify_button=self.button(self.nav,'Prettify',self.toggle_prettify)
        self.pretty=False
        self.switcher=tk.Frame(self.nav)
        self.tab_buttons={}
        for label,callback in [('Paths',self.show_history),('Recent',self.show_recent),('Downloads',self.show_downloads),('Adhoc',self.show_adhoc),('Projects',self.show_projects)]:
            button=self.button(self.switcher,label,callback);button.pack(side='left',padx=3);self.tab_buttons[label]=button
        self.switcher.pack(side='right',padx=7)
        self.input_frame=tk.Frame(self.frame);self.input_frame.pack(fill='x',pady=(0,10))
        self.path_input=tk.Text(self.input_frame,height=2,wrap='word',font=(self.font_family,self.font_size),bd=1,relief='solid',padx=10,pady=8,undo=True)
        self.path_input.pack(fill='x')
        self.path_input.bind('<<Modified>>',self.input_changed)
        for key in ('<Control-v>','<Control-V>','<Shift-Insert>','<<Paste>>'):self.path_input.bind(key,self.paste)
        self.content=tk.Frame(self.frame);self.content.pack(fill='both',expand=True)
        self.list_frame=tk.Frame(self.content)
        self.list_text=tk.Text(self.list_frame,wrap='none',font=(self.font_family,self.font_size),cursor='hand2',bd=0,padx=5,pady=2)
        self.list_scroll=ttk.Scrollbar(self.list_frame,command=self.list_text.yview)
        self.list_text.configure(yscrollcommand=self.list_scrolled)
        self.list_scroll.pack(side='right',fill='y');self.list_text.pack(fill='both',expand=True)
        self.html=HtmlFrame(self.content,messages_enabled=False,javascript_enabled=False,
                           request_func=local_resource,on_link_click=self.open_link,threading_enabled=False,
                           fontscale=self.font_scale,
                           selected_text_highlight_color=LIGHT['selected'],selected_text_color=LIGHT['selected_fg'])
        self.html.bind('<Button-3>',self.menu,add='+');self.html.html.bind('<Button-3>',self.menu,add='+')
        self.source_frame=tk.Frame(self.content)
        self.gutter=tk.Text(self.source_frame,width=5,wrap='none',state='disabled',bd=0,font=(self.font_family,self.font_size),padx=5,pady=12)
        self.gutter.pack(side='left',fill='y')
        self.source=tk.Text(self.source_frame,wrap='word',font=(self.font_family,self.font_size),bd=0,padx=12,pady=12)
        self.source_scroll=ttk.Scrollbar(self.source_frame,command=self.source_yview)
        self.source.configure(yscrollcommand=self.source_scrolled)
        self.source_scroll.pack(side='right',fill='y');self.source.pack(fill='both',expand=True)
        self.source.bind('<Button-3>',self.menu)
        self.image_frame=tk.Frame(self.content)
        self.image_canvas=tk.Canvas(self.image_frame,highlightthickness=0)
        self.image_v=ttk.Scrollbar(self.image_frame,orient='vertical',command=self.image_canvas.yview)
        self.image_h=ttk.Scrollbar(self.image_frame,orient='horizontal',command=self.image_canvas.xview)
        self.image_canvas.configure(yscrollcommand=self.image_v.set,xscrollcommand=self.image_h.set)
        self.image_canvas.grid(row=0,column=0,sticky='nsew');self.image_v.grid(row=0,column=1,sticky='ns');self.image_h.grid(row=1,column=0,sticky='ew')
        self.image_frame.rowconfigure(0,weight=1);self.image_frame.columnconfigure(0,weight=1)
        self.image_canvas.bind('<ButtonPress-1>',lambda e:self.image_canvas.scan_mark(e.x,e.y))
        self.image_canvas.bind('<B1-Motion>',lambda e:self.image_canvas.scan_dragto(e.x,e.y,gain=1))
        self.image_canvas.bind('<Button-3>',self.menu)
        for event,direction in [('<Button-4>',-1),('<Button-5>',1)]:
            self.image_canvas.bind(event,lambda e,d=direction:self.image_wheel(e,d))
        self.table=None;self.embedded_tables=[]
        self.status=tk.Label(self.frame,font=(self.font_family,9),anchor='w');self.status.pack(fill='x')
        root.bind('<Control-v>',self.paste);root.bind('<Control-V>',self.paste)
        root.bind('<Shift-Insert>',self.paste);root.bind('<<Paste>>',self.paste)
        root.bind('<Escape>',lambda e:self.escape());root.bind('<Control-l>',lambda e:self.escape())
        root.bind('<Control-f>',lambda e:self.find());root.bind('<Control-Shift-C>',lambda e:self.copy_content())
        for key in ('<Control-plus>','<Control-equal>','<Control-KP_Add>'):root.bind(key,lambda e:self.zoom(1))
        for key in ('<Control-minus>','<Control-KP_Subtract>'):root.bind(key,lambda e:self.zoom(-1))
        for key in ('<Control-0>','<Control-KP_0>'):root.bind(key,lambda e:self.zoom(0))
        root.protocol('WM_DELETE_WINDOW',self.close)
        self.apply_theme();self.show_history();self.tick();self.clock()

    def button(self,parent,text,command):
        return tk.Button(parent,text=text,command=command,font=(self.font_family,9),relief='solid',bd=1,
                         padx=8,pady=3,takefocus=False,cursor='hand2')

    def save_settings(self):
        self.settings_path.parent.mkdir(parents=True,exist_ok=True)
        self.settings_path.write_text(json.dumps(dict(dark=self.dark,top=self.top,font_size=self.font_size,font_scale=self.font_scale,font_family=self.font_family,geometry=self.root.geometry())))
        os.chmod(self.settings_path,0o600)

    def toggle_top(self):
        self.top=not self.top;self.root.attributes('-topmost',self.top)
        self.update_buttons();self.save_settings()

    def toggle_theme(self):
        self.dark=not self.dark;self.apply_theme();self.save_settings()
        if self.view=='list':self.render_list()
        elif self.view=='document':self.render_document()
        elif self.view=='source':self.show_source()

    def apply_theme(self):
        self.palette=DARK if self.dark else LIGHT;p=self.palette
        style=ttk.Style(self.root);style.theme_use('clam')
        for name in ('Vertical.TScrollbar','Horizontal.TScrollbar'):
            style.configure(name,background=p['line'],troughcolor=p['bg'],bordercolor=p['bg'],arrowcolor=p['muted'],lightcolor=p['line'],darkcolor=p['line'])
        def paint(widget):
            if isinstance(widget,(tk.Frame,tk.Label,tk.Button,tk.Text,tk.Canvas)):
                options={'bg':p['bg']}
                if isinstance(widget,(tk.Label,tk.Button,tk.Text)):options['fg']=p['fg']
                if isinstance(widget,tk.Button):options.update(activebackground=p['button'],activeforeground=p['fg'],highlightbackground=p['line'])
                if isinstance(widget,tk.Text):options.update(insertbackground=p['fg'],selectbackground=p['selected'],selectforeground=p['selected_fg'])
                try:widget.configure(**options)
                except tk.TclError:pass
            # HtmlFrame's internal nodes/widgets are managed by its renderer.
            if widget is self.html:return
            for child in widget.winfo_children():paint(child)
        self.root.configure(bg=p['bg']);paint(self.frame)
        self.date.configure(fg=p['muted']);self.status.configure(fg=p['muted']);self.gutter.configure(fg=p['muted'])
        self.html.configure(selected_text_highlight_color=p['selected'],selected_text_color=p['selected_fg'])
        self.update_buttons()

    def update_buttons(self):
        p=self.palette
        self.top_button.configure(text=('✓ ' if self.top else '')+'Top',bg=p['button'] if self.top else p['bg'],fg=p['green'])
        self.theme_button.configure(text='☀' if self.dark else '◐',fg=p['muted'])
        self.font_down_button.configure(bg=p['bg'],fg=p['muted'])
        self.font_up_button.configure(bg=p['bg'],fg=p['muted'])
        self.font_select_button.configure(bg=p['bg'],fg=p['muted'])
        self.folder_button.configure(bg=p['bg'],fg=p['green'] if self.current_folder else p['muted'])
        for label,button in self.tab_buttons.items():button.configure(bg=p['button'] if label==self.root_tab else p['bg'],fg=p['green'] if label==self.root_tab else p['muted'])

    def show_font_menu(self):
        menu=tk.Menu(self.root,tearoff=False,font=(self.font_family,10))
        for f in AVAILABLE_FONTS:
            menu.add_command(label=('✓ ' if f==self.font_family else '   ')+f,command=lambda fam=f:self.set_font_family(fam))
        x=self.font_select_button.winfo_rootx()
        y=self.font_select_button.winfo_rooty()+self.font_select_button.winfo_height()
        menu.tk_popup(x,y)

    def set_font_family(self,family):
        self.font_family=family
        self.source.configure(font=(family,self.font_size))
        self.gutter.configure(font=(family,self.font_size))
        self.list_text.configure(font=(family,self.font_size))
        self.path_input.configure(font=(family,self.font_size))
        self.date.configure(font=(family,9))
        self.status.configure(font=(family,9))
        self.top_button.configure(font=(family,9))
        self.back_button.configure(font=(family,9))
        self.copy_button.configure(font=(family,9))
        self.theme_button.configure(font=(family,9))
        self.font_up_button.configure(font=(family,9))
        self.font_down_button.configure(font=(family,9))
        self.font_select_button.configure(font=(family,9))
        self.folder_button.configure(font=(family,9))
        self.prettify_button.configure(font=(family,9))
        for b in self.tab_buttons.values():b.configure(font=(family,9))
        self.update_gutter()
        if self.table:self.table.set_font_family(family)
        for t in self.embedded_tables:t.set_font_family(family)
        if self.view=='document' and self.current and self.current.kind=='markdown':
            self.render_document()
        self.save_settings()

    def clock(self):
        if self.closing:return
        from datetime import datetime
        self.date.configure(text=datetime.now().strftime('%B %-d · %H:%M:%S'))
        self.clock_timer=self.root.after(1000,self.clock)

    def submit(self,function,callback):
        self.busy+=1
        identity=uuid.uuid4().hex;self.callbacks[identity]=callback
        results=self.results
        def completed(future):
            try:results.put((identity,future.result(),None))
            except Exception as exc:results.put((identity,None,str(exc)))
        self.executor.submit(function).add_done_callback(completed)

    def tick(self):
        if self.closing:return
        while True:
            try:identity,value,error=self.results.get_nowait()
            except queue.Empty:break
            self.busy-=1;callback=self.callbacks.pop(identity);callback(value,error)
        self.tick_timer=self.root.after(20 if self.busy else 80,self.tick)

    def input_changed(self,event=None):
        if not self.path_input.edit_modified():return
        self.path_input.edit_modified(False)
        if self.paste_timer:self.root.after_cancel(self.paste_timer)
        self.paste_timer=self.root.after(180,self.parse_input)

    def paste(self,event=None,text=None,image=None):
        if image is None and text is None:
            self.paste_generation+=1;generation=self.paste_generation
            if self.paste_timer:self.root.after_cancel(self.paste_timer);self.paste_timer=None
            self.status.configure(text='Reading clipboard…')
            def ready(value,error):
                if generation!=self.paste_generation:return
                if error:self.status.configure(text=error);return
                text,image=value;self.paste(text=text,image=image)
            self.submit(read_clipboard,ready)
            return 'break'
        self.escape()
        if isinstance(image,Image.Image):
            data=io.BytesIO();image.save(data,'PNG');png=data.getvalue()
            self.paste_generation+=1;generation=self.paste_generation;self.status.configure(text='Reading image…')
            def ready(value,error):
                if generation!=self.paste_generation:return
                warning='Image saved; text could not be read.' if error else 'Image saved; no readable text.' if not value.strip() else None
                self.resolve_dump(value or '',png,warning)
            self.submit(lambda:image_text(png),ready)
        elif text is not None:
            self.path_input.delete('1.0','end');self.path_input.insert('1.0',text)
        return 'break'

    def parse_input(self):
        self.paste_timer=None
        text=self.path_input.get('1.0','end-1c').strip()
        if text:self.resolve_dump(text)

    def resolve_dump(self,source,image=None,warning=None):
        self.paste_generation+=1;generation=self.paste_generation
        self.status.configure(text='Finding files…')
        resolver=self.resolver
        def resolve():
            paths=[]
            for name in pasted_paths(source):
                for path in resolver.resolve(name):
                    paths.append(path)
                    if path.is_dir():
                        paths.extend(sorted((p for p in path.iterdir() if p.is_file()),key=lambda p:p.name.casefold())[:2000])
            return list(dict.fromkeys(paths))
        def ready(paths,error):
            if generation!=self.paste_generation:return
            notice=warning
            if error:
                if image is None:self.status.configure(text=error);return
                paths=[];notice='Image saved; paths could not be resolved.'
            batch=self.history.add(source,paths,image);self.batches.insert(0,batch)
            self.show_history()
            if notice:self.status.configure(text=notice)
        self.submit(resolve,ready)

    def hide_views(self):
        if self.zoom_view:self.unzoom_image()
        for widget in self.content.winfo_children():widget.pack_forget()

    def show_history(self):
        self.navigation+=1;self.root_tab='Paths';self.context_label='Paths';self.context_stack=[];self.current_folder=None
        self.update_folder_button()
        self.view='list';self.show_list_controls();self.render_list()

    def show_list_controls(self):
        self.prettify_button.pack_forget()
        self.root.title('Folio');self.switcher.pack(side='right',padx=7)
        self.date.pack(side='left');self.folder_button.pack(side='left',padx=(8,0),after=self.date)
        self.input_frame.pack(fill='x',before=self.content,pady=(0,10))
        self.back_button.configure(state='disabled' if self.context_label=='Paths' else 'normal')
        self.update_buttons();self.hide_views();self.list_frame.pack(fill='both',expand=True);self.status.configure(text='')

    def label_path(self,path):
        path=Path(path)
        for destination in self.zip_roots.values():
            if path.is_relative_to(destination):return str(path.relative_to(destination))
        try:return str(path.relative_to(Path.home()))
        except ValueError:return str(path)

    def render_list(self):
        text=self.list_text;p=self.palette;text.configure(state='normal');text.delete('1.0','end')
        text.tag_configure('date',foreground=p['muted'],font=(self.font_family,9),spacing1=12,spacing3=7)
        text.tag_configure('file',foreground=p['green'],spacing1=5,spacing3=5)
        text.tag_configure('parent',foreground=p['muted'],spacing1=5,spacing3=5)
        self.list_actions=[]
        def item(label,action,date=False,parent=False):
            tag=f'item{len(self.list_actions)}';self.list_actions.append(action)
            text.insert('end',label+'\n',(tag,'date' if date else ('parent' if parent else 'file')))
            text.tag_bind(tag,'<Button-1>',lambda e,callback=action:callback())
        if self.context_label=='Paths':
            for batch in self.batches:
                item(('◩ ' if batch['has_image'] else '≡ ')+batch['stamp'],lambda b=batch:self.open_dump(b),True)
                for path in batch['paths']:item(('▸ ' if Path(path).is_dir() else '')+self.label_path(path),lambda p=path,b=batch:self.open_group_path(p,b))
                if not batch['paths']:item('Pasted content',lambda b=batch:self.open_dump(b))
        else:
            if self.current_folder and self.current_folder.parent and self.current_folder.parent!=self.current_folder and self.context_label not in ('Paths','Recent','Projects'):
                p_name=self.current_folder.parent.name or str(self.current_folder.parent)
                item(f'◂ .. ({p_name})',lambda:self.open_folder(self.current_folder.parent),parent=True)
            for path in self.context_entries:item(('▸ ' if path.is_dir() else '')+self.label_path(path),lambda p=path:self.load_path(p))
            if not self.context_entries:item('No files',lambda:None,True)
        text.configure(state='disabled')

    def list_scrolled(self,first,last):
        self.list_scroll.set(first,last)
        if self.view=='list' and self.context_label=='Paths' and float(first)>0 and float(last)>.98 and not self.history_complete:
            self.root.after_idle(self.load_more)

    def load_more(self):
        if self.view!='list' or self.context_label!='Paths' or self.history_complete:return
        more=self.history.recent(50,len(self.batches))
        if not more:self.history_complete=True;return
        position=self.list_text.yview()[0];self.batches.extend(more);self.render_list();self.list_text.yview_moveto(position)

    def open_group_path(self,path,batch):
        self.active_batch=batch['id'];self.context_entries=[Path(p) for p in batch['paths']];self.load_path(Path(path))

    def open_dump(self,batch):
        self.active_batch=batch['id'];self.context_entries=[Path(p) for p in batch['paths']]
        image=self.history.image(batch['id']) if batch['has_image'] else None
        self.select_document(Document(None,'image' if image else 'markdown',batch['source'],rows=image))

    def show_context(self,entries,label=None):
        self.navigation+=1
        if label:self.context_label=label
        self.context_entries=list(entries);self.active_batch=None;self.view='list'
        self.show_list_controls();self.render_list()

    def show_recent(self):
        self.context_stack=[];self.root_tab='Recent';self.current_folder=None
        self.update_folder_button()
        self.show_context(self.history.recent_files(),'Recent')

    def show_downloads(self,folder=None):
        self.root_tab='Downloads';self.context_stack=[]
        self.open_folder(Path(folder or Path.home()/'Downloads'),'Downloads',push=False,by_date=True)

    def show_adhoc(self):
        self.context_stack=[];self.root_tab='Adhoc';self.current_folder=None
        self.update_folder_button()
        paths=[repo/'adhoc' for repo in self.resolver.repos if (repo/'adhoc').is_dir()]
        self.show_context(paths,'Adhoc')

    def show_projects(self):
        self.context_stack=[];self.root_tab='Projects';self.current_folder=None
        self.update_folder_button()
        home=Path.home()
        priority_names=['astralane-quant','trailblazer','whisper-typer','grid-grinder','personal-finance']
        priority_paths=[home/name for name in priority_names if (home/name).is_dir()]
        other_projects=[]
        for p in sorted(home.iterdir(),key=lambda x:x.name.casefold()):
            if p.name.startswith('.') or not p.is_dir() or p in priority_paths:continue
            if (p/'.git').exists() or any((p/marker).exists() for marker in ('Cargo.toml','package.json','pyproject.toml','requirements.txt','Makefile')):
                other_projects.append(p)
        self.show_context(priority_paths+other_projects,'Projects')

    def update_folder_button(self):
        folder=self.current_folder
        if not folder and self.current and self.current.path:
            folder=self.current.path.parent
        if folder and folder.is_dir():
            name=folder.name or str(folder)
            if len(name)>18:name=name[:17]+'…'
            self.folder_button.configure(text=f'📁 {name} ▾',state='normal')
        else:
            self.folder_button.configure(text='📁 Folders ▾',state='normal')

    def open_folder_dropdown(self):
        folder=self.current_folder
        if not folder or not folder.is_dir():
            if self.current and self.current.path and self.current.path.parent.is_dir():
                folder=self.current.path.parent
            else:folder=Path.home()
        menu=tk.Menu(self.root,tearoff=False,font=(self.font_family,10))
        if folder.parent and folder.parent!=folder:
            p_name=folder.parent.name or str(folder.parent)
            menu.add_command(label=f'↑ .. ({p_name})',command=lambda p=folder.parent:self.open_folder(p))
            menu.add_separator()
        try:entries=sorted(folder.iterdir(),key=lambda p:(not p.is_dir(),p.name.casefold()))
        except OSError as e:
            menu.add_command(label=f'Error: {e}',state='disabled')
            x=self.folder_button.winfo_rootx();y=self.folder_button.winfo_rooty()+self.folder_button.winfo_height()
            menu.tk_popup(x,y);return
        visible=[p for p in entries if not p.name.startswith('.')] or entries
        dirs=[p for p in visible if p.is_dir()]
        files=[p for p in visible if not p.is_dir()]
        for d in dirs[:30]:
            menu.add_command(label=f'▸ {d.name}/',command=lambda p=d:self.open_folder(p))
        if dirs and files:menu.add_separator()
        for f in files[:40]:
            menu.add_command(label=f'  {f.name}',command=lambda p=f:self.load_path(p))
        if not dirs and not files:
            menu.add_command(label='(Empty folder)',state='disabled')
        x=self.folder_button.winfo_rootx()
        y=self.folder_button.winfo_rooty()+self.folder_button.winfo_height()
        menu.tk_popup(x,y)

    def open_folder(self,path,label=None,push=True,by_date=False):
        path=Path(path).resolve()
        self.current_folder=path;self.update_folder_button()
        self.navigation+=1;navigation=self.navigation
        previous=(self.context_label,list(self.context_entries))
        def ready(entries,error):
            if navigation!=self.navigation:return
            if error:self.status.configure(text=error);return
            if push:self.context_stack.append(previous)
            self.show_context(entries,label or self.label_path(path))
        def listing():
            if not path.is_dir():return []
            if by_date or path.name=='adhoc':return sorted(path.iterdir(),key=lambda p:p.stat().st_mtime,reverse=True)
            return sorted(path.iterdir(),key=lambda p:(not p.is_dir(),p.name.casefold()))
        self.submit(listing,ready)

    def go_back(self):
        if self.zoom_view:
            self.unzoom_image();return
        self.navigation+=1
        if self.view!='list':
            if not self.context_stack and (self.context_label=='Paths' or len(self.context_entries)<=1):
                self.show_history()
            else:
                self.view='list';self.show_list_controls();self.render_list()
        elif self.context_stack:
            label,entries=self.context_stack.pop();self.show_context(entries,label)
        else:self.show_history()

    def escape(self):
        if self.zoom_view:
            self.unzoom_image();return 'break'
        self.navigation+=1;self.paste_generation+=1
        if self.paste_timer:self.root.after_cancel(self.paste_timer);self.paste_timer=None
        self.path_input.delete('1.0','end');self.path_input.edit_modified(False)
        self.show_history();self.path_input.focus_set()
        return 'break'

    def open_link(self,url):
        parsed=urlparse(url)
        if parsed.scheme=='folio-image':
            img_id=parsed.path or parsed.netloc
            if img_id in self.zoom_images:
                src,alt=self.zoom_images[img_id]
                self.open_image_zoom(src,alt)
            return
        if parsed.scheme=='file':
            target_path=Path(unquote(parsed.path))
            if target_path.suffix.lower() in ('.png','.jpg','.jpeg','.gif','.bmp','.webp','.svg'):
                self.open_image_zoom(str(target_path),target_path.name)
                return
            self.load_path(target_path)
        elif parsed.scheme in ('https','http','mailto'):self.open_browser(url)

    def open_image_zoom(self,src,alt='Diagram'):
        title=f"{self.current.path.name} · {alt}" if (self.current and self.current.path) else alt
        if src.startswith('data:image/'):
            try:
                raw=base64.b64decode(src.split(',',1)[1])
                self.zoom_image(raw,title=title)
            except Exception as e:
                self.status.configure(text=f'Error decoding image: {e}')
        else:
            if src.startswith('file://'):p=Path(unquote(urlparse(src).path))
            elif self.current and self.current.path:p=(self.current.path.parent/src).resolve()
            else:p=Path(src).resolve()
            if p.exists():self.zoom_image(p,title=title)
            else:self.status.configure(text=f'Image not found: {src}')

    def zoom_image(self,source_input,title='Diagram'):
        if self.zoom_view:self.unzoom_image()
        self.zoom_view=ImageZoom(self.frame,source_input,None,title,ZoomStyle(self.palette),self.unzoom_image)

    def unzoom_image(self):
        if self.zoom_view:
            self.zoom_view.destroy();self.zoom_view=None

    def open_browser(self,url):
        command=shutil.which('firefox') or shutil.which('xdg-open')
        if not command:raise ValueError('Firefox is not installed.')
        subprocess.Popen([command,url],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)

    def load_path(self,path):
        path=Path(path).resolve();self.navigation+=1;navigation=self.navigation
        if path.is_dir():self.open_folder(path);return
        if path.suffix.lower()=='.pdf':
            try:self.open_browser(path.as_uri());self.history.opened(path);self.status.configure(text='Opened in Firefox')
            except Exception as exc:self.status.configure(text=str(exc))
            return
        if path.suffix.lower()=='.zip':
            if path in self.zip_roots:self.open_folder(self.zip_roots[path],label=path.name);return
            destination=Path(self.temp.name)/('zip-'+uuid.uuid4().hex)/path.stem
            def ready(entries,error):
                if navigation!=self.navigation:return
                if error:self.status.configure(text=error);return
                self.zip_roots[path]=destination;self.context_stack.append((self.context_label,list(self.context_entries)))
                self.show_context(entries,path.name)
            self.status.configure(text='Opening ZIP…');self.submit(lambda:unzip_contents(path,destination),ready);return
        if path in self.documents:self.select_document(self.documents[path]);return
        self.status.configure(text='Opening…')
        def ready(document,error):
            if navigation!=self.navigation:return
            if error:self.status.configure(text=error);return
            self.documents[path]=document
            self.select_document(document)
        self.submit(lambda:load_document(path),ready)

    def select_document(self,document):
        self.navigation+=1;self.current=document;self.view='document';self.pretty=False
        self.prettify_button.pack_forget()
        if document.path:
            self.current_folder=document.path.parent
            self.update_folder_button()
        if document.path and not document.path.is_relative_to(Path(self.temp.name)):self.history.opened(document.path)
        self.root.title(document.path.name if document.path else 'Folio')
        self.input_frame.pack_forget();self.date.pack_forget();self.switcher.pack_forget()
        self.folder_button.pack(side='left',padx=(0,8))
        self.back_button.configure(state='normal')
        self.status.configure(text='');self.render_document()

    def render_document(self):
        if not self.current:return
        self.hide_views();document=self.current;p=self.palette;navigation=self.navigation
        if self.table:self.table.destroy();self.table=None
        if document.kind=='image':
            self.image=Image.open(io.BytesIO(document.rows)).copy();self.image_scale=min(1.,max(100,self.content.winfo_width()-20)/self.image.width)
            self.image_frame.pack(fill='both',expand=True);self.draw_image();return
        if document.kind=='text':self.show_source();return
        if document.kind=='csv':
            self.table=Table(self.content,document.rows,p,self.copy_text,self.submit)
            self.table.font_family=self.font_family
            self.table.font_size=self.font_size
            self.table.row_height=max(24,int(self.font_size*3.0))
            self.table.pack(fill='both',expand=True)
            return
        self.html.pack(fill='both',expand=True)
        key=(str(document.path),document.text,self.dark)
        if key in self.rendered:self.show_html(self.rendered[key]);return
        self.status.configure(text='Rendering…')
        def ready(body,error):
            if navigation!=self.navigation or self.view!='document':return
            if error:
                try:
                    fallback = rendered_body(document.text)
                    self.rendered[key] = fallback
                    self.show_html(fallback)
                    self.status.configure(text='')
                    return
                except Exception:
                    self.status.configure(text=error);self.show_source();return
            self.rendered[key]=body;self.show_html(body);self.status.configure(text='')
        dark=self.dark
        self.submit(lambda:typeset(document.text,dark),ready)

    def show_html(self,body):
        for table in self.embedded_tables:
            # Detach nodes before destroying embedded Tk widgets.
            try:self.html.widget_to_element(table).widget=None
            except (KeyError,tk.TclError):pass
            table.destroy()
        self.embedded_tables=[];rows=[]
        def replace(match):
            parser=TableParser();parser.feed(match[0]);rows.append(parser.rows)
            height=min(500,30*(len(parser.rows)+1)+70)
            return f'<object id="folio-table-{len(rows)-1}" style="width:100%;height:{height}px"></object>'
        body=re.sub(r'<div class="(?:table-controls|table-stats)"[^>]*>[\s\S]*?</div>','',body)
        body=re.sub(r'<t[dh] class="row-grip"[^>]*>[\s\S]*?</t[dh]>','',body)
        body=re.sub(r'<table\b[^>]*>[\s\S]*?</table>',replace,body)
        body=re.sub(r'<button\b[^>]*>[\s\S]*?</button>','',body)
        self.zoom_images = {}
        def wrap_img(match):
            full_tag = match.group(0)
            if full_tag.startswith('<a'):
                return full_tag
            src_m = re.search(r'src=["\']([^"\']+)["\']', full_tag)
            if not src_m:
                return full_tag
            src = src_m.group(1)
            alt_m = re.search(r'alt=["\']([^"\']*)["\']', full_tag)
            alt = alt_m.group(1) if alt_m else 'Diagram'
            img_id = str(len(self.zoom_images))
            self.zoom_images[img_id] = (src, alt)
            return f'<a href="folio-image:{img_id}" style="cursor:zoom-in;display:inline-block">{full_tag}</a>'
        body = re.sub(r'(<a\b[^>]*>[\s\S]*?</a>)|(<img\b[^>]*>)', wrap_img, body)
        css=CSS.replace('article','.article').replace(':root {color-scheme:light}','')
        css=css.replace('"JetBrains Mono"',f'"{self.font_family}","JetBrains Mono"')
        # Tkhtml renders conservative HTML/CSS; advanced scripts never run here.
        max_w = 'none;width:100%' if rows else '1000px'
        css+=f'\n.article {{padding:25px 36px;max-width:{max_w}}} object {{display:block;width:100%}} img {{max-width:100%;cursor:pointer}}\n'
        css=theme_colors(css,self.dark)
        base=self.current.path.parent.as_uri()+'/' if self.current.path else Path.home().as_uri()+'/'
        self.html.load_html('<html><head><style>'+css+'</style></head><body><div class="article">'+body+'</div></body></html>',base_url=base)
        for index,values in enumerate(rows):
            table=Table(self.html,values,self.palette,self.copy_text,self.submit)
            table.font_family=self.font_family
            table.font_size=self.font_size
            table.row_height=max(24,int(self.font_size*3.0))
            self.html.document.getElementById('folio-table-'+str(index)).widget=table
            self.embedded_tables.append(table)

    def transpose(self):
        for table in ([self.table] if self.table else self.embedded_tables):table.transpose()

    def source_yview(self,*args):
        self.source.yview(*args);self.update_gutter()

    def source_scrolled(self,first,last):
        self.source_scroll.set(first,last);self.update_gutter()

    def update_gutter(self):
        if not self.source.winfo_ismapped():return
        self.gutter.configure(state='normal');self.gutter.delete('1.0','end')
        top=self.source.index('@0,0');bottom=self.source.index('@0,'+str(self.source.winfo_height()))
        for line in range(int(top.split('.')[0]),int(bottom.split('.')[0])+1):
            info=self.source.dlineinfo(f'{line}.0')
            if info:
                # A newline with spacing follows the wrapped height of its source line.
                end=self.source.index(f'{line}.0 lineend')
                last=self.source.dlineinfo(end)
                height=(last[1]+last[3]-info[1]) if last else info[3]
                tag='number'+str(line)
                self.gutter.tag_configure(tag,spacing3=max(0,height-info[3]))
                self.gutter.insert('end',str(line)+'\n',tag)
        self.gutter.configure(state='disabled')

    def show_source(self):
        self.view='source';self.hide_views();self.source_frame.pack(fill='both',expand=True)
        text=self.current.text
        if self.pretty:
            try:text=pretty_json(text)
            except (ValueError,RecursionError) as exc:self.pretty=False;self.status.configure(text='Cannot prettify: '+str(exc))
        self.source.configure(state='normal');self.source.delete('1.0','end');self.source.insert('1.0',text);self.source.configure(state='disabled')
        if self.current.path and self.current.path.suffix.lower()=='.json':
            self.prettify_button.configure(text='Original' if self.pretty else 'Prettify')
            self.prettify_button.pack(side='right',padx=5,after=self.theme_button)
        else:self.prettify_button.pack_forget()
        self.root.after_idle(self.update_gutter)
        self.highlight_source()

    def toggle_prettify(self):
        self.pretty=not self.pretty;self.show_source()

    def highlight_source(self):
        if not self.current.path:return
        text=self.source.get('1.0','end-1c')
        if not syntax_safe(text):
            self.status.configure(text='Raw text · syntax coloring skipped for large files or long lines.')
            return
        from pygments import lex
        from pygments.lexers import get_lexer_for_filename, guess_lexer
        from pygments.util import ClassNotFound
        try:lexer=get_lexer_for_filename(self.current.path.name, text)
        except ClassNotFound:
            try:lexer=guess_lexer(text)
            except ClassNotFound:return
        style=source_style(self.dark)
        document=self.current;navigation=self.navigation
        text=self.source.get('1.0','end-1c')
        def tokens():
            result=[];offset=0
            for token,value in lex(text,lexer):
                color=style.style_for_token(token)['color']
                if color:result.append((offset,offset+len(value),color))
                offset+=len(value)
            return result
        def ready(segments,error):
            if error or navigation!=self.navigation or self.current is not document or self.view!='source' or self.source.get('1.0','end-1c')!=text:return
            for start,end,color in segments:
                tag='syntax'+color;self.source.tag_configure(tag,foreground='#'+color)
                self.source.tag_add(tag,f'1.0+{start}c',f'1.0+{end}c')
        self.submit(tokens,ready)

    def draw_image(self):
        if self.image is None:return
        size=(max(1,int(self.image.width*self.image_scale)),max(1,int(self.image.height*self.image_scale)))
        self.image_photo=ImageTk.PhotoImage(self.image.resize(size,Image.Resampling.LANCZOS))
        self.image_canvas.delete('all');self.image_canvas.create_image(0,0,image=self.image_photo,anchor='nw')
        self.image_canvas.configure(scrollregion=(0,0,*size))

    def image_wheel(self,event,direction):
        if event.state & 4:self.zoom(-direction)
        elif event.state & 1:self.image_canvas.xview_scroll(direction,'units')
        else:self.image_canvas.yview_scroll(direction,'units')
        return 'break'

    def zoom(self,direction):
        if direction==0:
            self.font_size=10
            self.font_scale=1.0
            self.image_scale=1.0
        else:
            self.font_size=max(7,min(28,self.font_size+direction))
            self.font_scale=max(0.6,min(2.5,round(self.font_scale+direction*0.1,2)))
            if self.current and self.current.kind=='image':
                self.image_scale=max(.05,min(4,self.image_scale*(1.2 if direction>0 else 1/1.2)))
                self.draw_image()
        try:self.html.configure(fontscale=self.font_scale)
        except Exception:pass
        self.source.configure(font=(self.font_family,self.font_size))
        self.gutter.configure(font=(self.font_family,self.font_size))
        self.update_gutter()
        self.list_text.configure(font=(self.font_family,self.font_size))
        self.path_input.configure(font=(self.font_family,self.font_size))
        if self.table:
            self.table.font_size=self.font_size
            self.table.row_height=max(24,int(self.font_size*3.0))
            self.table.draw()
        for t in self.embedded_tables:
            t.font_size=self.font_size
            t.row_height=max(24,int(self.font_size*3.0))
            t.draw()
        self.save_settings()

    def copy_text(self,text):
        self.root.clipboard_clear();self.root.clipboard_append(text);self.root.update_idletasks()

    def copy_visible(self):
        if self.zoom_view:
            self.zoom_view.copy_action();return
        if self.view!='list' and self.current:
            if self.current.kind=='image' and self.view=='document':self.copy_image()
            else:self.copy_content()
        else:
            paths=[str(p) for batch in self.batches for p in batch['paths']] if self.context_label=='Paths' else [str(p) for p in self.context_entries]
            text='\n'.join(dict.fromkeys(paths))
            if not text and self.context_label=='Paths' and self.batches:text=self.batches[0]['source']
            if text:self.copy_text(text)

    def copy_content(self):
        if self.current:self.copy_text(self.current.text)

    def copy_image(self):
        if self.current and self.current.kind=='image':
            subprocess.run(['xclip','-selection','clipboard','-t','image/png','-i'],input=self.current.rows,timeout=3,check=True)

    def find(self):
        if self.current:
            from tkinter.simpledialog import askstring
            text=askstring('Find','Find in document:',parent=self.root)
            if text and self.view=='document' and self.current.kind=='markdown':self.html.find_text(text)
            elif text:
                self.show_source();index=self.source.search(text,'1.0',nocase=True)
                if index:self.source.see(index);self.source.tag_add('sel',index,f'{index}+{len(text)}c')
        return 'break'

    def menu(self,event):
        if not self.current:return
        menu=tk.Menu(self.root,tearoff=False,font=(self.font_family,10))
        if self.view=='document' and self.current.kind=='markdown':menu.add_command(label='Copy selection',command=lambda:self.copy_text(self.html.get_selection()))
        menu.add_command(label='Copy content',command=self.copy_content)
        if self.current.path:menu.add_command(label='Copy path',command=lambda:self.copy_text(str(self.current.path)))
        if self.current.kind=='image':menu.add_command(label='Copy image',command=self.copy_image)
        else:menu.add_command(label='Rendered' if self.view=='source' and self.current.kind=='markdown' else 'Source',command=self.return_rendered if self.view=='source' and self.current.kind=='markdown' else self.show_source)
        menu.add_command(label='Find',command=self.find)
        menu.add_separator()
        font_menu=tk.Menu(menu,tearoff=False,font=(self.font_family,10))
        for f in AVAILABLE_FONTS:
            font_menu.add_command(label=('✓ ' if f==self.font_family else '   ')+f,command=lambda fam=f:self.set_font_family(fam))
        menu.add_cascade(label='Font family',menu=font_menu)
        menu.add_command(label='Increase font · Ctrl++',command=lambda:self.zoom(1))
        menu.add_command(label='Decrease font · Ctrl+-',command=lambda:self.zoom(-1))
        menu.add_command(label='Reset font · Ctrl+0',command=lambda:self.zoom(0))
        menu.add_separator()
        menu.add_command(label='Back · Esc',command=self.escape);menu.tk_popup(event.x_root,event.y_root)

    def return_rendered(self):
        self.view='document';self.render_document()

    def close(self):
        if self.closing:return
        self.save_settings();self.closing=True
        for timer in (self.paste_timer,self.clock_timer,self.tick_timer):
            if timer:self.root.after_cancel(timer)
        self.executor.shutdown(wait=True,cancel_futures=True)
        self.callbacks.clear()
        for table in self.embedded_tables:
            try:self.html.widget_to_element(table).widget=None
            except (KeyError,tk.TclError):pass
        self.history.close();self.temp.cleanup();self.root.destroy()


def main():
    root=tk.Tk(className='Folio');app=Folio(root)
    if len(sys.argv)==3 and sys.argv[1]=='--read':app.load_path(Path(sys.argv[2]))
    elif len(sys.argv)==3 and sys.argv[1]=='--image':app.paste(image=Image.open(sys.argv[2]))
    elif len(sys.argv)>1:
        paths=[Path(p).expanduser().resolve() for p in sys.argv[1:] if Path(p).expanduser().exists()]
        if paths:
            app.context_entries=paths
            app.load_path(paths[0])
        else:
            app.paste(text='\n'.join(sys.argv[1:]))
    else:
        latest=latest_history_file(app.history, app.batches)
        if latest:
            app.load_path(latest)
    root.mainloop()


if __name__=='__main__':main()
