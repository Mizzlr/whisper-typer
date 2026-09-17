#!/usr/bin/python3
"""Floating dictation history with copy, export and on-demand meeting summaries.

Pasted and grammar-corrected versions use difflib word highlighting.
Grammar is performed by Whisper Typer's independent background queue.
"""
import argparse
import difflib
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import time
import tkinter as tk
from datetime import datetime


def diff_segments(original, corrected):
    """Highlight word edits; punctuation and capitalization stay unhighlighted."""
    words = lambda text: list(re.finditer(r"\w+(?:['’]\w+)*", text, re.UNICODE))
    a, b = words(original), words(corrected)
    removed, added = set(), set()
    for kind, i, j, k, l in difflib.SequenceMatcher(
            None, [m.group().casefold() for m in a], [m.group().casefold() for m in b], autojunk=False).get_opcodes():
        if kind in ('replace', 'delete'): removed.update(range(i, j))
        if kind in ('replace', 'insert'): added.update(range(k, l))
    def segments(text, matches, edits, tag):
        result, offset = [], 0
        for index, match in enumerate(matches):
            result.append((text[offset:match.start()], 'normal'))
            result.append((match.group(), tag if index in edits else 'normal'))
            offset = match.end()
        result.append((text[offset:], 'normal'))
        return result
    return segments(original, a, removed, 'removed'), segments(corrected, b, added, 'added')


def grammar_changed(original, corrected):
    if original == corrected:
        return False
    words = lambda text: [word.casefold() for word in re.findall(r"\w+(?:['’]\w+)*", text, re.UNICODE)]
    return words(original) != words(corrected)


@lru_cache(maxsize=16384)
def timestamp_seconds(value):
    return datetime.fromisoformat(value).timestamp()


class DictationStore:
    RECORD_FIELDS=('final_text','whisper_text','background_review','grammar_gate_decision','correction_accepted','total_latency_ms','ollama_latency_ms','grammar_gate_latency_ms')
    REVIEW_FIELDS=('pasted','corrected','status','accepted','fallback_reason','grammar_latency_ms')
    def __init__(self, history, reviews):
        self.history = Path(history)
        self.reviews = Path(reviews)
        self.records = {}
        self.corrections = {}
        self.tails = {}
        self.bad_lines = 0
        self.loaded_paths = set()
        self.row_cache={}

    def load_older_day(self, allow_archives=False):
        paths = sorted(self.history.glob('*.jsonl'), reverse=True)
        for path in paths:
            if ((allow_archives or datetime.fromtimestamp(time.time()-86400).strftime('%Y-%m-%d') <= path.stem) and path.stem <= datetime.now().strftime('%Y-%m-%d')
                    and path != self.reviews and path not in self.loaded_paths):
                self.loaded_paths.add(path)
                self.refresh()
                return True
        return False

    def release_archives(self):
        """Returning to live view releases old days as well as their Tk cards."""
        cutoff = datetime.fromtimestamp(datetime.now().timestamp()-900).strftime('%Y-%m-%d')
        released = {path for path in self.loaded_paths if path.stem < cutoff}
        self.loaded_paths.difference_update(released)
        days = {path.stem for path in released}
        for timestamp in list(self.records):
            if timestamp[:10] in days: del self.records[timestamp]
        for path in released: self.tails.pop(path,None)

    def read_tail(self, path):
        try:
            size = path.stat().st_size
            offset, partial = self.tails.get(path, (0, b''))
            if size < offset:
                offset, partial = 0, b''
            if size == offset:
                return []
            with path.open('rb') as f:
                f.seek(offset)
                chunk = f.read()
                offset = f.tell()
            lines = (partial + chunk).split(b'\n')
            self.tails[path] = (offset, lines.pop())
            result = []
            for line in lines:
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    if isinstance(value, dict): result.append(value)
                except (ValueError, UnicodeError):
                    self.bad_lines += 1
            return result
        except OSError:
            return []

    def refresh(self):
        changed = False
        paths = sorted((p for p in self.history.glob('*.jsonl') if p != self.reviews), reverse=True)
        if paths:
            if paths[0].stem >= datetime.fromtimestamp(time.time()-86400).strftime('%Y-%m-%d'):
                self.loaded_paths.add(paths[0])
            cutoff_day = datetime.fromtimestamp(datetime.now().timestamp() - 900).strftime('%Y-%m-%d')
            self.loaded_paths.update(path for path in paths if cutoff_day <= path.stem <= datetime.now().strftime('%Y-%m-%d'))
        for path in sorted(self.loaded_paths):
            if path == self.reviews:
                continue
            for record in self.read_tail(path):
                timestamp = record.get('timestamp')
                if timestamp and isinstance(record.get('whisper_text'), str):
                    self.records[timestamp] = record
                    changed = True
        for review in self.read_tail(self.reviews):
            timestamp = review.get('dictation_timestamp')
            if timestamp:
                self.corrections[timestamp] = review
                changed = True
        return changed

    def visible(self, query='', day=''):
        query = query.casefold()
        rows = []
        numbers, counts = {}, {}
        ordered=sorted(self.records)
        for timestamp in set(self.row_cache)-set(self.records):self.row_cache.pop(timestamp,None)
        for timestamp in ordered:
            date = timestamp[:10]
            counts[date] = counts.get(date, 0) + 1
            numbers[timestamp] = counts[date]
        now=time.time()
        for timestamp in reversed(ordered):
            record=self.records[timestamp]
            if day and not timestamp.startswith(day):
                continue
            review=self.corrections.get(timestamp)
            pending=False
            if record.get('background_review') and not review:
                try:pending=now-timestamp_seconds(timestamp)<120
                except ValueError:pending=True
            signature=(tuple(record.get(field) for field in self.RECORD_FIELDS),
                       tuple(review.get(field) for field in self.REVIEW_FIELDS) if review else None,
                       numbers[timestamp],pending)
            cached=self.row_cache.get(timestamp)
            if cached and cached[0]==signature:
                row=cached[1]
                if not query or query in (row['original']+' '+row['corrected']).casefold():rows.append(row.copy())
                continue
            # The baseline is what was already pasted, after spelling/punctuation.
            original = str(record.get('final_text') or record['whisper_text']).strip()
            corrected = original
            status = 'Recorded'
            if review:
                if isinstance(review.get('pasted'), str): original = review['pasted'].strip()
                if isinstance(review.get('corrected'), str): corrected = review['corrected'].strip()
                status = 'Grammar changed' if grammar_changed(original, corrected) else 'No grammar change'
                if review.get('status') == 'skipped': status = 'Grammar skipped'
                if review.get('accepted') is False and review.get('status') == 'unchanged':
                    rejected = review.get('fallback_reason') in {
                        'large_length_change', 'introduced_stutter', 'changed_numeric_fact', 'changed_url',
                        'empty_correction', 'wrapped_or_explained_output', 'removed_protected_term', 'changed_personal_reference', 'introduced_unknown_token',
                    }
                    status = 'Suggestion rejected' if rejected else 'Grammar unavailable'
            elif record.get('background_review'):
                status = 'Checking grammar…' if pending else 'No grammar result'
            elif record.get('grammar_gate_decision') == 'clean':
                status = 'Grammar passed'
            elif record.get('correction_accepted'):
                status = 'Grammar checked'
            if not grammar_changed(original, corrected): corrected = original
            judgment = record.get('grammar_gate_decision')
            if judgment: status += ' · Judgment: ' + str(judgment)
            timings = []
            def timing(label, value):
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    timings.append(f'{label}: {value:.0f} ms')
            timing('Until paste', record.get('total_latency_ms'))
            if review:
                timing('Grammar round trip', review.get('grammar_latency_ms'))
            elif record.get('background_review'):
                timings.append('Grammar: pending' if status.startswith('Checking') else 'Grammar: unavailable')
            else:
                timing('Grammar', record.get('ollama_latency_ms'))
            timing('Judge', record.get('grammar_gate_latency_ms'))
            if query and query not in (original+' '+corrected).casefold():
                continue
            row={'timestamp': timestamp, 'original': original, 'corrected': corrected, 'status': status,
                         'timings': ' · '.join(timings), 'paste_ms': record.get('total_latency_ms'),
                         'grammar_ms': review.get('grammar_latency_ms') if review else record.get('ollama_latency_ms'),
                         'reason': review.get('fallback_reason') if review else None, 'number': numbers[timestamp]}
            self.row_cache[timestamp]=(signature,row.copy())
            rows.append(row)
        return rows


class DictationWindow:
    FONT = 'JetBrains Mono'
    BG = '#171b22'
    PANEL = '#222832'
    FG = '#e8edf4'
    MUTED = '#9ba8ba'
    METADATA = '#778394'
    THEMES = {
        'dark': dict(BG=BG,PANEL=PANEL,FG=FG,MUTED=MUTED,METADATA=METADATA,
                     LINE='#303741',RAIL='#405347',SUCCESS='#a4e7bc',WARNING='#e6bd70',
                     ERROR='#ffaaa9',RECORD='#ef9292',RECORD_ACTIVE='#ffaaaa',
                     COPY_BG='#294036',COPY_FG='#c6e4d2',COPY_HOVER='#365548',
                     REMOVED_BG='#3d242b',ADDED_BG='#20392e'),
        # Triage Desk's warm light background, ink, borders and muted metadata.
        'light': dict(BG='#fdfbf5',PANEL='#f6f1e6',FG='#3a3326',MUTED='#756b55',METADATA='#8a7f66',
                      LINE='#efe9db',RAIL='#cfc6b3',SUCCESS='#2f6a3a',WARNING='#8a6300',
                      ERROR='#b4500c',RECORD='#c0322e',RECORD_ACTIVE='#a52724',
                      COPY_BG='#e3ecdd',COPY_FG='#2e5935',COPY_HOVER='#d5e3cd',
                      REMOVED_BG='#f8e5df',ADDED_BG='#e1efdb'),
    }

    def __init__(self, root, store, settings_path, clipboard_store=None):
        self.root, self.store = root, store
        self.clipboard_store = clipboard_store
        self.clipboard_items = clipboard_store.rows() if clipboard_store else []
        self.clipboard_monitor = None
        self.image_captioner = None
        self.zoom_view = None
        self.event_bridge = None
        self.recordings = None
        self.closing = False
        self.ui_metrics = {}
        self.content_widgets = []
        self.photos = {}
        self.tooltip = None
        self.settings_path = Path(settings_path)
        settings = {}
        try: settings = json.loads(self.settings_path.read_text())
        except (OSError, ValueError): pass
        self.theme=settings.get('theme','light')
        if self.theme not in self.THEMES:self.theme='light'
        for name,color in self.THEMES[self.theme].items():setattr(self,name,color)
        self.older_count = 0
        self.history_expanded = False
        self.fallback_categories=set()
        self.clipboard_archive_reader=DictationStore(self.store.history,self.store.reviews)
        self.clipboard_archive_texts={}
        self.window_start = self.window_end = 0
        self.rows = []
        self.loading_older = False
        self.page_timer = None
        self.header_resize_timer = None
        self.scroll_sync_pending = False
        self.pending_wheel_units = 0
        self.pending_view_args = None
        self.scroll_direction = 0
        self.rendered = None
        self.copy_buttons = {}
        self.row_headers = []
        self.row_starts = {}
        self.row_blocks = {}
        self.row_serial = 0
        self.last_row_day = None
        self.display_day = datetime.now().date()
        self.root.title('Whisper Typer')
        if self.root.tk.call('tk','windowingsystem') == 'x11':
            self.root.wm_attributes('-type','normal')
            # Globally active ICCCM focus model: decline WM_TAKE_FOCUS and
            # advertise input=False, so workspace switches do not activate us.
            self.root.focusmodel('active')
            self.root.protocol('WM_TAKE_FOCUS',lambda:None)
        self.root.configure(background=self.BG)
        self.root.minsize(540, 240)
        geometry = settings.get('geometry', '660x430')
        if re.fullmatch(r'\d+x\d+(?:[+-]\d+[+-]\d+)?', str(geometry)):
            self.root.geometry(geometry)
        self.topmost = tk.BooleanVar(master=root,value=settings.get('topmost', True))
        self.show_clipboard = tk.BooleanVar(master=root,value=settings.get('show_clipboard', False))
        self.show_dictations = tk.BooleanVar(master=root,value=settings.get('show_dictations', True))
        self.show_recordings = tk.BooleanVar(master=root,value=settings.get('show_recordings', True))
        self.root.attributes('-topmost', self.topmost.get())
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.toolbar = tk.Frame(root, background=self.BG)
        self.toolbar.pack(fill='x', padx=8, pady=(2, 0))
        self.toolbar.columnconfigure(0, weight=1)
        controls = self.controls = tk.Frame(self.toolbar, background=self.BG)
        controls.grid(row=0,column=1,sticky='ew')
        self.compact_toolbar = None
        self.clock = tk.StringVar(master=root)
        self.clock_label=tk.Label(self.toolbar, textvariable=self.clock, bg=self.BG, fg=self.METADATA,
                                  font=(self.FONT, 9), anchor='w')
        self.clock_label.grid(row=0,column=0,sticky='w',padx=(12,0))
        self.topmost_button = tk.Button(controls, command=lambda: self.toggle_control(self.topmost, self.set_topmost),
                                       font=(self.FONT, 9), relief='flat', padx=3, pady=0, takefocus=False)
        self.topmost_button.pack(side='right', padx=5)
        self.clipboard_button = tk.Button(controls, command=lambda: self.toggle_control(self.show_clipboard, self.change_view),
                                         font=(self.FONT, 9), relief='flat', padx=3, pady=0, takefocus=False)
        self.recordings_button=tk.Button(controls,command=lambda:self.toggle_control(self.show_recordings,self.change_view),
                                         font=(self.FONT,9),relief='flat',padx=3,pady=0,takefocus=False)
        self.recordings_button.pack(side='right',padx=5)
        self.dictations_button=tk.Button(controls,command=lambda:self.toggle_control(self.show_dictations,self.change_view),
                                         font=(self.FONT,9),relief='flat',padx=3,pady=0,takefocus=False)
        self.dictations_button.pack(side='right',padx=5)
        self.clipboard_button.pack(side='right', padx=5)
        self.update_control_buttons()
        self.record_button = tk.Button(controls, text='● Record', command=self.toggle_recording, font=(self.FONT,9),
                                      bg=self.BG, fg=self.RECORD, activebackground=self.PANEL, activeforeground=self.RECORD_ACTIVE,
                                      relief='flat', padx=3, pady=0, takefocus=False)
        self.record_button.pack(side='right', padx=5)
        self.theme_button=tk.Button(controls,text='◐',command=self.toggle_theme,font=(self.FONT,9),
                                    bg=self.BG,fg=self.METADATA,activebackground=self.PANEL,
                                    activeforeground=self.FG,relief='flat',padx=3,pady=0,takefocus=False)
        self.theme_button.pack(side='right',padx=2)
        body = tk.Frame(root, bg=self.BG)
        self.body=body
        body.pack(fill='both', expand=True, padx=12)
        scroll = tk.Scrollbar(body)
        scroll.pack(side='right', fill='y')
        self.text = tk.Text(body, height=1, width=1, wrap='word', background=self.BG, foreground=self.FG, relief='flat',
                            borderwidth=0, highlightthickness=0, font=(self.FONT, 11),
                            yscrollcommand=scroll.set, cursor='arrow', padx=8, pady=6)
        self.text.pack(side='left', fill='both', expand=True)
        self.toolbar.pack_configure(padx=(8,12+scroll.winfo_reqwidth()+int(self.text.cget('padx'))-5))
        self.text.bind('<Configure>', self.queue_header_resize)
        self.sync_scroll_command = root.register(self.flush_pending_scroll)
        scroll.configure(command=self.scroll)
        self.text.bind('<Button-4>', lambda event: self.wheel(-1))
        self.text.bind('<Button-5>', lambda event: self.wheel(1))
        self.text.bind('<MouseWheel>', lambda event: self.wheel(-1 if event.delta > 0 else 1))
        root.bind('<Button-4>', lambda event: self.wheel(-1))
        root.bind('<Button-5>', lambda event: self.wheel(1))
        root.bind('<MouseWheel>', lambda event: self.wheel(-1 if event.delta > 0 else 1))
        self.text.tag_configure('timestamp', foreground=self.METADATA, spacing1=12, spacing3=5, font=(self.FONT, 9))
        self.text.tag_configure('label', foreground=self.MUTED)
        self.text.tag_configure('removed', foreground=self.ERROR, background=self.REMOVED_BG)
        self.text.tag_configure('added', foreground=self.SUCCESS, background=self.ADDED_BG)
        self.text.tag_configure('normal', foreground=self.FG)
        self.text.tag_configure('space', spacing3=10)
        self.save_timer = None
        self.root.bind('<Configure>', self.moved)
        self.update_clock()
        self.root.after(100, self.poll)

    def update_clock(self):
        now = datetime.now()
        self.clock.set(f'{now:%B} {now.day}  ·  {now:%H:%M:%S}')
        if now.date() != self.display_day:
            self.display_day = now.date()
            self.rendered = None
        self.root.after(1000 - now.microsecond // 1000, self.update_clock)

    def moved(self, event):
        if event.widget is self.root:
            if self.save_timer is not None:
                self.root.after_cancel(self.save_timer)
            self.save_timer = self.root.after(750, self.save_settings)
            self.layout_toolbar()

    def layout_toolbar(self):
        compact = self.root.winfo_width() < self.clock_label.winfo_reqwidth() + self.controls.winfo_reqwidth() + 50
        if compact == self.compact_toolbar: return
        self.compact_toolbar = compact
        self.clock_label.grid_configure(row=0,column=0,columnspan=2 if compact else 1)
        self.controls.grid_configure(row=1 if compact else 0,column=0 if compact else 1,columnspan=2 if compact else 1)

    def set_topmost(self):
        self.root.attributes('-topmost', self.topmost.get())
        self.update_control_buttons()
        self.save_settings()

    def toggle_theme(self):
        previous=self.THEMES[self.theme]
        self.theme='dark' if self.theme=='light' else 'light'
        palette=self.THEMES[self.theme]
        colors={color:palette[name] for name,color in previous.items()}
        for name,color in palette.items():setattr(self,name,color)
        def recolor(widget):
            options={}
            for option in ('background','foreground','activebackground','activeforeground',
                           'disabledforeground','selectbackground','selectforeground','insertbackground','troughcolor'):
                try:value=widget.cget(option)
                except tk.TclError:continue
                if value in colors:options[option]=colors[value]
            if options:widget.configure(**options)
            if isinstance(widget,tk.Text):
                for tag in widget.tag_names():
                    for option in ('foreground','background'):
                        value=widget.tag_cget(tag,option)
                        if value in colors:widget.tag_configure(tag,**{option:colors[value]})
            if isinstance(widget,tk.Canvas):
                for item in widget.find_all():
                    value=widget.itemcget(item,'fill')
                    if value in colors:widget.itemconfigure(item,fill=colors[value])
            for child in widget.winfo_children():recolor(child)
        recolor(self.root)
        self.update_control_buttons()
        self.save_settings()

    def toggle_control(self, variable, command):
        variable.set(not variable.get())
        command()

    def update_control_buttons(self):
        for button, variable, label in ((self.topmost_button, self.topmost, 'Top'),
                                        (self.clipboard_button, self.show_clipboard, 'Clipboard'),
                                        (self.dictations_button,self.show_dictations,'Dictations'),
                                        (self.recordings_button,self.show_recordings,'Recordings')):
            selected = variable.get()
            button.configure(text=('✓ ' if selected else '  ') + label,
                             bg=self.COPY_BG if selected else self.BG,
                             fg=self.COPY_FG if selected else self.MUTED,
                             activebackground=self.COPY_HOVER if selected else self.PANEL,
                             activeforeground=self.FG)

    def change_view(self):
        self.update_control_buttons()
        if self.page_timer is not None:
            self.root.after_cancel(self.page_timer);self.page_timer=None
        self.pending_view_args=None;self.pending_wheel_units=0
        self.older_count = 0
        self.history_expanded = False
        self.fallback_categories=set()
        self.window_start = self.window_end = 0
        self.rendered = None
        self.text.yview_moveto(0)
        self.render()
        self.save_settings()

    MAX_RENDERED = 100
    PAGE_SIZE = 12
    FALLBACK_COUNT = 20

    def load_older(self):
        if self.loading_older: return
        self.loading_older = True
        try:
            self.history_expanded=True
            self.rows=self.collect_rows()
            while self.show_dictations.get() and len(self.rows) <= self.window_end:
                if not self.store.load_older_day(allow_archives=True): break
                self.rows = self.collect_rows()
            self.window_end = min(len(self.rows), self.window_end + self.PAGE_SIZE)
            self.window_start = max(self.window_start, self.window_end - self.MAX_RENDERED)
            self.render(self.rows)
        finally:
            self.loading_older = False

    def load_newer(self):
        if self.loading_older or not self.window_start: return
        self.loading_older = True
        try:
            self.window_start = max(0, self.window_start - self.PAGE_SIZE)
            self.window_end = min(self.window_end, self.window_start + self.MAX_RENDERED)
            self.render(self.rows)
        finally:
            self.loading_older = False

    @staticmethod
    def recent_count(rows):
        now = time.time()
        count = 0
        for row in rows:
            if row.get('recording_active'):
                count += 1
                continue
            try:
                if now-timestamp_seconds(row.get('sort_timestamp',row['timestamp']))>900:break
            except ValueError: break
            count += 1
        return count

    @classmethod
    def initial_count(cls, rows):
        count=cls.recent_count(rows)
        if not count:return min(cls.FALLBACK_COUNT,len(rows))
        categories={cls.row_category(row) for row in rows}
        for category in categories:
            indices=[index for index,row in enumerate(rows) if cls.row_category(row)==category]
            if not cls.recent_count([rows[index] for index in indices]):
                count=max(count,indices[min(cls.FALLBACK_COUNT,len(indices))-1]+1)
        return count

    def scroll(self, *args):
        if self.text.tk.getboolean(self.text.tk.call(self.text._w,'pendingsync')):
            self.pending_view_args=args
            self.pending_wheel_units=0
            self.wait_for_scroll_layout()
            return
        before = self.text.yview()[0]
        self.text.yview(*args)
        if args[0] == 'scroll':
            direction = 1 if int(args[1]) > 0 else -1
        else:
            direction = 1 if float(args[1]) > before else -1
        self.queue_history_page(direction)

    def wheel(self, direction):
        if self.text.tk.getboolean(self.text.tk.call(self.text._w,'pendingsync')):
            self.pending_wheel_units += direction * 2
            self.wait_for_scroll_layout()
            return 'break'
        self.text.yview_scroll(direction * 2, 'units')
        self.queue_history_page(direction)
        return 'break'

    def wait_for_scroll_layout(self):
        if self.scroll_sync_pending or self.closing: return
        self.scroll_sync_pending=True
        self.text.tk.call(self.text._w,'sync','-command',self.sync_scroll_command)

    def flush_pending_scroll(self):
        self.scroll_sync_pending=False
        if self.closing: return
        args,self.pending_view_args=self.pending_view_args,None
        units,self.pending_wheel_units=self.pending_wheel_units,0
        if args:self.scroll(*args)
        if units:
            self.text.yview_scroll(units,'units')
            self.queue_history_page(1 if units>0 else -1)

    def queue_history_page(self, direction):
        # A mouse can deliver many events before Tk gets an idle turn. Never
        # enqueue one expensive render per event, and honor direction changes.
        self.scroll_direction = direction
        if self.closing or self.page_timer is not None: return
        top, bottom = self.text.yview()
        if (direction > 0 and bottom >= .98) or (direction < 0 and top < .001):
            self.page_timer = self.root.after(16, self.page_history)

    def page_history(self):
        self.page_timer = None
        if self.closing: return
        if self.text.tk.getboolean(self.text.tk.call(self.text._w,'pendingsync')):
            self.page_timer=self.root.after(16,self.page_history)
            return
        top, bottom = self.text.yview()
        if self.scroll_direction > 0 and bottom >= .98:
            self.load_older()
        elif self.scroll_direction < 0 and top < .001:
            self.trim_history()

    def trim_history(self):
        if self.loading_older or self.text.yview()[0] >= .001: return
        if self.window_start:
            self.load_newer()
        elif self.older_count:
            self.window_end = min(self.MAX_RENDERED, self.initial_count(self.rows))
            if self.recent_count(self.rows) and 'dictation' not in self.fallback_categories:self.store.release_archives()
            self.render()

    def within_history_window(self,row):
        if self.history_expanded or row.get('recording_active') or self.row_category(row) in self.fallback_categories: return True
        try:
            stamp = timestamp_seconds(row.get('sort_timestamp',row['timestamp']))
            return stamp >= time.time() - 86400
        except (ValueError, TypeError):
            return False

    def copy(self, row, button):
        if row.get('kind') in ('text', 'image') and self.clipboard_monitor:
            self.clipboard_monitor.copy(row['item'])
        else:
            self.root.clipboard_clear()
            self.root.clipboard_append(row['corrected'])
        self.root.update_idletasks()
        button.configure(text='Copied')
        self.root.after(1500, lambda: button.configure(text='Copy') if button.winfo_exists() else None)

    def queue_header_resize(self, _=None):
        if self.header_resize_timer is None and not self.closing:
            self.header_resize_timer=self.root.after_idle(self.finish_header_resize)

    def finish_header_resize(self):
        self.header_resize_timer=None
        if not self.closing:
            self.resize_headers()
            self.text.tk.call(self.text._w,'sync')
            self.redraw_history()

    def redraw_history(self):
        # Embedded widgets and the text body must repaint together after layout;
        # invalidate visible regions instead of rebuilding any cards.
        pending=[self.text]
        while pending:
            widget=pending.pop()
            if not widget.winfo_ismapped():continue
            widget.event_generate('<Expose>',x=0,y=0,width=widget.winfo_width(),height=widget.winfo_height())
            pending.extend(widget.winfo_children())

    def resize_headers(self, _=None):
        width = max(100, self.text.winfo_width() - 2 * int(self.text.cget('padx')))
        for header in self.row_headers+self.content_widgets:
            if hasattr(header,'caption_title'):
                header.resize(width);continue
            if hasattr(header, 'resize'):
                height=max(90,min(350,self.root.winfo_height()//2-40)) if header.segments or header.summary else 48
                size=(width,height)
                if getattr(header,'_display_size',None)!=size:
                    header._display_size=size;header.resize(*size)
                continue
            if getattr(header, '_display_width', None) == width: continue
            header._display_width = width
            header.configure(width=width)
            if hasattr(header, 'recording_text'):
                header.recording_text.configure(wraplength=max(80,width-20))
                header.columnconfigure(1, minsize=max(80,width-12))

    @staticmethod
    def row_category(row):
        return 'recording' if row.get('kind')=='recording' else 'clipboard' if row.get('kind') in ('text','image') else 'dictation'

    def selected_rows(self, rows):
        enabled={'recording':self.show_recordings.get(),'clipboard':self.show_clipboard.get(),'dictation':self.show_dictations.get()}
        return (row for row in rows if enabled[self.row_category(row)])

    def prepare_category_fallback(self):
        groups={'dictation':self.store.visible(),'recording':self.recordings.rows() if self.recordings else [],
                'clipboard':sorted(self.clipboard_items,key=lambda item:item['timestamp'],reverse=True)}
        owned=self.voice_owned_text(groups['dictation']+groups['recording'])|self.clipboard_archive_owners()
        groups['clipboard']=[item for item in groups['clipboard'] if item['kind']!='text' or item['text'].strip() not in owned]
        enabled={'recording':self.show_recordings.get(),'clipboard':self.show_clipboard.get(),'dictation':self.show_dictations.get()}
        self.fallback_categories={category for category,rows in groups.items() if enabled[category] and not self.recent_count(sorted(rows,key=self.row_sort_key,reverse=True))}

    @staticmethod
    def voice_owned_text(rows):
        # Ownership is independent of visibility: hiding a voice category must
        # not let the clipboard monitor reintroduce its text under another icon.
        return {text.strip() for row in rows for text in
                (row['original'],row['corrected'],*(segment['text'] for segment in row.get('segments',[]))) if text.strip()}

    def clipboard_archive_owners(self):
        if not self.show_clipboard.get():return set()
        paths=set()
        for item in self.clipboard_items:
            if item['kind']!='text':continue
            try:day=datetime.fromisoformat(item['timestamp']).astimezone().strftime('%Y-%m-%d')
            except ValueError:continue
            path=self.store.history/(day+'.jsonl')
            if path not in self.store.loaded_paths:paths.add(path)
        for path in set(self.clipboard_archive_texts)-paths:
            self.clipboard_archive_texts.pop(path,None);self.clipboard_archive_reader.tails.pop(path,None)
        for path in paths:
            owned=self.clipboard_archive_texts.setdefault(path,set())
            for record in self.clipboard_archive_reader.read_tail(path):
                if isinstance(record.get('whisper_text'),str):
                    for field in ('whisper_text','final_text'):
                        value=record.get(field)
                        if isinstance(value,str) and value.strip():owned.add(value.strip())
        result=set().union(*self.clipboard_archive_texts.values())
        for review in self.store.corrections.values():
            for field in ('pasted','corrected'):
                value=review.get(field)
                if isinstance(value,str) and value.strip():result.add(value.strip())
        return result

    def collect_rows(self):
        rows = self.store.visible()
        if self.recordings: rows.extend(self.recordings.rows())
        if not self.clipboard_store or not self.show_clipboard.get():
            return sorted(filter(self.within_history_window, self.selected_rows(rows)),key=self.row_sort_key,reverse=True)
        dictated=self.voice_owned_text(rows)|self.clipboard_archive_owners()
        for item in self.clipboard_items:
            # Automatic pastes and later copies remain owned by their voice
            # category, even when that category is hidden or outside the window.
            if item['kind'] == 'text' and item['text'].strip() in dictated:
                continue
            text = item['text'] or ''
            rows.append({'timestamp': item['timestamp'], 'key': 'clip:'+item['id'], 'kind': item['kind'],
                         'item': item, 'original': text, 'corrected': text, 'status': 'Clipboard',
                         'number':item.get('image_number') if item['kind']=='image' else None,
                         'image_title':(item.get('title') or 'Screenshot') if item['kind']=='image' else None,
                         'image_description':item.get('description') or '',
                         'image_busy':bool(self.image_captioner and item['id'] in self.image_captioner.inflight),
                         'timings': '', 'paste_ms': None, 'grammar_ms': None, 'reason': None})
        return sorted(filter(self.within_history_window, self.selected_rows(rows)), key=self.row_sort_key, reverse=True)

    @staticmethod
    def row_sort_key(row):
        return (bool(row.get('recording_active')), row.get('sort_timestamp',row['timestamp']))

    def clipboard_changed(self):
        self.clipboard_items = self.clipboard_store.rows()
        if self.image_captioner:self.image_captioner.request_scan()
        self.render()

    def receive_events(self, events):
        for event in events:
            payload = event['payload']
            if event['type']=='window':
                self.root.deiconify();self.root.lift()
            elif event['type'] == 'dictation':
                self.store.records[payload['timestamp']] = payload
            elif event['type'] == 'grammar_review':
                self.store.corrections[payload['dictation_timestamp']] = payload
            elif self.recordings: self.recordings.receive(payload)
        self.update_record_control()
        self.render()

    def toggle_recording(self):
        if not self.recordings: return
        if self.recordings.state == 'idle': self.recordings.start()
        else: self.recordings.stop()
        self.update_record_control(); self.render()

    def update_record_control(self):
        state = self.recordings.state if self.recordings else 'idle'
        self.record_button.configure(text='● Record' if state == 'idle' else 'Finishing…' if state == 'finishing' else '■ Stop',
                                     state='disabled' if state == 'finishing' else 'normal')

    def hide_tooltip(self, *_):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

    def show_tooltip(self, widget, row):
        self.hide_tooltip()
        message = row['status']
        if row.get('kind')=='recording':
            activity=row.get('activity',{})
            message+=' · '+activity.get('vad_mode','Voice detector')+' · split after 900 ms silence'
            probability=activity.get('speech_probability')
            if isinstance(probability,(int,float)):message+=f' · speech {probability:.0%}'
        if row.get('reason'):
            reason = {'large_length_change': 'proposed correction changed the length too much'}.get(row['reason'], row['reason'].replace('_', ' '))
            message += ': '+reason+'; original retained'
        self.tooltip = tk.Toplevel(self.root)
        self.tooltip.overrideredirect(True)
        if self.root.tk.call('tk','windowingsystem') == 'x11':
            self.tooltip.wm_attributes('-type','tooltip')
            self.tooltip.focusmodel('active')
            self.tooltip.protocol('WM_TAKE_FOCUS',lambda:None)
        self.tooltip.attributes('-topmost', True)
        self.tooltip.geometry(f'+{widget.winfo_rootx()+12}+{widget.winfo_rooty()+22}')
        tk.Label(self.tooltip, text=message, bg=self.PANEL, fg=self.FG,
                 font=(self.FONT, 9), padx=8, pady=4, wraplength=350).pack()

    def render(self, rows=None):
        started = time.perf_counter()
        if not self.window_end:self.prepare_category_fallback()
        self.rows = self.collect_rows() if rows is None else list(filter(self.within_history_window, self.selected_rows(rows)))
        if not self.window_end:
            while 'dictation' in self.fallback_categories and sum(self.row_category(row)=='dictation' for row in self.rows)<self.FALLBACK_COUNT:
                if not self.store.load_older_day(allow_archives=True):break
                self.rows=self.collect_rows()
        if not self.loading_older and self.window_start and self.rendered:
            first_key = self.rendered[0][0]
            for index, row in enumerate(self.rows):
                if row.get('key',row['timestamp']) == first_key:
                    delta = index - self.window_start
                    self.window_start += delta
                    self.window_end += delta
                    break
        if not self.window_start:
            self.window_end = max(self.window_end, min(self.MAX_RENDERED, self.initial_count(self.rows)))
        self.window_end = min(len(self.rows), self.window_end)
        self.window_start = max(0, min(self.window_start, self.window_end))
        self.window_end = min(self.window_end, self.window_start + self.MAX_RENDERED)
        self.older_count = max(0, self.window_end - self.recent_count(self.rows))
        visible = self.rows[self.window_start:self.window_end]
        if self.recordings:
            for row in visible:
                if row.get('kind')=='recording' and self.recordings.request_title(row['key'].split(':',1)[1]):row['title_busy']=True
        last_day = None
        for row in visible:
            try: day = datetime.fromisoformat(row['timestamp']).astimezone().date()
            except ValueError: day = self.display_day
            row['_heading'] = day if day != last_day and day != self.display_day else None
            last_day = day
        signature = [(r.get('key',r['timestamp']), r['original'], r['corrected'], r['status'], r['timings'],
                      r.get('reason'), r.get('number'), r.get('_heading'),
                      tuple((s['seq'],s['timestamp'],s['text'],s.get('end_reason')) for s in r.get('segments',[])),
                      r.get('summary'),r.get('summary_title'),r.get('title_busy'),r.get('summary_busy'),r.get('summary_error'),r.get('image_title'),r.get('image_description'),r.get('image_busy')) for r in visible]
        if signature == self.rendered:
            self.record_render_metrics(started)
            return
        previous = self.rendered or []
        position = self.text.index('@0,0')
        at_top = self.text.yview()[0] < .001
        anchor = None
        for key, mark in self.row_starts.items():
            if self.text.compare(mark, '<=', position) and (anchor is None or self.text.compare(mark, '>', self.row_starts[anchor[0]])):
                distance = self.text.count(mark, position, 'indices')
                anchor = (key, distance[0] if distance else 0)
        old = {entry[0]:entry for entry in previous}
        new = {entry[0]:entry for entry in signature}
        old_keys, new_keys = list(old), list(new)
        retained = set()
        for block in difflib.SequenceMatcher(None,old_keys,new_keys,autojunk=False).get_matching_blocks():
            retained.update(old_keys[block.a:block.a+block.size])
        self.text.configure(state='normal')
        self.hide_tooltip()
        updated_recordings = set()
        for row in visible:
            key = row.get('key',row['timestamp'])
            if key in retained and key in self.row_blocks and old.get(key) != new.get(key):
                if row.get('kind') == 'recording' and self.update_recording_row(row):
                    updated_recordings.add(key)
                elif row.get('kind')=='image' and self.update_image_row(row):
                    updated_recordings.add(key)
        # Keep every unchanged card's real Tk widgets, thumbnails, and marks.
        # Only changed, removed, or reordered cards are rebuilt.
        for key in list(self.row_blocks):
            if key not in retained or (old.get(key) != new.get(key) and key not in updated_recordings): self.remove_row(key)
        if not previous: self.text.delete('1.0','end')
        next_key = None
        for row in reversed(visible):
            key = row.get('key',row['timestamp'])
            if key not in self.row_blocks:
                location = self.row_blocks[next_key]['start'] if next_key else 'end-1c'
                self.insert_row(row,location)
            next_key = key
        if not signature:
            self.text.delete('1.0','end')
            self.text.insert('end','No items.' if self.show_dictations.get() or self.show_recordings.get() or self.show_clipboard.get() else 'No categories selected.','label')
        self.rendered = signature
        self.text.configure(state='disabled')
        self.resize_headers()
        # A bounded render changes embedded window sizes and wrapped line heights.
        # Reconcile those metrics before restoring its anchor or reading yview.
        self.text.tk.call(self.text._w,'sync')
        if previous and (not at_top or self.loading_older):
            if anchor and anchor[0] in self.row_starts:
                self.text.yview(self.row_starts[anchor[0]]+f'+{anchor[1]}c')
            else: self.text.yview(position)
        else: self.text.yview_moveto(0)
        self.redraw_history()
        self.record_render_metrics(started)

    def record_render_metrics(self, started):
        self.ui_metrics = {'render_ms':round((time.perf_counter()-started)*1000,3),
                           'visible_cards':len(self.row_blocks),'loaded_dictations':len(self.store.records),
                           'older_cards':self.older_count,'window_start':self.window_start,'window_end':self.window_end,
                           'visible_categories':{category:sum(self.row_category(row)==category for row in self.rows[self.window_start:self.window_end]) for category in ('clipboard','dictation','recording')},
                           'visible_recording_titles':sum(bool(row.get('summary_title')) for row in self.rows[self.window_start:self.window_end]),
                           'pending_recording_titles':sum(bool(session.get('title_busy')) for session in self.recordings.sessions.values()) if self.recordings else 0}

    def remove_row(self, key):
        block = self.row_blocks.pop(key)
        self.text.delete(block['start'],block['end'])
        self.text.mark_unset(block['start'],block['end'])
        if block.get('recording_tail'): self.text.mark_unset(block['recording_tail'])
        for widget in block['headers']:
            self.row_headers.remove(widget); widget.destroy()
        for widget in block['content']:
            self.content_widgets.remove(widget); widget.destroy()
        self.row_starts.pop(key,None); self.copy_buttons.pop(key,None); self.photos.pop(key,None)

    def insert_row(self, row, location):
        self.row_serial += 1
        start_mark, end_mark = f'row_start_{self.row_serial}', f'row_end_{self.row_serial}'
        self.text.mark_set('row_insert',location)
        self.text.mark_gravity('row_insert','right')
        self.text.mark_set(start_mark,'row_insert'); self.text.mark_gravity(start_mark,'left')
        header_offset, content_offset = len(self.row_headers), len(self.content_widgets)
        row_key = row.get('key', row['timestamp'])
        stamp = row['timestamp'].replace('T', ' ')
        try:
            recorded = datetime.fromisoformat(row['timestamp']).astimezone()
            if row.get('_heading'):
                self.text.insert('row_insert', f'{recorded:%B} {recorded.day}, {recorded.year}\n', 'timestamp')
            self.last_row_day = recorded.date()
            stamp = recorded.strftime('%H:%M:%S')
        except ValueError: pass
        self.row_starts[row_key] = start_mark
        header = tk.Frame(self.text, bg=self.BG, height=32)
        header.pack_propagate(False)
        self.row_headers.append(header)
        button = tk.Button(header, text='Copy', width=6, font=(self.FONT, 9), bg=self.COPY_BG, fg=self.COPY_FG, relief='flat', padx=2, pady=1,
                           activebackground=self.COPY_HOVER, activeforeground=self.COPY_FG, takefocus=False)
        button.pack(side='right')
        recording_actions={}
        image_action=None
        if row.get('kind')=='image':
            image_action=tk.Button(header,text='Describing…' if row.get('image_busy') else 'Described' if row.get('image_description') else 'Describe',
                                   state='disabled' if row.get('image_busy') or row.get('image_description') else 'normal',
                                   command=lambda r=row:self.describe_image(r),font=(self.FONT,9),
                                   bg=self.BG,fg=self.MUTED,disabledforeground=self.METADATA,
                                   activebackground=self.PANEL,activeforeground=self.FG,relief='flat',
                                   padx=3,pady=1,takefocus=False)
            image_action.pack(side='right',padx=3)
        if row.get('kind')=='recording':
            for label,command in (('Download',lambda r=row:self.download_recording(r)),
                                  ('Summarize',lambda r=row:self.summarize_recording(r))):
                action=tk.Button(header,text=label,font=(self.FONT,9),bg=self.BG,fg=self.MUTED,
                                 activebackground=self.PANEL,activeforeground=self.FG,relief='flat',
                                 padx=3,pady=1,takefocus=False,command=command)
                action.pack(side='right',padx=3)
                recording_actions[label]=action
            recording_actions['Download'].configure(state='normal' if row['corrected'] else 'disabled')
            recording_actions['Summarize'].configure(state='normal' if row['corrected'] and not row.get('summary_busy') else 'disabled')
        changed = grammar_changed(row['original'], row['corrected'])
        color = self.SUCCESS if changed else self.WARNING if row['status'].startswith('Checking') else self.MUTED
        if row['status'].startswith(('Grammar unavailable', 'Grammar skipped', 'No grammar result')):
            color = self.ERROR
        if row.get('kind') == 'recording':
            color = self.RECORD if row.get('recording_error') or row['recording_state'] in ('starting','started','chunk','finishing') else self.MUTED
            if not row['corrected']: button.configure(state='disabled')
        marker = tk.Label(header, text='▣' if row.get('kind') == 'image' else '▤' if row.get('kind') == 'text' else '◉' if row.get('kind') == 'recording' else '●', bg=self.BG, fg=color)
        marker.pack(side='left', padx=(0, 6))
        marker.bind('<Enter>', lambda _, w=marker, r=row: self.show_tooltip(w, r))
        marker.bind('<Leave>', self.hide_tooltip)
        millis = lambda value: f'{value:.0f} ms' if isinstance(value, (int, float)) else '—'
        timing = '  ·  '+row['status'] if row.get('kind') == 'recording' else '' if row.get('kind') in ('text', 'image') else '  ·  '+millis(row['paste_ms'])+' / '+millis(row['grammar_ms'])
        number = f"  ·  #{row['number']}" if row.get('number') else ''
        metadata = tk.Label(header, text=stamp+number+timing, bg=self.BG, fg=self.METADATA,
                            font=(self.FONT, 9), anchor='w')
        metadata.pack(side='left', fill='x', expand=True)
        button.configure(command=lambda r=row, b=button: self.copy(r, b))
        self.copy_buttons[row_key] = button
        self.text.window_create('row_insert', window=header)
        self.text.insert('row_insert', '\n')
        recording_view = None
        if row.get('kind') == 'recording' and (row.get('recording_active') or row.get('segments') or row.get('summary')):
            from recording_view import RecordingView
            recording_view=RecordingView(self.text,row,self,self.copy_summary)
            self.row_headers.append(recording_view)
            self.text.window_create('row_insert',window=recording_view)
            self.text.insert('row_insert','\n')
        elif row.get('kind') == 'image':
            image_view=None
            try:
                photo = tk.PhotoImage(file=str(self.clipboard_store.images / row['item']['thumbnail']))
                self.photos[row_key] = photo
                from image_view import ImageView
                preview=ImageView(self.text,photo,row['image_title'],row['image_description'],self,lambda r=row:self.zoom_image(r))
                image_view=preview
                self.content_widgets.append(preview)
                self.text.window_create('row_insert', window=preview)
            except tk.TclError:
                self.text.insert('row_insert', 'Image unavailable', 'label')
            self.text.insert('row_insert', '\n')
        elif row.get('kind') == 'text':
            preview = '\n'.join(row['original'].splitlines()[:4])
            if len(preview) > 400: preview = preview[:400]+'…'
            elif preview != row['original']: preview += '…'
            self.text.insert('row_insert', preview+'\n', 'normal')
        else:
            left, right = diff_segments(row['original'], row['corrected'])
            lines = [('− ', left), ('+ ', right)] if changed else [('', left)]
            for label, segments in lines:
                self.text.insert('row_insert', label, 'label')
                for phrase, tag in segments: self.text.insert('row_insert', phrase, tag)
                self.text.insert('row_insert', '\n')
        divider = tk.Frame(self.text, bg=self.LINE, height=1, borderwidth=0)
        self.row_headers.append(divider)
        self.text.window_create('row_insert', window=divider, pady=5)
        self.text.insert('row_insert', '\n')
        self.text.mark_set(end_mark,'row_insert'); self.text.mark_gravity(end_mark,'left')
        self.text.mark_gravity(start_mark,'right')
        self.row_blocks[row_key] = {'start':start_mark,'end':end_mark,
                                    'headers':self.row_headers[header_offset:], 'content':self.content_widgets[content_offset:]}
        if recording_view:
            recording_view.bind_wheel(header,*header.winfo_children(),divider)
            self.row_blocks[row_key].update(view=recording_view, segments=list(row['segments']),actions=recording_actions,
                                           metadata=metadata,marker=marker,stamp=stamp,heading=row.get('_heading'))
        elif row.get('kind')=='image':
            self.row_blocks[row_key].update(image_view=image_view,image_action=image_action,
                                           image_timestamp=row['timestamp'],image_file=row['item']['image'])

    def update_image_row(self,row):
        block=self.row_blocks[row['key']]
        if block.get('image_timestamp')!=row['timestamp'] or block.get('image_file')!=row['item']['image'] or not block.get('image_view'):return False
        block['image_view'].update_caption(row['image_title'],row['image_description'])
        block['image_action'].configure(text='Describing…' if row.get('image_busy') else 'Described' if row.get('image_description') else 'Describe',state='disabled' if row.get('image_busy') or row.get('image_description') else 'normal')
        return True

    def describe_image(self,row):
        if self.image_captioner:self.image_captioner.request(row['item']['id'])

    def zoom_image(self,row):
        if self.zoom_view:self.unzoom_image();return
        from image_zoom import ImageZoom
        self.zoom_view=ImageZoom(self.body,self.clipboard_store.images/row['item']['image'],
                                 self.photos[row['key']],row['image_title'],self,self.unzoom_image,
                                 lambda button,r=row:self.copy(r,button))

    def unzoom_image(self):
        if self.zoom_view:self.zoom_view.destroy();self.zoom_view=None

    def update_recording_row(self, row):
        block = self.row_blocks[row['key']]
        previous = block.get('segments')
        segments = row['segments']
        if previous is None or block['heading'] != row.get('_heading') or not (row.get('recording_active') or segments or row.get('summary')):
            return False
        block['view'].update_row(row)
        block['segments'] = list(segments)
        block['metadata'].configure(text=block['stamp']+'  ·  '+row['status'])
        block['marker'].configure(fg=self.RECORD if row.get('recording_error') or row['recording_active'] else self.MUTED)
        block['marker'].bind('<Enter>',lambda _,w=block['marker'],r=row:self.show_tooltip(w,r))
        button = self.copy_buttons[row['key']]
        button.configure(state='normal' if row['corrected'] else 'disabled',command=lambda r=row,b=button:self.copy(r,b))
        block['actions']['Download'].configure(state='normal' if row['corrected'] else 'disabled',command=lambda r=row:self.download_recording(r))
        block['actions']['Summarize'].configure(state='normal' if row['corrected'] and not row.get('summary_busy') else 'disabled',command=lambda r=row:self.summarize_recording(r))
        return True

    def copy_summary(self,text):
        self.root.clipboard_clear();self.root.clipboard_append(text)

    def summarize_recording(self,row):
        if not self.recordings:return
        self.recordings.summarize(row['key'].split(':',1)[1]);self.render()

    def download_recording(self,row):
        from tkinter import filedialog,messagebox
        from recording_summary import transcript_export
        filename=filedialog.asksaveasfilename(parent=self.root,title='Save transcript',defaultextension='.txt',
                    initialfile='meeting-'+row['timestamp'][:19].replace(':','-')+'.txt',filetypes=[('Text','*.txt')])
        if not filename:return
        try:
            fd=os.open(filename,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
            with os.fdopen(fd,'w',encoding='utf-8') as out:out.write(transcript_export(row))
        except OSError:messagebox.showerror('Save transcript','Could not save the transcript.',parent=self.root)

    def poll(self):
        changed=self.store.refresh()
        now=time.time()
        expired=False
        for row in self.rows:
            try:
                pending_expired=row['status'].startswith('Checking grammar') and now-timestamp_seconds(row['timestamp'])>=120
            except (ValueError,TypeError):
                pending_expired=False
            if pending_expired or not self.within_history_window(row):
                expired=True
                break
        if changed or expired or self.rendered is None:
            self.render()
        self.root.after(5000, self.poll)

    def save_settings(self):
        try:
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.settings_path.with_suffix('.tmp')
            fd = os.open(tmp, os.O_WRONLY|os.O_CREAT|os.O_TRUNC, 0o600)
            with os.fdopen(fd, 'w') as f:
                json.dump({'geometry':self.root.geometry(), 'topmost':self.topmost.get(),
                           'show_clipboard':self.show_clipboard.get(),'show_dictations':self.show_dictations.get(),
                           'show_recordings':self.show_recordings.get(),'theme':self.theme}, f)
            os.replace(tmp, self.settings_path)
        except OSError:
            pass

    def close(self):
        if self.closing: return
        self.closing = True
        self.unzoom_image()
        if self.recordings: self.recordings.close()
        if self.clipboard_monitor: self.clipboard_monitor.close()
        if self.image_captioner:self.image_captioner.close()
        self.hide_tooltip()
        self.save_settings()
        for timer in self.root.tk.call('after','info'):
            # Each widget owns its registered callback command. Cancel the timer
            # here and let that widget destroy its command exactly once.
            self.root.tk.call('after','cancel',timer)
        # Hide immediately, but keep the event loop alive until the recorder
        # persists its final chunks. Exiting first would let systemd kill it.
        self.root.withdraw()
        self.finish_close()

    def finish_close(self):
        if self.recordings and self.recordings.process and self.recordings.process.poll() is None:
            self.root.after(50,self.finish_close)
            return
        if self.event_bridge: self.event_bridge.close()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history-dir', type=Path, default=Path.home()/'.whisper-typer-history')
    parser.add_argument('--reviews', type=Path, default=Path.home()/'.cache/whisper-typer/grammar-review.jsonl')
    parser.add_argument('--settings', type=Path, default=Path.home()/'.cache/whisper-typer/dictation-window.json')
    parser.add_argument('--clipboard-dir', type=Path, default=Path.home()/'.cache/whisper-typer/clipboard')
    parser.add_argument('--vision-host',default='http://192.168.0.103:11434')
    parser.add_argument('--vision-model',default='qwen3-vl:2b-instruct')
    parser.add_argument('--event-port', type=int, default=8768)
    parser.add_argument('--recordings-dir', type=Path, default=Path.home()/'.cache/whisper-typer/recordings')
    parser.add_argument('--recorder', type=Path, default=Path.home()/'.local/lib/whisper-typer/voice-journal-recorder')
    parser.add_argument('--config', type=Path, default=Path.home()/'whisper-typer/config.yaml')
    args = parser.parse_args()
    root = tk.Tk(className='WhisperTyper')
    from clipboard_history import ClipboardStore, ClipboardMonitor
    clips = ClipboardStore(args.clipboard_dir)
    app = DictationWindow(root, DictationStore(args.history_dir, args.reviews), args.settings, clips)
    app.clipboard_monitor = ClipboardMonitor(root, clips, app.clipboard_changed)
    from image_caption import ImageCaptioner
    app.image_captioner=ImageCaptioner(root,clips,app.clipboard_changed,args.vision_host,args.vision_model)
    app.store.refresh(); app.render()
    from ui_events import UiEventBridge
    app.event_bridge = UiEventBridge(root, app.receive_events, args.event_port,lambda:app.ui_metrics)
    from recording_session import RecordingSessions
    app.recordings = RecordingSessions(args.recordings_dir,args.recorder,args.config,app.event_bridge.enqueue)
    app.render()
    root.mainloop()


if __name__ == '__main__':
    main()
