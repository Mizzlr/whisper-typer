"""Scrollable in-window image viewer with bounded background resize work, adapted for Folio."""
import io
import math
import queue
import subprocess
import threading
import tkinter as tk
from pathlib import Path
from PIL import Image


def limits(size, bounds):
    width, height = size
    maximum = min(8, 8192 / max(1, width), 8192 / max(1, height),
                  math.sqrt(16_000_000 / max(1, width * height)))
    fit = min(1, bounds[0] / max(1, width), bounds[1] / max(1, height), maximum)
    return fit / 4, maximum, fit


def put_latest(channel, value):
    try:
        channel.put_nowait(value)
    except queue.Full:
        try: channel.get_nowait()
        except queue.Empty: pass
        channel.put_nowait(value)


class ZoomStyle:
    def __init__(self, p=None):
        p = p or {}
        self.BG = p.get('bg', '#ffffff')
        self.FG = p.get('fg', '#111111')
        self.PANEL = p.get('panel', '#ffffff')
        self.MUTED = p.get('muted', '#595959')
        self.LINE = p.get('line', '#d0d0d0')
        self.FONT = 'JetBrains Mono'
        self.COPY_BG = p.get('button', '#eeeeee')
        self.COPY_FG = p.get('fg', '#111111')
        self.COPY_HOVER = p.get('selected', '#cbdff5')


class ImageZoom(tk.Frame):
    def __init__(self, parent, source_input, thumbnail, title, style, close, copy=None):
        super().__init__(parent, bg=style.BG)
        self.source_input = source_input
        self.photo = thumbnail
        self.results = queue.Queue(maxsize=1)
        self.tasks = queue.Queue(maxsize=1)
        self.stop = threading.Event()
        self.serial = 0
        self.poll_timer = self.resize_timer = None
        self.closed = False
        self.scale = self.target_scale = None
        self.source_size = None
        self.focal = None
        self.offset = (0, 0)
        self.previous_focus = self.focus_get()
        self.previous_x_focus = None
        if self.tk.call('tk', 'windowingsystem') == 'x11':
            try:
                from Xlib import display
                connection = display.Display()
                focus = connection.get_input_focus().focus
                self.previous_x_focus = getattr(focus, 'id', focus)
                connection.close()
            except (ImportError, OSError): pass
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        toolbar = tk.Frame(self, bg=style.BG)
        toolbar.grid(row=0, column=0, columnspan=2, sticky='ew', padx=8, pady=6)
        self.title = tk.Label(toolbar, text=title, bg=style.BG, fg=style.MUTED,
                              font=(style.FONT, 9), anchor='w', width=1)
        self.title.pack(side='left', fill='x', expand=True)

        def button(label, command):
            widget = tk.Button(toolbar, text=label, command=command, font=(style.FONT, 9),
                               bg=style.COPY_BG, fg=style.COPY_FG, relief='flat',
                               activebackground=style.COPY_HOVER, padx=5, pady=1, takefocus=False)
            widget.pack(side='right', padx=2)
            return widget

        def default_copy(btn):
            try:
                if isinstance(source_input, (bytes, bytearray)):
                    raw = source_input
                elif isinstance(source_input, Image.Image):
                    bio = io.BytesIO()
                    source_input.save(bio, format='PNG')
                    raw = bio.getvalue()
                else:
                    raw = Path(source_input).read_bytes()
                subprocess.run(['xclip', '-selection', 'clipboard', '-t', 'image/png', '-i'], input=raw, timeout=3, check=True)
                if btn and btn.winfo_exists():
                    btn.configure(text='Copied')
                    self.after(1500, lambda: btn.configure(text='Copy') if not self.closed and btn.winfo_exists() else None)
            except Exception: pass

        self.copy_action = lambda: (copy or default_copy)(self.copy_button)
        self.close_button = button('Close', close)
        self.copy_button = button('Copy', self.copy_action)
        self.fit_button = button('Fit', self.fit)
        self.plus_button = button('+', lambda: self.zoom(1.25))
        self.percent = tk.Label(toolbar, text='…', width=5, font=(style.FONT, 9),
                                bg=style.BG, fg=style.MUTED)
        self.percent.pack(side='right', padx=2)
        self.minus_button = button('−', lambda: self.zoom(1 / 1.25))
        self.canvas = tk.Canvas(self, bg=style.BG, borderwidth=0, highlightthickness=0,
                                cursor='fleur', width=1, height=1)
        self.canvas.grid(row=1, column=0, sticky='nsew')
        horizontal = tk.Scrollbar(self, orient='horizontal', command=self.canvas.xview)
        vertical = tk.Scrollbar(self, command=self.canvas.yview)
        horizontal.grid(row=2, column=0, sticky='ew')
        vertical.grid(row=1, column=1, sticky='ns')
        self.horizontal = horizontal; self.vertical = vertical
        self.canvas.configure(xscrollcommand=horizontal.set, yscrollcommand=vertical.set)
        self.image_item = (self.canvas.create_image(0, 0, image=thumbnail, anchor='nw')
                           if thumbnail is not None else self.canvas.create_image(0, 0, anchor='nw'))
        self.canvas.bind('<Configure>', self.resized)
        self.canvas.bind('<Button-1>', self.pan_start)
        self.canvas.bind('<B1-Motion>', self.pan_move)

        def context_menu(event):
            menu = tk.Menu(self, tearoff=False, font=(style.FONT, 9), bg=style.BG, fg=style.FG,
                           activebackground=style.PANEL, activeforeground=style.FG)
            menu.add_command(label='Copy image (Ctrl+C)', command=self.copy_action)
            menu.add_command(label='Fit to window (0)', command=self.fit)
            menu.add_command(label='Close (Esc)', command=close)
            menu.tk_popup(event.x_root, event.y_root)

        self.canvas.bind('<Button-3>', context_menu)
        self.bind('<Button-3>', context_menu)

        for widget in (self, toolbar, *toolbar.winfo_children(), self.canvas, horizontal, vertical):
            widget.bind('<Button-4>', lambda event: self.wheel(event, 1.25))
            widget.bind('<Button-5>', lambda event: self.wheel(event, 1 / 1.25))
            widget.bind('<MouseWheel>', lambda event: self.wheel(event, 1.25 if event.delta > 0 else 1 / 1.25))
            widget.bind('<Shift-Button-4>', lambda _: self.pan('x', -3))
            widget.bind('<Shift-Button-5>', lambda _: self.pan('x', 3))
            widget.bind('<Shift-MouseWheel>', lambda event: self.pan('x', -3 if event.delta > 0 else 3))
        for scrollbar, axis in ((horizontal, 'x'), (vertical, 'y')):
            scrollbar.bind('<Button-4>', lambda _, a=axis: self.pan(a, -3))
            scrollbar.bind('<Button-5>', lambda _, a=axis: self.pan(a, 3))
            scrollbar.bind('<MouseWheel>', lambda event, a=axis: self.pan(a, -3 if event.delta > 0 else 3))
        self.root = self.winfo_toplevel()
        self.key_bindings = []
        for sequence, command in (('<Escape>', close), ('<plus>', lambda: self.zoom(1.25)),
                                  ('<equal>', lambda: self.zoom(1.25)),
                                  ('<minus>', lambda: self.zoom(1 / 1.25)), ('<Key-0>', self.fit),
                                  ('<Left>', lambda: self.pan('x', -3)), ('<Right>', lambda: self.pan('x', 3)),
                                  ('<Up>', lambda: self.pan('y', -3)), ('<Down>', lambda: self.pan('y', 3)),
                                  ('<Control-c>', self.copy_action),
                                  ('<Control-C>', self.copy_action),
                                  ('<Key-c>', self.copy_action),
                                  ('<Key-C>', self.copy_action)):
            def handler(_, action=command):
                action()
                return 'break'
            self.key_bindings.append((sequence, self.root.bind(sequence, handler, add='+')))
        self.worker = threading.Thread(target=self.decode, args=(source_input, self.tasks, self.results, self.stop),
                                       daemon=True, name='folio-image-zoom')
        self.worker.start()
        self.place(x=0, y=0, relwidth=1, relheight=1)
        self.poll_timer = self.after(20, self.poll)
        self.focus_timer = self.after_idle(self.focus_viewer)

    def focus_viewer(self):
        self.focus_timer = None
        if not self.closed: self.canvas.focus_force()

    def bounds(self):
        return max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())

    def resized(self, _):
        if self.closed: return
        if self.resize_timer: self.after_cancel(self.resize_timer)
        self.resize_timer = self.after(60, self.load)

    def fit(self):
        self.target_scale = None
        self.focal = None
        self.load()
        return 'break'

    def zoom(self, factor, point=None):
        if self.closed or not self.source_size: return
        minimum, maximum, fit = limits(self.source_size, self.bounds())
        requested = max(minimum, min(maximum, (self.target_scale or fit) * factor))
        if requested == self.target_scale: return
        x, y = point or (self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2)
        pw = self.photo.width() if self.photo else max(1, self.canvas.winfo_width())
        ph = self.photo.height() if self.photo else max(1, self.canvas.winfo_height())
        self.focal = (max(0, min(1, (self.canvas.canvasx(x) - self.offset[0]) / pw)),
                      max(0, min(1, (self.canvas.canvasy(y) - self.offset[1]) / ph)), x, y)
        self.target_scale = requested
        self.load()

    def wheel(self, event, factor):
        point = (event.x, event.y) if event.widget == self.canvas else None
        self.zoom(factor, point)
        return 'break'

    def pan(self, axis, units):
        if not self.closed:
            (self.canvas.xview_scroll if axis == 'x' else self.canvas.yview_scroll)(units, 'units')
        return 'break'

    def pan_start(self, event):
        self.canvas.focus_set()
        self.canvas.scan_mark(event.x, event.y)
        return 'break'

    def pan_move(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        return 'break'

    def load(self):
        if self.closed: return
        if self.resize_timer:
            self.after_cancel(self.resize_timer)
            self.resize_timer = None
        self.serial += 1
        put_latest(self.tasks, (self.serial, self.bounds(), self.target_scale))

    @staticmethod
    def decode(source_input, tasks, results, stop):
        source = None
        try:
            while not stop.is_set():
                try: serial, bounds, requested = tasks.get(timeout=.1)
                except queue.Empty: continue
                try:
                    if source is None:
                        if isinstance(source_input, (bytes, bytearray)):
                            with Image.open(io.BytesIO(source_input)) as original:
                                source = original.convert('RGBA')
                        elif isinstance(source_input, Image.Image):
                            source = source_input.convert('RGBA')
                        else:
                            with Image.open(source_input) as original:
                                source = original.convert('RGBA')
                    minimum, maximum, fit = limits(source.size, bounds)
                    scale = fit if requested is None else max(minimum, min(maximum, requested))
                    size = tuple(max(1, round(dimension * scale)) for dimension in source.size)
                    image = source.resize(size, Image.Resampling.LANCZOS)
                    output = io.BytesIO()
                    image.save(output, format='PNG', compress_level=1)
                    image.close()
                    result = (serial, output.getvalue(), scale, source.size)
                except (OSError, ValueError): result = (serial, None, None, None)
                if not stop.is_set(): put_latest(results, result)
        finally:
            if source is not None: source.close()

    def poll(self):
        self.poll_timer = None
        if self.closed: return
        while not self.results.empty():
            serial, data, scale, size = self.results.get_nowait()
            if serial != self.serial: continue
            if not data:
                self.percent.configure(text='Error')
                continue
            self.photo = tk.PhotoImage(data=data, master=self)
            self.scale, self.source_size = scale, size
            if self.target_scale is not None: self.target_scale = scale
            self.percent.configure(text=f'{scale:.0%}')
            self.canvas.itemconfigure(self.image_item, image=self.photo)
            width, height = self.bounds()
            self.offset = (max(0, (width - self.photo.width()) / 2),
                           max(0, (height - self.photo.height()) / 2))
            self.canvas.coords(self.image_item, *self.offset)
            region = (max(width, self.photo.width()), max(height, self.photo.height()))
            self.canvas.configure(scrollregion=(0, 0, *region))
            u, v, x, y = self.focal or (.5, .5, width / 2, height / 2)
            self.canvas.xview_moveto((self.offset[0] + u * self.photo.width() - x) / region[0])
            self.canvas.yview_moveto((self.offset[1] + v * self.photo.height() - y) / region[1])
        self.poll_timer = self.after(20, self.poll)

    def destroy(self):
        self.closed = True
        self.stop.set()
        self.canvas.itemconfigure(self.image_item, image='')
        self.photo = None
        for timer in (self.poll_timer, self.resize_timer, self.focus_timer):
            if timer: self.after_cancel(timer)
        for sequence, binding in self.key_bindings:
            try: self.root.unbind(sequence, binding)
            except Exception: pass
        if self.previous_x_focus is not None:
            try:
                from Xlib import X, display
                connection = display.Display()
                connection.set_input_focus(self.previous_x_focus, X.RevertToParent, X.CurrentTime, onerror=lambda *_: None)
                connection.sync(); connection.close()
            except (ImportError, OSError): pass
        elif self.previous_focus and self.previous_focus.winfo_exists():
            self.previous_focus.focus_force()
        super().destroy()
