"""Private, bounded text/image clipboard history and asynchronous X11 capture."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
import hashlib
import io
import itertools
import os
from pathlib import Path
import queue
import shutil
import sqlite3
import subprocess
import time

from PIL import Image, ImageOps


class ClipboardStore:
    MAX_BYTES = 32 * 1024 * 1024

    def __init__(self, directory, limit=200):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.directory.chmod(0o700)
        self.images = self.directory / 'images'
        self.images.mkdir(exist_ok=True, mode=0o700)
        self.images.chmod(0o700)
        self.database = self.directory / 'history.sqlite'
        os.close(os.open(self.database, os.O_CREAT | os.O_WRONLY, 0o600))
        self.database.chmod(0o600)
        self.limit = limit
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('CREATE TABLE IF NOT EXISTS items (id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, kind TEXT NOT NULL, text TEXT, image TEXT, thumbnail TEXT, width INTEGER, height INTEGER)')
            db.execute('CREATE TABLE IF NOT EXISTS sources (path TEXT PRIMARY KEY, mtime_ns INTEGER, size INTEGER)')
            db.execute("CREATE TABLE IF NOT EXISTS image_titles (id TEXT PRIMARY KEY, title TEXT NOT NULL, model TEXT NOT NULL, latency_ms REAL NOT NULL, description TEXT NOT NULL DEFAULT '')")
            if 'description' not in {row['name'] for row in db.execute('PRAGMA table_info(image_titles)')}:
                db.execute("ALTER TABLE image_titles ADD COLUMN description TEXT NOT NULL DEFAULT ''")
            db.execute('CREATE TABLE IF NOT EXISTS image_numbers (id TEXT PRIMARY KEY, day TEXT NOT NULL, number INTEGER NOT NULL, UNIQUE(day,number))')
            db.execute('CREATE TABLE IF NOT EXISTS image_counters (day TEXT PRIMARY KEY, value INTEGER NOT NULL)')
            for item in list(db.execute("SELECT id,timestamp FROM items WHERE kind='image' ORDER BY timestamp,id")):
                self.assign_image_number(db,item['id'],item['timestamp'])
            db.execute('DELETE FROM sources WHERE mtime_ns < ?', (int((time.time()-172800)*1_000_000_000),))

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.database, timeout=1)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def write_private(path, data):
        temp = path.with_suffix('.tmp')
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'wb') as out:
            out.write(data)
        os.replace(temp, path)

    def add(self, kind, payload, timestamp=None):
        """Exact text; images normalized to PNG, with separately cached thumbnails."""
        stamp = timestamp or datetime.now().isoformat()
        if kind == 'text':
            if not isinstance(payload, str) or not payload.strip() or len(payload.encode()) > 1024 * 1024:
                return False
            digest = hashlib.sha256(b'text\0' + payload.encode()).hexdigest()
            text, image, thumbnail, width, height = payload, None, None, None, None
        else:
            if not payload or len(payload) > self.MAX_BYTES:
                return False
            try:
                with Image.open(io.BytesIO(payload)) as source:
                    if source.width * source.height > 40_000_000:
                        return False
                    normalized = ImageOps.exif_transpose(source).convert('RGBA')
                    width, height = normalized.size
                    digest = hashlib.sha256(b'image\0' + str(normalized.size).encode() + normalized.tobytes()).hexdigest()
                    image, thumbnail = digest + '.png', digest + '-thumb.png'
                    if not (self.images/image).exists() or not (self.images/thumbnail).exists():
                        full = io.BytesIO(); normalized.save(full, format='PNG')
                        normalized.thumbnail((144, 88), Image.Resampling.LANCZOS)
                        small = io.BytesIO(); normalized.save(small, format='PNG')
                        self.write_private(self.images / image, full.getvalue())
                        self.write_private(self.images / thumbnail, small.getvalue())
                text = None
            except (OSError, ValueError, Image.DecompressionBombError):
                return False
        with self.connect() as db:
            previous = db.execute('SELECT timestamp FROM items WHERE id=?', (digest,)).fetchone()
            if previous:
                return False
            db.execute('INSERT INTO items VALUES (?,?,?,?,?,?,?,?)',
                       (digest, stamp, kind, text, image, thumbnail, width, height))
            if kind!='text':self.assign_image_number(db,digest,stamp)
            removed = list(db.execute('SELECT image,thumbnail FROM items ORDER BY timestamp DESC,id DESC LIMIT -1 OFFSET ?', (self.limit,)))
            db.execute('DELETE FROM items WHERE id IN (SELECT id FROM items ORDER BY timestamp DESC,id DESC LIMIT -1 OFFSET ?)', (self.limit,))
            db.execute('DELETE FROM image_titles WHERE id NOT IN (SELECT id FROM items)')
            db.execute('DELETE FROM image_numbers WHERE id NOT IN (SELECT id FROM items)')
        for item in removed:
            for name in (item['image'], item['thumbnail']):
                if name: (self.images / name).unlink(missing_ok=True)
        return True

    def rows(self):
        with self.connect() as db:
            return [dict(row) for row in db.execute('SELECT i.*,t.title,t.description,t.model AS caption_model,t.latency_ms AS caption_ms,n.number AS image_number,n.day AS image_day FROM items i LEFT JOIN image_titles t ON t.id=i.id LEFT JOIN image_numbers n ON n.id=i.id ORDER BY i.timestamp DESC,i.id DESC')]

    @staticmethod
    def assign_image_number(db,identity,stamp):
        day=datetime.fromisoformat(stamp).astimezone().date().isoformat()
        previous=db.execute('SELECT day FROM image_numbers WHERE id=?',(identity,)).fetchone()
        if previous and previous['day']==day:return
        db.execute('INSERT INTO image_counters VALUES (?,1) ON CONFLICT(day) DO UPDATE SET value=value+1',(day,))
        number=db.execute('SELECT value FROM image_counters WHERE day=?',(day,)).fetchone()['value']
        db.execute('INSERT INTO image_numbers VALUES (?,?,?) ON CONFLICT(id) DO UPDATE SET day=excluded.day,number=excluded.number',(identity,day,number))

    def save_caption(self, identity, title, description, model, latency_ms):
        """Cache by image content; a result arriving after retention is ignored."""
        with self.connect() as db:
            if not db.execute("SELECT 1 FROM items WHERE id=? AND kind='image'",(identity,)).fetchone():return False
            return db.execute('INSERT INTO image_titles VALUES (?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET title=excluded.title,description=excluded.description,model=excluded.model,latency_ms=excluded.latency_ms',
                              (identity,title,model,latency_ms,description)).rowcount>0

    def pending_images(self):
        with self.connect() as db:
            return [dict(row) for row in db.execute("SELECT i.id,i.image,i.timestamp FROM items i LEFT JOIN image_titles t ON t.id=i.id WHERE i.kind='image' AND t.id IS NULL ORDER BY i.timestamp DESC,i.id DESC")]

    def image_item(self,identity):
        with self.connect() as db:
            row=db.execute("SELECT i.id,i.image,i.timestamp,t.description FROM items i LEFT JOIN image_titles t ON t.id=i.id WHERE i.id=? AND i.kind='image'",(identity,)).fetchone()
            return dict(row) if row else None

    def file_signatures(self):
        with self.connect() as db:
            return {(Path(row['path']),row['mtime_ns'],row['size']) for row in db.execute('SELECT * FROM sources')}

    def remember_file(self, signature):
        with self.connect() as db:
            db.execute('INSERT OR REPLACE INTO sources VALUES (?,?,?)', (str(signature[0]),signature[1],signature[2]))


class ClipboardMonitor:
    def __init__(self, root, store, changed, screenshot_dirs=None):
        import gi
        gi.require_version('Gtk', '3.0')
        gi.require_version('Gdk', '3.0')
        gi.require_version('GdkPixbuf', '2.0')
        from gi.repository import Gdk, GdkPixbuf, GLib, Gtk
        self.root, self.store, self.changed = root, store, changed
        self.Gdk, self.GdkPixbuf = Gdk, GdkPixbuf
        self.context = GLib.MainContext.default()
        self.clipboard = Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='clipboard-history')
        self.tasks = queue.PriorityQueue()
        self.sequence = itertools.count()
        self.executor.submit(self.run_tasks, self.tasks)
        self.results = queue.Queue()
        self.pending = 0
        self.generation = 0
        self.closed = False
        self.screenshot_dirs = screenshot_dirs if screenshot_dirs is not None else [Path.home() / 'Pictures/Screenshots', Path('/tmp')]
        self.seen_files = store.file_signatures()
        self.handler = self.clipboard.connect('owner-change', self.owner_changed)
        self.pump_timer = root.after(20, self.pump)
        self.scan_timer = root.after(100, self.scan_screenshots)
        self.owner_changed()

    @staticmethod
    def run_tasks(tasks):
        while True:
            _, _, save = tasks.get()
            if save is None: return
            save()

    def submit(self, kind, data, stamp=None):
        if self.closed or self.pending >= 64: return
        self.pending += 1
        store, results = self.store, self.results
        def save():
            try:
                signature = None
                if isinstance(data, Path):
                    stat = data.stat(); signature = (data,stat.st_mtime_ns,stat.st_size)
                payload = data.read_bytes() if isinstance(data, Path) else data
                result = store.add(kind, payload, stamp)
                if signature: store.remember_file(signature)
            except (OSError, ValueError, sqlite3.Error):
                result = False
            results.put(result)
        # New copies take precedence over importing older screenshot files.
        self.tasks.put((10 if isinstance(data, Path) else 0, next(self.sequence), save))

    def owner_changed(self, *_):
        self.generation += 1
        generation = self.generation
        stamp = datetime.now().isoformat()
        def targets(clipboard, atoms, *_):
            if generation != self.generation or self.closed: return
            names = {atom.name() for atom in (atoms or [])}
            if 'x-kde-passwordManagerHint' in names: return
            image_type = next((name for name in ('image/png', 'image/jpeg', 'image/bmp', 'image/tiff') if name in names), None)
            if image_type:
                def image_ready(_, selection, *__):
                    if generation == self.generation and not self.closed and 0 < selection.get_length() <= self.store.MAX_BYTES:
                        self.submit('image', selection.get_data(), stamp)
                clipboard.request_contents(self.Gdk.Atom.intern(image_type, False), image_ready)
            elif names.intersection({'UTF8_STRING', 'STRING', 'TEXT', 'text/plain', 'text/plain;charset=utf-8'}):
                def text_ready(_, text, *__):
                    if generation == self.generation and text and not self.closed:
                        self.submit('text', text, stamp)
                clipboard.request_text(text_ready)
        self.clipboard.request_targets(targets)

    def pump(self):
        if self.closed: return
        for _ in range(20):
            if not self.context.pending(): break
            self.context.iteration(False)
        changed = False
        while not self.results.empty():
            changed |= self.results.get_nowait()
            self.pending -= 1
        if changed: self.changed()
        self.pump_timer = self.root.after(20, self.pump)

    def scan_screenshots(self):
        if self.closed: return
        candidates = []
        for directory in self.screenshot_dirs:
            pattern = 'codex-clipboard-*.png' if directory == Path('/tmp') else '*'
            try:
                for path in directory.glob(pattern):
                    stat = path.stat()
                    signature = (path, stat.st_mtime_ns, stat.st_size)
                    if signature not in self.seen_files and path.is_file() and path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'} and time.time()-stat.st_mtime < 86400 and 0 < stat.st_size <= self.store.MAX_BYTES:
                        candidates.append((stat.st_mtime, path, signature))
            except OSError: pass
        for mtime, path, signature in sorted(candidates, reverse=True)[:32]:
            if self.pending >= 8: break
            self.seen_files.add(signature)
            self.submit('image', path, datetime.fromtimestamp(mtime).isoformat())
        self.scan_timer = self.root.after(2000, self.scan_screenshots)

    def copy(self, item):
        if item['kind'] == 'text':
            self.clipboard.set_text(item['text'], -1)
            try: self.clipboard.store()
            except Exception: pass
        else:
            image_path = self.store.images / item['image']
            copied = False
            if shutil.which('xclip') and image_path.exists():
                try:
                    subprocess.run(['xclip', '-selection', 'clipboard', '-target', 'image/png', str(image_path)],
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2, check=True)
                    copied = True
                except (subprocess.SubprocessError, OSError):
                    copied = False
            if not copied and shutil.which('wl-copy') and image_path.exists():
                try:
                    subprocess.run(['wl-copy', '--type', 'image/png'],
                                   input=image_path.read_bytes(), timeout=2, check=True)
                    copied = True
                except (subprocess.SubprocessError, OSError):
                    copied = False
            if not copied:
                try:
                    pixbuf = self.GdkPixbuf.Pixbuf.new_from_file(str(image_path))
                    self.clipboard.set_image(pixbuf)
                    self.clipboard.store()
                except Exception:
                    pass

    def close(self):
        self.closed = True
        self.clipboard.disconnect(self.handler)
        for timer in (self.pump_timer, self.scan_timer):
            self.root.after_cancel(timer)
        self.tasks.put((-1, -1, None))
        self.executor.shutdown(wait=False, cancel_futures=True)
