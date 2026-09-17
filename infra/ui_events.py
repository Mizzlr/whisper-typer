"""Loopback HTTP push with a socket wakeup into Tk's main thread."""
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import queue
import socket
import threading
import time
import tkinter as tk


class UiEventBridge:
    MAX_BODY = 1024 * 1024

    def __init__(self, root, receive, port=8768, metrics=None):
        self.root, self.receive = root, receive
        self.metrics = metrics or (lambda:{})
        self.events = queue.Queue(maxsize=256)
        self.reader, self.writer = socket.socketpair()
        self.reader.setblocking(False); self.writer.setblocking(False)
        self.enqueue_closed=threading.Event()
        self.enqueue=self.make_sender(self.events,self.writer,self.enqueue_closed)
        self.closed = False
        self.counts = {'dictation': 0, 'grammar_review': 0, 'recording': 0, 'window':0}
        self.last_dispatch_ms = None
        self.last_event = None
        self.last_timestamp = None
        self.lock = threading.Lock()
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def setup(self):
                super().setup()
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.connection.settimeout(30)

            def log_message(self, *_):
                pass  # Request bodies, transcripts, and headers are never logged.

            def respond(self, status, data):
                body = json.dumps(data).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                if status >= 400:
                    self.send_header('Connection', 'close'); self.close_connection = True
                self.end_headers()
                try: self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError): pass

            def do_GET(self):
                if self.path != '/health': return self.respond(404, {'error': 'not_found'})
                with bridge.lock:
                    self.respond(200, {'pid': os.getpid(), 'counts': dict(bridge.counts),
                                      'last_dispatch_ms': bridge.last_dispatch_ms,
                                      'last_event': bridge.last_event, 'last_timestamp': bridge.last_timestamp,
                                      **bridge.metrics()})

            def do_POST(self):
                if self.path not in ('/events','/show'): return self.respond(404, {'error': 'not_found'})
                # Browser-origin writes are not accepted; local Rust sends JSON.
                if self.headers.get('Origin') or self.headers.get_content_type() != 'application/json':
                    return self.respond(403, {'error': 'local_json_only'})
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                    if not 0 < length <= bridge.MAX_BODY: raise ValueError()
                    data = json.loads(self.rfile.read(length))
                    if self.path=='/show':
                        data={'type':'window','payload':{'action':'show','timestamp':datetime.now().astimezone().isoformat()}}
                    if not bridge.valid(data): raise ValueError()
                except (ValueError, TypeError, OSError):
                    return self.respond(400, {'error': 'invalid_event'})
                if not bridge.enqueue(data): return self.respond(429, {'error': 'queue_full_or_closed'})
                self.respond(202, {'accepted': True})

        try:
            self.server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        except OSError:
            self.reader.close(); self.writer.close()
            raise
        self.server.daemon_threads = True
        self.root.createfilehandler(self.reader, tk.READABLE, self.deliver)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={'poll_interval': .05}, name='ui-http-events', daemon=True)
        self.thread.start()

    @staticmethod
    def valid(data):
        if not isinstance(data, dict) or data.get('type') not in {'dictation', 'grammar_review', 'recording','window'}:
            return False
        payload = data.get('payload')
        if not isinstance(payload, dict): return False
        key = 'dictation_timestamp' if data['type'] == 'grammar_review' else 'timestamp'
        stamp = payload.get(key)
        if not isinstance(stamp, str) or len(stamp) > 64: return False
        try: datetime.fromisoformat(stamp)
        except ValueError: return False
        if data['type']=='window':return payload.get('action')=='show'
        if data['type'] == 'recording':
            return isinstance(payload.get('session_id'), str) and payload.get('status') in {'started','chunk','finishing','stopped','error','filtered','activity','summary_ready','summary_error','title_ready','title_error'}
        if data['type'] == 'dictation':
            return isinstance(payload.get('whisper_text'), str) and isinstance(payload.get('final_text'), str)
        return payload.get('status') in {'changed', 'unchanged', 'skipped'}

    @staticmethod
    def make_sender(events,writer,closed):
        """Workers hold only queue/socket primitives, never a Tk interpreter."""
        def enqueue(data):
            if closed.is_set():return False
            try:events.put_nowait((time.perf_counter(),data))
            except queue.Full:return False
            try:writer.send(b'x')
            except BlockingIOError:pass
            except OSError:return False
            return True
        return enqueue

    def deliver(self, *_):
        while True:
            try:
                if not self.reader.recv(4096): break
            except BlockingIOError: break
        batch = []
        while True:
            try: batch.append(self.events.get_nowait())
            except queue.Empty: break
        if not batch: return
        self.receive([data for _, data in batch])
        completed = time.perf_counter()
        with self.lock:
            for _, data in batch: self.counts[data['type']] += 1
            self.last_dispatch_ms = round((completed - batch[-1][0]) * 1000, 3)
            self.last_event = batch[-1][1]['type']
            payload = batch[-1][1]['payload']
            self.last_timestamp = payload.get('dictation_timestamp') or payload.get('timestamp')

    def close(self):
        if self.closed: return
        self.closed = True
        self.enqueue_closed.set()
        self.root.deletefilehandler(self.reader)
        self.server.shutdown(); self.server.server_close()
        self.reader.close(); self.writer.close()
