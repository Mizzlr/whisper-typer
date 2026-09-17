#!/usr/bin/python3
"""Open/reveal the existing window from GNOME's application overview."""
import subprocess
import time
import urllib.request

subprocess.run(['systemctl','--user','start','whisper-dictation-window.service'],check=True)
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for attempt in range(40):
    try:
        request=urllib.request.Request('http://127.0.0.1:8768/show',b'{}',{'Content-Type':'application/json'})
        with opener.open(request,timeout=.5) as response:
            assert response.status==202
        break
    except (OSError,AssertionError):time.sleep(.1)
else:raise SystemExit('Whisper Typer window did not become available.')
try:
    from Xlib import X,display,protocol
    d=display.Display();root=d.screen().root
    clients=root.get_full_property(d.intern_atom('_NET_CLIENT_LIST'),X.AnyPropertyType)
    for identity in clients.value if clients is not None else []:
        window=d.create_resource_object('window',int(identity))
        if 'whispertyper' not in [value.lower() for value in (window.get_wm_class() or ())]:continue
        # Explicit launcher activation lets GNOME reveal the app/workspace;
        # background transcript arrivals never request activation.
        event=protocol.event.ClientMessage(window=window,client_type=d.intern_atom('_NET_ACTIVE_WINDOW'),
                                          data=(32,[2,X.CurrentTime,0,0,0]))
        root.send_event(event,event_mask=X.SubstructureRedirectMask|X.SubstructureNotifyMask)
        d.flush();break
    d.close()
except ImportError:pass
