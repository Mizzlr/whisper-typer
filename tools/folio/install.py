#!/usr/bin/python3
"""Install Folio and checksum-pinned offline renderers, without altering dictation."""
import hashlib
import io
import os
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path


PACKAGES = [
    ('mermaid', '10.9.3', 'd9458cd3744b9f30ebafd6afb07e075fe1def8a4d1dba486c9a4c518143692b3'),
    ('mathjax', '3.2.2', '1b9c0a1c44df864e915690558e72adb9cc5203360daefd385084ced3b6c64c09'),
]


def install():
    # Fail before copying anything if UI/text dependencies are absent.
    import bleach  # noqa: F401
    import markdown  # noqa: F401
    from PIL import Image  # noqa: F401
    from PyQt5.QtWebEngineWidgets import QWebEngineView  # noqa: F401
    from PyQt5.QtSvg import QSvgRenderer  # noqa: F401
    import tkinter  # noqa: F401
    import pygments  # noqa: F401
    for command in ('pdfinfo', 'pdftoppm', 'pdftotext', 'tesseract','xvfb-run','xclip','firefox'):
        if not shutil.which(command):
            raise RuntimeError(f'Missing {command}; see README.md dependencies.')
    home = Path.home()
    share = home / '.local/share/folio'
    python=share/'venv/bin/python'
    if not python.exists():
        subprocess.run(['/usr/bin/python3','-m','venv','--system-site-packages',str(share/'venv')],check=True)
    version=subprocess.run([str(python),'-c','import importlib.metadata; print(importlib.metadata.version("tkinterweb"))'],capture_output=True,text=True)
    if version.returncode or version.stdout.strip()!='4.25.4':
        subprocess.run([str(python),'-m','pip','install','tkinterweb==4.25.4','tkinterweb-tkhtml==2.1.1'],check=True)
    for package, version, checksum in PACKAGES:
        destination = share / 'vendor' / package
        marker = destination / '.archive-sha256'
        if marker.exists() and marker.read_text().strip() == checksum:
            continue
        data = urllib.request.urlopen(f'https://registry.npmjs.org/{package}/-/{package}-{version}.tgz', timeout=60).read()
        if hashlib.sha256(data).hexdigest() != checksum:
            raise RuntimeError(f'{package} checksum does not match the pinned release.')
        with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
            for member in archive.getmembers():
                rel = member.name.removeprefix('package/')
                allowed = (rel == 'dist/mermaid.min.js' or rel.lower().startswith('license')) if package == 'mermaid' else (rel.startswith('es5/') or rel.lower().startswith('license'))
                if not allowed or not member.isfile() or '..' in Path(rel).parts or Path(rel).is_absolute():
                    continue
                target = destination / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())
        marker.write_text(checksum + '\n')
    source = Path(__file__).resolve().parent
    library = home / '.local/lib/folio'
    library.mkdir(parents=True, exist_ok=True)
    for name in ('app.py','tk_widgets.py','image_zoom.py','render_worker.py','files.py','rendering.py','cell_stats.py','history.py','clipboard.py','syntax.py'):
        shutil.copyfile(source / name, library / name)
    (library / 'qt_app.py').unlink(missing_ok=True)
    executable = home / '.local/bin/folio'
    executable.parent.mkdir(parents=True, exist_ok=True)
    executable.write_text('#!/bin/sh\nexec "$HOME/.local/share/folio/venv/bin/python" "$HOME/.local/lib/folio/app.py" "$@"\n')
    executable.chmod(0o755)
    FOLIO_MIME_TYPES = [
        'text/plain', 'application/x-zerosize',
        'text/markdown', 'text/x-markdown',
        'application/json', 'application/x-json', 'application/jsonlines', 'text/json',
        'text/rust', 'text/x-rust', 'application/rust', 'text/x-rustsrc',
        'text/x-python', 'text/x-python3',
        'text/csv', 'text/x-csv', 'application/csv', 'text/tab-separated-values',
        'application/toml', 'text/x-toml', 'text/toml',
        'application/yaml', 'application/x-yaml', 'text/yaml', 'text/x-yaml',
        'text/x-sh', 'text/x-shellscript', 'application/x-shellscript', 'text/x-bash', 'text/x-zsh',
        'application/sql', 'text/x-sql', 'text/sql', 'application/x-sql',
        'text/x-c', 'text/x-c++', 'text/x-csrc', 'text/x-c++src', 'text/x-chdr', 'text/x-c++hdr',
        'text/x-go', 'text/x-java', 'text/x-java-source', 'text/x-kotlin',
        'application/javascript', 'text/javascript', 'application/x-javascript',
        'application/typescript', 'text/typescript', 'application/x-typescript',
        'text/css', 'application/xml', 'text/xml',
        'text/x-log', 'text/x-config', 'text/x-ini',
        'text/x-systemd-unit', 'text/x-dbus-service',
        'application/pdf',
    ]
    mimes_str = ';'.join(FOLIO_MIME_TYPES) + ';'
    desktop = home / '.local/share/applications/folio.desktop'
    desktop.parent.mkdir(parents=True, exist_ok=True)
    desktop.write_text(f'''[Desktop Entry]
Type=Application
Name=Folio
Comment=A local reading desk for files, diagrams and mathematics
Exec={executable} %F
Icon=accessories-text-editor
Terminal=false
StartupWMClass=Folio
Categories=Utility;Office;
MimeType={mimes_str}
''')
    (desktop.parent / 'folio-qt.desktop').unlink(missing_ok=True)
    for mime_file in (home / '.config/mimeapps.list', home / '.config/gnome-mimeapps.list'):
        lines = mime_file.read_text().splitlines() if mime_file.exists() else []
        header_idx = next((i for i, l in enumerate(lines) if l.strip() == '[Default Applications]'), -1)
        if header_idx == -1:
            lines.insert(0, '[Default Applications]')
            header_idx = 0
        defaults = {}
        end_idx = len(lines)
        for i in range(header_idx + 1, len(lines)):
            line = lines[i].strip()
            if line.startswith('[') and line.endswith(']'):
                end_idx = i
                break
            if '=' in line:
                k, v = line.split('=', 1)
                defaults[k.strip()] = v.strip()
        for m in FOLIO_MIME_TYPES:
            defaults[m] = 'folio.desktop'
        new_defaults = [f'{k}={v}' for k, v in defaults.items()]
        mime_file.write_text('\n'.join(lines[:header_idx + 1] + new_defaults + lines[end_idx:]) + '\n')
    if shutil.which('update-desktop-database'):
        subprocess.run(['update-desktop-database', str(desktop.parent)], check=False, capture_output=True)
    print('Installed Folio. Launch from Applications or run: folio')


if __name__ == '__main__':
    install()
