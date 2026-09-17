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
    for name in ('app.py','qt_app.py','tk_widgets.py','render_worker.py','files.py','rendering.py','cell_stats.py','history.py','clipboard.py','syntax.py'):
        shutil.copyfile(source / name, library / name)
    executable = home / '.local/bin/folio'
    executable.parent.mkdir(parents=True, exist_ok=True)
    executable.write_text('#!/bin/sh\ncase "$1" in\n  --qt) shift; exec /usr/bin/python3 "$HOME/.local/lib/folio/qt_app.py" "$@" ;;\n  *) exec "$HOME/.local/share/folio/venv/bin/python" "$HOME/.local/lib/folio/app.py" "$@" ;;\nesac\n')
    executable.chmod(0o755)
    desktop = home / '.local/share/applications/folio.desktop'
    desktop.parent.mkdir(parents=True, exist_ok=True)
    desktop.write_text(f'''[Desktop Entry]
Type=Application
Name=Folio
Comment=A local reading desk for files, diagrams and mathematics
Exec="{executable}" %F
Icon=accessories-text-editor
Terminal=false
StartupWMClass=Folio
Categories=Utility;Office;
MimeType=text/markdown;text/csv;text/plain;application/pdf;
''')
    (desktop.parent/'folio-qt.desktop').write_text(f'''[Desktop Entry]
Type=Application
Name=Folio (Qt)
Comment=Compare the Qt reading interface
Exec="{executable}" --qt %F
Icon=accessories-text-editor
Terminal=false
StartupWMClass=FolioQt
Categories=Utility;Office;
''')
    print('Installed Folio. Launch from Applications or run: folio')


if __name__ == '__main__':
    install()
