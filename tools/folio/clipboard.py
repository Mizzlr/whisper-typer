"""Read a clipboard snapshot without Tk calls or unbounded subprocess waits."""
import io
import os
import shutil
import subprocess
from PIL import Image

IMAGE_TYPES=('image/png','image/jpeg','image/bmp','image/tiff','image/webp')


def read_clipboard():
    # Tk uses X11, including under Xwayland. Read the same selection it uses.
    if os.getenv('DISPLAY') and shutil.which('xclip'):
        prefix=['xclip','-selection','clipboard']
        listing=prefix+['-t','TARGETS','-o']
        def request(target):return prefix+['-t',target,'-o']
    elif shutil.which('wl-paste'):
        listing=['wl-paste','--list-types']
        def request(target):return ['wl-paste','--no-newline','--type',target]
    else:raise ValueError('No clipboard reader is available.')
    def read(command):
        result=subprocess.run(command,capture_output=True,timeout=3)
        if result.returncode:raise ValueError('Clipboard owner could not provide its contents. Try pasting again.')
        return result.stdout
    targets=read(listing).decode('utf-8',errors='replace').splitlines()
    image_type=next((kind for kind in IMAGE_TYPES if kind in targets),None)
    if image_type:
        image=Image.open(io.BytesIO(read(request(image_type))));image.load()
        return None,image.copy()
    target=next((kind for kind in ('UTF8_STRING','text/plain;charset=utf-8','text/plain;charset=UTF-8','text/plain','STRING') if kind in targets),None)
    if target:
        return read(request(target)).decode('latin-1' if target=='STRING' else 'utf-8'),None
    raise ValueError('The clipboard contains no supported image or text.')
