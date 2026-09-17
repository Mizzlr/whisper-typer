"""Local path discovery and document loading; no content leaves the machine."""
import csv
import os
import io
import threading
import time
import re
import subprocess
import zipfile
import stat
import shutil
import json
from decimal import Decimal
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import unquote, urlparse


EXTENSIONS = r'(?:md|markdown|csv|tsv|txt|log|json|jsonl|yaml|yml|toml|rs|py|html|pdf|sql|sh|zip)'


def pasted_paths(text):
    """Keep exact paths (including spaces); also understand copied report prose."""
    paths = []
    # Terminal output can wrap a path after a slash (including within a
    # Markdown link). Join those continuations before extracting candidates.
    text = re.sub(r'(?<=[/\\-])\s*\n\s*(?=[\w.@+-])', '', text)
    text = re.sub(r'\(([^()]*\n[^()]*)\)',
                  lambda m:'('+re.sub(r'\s*\n\s*','',m[1])+')' if '/' in m[1] else m[0],text)
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('```'):
            continue
        exact = line.strip('`\"\'')
        if Path(exact).expanduser().exists():
            paths.append(exact)
            continue
        links = re.findall(r'\[[^\]]*\]\((?:<([^>]+)>|([^\s)]+))(?:\s+"[^"]*")?\)', line)
        quoted = re.findall(r'`([^`]+)`', line)
        if links:
            paths.extend(a or b for a, b in links)
        elif quoted:
            paths.extend(quoted)
        else:
            found = re.findall(r'(?:file://|~/|/)?[\w.@+~/%-]+\.' + EXTENSIONS + r'(?::\d+(?::\d+)?|#L\d+(?:-L?\d+)?)?|(?:file://|~/|/)?[\w.@+~%-]+(?:/[\w.@+%-]+)+/?', line, re.I)
            paths.extend(found or ([exact] if '\n' not in text and re.fullmatch(r'[\w.@+-]+',exact) else []))
    return list(dict.fromkeys(paths))


def image_text(png):
    from PIL import Image
    # The red channel preserves contrast in turquoise terminal links on cream
    # paper; ordinary grayscale makes those links too pale for reliable OCR.
    image=Image.open(io.BytesIO(png)).convert('RGB').getchannel('R')
    scale=min(2,4000/max(image.size))
    if scale>1:
        image=image.resize((int(image.width*scale),int(image.height*scale)))
    data=io.BytesIO()
    image.save(data,format='PNG')
    result = subprocess.run(['tesseract', 'stdin', 'stdout', '--psm', '6'],
                            input=data.getvalue(), capture_output=True, check=True, timeout=45)
    return result.stdout.decode('utf-8')


def clean_path(value):
    value = value.strip().strip('`\"\'<>')
    if value.startswith('file://'):
        parsed = urlparse(value)
        if parsed.netloc not in ('', 'localhost'):
            raise ValueError('Only local file paths are supported.')
        value = unquote(parsed.path)
    return re.sub(r'(?:#L\d+(?:-L?\d+)?|:\d+(?::\d+)?)$', '', value)


class PathResolver:
    def __init__(self, home=None, cwd=None):
        self.home = Path(home or Path.home())
        self.cwd = Path(cwd or Path.cwd())
        self.repos = []
        self.index = {}
        self.indexed_at={}
        self.results={}
        self.lock=threading.RLock()
        self.discover()

    def discover(self):
        # Two directory levels cover ~/repo and ~/projects/repo without scanning
        # data stores, node_modules, or the contents of every repository.
        repos = []
        for folder in self.home.iterdir():
            if folder.name.startswith('.') or not folder.is_dir():
                continue
            if (folder / '.git').exists():
                repos.append(folder)
            else:
                try:
                    repos.extend(child for child in folder.iterdir()
                                 if child.is_dir() and (child / '.git').exists())
                except OSError:
                    pass
        self.repos = sorted(set(p.resolve() for p in repos))

    def resolve(self, value, context=None):
        key=(value,str(context) if context else None)
        with self.lock:
            previous=self.results.get(key)
            if previous and time.monotonic()-previous[0]<3:
                return list(previous[1])
            result=self._resolve(value,context)
            self.results[key]=(time.monotonic(),result)
            return list(result)

    def _resolve(self, value, context=None):
        name = clean_path(value)
        path = Path(name).expanduser()
        if path.is_absolute():
            if path.is_file() or path.is_dir():
                return [path.resolve()]
        roots = [self.cwd, self.home, *self.repos]
        if context:
            base=Path(context) if Path(context).is_dir() else Path(context).parent
            local = base / path
            if local.is_file() or local.is_dir():
                return [local.resolve()]
        matches = set()
        for root in roots:
            candidate = root / path
            if candidate.is_file() or candidate.is_dir():
                matches.add(candidate.resolve())
        if matches:
            return sorted(matches)
        if name.startswith('refs/'):
            return []
        # A basename or suffix copied from a report may omit its directory.
        # Cache tracked filenames only; never crawl file contents.
        for repo in self.repos:
            if repo not in self.index or time.monotonic()-self.indexed_at.get(repo,0)>30:
                try:
                    result = subprocess.run(['git', '-C', str(repo), 'ls-files', '-z'],
                                            capture_output=True, timeout=10, check=True)
                    self.index[repo] = result.stdout.decode('utf-8', 'surrogateescape').split('\0')
                except (OSError, subprocess.SubprocessError):
                    self.index[repo] = []
                # Reports are often ignored or not committed yet. Index names
                # from the latest date directories too, without reading data.
                adhoc=repo/'adhoc'
                if adhoc.is_dir():
                    dates=sorted((p for p in adhoc.iterdir() if p.is_dir() and re.fullmatch(r'\d{4}-\d{2}-\d{2}',p.name)),reverse=True)[:14]
                    extras=[]
                    for date in dates:
                        for folder,dirs,files in os.walk(date):
                            base=Path(folder)
                            dirs[:]=[d for d in dirs if not d.startswith('.') and d not in ('node_modules','__pycache__','target')]
                            if len(base.relative_to(date).parts)>=5:
                                dirs.clear()
                            extras.append(str(base.relative_to(repo)))
                            extras.extend(str((base/name).relative_to(repo)) for name in files)
                            if len(extras)>60000:
                                break
                        if len(extras)>60000:
                            break
                    self.index[repo]=list(dict.fromkeys(extras+self.index[repo]))
                self.indexed_at[repo]=time.monotonic()
            for entry in self.index[repo]:
                if entry == name or entry.endswith('/' + name):
                    candidate = repo / entry
                    if candidate.is_file() or candidate.is_dir():
                        matches.add(candidate.resolve())
        if matches:
            return sorted(matches,reverse=True)
        return self.fuzzy(name)

    def fuzzy(self,name):
        def normalized(value):
            return re.sub(r'[^a-z0-9/]', '', value.casefold().replace('@','0'))
        query=Path(name).parts
        query_name=Path(name).name
        suffix=Path(query_name).suffix.lower()
        scores=[]
        for repo,entries in self.index.items():
            for entry in entries:
                if not entry:
                    continue
                candidate=repo/entry
                if suffix and Path(entry).suffix.lower()!=suffix:
                    continue
                target=Path(entry).parts
                # Most repository names can be rejected without the expensive
                # full sequence comparison, especially for long report paths.
                if SequenceMatcher(None,normalized(query_name),normalized(candidate.name),autojunk=False).quick_ratio()<.72:
                    continue
                if len(query)==1:
                    left,right=normalized(query_name),normalized(candidate.name)
                else:
                    left=normalized('/'.join(query[-min(4,len(query)):]))
                    right=normalized('/'.join(target[-min(4,len(query)):]))
                matcher=SequenceMatcher(None,left,right,autojunk=False)
                if matcher.quick_ratio()<.84:
                    continue
                score=matcher.ratio()
                if score>=.84 and (candidate.is_file() or candidate.is_dir()):
                    scores.append((score,candidate.resolve()))
        if not scores:
            return []
        best=max(score for score,_ in scores)
        candidates={p for score,p in scores if score>=best-.035}
        # Recent date folders sort first; alternatives remain selectable.
        return sorted(candidates,reverse=True)[:30]


@dataclass
class Document:
    path: Path
    kind: str
    text: str = ''
    rows: object = None
    pages: int = 0


def load_document(path):
    path = Path(path)
    if path.suffix.lower() == '.pdf':
        info = subprocess.run(['pdfinfo', str(path)], capture_output=True, check=True, timeout=30)
        pages = re.search(rb'^Pages:\s+(\d+)', info.stdout, re.M)
        if not pages:
            raise ValueError('Cannot determine PDF page count.')
        text = subprocess.run(['pdftotext', '-layout', str(path), '-'],
                              capture_output=True, check=True, timeout=60)
        return Document(path, 'pdf', text.stdout.decode('utf-8'), pages=int(pages[1]))
    raw = path.read_bytes()
    if b'\0' in raw and not raw.startswith((b'\xff\xfe', b'\xfe\xff')):
        raise ValueError('This file is binary; choose a text document or PDF.')
    try:
        text = raw.decode('utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig')
    except UnicodeDecodeError:
        # A visible encoding choice is preferable to silently mangling content.
        raise ValueError('This text is not UTF-8 or UTF-16. Convert its encoding to view it.')
    ext = path.suffix.lower()
    if ext in ('.csv', '.tsv'):
        delimiter = '\t' if ext == '.tsv' else ','
        try:
            dialect = csv.Sniffer().sniff(text[:65536], delimiters=',;\t|') if text.strip() else None
        except csv.Error:
            dialect = None
        rows = list(csv.reader(io.StringIO(text), dialect=dialect)) if dialect else list(csv.reader(io.StringIO(text), delimiter=delimiter))
        return Document(path, 'csv', text, rows)
    return Document(path, 'markdown' if ext in ('.md', '.markdown', '.txt') else 'text', text)


def unzip_contents(path, destination):
    """Extract ordinary files into an isolated directory, never outside it."""
    destination=Path(destination).resolve()
    with zipfile.ZipFile(path) as archive:
        members=archive.infolist()
        if len(members)>5000 or sum(m.file_size for m in members)>200*1024*1024:
            raise ValueError('ZIP is too large to preview (5,000 entries or 200 MB).')
        targets=[]
        for member in members:
            target=(destination/member.filename).resolve()
            if not target.is_relative_to(destination) or stat.S_ISLNK(member.external_attr>>16):
                raise ValueError('ZIP contains an unsafe path or symbolic link.')
            targets.append((member,target))
        destination.mkdir(parents=True,exist_ok=True)
        for member,target in targets:
            if member.is_dir():
                target.mkdir(parents=True,exist_ok=True)
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                with archive.open(member) as source,target.open('wb') as output:
                    shutil.copyfileobj(source,output)
    return sorted(destination.iterdir(),key=lambda p:(not p.is_dir(),p.name.casefold()))


def pretty_json(source):
    """Indent JSON while preserving literal numbers, key order and strings."""
    def invalid_constant(value):raise ValueError('Invalid JSON constant: '+value)
    json.loads(source,parse_float=Decimal,parse_int=Decimal,parse_constant=invalid_constant)
    tokens=re.findall(r'"(?:\\.|[^"\\])*"|[{}\[\],:]|[^\s{}\[\],:]+',source)
    result=[];depth=0
    for index,token in enumerate(tokens):
        previous=tokens[index-1] if index else None
        following=tokens[index+1] if index+1<len(tokens) else None
        if token in ('{','['):
            result.append(token);depth+=1
            if following not in ('}',']'):result.append('\n'+'  '*depth)
        elif token in ('}',']'):
            depth-=1
            if previous not in ('{','['):result.append('\n'+'  '*depth)
            result.append(token)
        elif token==',':result.append(',\n'+'  '*depth)
        elif token==':':result.append(': ')
        else:result.append(token)
    return ''.join(result)


def pdf_page(path, page, dpi):
    # Poppler writes a single PNG to stdout when no output prefix is supplied.
    result = subprocess.run(['pdftoppm', '-f', str(page), '-l', str(page), '-singlefile',
                             '-scale-to', '4096', '-r', str(dpi), '-png', str(path)],
                            capture_output=True, check=True, timeout=60)
    return result.stdout
