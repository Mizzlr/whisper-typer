"""Local path discovery and document loading; no content leaves the machine."""
import csv
import io
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse


EXTENSIONS = r'(?:md|markdown|csv|tsv|txt|log|json|jsonl|yaml|yml|toml|rs|py|html|pdf|sql|sh)'


def pasted_paths(text):
    """Keep exact paths (including spaces); also understand copied report prose."""
    paths = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('```'):
            continue
        exact = line.strip('`\"\'')
        if Path(exact).expanduser().is_file():
            paths.append(exact)
            continue
        links = re.findall(r'\[[^\]]*\]\((?:<([^>]+)>|([^\s)]+))(?:\s+"[^"]*")?\)', line)
        quoted = re.findall(r'`([^`]+)`', line)
        if links:
            paths.extend(a or b for a, b in links)
        elif quoted:
            paths.extend(quoted)
        else:
            found = re.findall(r'(?:file://|~/|/)?[\w.@+~/-]+\.' + EXTENSIONS + r'(?::\d+(?::\d+)?|#L\d+(?:-L?\d+)?)?', line, re.I)
            paths.extend(found or [exact])
    return list(dict.fromkeys(paths))


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
        name = clean_path(value)
        path = Path(name).expanduser()
        if path.is_absolute():
            return [path.resolve()] if path.is_file() else []
        roots = [self.cwd, self.home, *self.repos]
        if context:
            local = (Path(context).parent / path)
            if local.is_file():
                return [local.resolve()]
        matches = set()
        for root in roots:
            candidate = root / path
            if candidate.is_file():
                matches.add(candidate.resolve())
        if matches:
            return sorted(matches)
        # A basename or suffix copied from a report may omit its directory.
        # Cache tracked filenames only; never crawl file contents.
        for repo in self.repos:
            if repo not in self.index:
                try:
                    result = subprocess.run(['git', '-C', str(repo), 'ls-files', '-z'],
                                            capture_output=True, timeout=10, check=True)
                    self.index[repo] = result.stdout.decode('utf-8', 'surrogateescape').split('\0')
                except (OSError, subprocess.SubprocessError):
                    self.index[repo] = []
            for entry in self.index[repo]:
                if entry == name or entry.endswith('/' + name):
                    candidate = repo / entry
                    if candidate.is_file():
                        matches.add(candidate.resolve())
        return sorted(matches)


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
    return Document(path, 'markdown' if ext in ('.md', '.markdown') else 'text', text)


def pdf_page(path, page, dpi):
    # Poppler writes a single PNG to stdout when no output prefix is supplied.
    result = subprocess.run(['pdftoppm', '-f', str(page), '-l', str(page), '-singlefile',
                             '-scale-to', '4096', '-r', str(dpi), '-png', str(path)],
                            capture_output=True, check=True, timeout=60)
    return result.stdout
