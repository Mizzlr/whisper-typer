# Folio

Paste context, choose a file, read it, press Escape. JetBrains Mono, cream paper
or a dark theme, and a compact top line similar to Whisper Typer.

Install: `/usr/bin/python3 tools/folio/install.py`. Launch **Folio** from Applications.
CLI: `folio [paths…]`.

## Paste and read

Paste text or a clipboard screenshot anywhere. No Open or Parse step is needed.
A paste replaces the input, extracts paths, and adds a dated group of plain links.
Click a path to read. A small file switcher above the document jumps between files
from the same paste or folder. Copy copies the document (or the original image); on a list it copies the listed
paths. Back stays in the same top-right position and is disabled at home.
Back returns to that list; Escape returns to Paths,
clears the input and prepares the next paste. Earlier groups stay available.
Click a group's timestamp to view its original dump or screenshot.

Top-line controls: **Paths**, **Recent**, **Downloads**, **Adhoc**, **Projects**, theme, and
**Top**. Recent lists files opened in Folio. Downloads lists newest downloads first.
Adhoc lists the discovered repos' `adhoc` folders. Projects lists prioritized project folders.
Theme, font family, font size, and optional always-on-top are saved in `~/.config/folio/settings.json`.
Top keeps the window above others without repeatedly forcing keyboard focus.

Folders list their files/subfolders. ZIP files open as browsable contents in a
private temporary directory. Extraction rejects escaping paths, symbolic links,
more than 5,000 entries or more than 200 MB of uncompressed data. Nothing executes.
PDFs open in Firefox, preserving its familiar reading/search/zoom controls.

Path lookup understands absolute, home-relative, repo-relative, basename,
Markdown/backticked and terminal-wrapped paths, including line/column suffixes.
Home-folder Git repos and repos one level deeper are discovered. Tracked names and
the fourteen latest `adhoc/YYYY-MM-DD` directories are indexed lazily, including
untracked reports. Shortened/typo paths use fuzzy filename/suffix matching;
plausible alternatives are offered separately. The index refreshes after thirty
seconds when queried. Screenshot OCR uses local Tesseract, enhancing colored
terminal links. Original screenshots/dumps remain available when OCR is imperfect.

Markdown and `.txt` files render as Markdown. Mermaid and mathematical LaTeX use
local, pinned renderer assets. Tkinter displays the typeset diagrams/math as images
from a separate worker on an isolated display. Full original content remains copyable from the menu. Source files use
Pygments, theme-aware colors, original line numbers and word wrapping. JSON has a reversible **Prettify / Original** display control. Prettifying keeps
exact number/string literals and does not change the file or full-content copy.
Other text must be UTF-8 or BOM-marked UTF-16; unsupported encodings/binary files report errors.

CSV/TSV and Markdown tables support cell/rectangle/row/column selections,
count, sum, average, median, minimum and maximum, plus TSV copying. Ctrl/Shift extend
selections. **Transpose**, directly above each table at its top right, changes only that
table without altering the file or full-content copy. **Shift + scroll** moves horizontally; ordinary scroll moves vertically.
Tkinter draws only visible table cells. Statistics use Decimal and exclude unknown
units such as percentage/currency strings rather than treating them as numbers.

Right-click for source, copying, zoom and search. Ctrl+C copies a selection;
Ctrl+Shift+C copies full content. Ctrl+F finds; Ctrl+L returns to Paths.
Ctrl++/Ctrl+- zoom. Screenshot views support two-axis drag/scroll panning.

## Private data and dependencies

Paste groups, OCR text, original screenshots and recent paths live in a mode-0600
SQLite database: `~/.local/share/folio/history.sqlite3`, outside Git. Temporary HTML
and ZIP contents are removed on normal exit. Settings live in
`~/.config/folio/settings.json`.
Tests use synthetic temporary fixtures only. Private reports, screenshots, OCR
output, history and credentials must never become public fixtures or commits.

System dependencies: `python3-tk python3-venv python3-pyqt5
python3-pyqt5.qtwebengine python3-pyqt5.qtsvg python3-markdown python3-bleach
python3-pil python3-pygments poppler-utils tesseract-ocr xvfb xclip firefox`.
The installer creates a system-package-enabled local venv with pinned TkinterWeb
4.25.4/Tkhtml 2.1.1 and installs checksum-pinned Mermaid 10.9.3/MathJax 3.2.2 with
licenses under `~/.local/share/folio/vendor`. No network is needed for rendering.
Embedded network resources and untrusted scripts are blocked. Explicitly clicked
external links open Firefox. Background workers resolve/load/OCR/typeset files.

Tests (isolated displays; no desktop input injection):

```sh
xvfb-run -a ~/.local/share/folio/venv/bin/python tools/folio/test_folio.py --render
```

[Folio's Tk HTML widget](https://tkinterweb.readthedocs.io/en/latest/api/htmlframe.html),
[Mermaid](https://mermaid.js.org/config/usage),
[MathJax](https://docs.mathjax.org/en/v3.2/web/configuration.html).

Image pastes prefer image data when the clipboard also offers alternate text.
Encoded PNG/JPEG/BMP/TIFF/WebP formats are supported. Tk reads the X11 selection
in a worker with bounded subprocess waits. Original images remain in private history
even when OCR yields no text or fails. Folder links carry a small ▸ arrow.

Folio uses cream paper/light and black/dark backgrounds, neutral controls and
[Coldark-inspired syntax colors](https://github.com/PrismJS/prism-themes).
Source views skip tokenization for files above 150,000 characters or lines above
2,000 characters, displaying the complete raw text instead. Full-content Copy
always retains the original. This bounds syntax work, not file contents.
