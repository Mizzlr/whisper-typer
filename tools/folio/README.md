# Folio

A quick, local reader: paste context, choose a file, read it, press Escape.
Cream paper and JetBrains Mono throughout. No sidebar, welcome screen, branding
inside the app, or document toolbar.

Install: `/usr/bin/python3 tools/folio/install.py`. Launch **Folio** from
Applications or run `folio [paths…]`.

## Paste and read

Paste text or a clipboard screenshot anywhere. No Open or Parse step is needed.
A paste replaces the input, extracts paths, and adds a dated group to the list.
Click a path to read the whole file. Escape closes the document, returns to the
list, clears the input, and prepares the next paste. Earlier groups stay available.
Click the date/icon of a group to view its original text or screenshot. Images
support zoom and horizontal/vertical panning.

Folders list their immediate files and subfolders; click a subfolder to browse.
Recent shows files opened in Folio. Downloads lists the Downloads folder with
newest entries first. Escape returns either view to the paste history.

Path resolution understands absolute, home-relative, repo-relative, basename,
Markdown/backticked and terminal-wrapped paths, including line/column suffixes.
Home-folder Git repos and repos one level deeper are discovered. Tracked filenames
and the fourteen latest `adhoc/YYYY-MM-DD` directories are indexed lazily, including
untracked reports. Shortened/typo paths use fuzzy filename/suffix matching;
multiple plausible existing matches are offered as separate entries rather than
silently choosing one. The index refreshes after thirty seconds when queried.
Screenshot OCR uses local Tesseract; colored terminal links are enhanced before
recognition. OCR can still fail on unreadable/cropped images; the original remains
available after a successful read, and ambiguous matches require selection.

Markdown, Mermaid diagrams, mathematical LaTeX, CSV/TSV, text and PDF render
locally. Pasted Markdown or bare Mermaid without paths renders directly and is
also kept in its paste group. TeX is mathematical typesetting inside Markdown,
not a full LaTeX document compiler.

CSV and Markdown tables support cell/row/column selections, copying and compact
selection statistics: count, numeric count, sum, average, minimum and maximum.
CSV: Shift extends, Ctrl adds, row/column headers select, Ctrl+C copies TSV.
Markdown: click/Shift rectangles/Ctrl cells, header columns, row numbers.
Calculations use decimals, including standard thousands commas. Labels, empty
cells, percentages and currency-marked values are excluded rather than assuming
units; compact statistics show ten significant digits.

Text/source views wrap to the window width and number original file lines;
wrapped continuations do not create extra line numbers or change copied text.

While reading, only content and requested selection statistics appear. Right-click
for source, copy selection/content/path, zoom, search and PDF paging. Ctrl+C copies
selected text/cells; Ctrl+Shift+C copies full contents. Ctrl+F finds; Ctrl+L returns
to the paste list. Ctrl++/Ctrl+- zoom; Alt+Left/Alt+Right page PDFs. Images/PDFs pan
by dragging, scrollbar, or Shift+wheel horizontally. PDF search switches to extracted
text. Scanned PDFs need OCR for searchable/copyable text. PDF images cap at 4096
pixels. Text must be UTF-8 or BOM-marked UTF-16; unsupported encodings/binary files
report errors instead of silently changing content.

## Data and dependencies

Paste groups, extracted text, original screenshot PNGs and recently opened paths
are kept in a private, mode-0600 SQLite database:
`~/.local/share/folio/history.sqlite3`. Documents are read when clicked. Temporary
rendered HTML lives in a private temporary directory and is removed on normal exit.
Window geometry is stored in `~/.config/Folio/Reader.conf`. These private runtime
files stay outside the repository. Tests use synthetic temporary files and images;
private screenshots, reports, OCR output and credential files are never fixtures.

Dependencies on Debian/Ubuntu: `python3-pyqt5 python3-pyqt5.qtwebengine
python3-markdown python3-bleach python3-pil poppler-utils tesseract-ocr`.
The installer downloads checksum-pinned Mermaid 10.9.3 and MathJax 3.2.2 with their
licenses into `~/.local/share/folio/vendor`. No network is needed for rendering.
External embedded resources are blocked; explicitly clicked external links open
your default browser. Background workers load files, resolve paths, OCR images,
typeset documents and rasterize PDF pages independently of the UI.

Tests: run `tools/folio/test_folio.py --render` with system Python under isolated
Xvfb and software GL. Tests do not inject input into your desktop. No Whisper Typer
services or pipelines are changed by installing Folio.

Renderer references: [Mermaid](https://mermaid.js.org/config/usage) and
[MathJax](https://docs.mathjax.org/en/v3.2/web/configuration.html).
