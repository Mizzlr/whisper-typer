# Folio

A separate, local desktop reading desk. Cream paper, readable typography, a file
list and a generous document pane. Built with Python, Qt and an embedded offline
document renderer. It does not change Whisper Typer or the dictation pipeline.

Install with `/usr/bin/python3 tools/folio/install.py`; launch **Folio** from
Applications or run `folio [paths…]`. Dependencies on Debian/Ubuntu:
`python3-pyqt5 python3-pyqt5.qtwebengine python3-markdown python3-bleach poppler-utils`.
The installer downloads checksum-pinned Mermaid 10.9.3 and MathJax 3.2.2 assets
and their licenses into `~/.local/share/folio/vendor`. No network is needed to
render documents. External document resource requests are blocked; clicking an
external link explicitly opens your default browser.

- **Paste paths** takes clipboard text, Markdown links, backticked paths, or
  paths within report prose. Enter opens; Shift+Enter adds another input line.
  Absolute paths, `file://`, `~/`, repo-prefixed paths and relative repo paths
  work. `:line:column` and `#Lline` suffixes are removed. Paths with spaces work
  when pasted alone, quoted/backticked, or in a Markdown `<path>` link.
- Home-folder Git repos and repos one directory deeper are discovered at launch.
  Exact existing paths are tried first; tracked filename suffixes are indexed
  lazily for shorter names. Multiple matches require your selection. File
  contents are never indexed or uploaded. Linked files resolve relative to the
  current document.
- Markdown displays headings, tables, code, local images, fenced Mermaid diagrams
  and inline/display TeX (`$…$`, `$$…$$`, `\(…\)`, `\[…\]`). **Paste content**
  also accepts bare Mermaid source. This is mathematical typesetting inside
  Markdown, not a full LaTeX document compiler.
- CSV/TSV uses a table model without creating widgets for every cell. All rows,
  including the first row, remain present. Click row/column headers to select;
  Shift extends and Ctrl adds selections. **Ctrl+C** copies selected cells as
  TSV. The bottom line shows cell count, numeric count, sum, average, min and max.
  CSV calculations use decimals; numeric strings and standard thousands commas
  work. Empty cells, labels, percentages and currency-marked values are excluded
  rather than assuming units. The compact display uses ten significant digits.
- Markdown tables support click/Shift rectangle/Ctrl selection, header column
  selection, row-number selection, statistics and **Copy selection**. Calculations
  use the same decimal implementation as CSVs.
- PDF pages render on demand with page navigation, zoom, fit width, horizontal
  and vertical scrolling, and drag panning. **Source** shows extracted text;
  **Copy content** copies all extracted text. Scanned PDFs can be viewed but do
  not gain searchable/copyable text without OCR. PDF rendering caps the page
  image at 4096 pixels; zoom enlarges that image.
- **Copy content** copies the entire original document, not only the viewport.
  **Copy path** copies the resolved absolute path. Source is selectable; rendered
  paragraphs retain ordinary text selection and code blocks have a Copy button.
  Ctrl+L focuses paths; Ctrl+O opens a file chooser; Ctrl+F searches. PDF search
  switches to extracted text. Right-click a file to reload or close it.

Loading, path indexes, Markdown processing and PDF rasterization run off the UI
thread. Opened documents are held in memory until closed. Temporary rendered HTML
is stored in a private temporary directory and removed on normal exit. Only
window geometry is saved (`~/.config/Folio/Reader.conf`); no document history or
clipboard contents are persisted. Text files must be UTF-8 or BOM-marked UTF-16;
unsupported encodings and binary files report an error instead of silently
changing content.

Tests: `QT_QPA_PLATFORM=offscreen /usr/bin/python3 tools/folio/test_folio.py`.
For embedded rendering checks, run in isolated Xvfb with software GL; see the
test's `--render` mode. Tests never inject input into your desktop.

Renderer references: [Mermaid usage](https://mermaid.js.org/config/usage) and
[MathJax 3 configuration](https://docs.mathjax.org/en/v3.2/web/configuration.html).
