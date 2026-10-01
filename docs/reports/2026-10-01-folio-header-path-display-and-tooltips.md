# Folio Header Path Display & Hover Tooltips

**Date**: 2026-10-01  
**Topic**: Utilizing empty toolbar space in Folio to display active document path with responsive truncation, copy-on-click, and hover tooltips for paths and directories.

---

## 1. Problem Statement

* In Folio document view, the navigation header bar had a large unused empty gap between the left folder dropdown (`📁 <folder> ▾`) and the right-hand controls (`Aa ▾`, `A-`, `A+`, `◐`, `Copy`, `← Back`, `Top`).
* Long folder names were truncated to 18 characters (e.g. `📁 september_revenue… ▾`), making it impossible to see the parent directory name without copying the path from the context menu.
* The document filename was only shown in the OS window title bar, and the full filesystem path was not visible anywhere in the window.

---

## 2. Implementation Details

* **Files**:
  * `tools/folio/tk_widgets.py`
  * `tools/folio/app.py`
  * `tools/folio/test_folio.py`

* **Hover Tooltip Widget (`tk_widgets.py`)**:
  * Implemented a lightweight, theme-aware `Tooltip` class using borderless `tk.Toplevel` with `-topmost`.
  * Features delayed hover trigger (300ms), screen edge constraint handling, automatic cleanup on leave, click, destruction, or unmap, and dynamic font/palette integration.

* **Header Path Display & Interaction (`app.py`)**:
  * Added `self.path_label` in `self.nav`, positioned between the folder button and right-hand controls with `fill='x', expand=True`.
  * Implemented `fit_path_text()` with dynamic path fitting:
    * Displays full path when available width permits.
    * Substitutes `~` for the user home directory.
    * Uses middle truncation (`~/…/<dir>/<file>`) when width is constrained.
    * Progressively truncates to keep the filename visible on narrow windows.
  * Added click-to-copy handler (`copy_path_click()`) that copies the resolved full path to the clipboard and provides status bar confirmation.
  * Integrated tooltips on both the path label (full resolved file path) and the folder button (full resolved directory path).
  * Styled with hand cursor, theme palette foreground, hover color transitions, and proper hide/cleanup routines on view switching and application close.

* **Test Coverage (`test_folio.py`)**:
  * Added `test_path_label_and_tooltips_in_document_view`: verifies path label mapping, text content, tooltip text resolution, clipboard copying on click, and proper hide behavior in list view or pathless documents.
  * Added `test_fit_path_text`: verifies progressive truncation across wide, medium, and narrow window widths.
  * All 35 tests in the test suite pass cleanly.

---

## 3. Verification & Deployment

* **Unit Tests**: All 35 unit tests in `test_folio.py` executed and passed cleanly.
* **Installation**: Installed via `python3 tools/folio/install.py` to `~/.local/lib/folio/`. Verified zero diff between repository sources and installed library files.
* **Runtime Verification**: Tested in Xvfb with live files (`stages.rs` and `september_sol_non_sol_daily.tsv`), confirming correct path display, tooltip generation, and tooltip geometry constraints.
