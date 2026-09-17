#!/usr/bin/python3
"""Folio — a local reading desk for documents and pasted ideas."""
import csv
import io
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtWebEngineCore import QWebEngineUrlRequestInterceptor
from PyQt5.QtWebEngineWidgets import QWebEnginePage, QWebEngineProfile, QWebEngineView

from files import Document, PathResolver, load_document, pasted_paths, pdf_page
from rendering import VENDOR, document_html
from cell_stats import selection_summary, display_summary


class Signals(QtCore.QObject):
    done = QtCore.pyqtSignal(object, object)


class Job(QtCore.QRunnable):
    def __init__(self, function, callback):
        super().__init__()
        self.function = function
        self.signals = Signals()
        self.signals.done.connect(callback)

    def run(self):
        try:
            result, error = self.function(), None
        except Exception as exc:
            result, error = None, str(exc)
        self.signals.done.emit(result, error)


class LocalRequests(QWebEngineUrlRequestInterceptor):
    def interceptRequest(self, info):
        if info.requestUrl().scheme() not in ('file', 'data', 'about', 'qrc'):
            info.block(True)


class Page(QWebEnginePage):
    file_link = QtCore.pyqtSignal(str)

    def acceptNavigationRequest(self, url, kind, main_frame):
        if main_frame and kind == self.NavigationTypeLinkClicked:
            if url.isLocalFile():
                if url.toLocalFile() == self.url().toLocalFile() and url.hasFragment():
                    return True
                self.file_link.emit(url.toLocalFile())
            elif url.scheme() in ('http', 'https', 'mailto'):
                QtGui.QDesktopServices.openUrl(url)
            return False
        return True


class Bridge(QtCore.QObject):
    @QtCore.pyqtSlot(str, result=str)
    def stats(self, values):
        return display_summary(selection_summary(json.loads(values)))

    @QtCore.pyqtSlot(str)
    def copy(self, text):
        QtWidgets.QApplication.clipboard().setText(text)


class CsvModel(QtCore.QAbstractTableModel):
    def __init__(self, rows):
        super().__init__()
        self.rows = rows
        self.columns = max((len(row) for row in rows), default=0)

    def rowCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else self.columns

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role in (QtCore.Qt.DisplayRole, QtCore.Qt.ToolTipRole) and index.isValid():
            row = self.rows[index.row()]
            return row[index.column()] if index.column() < len(row) else ''
        if role == QtCore.Qt.BackgroundRole and index.row() == 0:
            return QtGui.QColor('#e6ede2')

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole:
            return str(section + 1) if orientation == QtCore.Qt.Vertical else self.column_label(section)

    @staticmethod
    def column_label(number):
        result = ''
        number += 1
        while number:
            number, digit = divmod(number - 1, 26)
            result = chr(65 + digit) + result
        return result


class CsvTable(QtWidgets.QTableView):
    def keyPressEvent(self, event):
        if event.matches(QtGui.QKeySequence.Copy):
            indexes = self.selectionModel().selectedIndexes()
            if indexes:
                selected = {(x.row(), x.column()) for x in indexes}
                rs, cs = [x.row() for x in indexes], [x.column() for x in indexes]
                output = io.StringIO()
                writer = csv.writer(output, delimiter='\t', lineterminator='\n')
                for row in range(min(rs), max(rs)+1):
                    writer.writerow([self.model().index(row, col).data() if (row, col) in selected else ''
                                     for col in range(min(cs), max(cs)+1)])
                QtWidgets.QApplication.clipboard().setText(output.getvalue())
            return
        super().keyPressEvent(event)


class PdfScroll(QtWidgets.QScrollArea):
    """Drag to pan; ordinary wheel is vertical, Shift+wheel is horizontal."""
    def __init__(self):
        super().__init__()
        self.drag = None
        self.viewport().installEventFilter(self)
        self.setAlignment(QtCore.Qt.AlignCenter)

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.MouseButtonPress and event.button() == QtCore.Qt.LeftButton:
            self.drag = (event.pos(), self.horizontalScrollBar().value(), self.verticalScrollBar().value())
            self.viewport().setCursor(QtCore.Qt.ClosedHandCursor)
            return True
        if event.type() == QtCore.QEvent.MouseMove and self.drag:
            pos, horizontal, vertical = self.drag
            delta = event.pos() - pos
            self.horizontalScrollBar().setValue(horizontal - delta.x())
            self.verticalScrollBar().setValue(vertical - delta.y())
            return True
        if event.type() == QtCore.QEvent.MouseButtonRelease:
            self.drag = None
            self.viewport().unsetCursor()
        if event.type() == QtCore.QEvent.Wheel and event.modifiers() & QtCore.Qt.ShiftModifier:
            bar = self.horizontalScrollBar()
            bar.setValue(bar.value() - event.angleDelta().y())
            return True
        return super().eventFilter(obj, event)


class Folio(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Folio')
        self.setWindowIcon(QtGui.QIcon.fromTheme('accessories-text-editor'))
        self.resize(1180, 820)
        self.resolver = PathResolver()
        self.pool = QtCore.QThreadPool(self)
        self.pool.setMaxThreadCount(3)
        self.documents = {}
        self.current = None
        self.closing = False
        self.generation = 0
        self.pdf_generation = 0
        self.stats_generation = 0
        self.pdf_image = None
        self.pdf_zoom = .65
        self.temp = tempfile.TemporaryDirectory(prefix='folio-')
        self.settings = QtCore.QSettings('Folio', 'Reader')
        body = QtWidgets.QWidget()
        self.setCentralWidget(body)
        layout = QtWidgets.QVBoxLayout(body)
        layout.setContentsMargins(22, 18, 22, 12)
        title = QtWidgets.QHBoxLayout()
        name = QtWidgets.QLabel('Folio')
        name.setObjectName('brand')
        title.addWidget(name)
        title.addWidget(QtWidgets.QLabel('A quiet place for your files.'), 1)
        title.addWidget(self.button('Paste content', self.paste_content))
        title.addWidget(self.button('Open file…', self.choose_files))
        layout.addLayout(title)
        pathrow = QtWidgets.QHBoxLayout()
        self.path_input = QtWidgets.QPlainTextEdit()
        self.path_input.setPlaceholderText('Paste paths, Markdown links, or a few lines from a report…  Enter to open · Shift+Enter for a new line')
        self.path_input.setMaximumHeight(64)
        self.path_input.installEventFilter(self)
        pathrow.addWidget(self.path_input, 1)
        pathrow.addWidget(self.button('Paste paths', self.paste_paths))
        self.open_button = self.button('Open', self.open_paths)
        pathrow.addWidget(self.open_button)
        layout.addLayout(pathrow)
        split = QtWidgets.QSplitter()
        self.file_list = QtWidgets.QListWidget()
        self.file_list.setMinimumWidth(160)
        self.file_list.setMaximumWidth(360)
        self.file_list.currentItemChanged.connect(self.select_document)
        self.file_list.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.file_list.customContextMenuRequested.connect(self.file_menu)
        split.addWidget(self.file_list)
        reading = QtWidgets.QWidget()
        reader = QtWidgets.QVBoxLayout(reading)
        reader.setContentsMargins(16, 0, 0, 0)
        self.file_title = QtWidgets.QLabel('Your reading desk')
        self.file_title.setObjectName('documentTitle')
        self.location = QtWidgets.QLabel('Open a file, or paste something worth reading.')
        self.location.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.location.setWordWrap(True)
        self.location.setObjectName('location')
        reader.addWidget(self.file_title)
        reader.addWidget(self.location)
        actions = QtWidgets.QHBoxLayout()
        self.source_toggle = self.button('Source', self.refresh_view, checkable=True)
        actions.addWidget(self.source_toggle)
        self.copy_button = self.button('Copy content', self.copy_content)
        actions.addWidget(self.copy_button)
        self.path_button = self.button('Copy path', self.copy_path)
        actions.addWidget(self.path_button)
        actions.addStretch()
        self.find = QtWidgets.QLineEdit()
        self.find.setPlaceholderText('Find in document')
        self.find.setMaximumWidth(220)
        self.find.returnPressed.connect(self.find_next)
        actions.addWidget(self.find)
        actions.addWidget(self.button('−', lambda:self.zoom(-1)))
        actions.addWidget(self.button('+', lambda:self.zoom(1)))
        reader.addLayout(actions)
        self.pdf_controls = QtWidgets.QWidget()
        controls = QtWidgets.QHBoxLayout(self.pdf_controls)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.addWidget(self.button('←', lambda:self.change_page(-1)))
        self.page_number = QtWidgets.QSpinBox()
        self.page_number.valueChanged.connect(self.render_pdf)
        controls.addWidget(self.page_number)
        self.page_count = QtWidgets.QLabel()
        controls.addWidget(self.page_count)
        controls.addWidget(self.button('→', lambda:self.change_page(1)))
        controls.addWidget(self.button('Fit width', self.fit_pdf))
        controls.addStretch()
        controls.addWidget(QtWidgets.QLabel('Drag to pan · Shift+wheel for horizontal scroll'))
        reader.addWidget(self.pdf_controls)
        self.views = QtWidgets.QStackedWidget()
        self.web = QWebEngineView()
        # The central widget (including its page) must die before the profile.
        self.profile = QWebEngineProfile(self)
        self.interceptor = LocalRequests(self.profile)
        self.profile.setUrlRequestInterceptor(self.interceptor)
        self.web_page = Page(self.profile, self.web)
        self.web_page.file_link.connect(lambda p:self.open_paths(p, context=self.current.path if self.current else None))
        self.web.setPage(self.web_page)
        self.channel = QWebChannel(self.web_page)
        self.bridge = Bridge(self.channel)
        self.channel.registerObject('folio', self.bridge)
        self.web_page.setWebChannel(self.channel)
        self.source = QtWidgets.QPlainTextEdit()
        self.source.setReadOnly(True)
        self.source.setFont(QtGui.QFont('JetBrains Mono', 11))
        self.source.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.table = CsvTable()
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(False)
        self.table.setSortingEnabled(False)
        self.table.horizontalHeader().setDefaultSectionSize(190)
        self.table.verticalHeader().setDefaultSectionSize(32)
        self.pdf = PdfScroll()
        self.pdf_label = QtWidgets.QLabel()
        self.pdf_label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.pdf.setWidget(self.pdf_label)
        for widget in (self.web, self.source, self.table, self.pdf):
            self.views.addWidget(widget)
        reader.addWidget(self.views, 1)
        self.stats = QtWidgets.QLabel('Select cells, rows, or columns to see their statistics.')
        self.stats.setWordWrap(True)
        self.stats.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.stats.setObjectName('location')
        reader.addWidget(self.stats)
        self.stats.hide()
        split.addWidget(reading)
        split.setSizes([220, 920])
        layout.addWidget(split, 1)
        self.status = QtWidgets.QLabel(f'{len(self.resolver.repos)} repositories · Everything stays local')
        self.status.setObjectName('location')
        layout.addWidget(self.status)
        self.setStyleSheet('''
QMainWindow,QWidget {background:#fbf8f1;color:#353b33;font-family:"DejaVu Sans";font-size:13px}
QLabel#brand {font-size:30px;font-weight:600;color:#315543;padding-right:18px}
QLabel#documentTitle {font-size:22px;font-weight:600;padding-top:8px}
QLabel#location {color:#858879;font-size:11px;padding:4px 0 9px}
QPushButton {background:#f4f1e8;border:1px solid #dddccd;border-radius:7px;padding:8px 13px}
QPushButton:hover {background:#e9eee2;border-color:#a8b8a1}
QPushButton:checked {background:#e1ebdc;color:#315543;border-color:#b0c1a8}
QPushButton:disabled {color:#aaa99e}
QLineEdit,QPlainTextEdit {background:#fffdf8;border:1px solid #dedbcd;border-radius:7px;padding:10px;selection-background-color:#cedfca}
QListWidget {border:0;background:#f1eee4;border-radius:10px;padding:8px}
QListWidget::item {padding:12px 8px;border-radius:6px}
QListWidget::item:selected {background:#dfe8d9;color:#2b4d37}
QTableView {background:#fffdf8;alternate-background-color:#f4f1e9;border:1px solid #dedbcd;gridline-color:#e5e2d8;selection-background-color:#d8e6d0}
QHeaderView::section {background:#eeede3;border:0;padding:8px;color:#6b7364}
QScrollArea {border:0;background:#eeeae0}
QSplitter::handle {background:#e5e0d4;width:1px}
''')
        self.pdf_controls.hide()
        self.source_toggle.setEnabled(False)
        self.copy_button.setEnabled(False)
        self.path_button.setEnabled(False)
        self.web.setHtml('<html><body style="background:#fbf8f1;color:#899080;font:18px sans-serif;padding:80px"><h1 style="color:#365444">Less hunting. More reading.</h1><p>Markdown, tables, text, PDFs — and ideas from your clipboard.</p><p style="font-size:14px">Ctrl+L · Paste a path &nbsp;&nbsp; Ctrl+O · Open a file &nbsp;&nbsp; Ctrl+F · Find</p></body></html>')
        for shortcut, callback in [('Ctrl+L', self.path_input.setFocus), ('Ctrl+O', self.choose_files), ('Ctrl+F', self.find.setFocus)]:
            QtWidgets.QShortcut(QtGui.QKeySequence(shortcut), self, activated=callback)
        geometry = self.settings.value('geometry')
        if geometry:
            self.restoreGeometry(geometry)

    def button(self, label, callback, checkable=False):
        button = QtWidgets.QPushButton(label)
        button.setCheckable(checkable)
        button.clicked.connect(lambda checked=False:callback())
        return button

    def submit(self, function, callback):
        job = Job(function, lambda result,error:callback(result,error) if not self.closing else None)
        self.pool.start(job)

    def eventFilter(self, obj, event):
        if obj == self.path_input and event.type() == QtCore.QEvent.KeyPress and event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter) and not event.modifiers() & QtCore.Qt.ShiftModifier:
            self.open_paths()
            return True
        return super().eventFilter(obj, event)

    def paste_paths(self):
        self.path_input.setPlainText(QtWidgets.QApplication.clipboard().text())
        self.open_paths()

    def choose_files(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, 'Open in Folio', str(Path.home()), 'Documents (*.md *.markdown *.csv *.tsv *.txt *.pdf *.log *.json *.yaml *.toml);;All files (*)')
        for path in paths:
            self.load_path(Path(path))

    def open_paths(self, text=None, context=None):
        paths = pasted_paths(text if text is not None else self.path_input.toPlainText())
        if not paths:
            self.status.setText('Paste a file path first, or use Paste content for Markdown and diagrams.')
            return
        self.open_button.setEnabled(False)
        self.status.setText('Resolving paths…')
        self.submit(lambda:[(name, self.resolver.resolve(name, context)) for name in paths], self.paths_ready)

    def paths_ready(self, results, error):
        self.open_button.setEnabled(True)
        if error:
            self.status.setText(error)
            return
        missing = []
        for name, matches in results:
            if not matches:
                missing.append(name)
            elif len(matches) == 1:
                self.load_path(matches[0])
            else:
                path, ok = QtWidgets.QInputDialog.getItem(self, 'Choose the matching file', name, [str(p) for p in matches], editable=False)
                if ok:
                    self.load_path(Path(path))
        if missing:
            self.status.setText('Not found: ' + ' · '.join(missing))

    def load_path(self, path):
        key = str(path.resolve())
        if key in self.documents:
            for index in range(self.file_list.count()):
                item = self.file_list.item(index)
                if item.data(QtCore.Qt.UserRole) == key:
                    self.file_list.setCurrentItem(item)
                    return
        self.status.setText('Opening ' + path.name + '…')
        self.submit(lambda:load_document(path), self.document_ready)

    def document_ready(self, document, error):
        if error:
            self.status.setText('Could not open: ' + error)
            return
        key = str(document.path) if document.path else f'paste:{uuid.uuid4().hex}'
        if key not in self.documents:
            item = QtWidgets.QListWidgetItem(document.path.name if document.path else 'Pasted Markdown')
            item.setData(QtCore.Qt.UserRole, key)
            item.setToolTip(str(document.path) if document.path else 'Unsaved clipboard content')
            self.documents[key] = document
            self.file_list.addItem(item)
        else:
            item = next(self.file_list.item(i) for i in range(self.file_list.count()) if self.file_list.item(i).data(QtCore.Qt.UserRole) == key)
        self.file_list.setCurrentItem(item)

    def select_document(self, item, previous=None):
        if item is None:
            self.current = None
            self.copy_button.setEnabled(False)
            self.path_button.setEnabled(False)
            self.source_toggle.setEnabled(False)
            self.source.clear()
            self.views.setCurrentWidget(self.source)
            self.pdf_controls.hide()
            self.stats.hide()
            self.file_title.setText('Your reading desk')
            self.location.setText('Open a file, or paste something worth reading.')
            return
        self.current = self.documents[item.data(QtCore.Qt.UserRole)]
        self.file_title.setText(self.current.path.name if self.current.path else 'Pasted Markdown')
        self.location.setText(str(self.current.path) if self.current.path else 'Clipboard content · Not saved to a project')
        self.copy_button.setEnabled(True)
        self.path_button.setEnabled(bool(self.current.path))
        self.source_toggle.setEnabled(self.current.kind != 'text')
        self.source_toggle.setChecked(False)
        self.page_number.blockSignals(True)
        self.page_number.setRange(1, max(1, self.current.pages))
        self.page_number.setValue(1)
        self.page_number.blockSignals(False)
        self.page_count.setText(f'of {self.current.pages}')
        self.pdf_image = None
        self.refresh_view()

    def refresh_view(self):
        if not self.current:
            return
        doc = self.current
        self.generation += 1
        generation = self.generation
        self.pdf_generation += 1
        source = self.source_toggle.isChecked() or doc.kind == 'text'
        self.pdf_controls.setVisible(doc.kind == 'pdf' and not source)
        self.stats.setVisible(doc.kind == 'csv' and not source)
        self.status.setText(f'{doc.kind.upper()} · {len(doc.text):,} characters' + (f' · {len(doc.rows):,} rows' if doc.kind == 'csv' else ''))
        if source:
            self.source.setPlainText(doc.text)
            self.views.setCurrentWidget(self.source)
        elif doc.kind == 'csv':
            self.table.setModel(CsvModel(doc.rows))
            self.table.selectionModel().selectionChanged.connect(self.selection_changed)
            self.selection_changed()
            self.views.setCurrentWidget(self.table)
        elif doc.kind == 'pdf':
            self.views.setCurrentWidget(self.pdf)
            self.render_pdf()
        else:
            self.views.setCurrentWidget(self.web)
            self.status.setText('Rendering…')
            def ready(value, error):
                if generation != self.generation:
                    return
                if error:
                    self.status.setText('Could not render: ' + error)
                    return
                target = Path(self.temp.name) / f'view-{generation}.html'
                # A base URL keeps relative report links and images useful.
                base = doc.path.parent.as_uri() + '/' if doc.path else Path.home().as_uri() + '/'
                value = value.replace('<head>', '<head><base href="' + base.replace('"', '%22') + '">', 1)
                target.write_text(value)
                os.chmod(target, 0o600)
                self.web.load(QtCore.QUrl.fromLocalFile(str(target)))
                self.status.setText(f'Markdown · {len(doc.text):,} characters · Offline rendering')
            self.submit(lambda:document_html(doc.text), ready)

    def render_pdf(self, unused=None):
        if not self.current or self.current.kind != 'pdf' or self.source_toggle.isChecked():
            return
        self.pdf_generation += 1
        generation = self.pdf_generation
        doc, page = self.current, self.page_number.value()
        self.status.setText(f'Rendering page {page}…')
        def ready(data, error):
            if generation != self.pdf_generation:
                return
            if error:
                self.status.setText('PDF: ' + error)
                return
            image = QtGui.QPixmap()
            if not image.loadFromData(data):
                self.status.setText('PDF page could not be decoded.')
                return
            self.pdf_image = image
            self.fit_pdf()
            self.pdf.verticalScrollBar().setValue(0)
            self.pdf.horizontalScrollBar().setValue(0)
            self.status.setText(f'PDF · Page {page} of {doc.pages} · Copy content copies extracted text')
        self.submit(lambda:pdf_page(doc.path, page, 144), ready)

    def selection_changed(self, unused=None, previous=None):
        self.stats_generation += 1
        generation = self.stats_generation
        ranges = [(s.top(),s.bottom(),s.left(),s.right()) for s in self.table.selectionModel().selection()]
        if not ranges:
            self.stats.setText('Select cells, rows, or columns · Ctrl adds a selection · Ctrl+C copies it')
            return
        rows = self.current.rows
        self.stats.setText('Calculating selection…')
        def calculate():
            return display_summary(selection_summary(rows[r][c] if c<len(rows[r]) else ''
                for top,bottom,left,right in ranges for r in range(top,bottom+1) for c in range(left,right+1)))
        def ready(result, error):
            if generation == self.stats_generation:
                self.stats.setText(result if not error else 'Statistics: '+error)
        self.submit(calculate, ready)

    def scale_pdf(self):
        if self.pdf_image:
            size = self.pdf_image.size() * self.pdf_zoom
            pixmap = self.pdf_image.scaled(size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.pdf_label.setPixmap(pixmap)
            self.pdf_label.resize(pixmap.size())

    def fit_pdf(self):
        if self.pdf_image:
            self.pdf_zoom = max(.05, (self.pdf.viewport().width()-32) / self.pdf_image.width())
            self.scale_pdf()

    def change_page(self, amount):
        self.page_number.setValue(self.page_number.value() + amount)

    def zoom(self, direction):
        if self.views.currentWidget() == self.pdf:
            self.pdf_zoom = min(4, max(.05, self.pdf_zoom * (1.2 if direction > 0 else 1/1.2)))
            self.scale_pdf()
        elif self.views.currentWidget() == self.web:
            self.web.setZoomFactor(min(3, max(.5, self.web.zoomFactor() + direction*.1)))
        else:
            widget = self.views.currentWidget()
            font = widget.font()
            font.setPointSize(max(7, min(28, font.pointSize()+direction)))
            widget.setFont(font)

    def find_next(self):
        term = self.find.text()
        if not term or not self.current:
            return
        widget = self.views.currentWidget()
        if widget == self.web:
            self.web.findText(term)
        elif widget == self.source:
            if not self.source.find(term):
                self.source.moveCursor(QtGui.QTextCursor.Start)
                self.source.find(term)
        elif widget == self.table:
            rows = self.current.rows
            start = self.table.currentIndex().row()+1
            for row in [*range(start, len(rows)), *range(0, start)]:
                for col, value in enumerate(rows[row]):
                    if term.casefold() in value.casefold():
                        index = self.table.model().index(row, col)
                        self.table.setCurrentIndex(index)
                        self.table.scrollTo(index)
                        return
        else:
            self.source_toggle.setChecked(True)
            self.refresh_view()
            self.find_next()

    def copy_content(self):
        if self.current:
            QtWidgets.QApplication.clipboard().setText(self.current.text)
            self.status.setText('Full content copied' if self.current.kind != 'pdf' else 'PDF text copied · Scanned pages need OCR')

    def copy_path(self):
        if self.current and self.current.path:
            QtWidgets.QApplication.clipboard().setText(str(self.current.path))
            self.status.setText('Full path copied')

    def paste_content(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle('Paste into Folio')
        dialog.resize(720, 520)
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(QtWidgets.QLabel('Markdown, Mermaid, or LaTeX math — paste below and render.'))
        editor = QtWidgets.QPlainTextEdit()
        editor.setFont(QtGui.QFont('JetBrains Mono', 11))
        editor.setPlainText(QtWidgets.QApplication.clipboard().text())
        layout.addWidget(editor)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.button(QtWidgets.QDialogButtonBox.Ok).setText('Render')
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec_() == QtWidgets.QDialog.Accepted and editor.toPlainText().strip():
            self.document_ready(Document(None, 'markdown', editor.toPlainText()), None)

    def file_menu(self, position):
        item = self.file_list.itemAt(position)
        if not item:
            return
        menu = QtWidgets.QMenu(self)
        reload_action = menu.addAction('Reload file')
        reload_action.setEnabled(bool(self.documents[item.data(QtCore.Qt.UserRole)].path))
        close_action = menu.addAction('Close')
        action = menu.exec_(self.file_list.mapToGlobal(position))
        key = item.data(QtCore.Qt.UserRole)
        if action == reload_action:
            path = self.documents[key].path
            self.submit(lambda:load_document(path), lambda doc, error:self.reload_ready(key, doc, error))
        elif action == close_action:
            self.file_list.takeItem(self.file_list.row(item))
            self.documents.pop(key, None)

    def reload_ready(self, key, document, error):
        if error:
            self.status.setText('Reload failed: ' + error)
        elif key in self.documents:
            self.documents[key] = document
            if self.file_list.currentItem() and self.file_list.currentItem().data(QtCore.Qt.UserRole) == key:
                self.select_document(self.file_list.currentItem())

    def closeEvent(self, event):
        self.settings.setValue('geometry', self.saveGeometry())
        self.closing = True
        self.hide()
        self.pool.clear()
        # Workers hold callbacks into this window. Defer deletion until they
        # finish rather than destroying Qt objects under a running callback.
        self.pool.waitForDone()
        self.temp.cleanup()
        super().closeEvent(event)


def main():
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    application = QtWidgets.QApplication(sys.argv)
    application.setApplicationName('Folio')
    window = Folio()
    window.show()
    if len(sys.argv) > 1:
        for name in sys.argv[1:]:
            window.open_paths(name)
    return application.exec_()


if __name__ == '__main__':
    sys.exit(main())
