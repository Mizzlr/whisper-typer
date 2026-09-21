#!/usr/bin/python3
"""Folio — a local reading desk for documents and pasted ideas."""
import base64
import csv
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse, unquote

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtWebEngineCore import QWebEngineUrlRequestInterceptor
from PyQt5.QtWebEngineWidgets import QWebEnginePage, QWebEngineProfile, QWebEngineView

from files import Document, PathResolver, load_document, pasted_paths, pdf_page, image_text, unzip_contents, pretty_json, latest_history_file
from rendering import VENDOR, document_html, theme_colors
from cell_stats import selection_summary, display_summary
from history import History
from syntax import syntax_safe, source_style


AVAILABLE_FONTS = [
    'JetBrains Mono',
    'Google Sans Mono',
    'DejaVu Sans Mono',
    'Ubuntu Sans Mono',
    'Liberation Mono',
]


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
    zoom_requested = QtCore.pyqtSignal(str)

    @QtCore.pyqtSlot(str, result=str)
    def stats(self, values):
        return display_summary(selection_summary(json.loads(values)))

    @QtCore.pyqtSlot(str)
    def copy(self, text):
        QtWidgets.QApplication.clipboard().setText(text)

    @QtCore.pyqtSlot(str)
    def zoom_image(self, src):
        self.zoom_requested.emit(src)


class CsvModel(QtCore.QAbstractTableModel):
    def __init__(self, rows, dark=False):
        super().__init__()
        self.dark=dark
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
            return QtGui.QColor('#202020' if self.dark else '#eeeeee')

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
    def wheelEvent(self,event):
        if event.modifiers() & QtCore.Qt.ShiftModifier:
            delta=event.pixelDelta().x() or event.pixelDelta().y() or event.angleDelta().x() or event.angleDelta().y()
            bar=self.horizontalScrollBar()
            bar.setValue(bar.value()-delta)
            event.accept()
        else:super().wheelEvent(event)

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


class PasteInput(QtWidgets.QPlainTextEdit):
    image_pasted = QtCore.pyqtSignal(object)

    def insertFromMimeData(self, source):
        image=QtGui.QImage()
        if source.hasImage():
            value=source.imageData()
            if isinstance(value,QtGui.QPixmap):image=value.toImage()
            elif isinstance(value,QtGui.QImage):image=value.copy()
        if image.isNull():
            for kind in ('image/png','image/jpeg','image/bmp','image/tiff','image/webp'):
                if source.hasFormat(kind) and image.loadFromData(source.data(kind)):break
        if not image.isNull():
            self.image_pasted.emit(image)
        elif source.hasText():
            # A new paste is a new request; no Select All step is required.
            self.setPlainText(source.text())


class LineNumbers(QtWidgets.QWidget):
    def __init__(self,editor):
        super().__init__(editor)
        self.editor=editor

    def paintEvent(self,event):
        editor=self.editor
        painter=QtGui.QPainter(self)
        painter.fillRect(event.rect(),QtGui.QColor('#101010' if editor.property('dark') else '#f6f6f6'))
        painter.setPen(QtGui.QColor('#a8a8a8' if editor.property('dark') else '#595959'))
        painter.setFont(editor.font())
        block=editor.firstVisibleBlock()
        top=int(editor.blockBoundingGeometry(block).translated(editor.contentOffset()).top())
        while block.isValid() and top<=event.rect().bottom():
            height=int(editor.blockBoundingRect(block).height())
            if block.isVisible() and top+height>=event.rect().top():
                painter.drawText(0,top,self.width()-8,editor.fontMetrics().height(),
                                 QtCore.Qt.AlignRight,str(block.blockNumber()+1))
            top+=height
            block=block.next()


class NumberedText(QtWidgets.QPlainTextEdit):
    def __init__(self):
        super().__init__()
        self.gutter=LineNumbers(self)
        self.setLineWrapMode(self.WidgetWidth)
        self.setWordWrapMode(QtGui.QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.blockCountChanged.connect(self.update_gutter)
        self.updateRequest.connect(self.repaint_gutter)
        self.update_gutter()

    def gutter_width(self):
        return 16+self.fontMetrics().horizontalAdvance('9')*len(str(max(1,self.blockCount())))

    def update_gutter(self,unused=None):
        self.setViewportMargins(self.gutter_width(),0,0,0)
        self.place_gutter()
        self.gutter.update()

    def place_gutter(self):
        rect=self.viewport().geometry()
        self.gutter.setGeometry(rect.x()-self.gutter_width(),rect.y(),self.gutter_width(),rect.height())

    def repaint_gutter(self,rect,dy):
        if dy:
            self.gutter.scroll(0,dy)
        else:
            self.gutter.update(0,rect.y(),self.gutter_width(),rect.height())

    def resizeEvent(self,event):
        super().resizeEvent(event)
        if hasattr(self,'gutter'):
            self.place_gutter()

    def changeEvent(self,event):
        super().changeEvent(event)
        if event.type()==QtCore.QEvent.FontChange and hasattr(self,'gutter'):
            self.update_gutter()


class SourceHighlight(QtGui.QSyntaxHighlighter):
    def __init__(self,document,path,dark):
        super().__init__(document)
        from pygments import lex
        from pygments.lexers import get_lexer_for_filename, guess_lexer
        from pygments.util import ClassNotFound
        self.lines={}
        text=document.toPlainText()
        if not path or not syntax_safe(text):return
        try:lexer=get_lexer_for_filename(path.name, text)
        except ClassNotFound:
            try:lexer=guess_lexer(text)
            except ClassNotFound:return
        style=source_style(dark)
        line=0;offset=0
        for token,value in lex(document.toPlainText(),lexer):
            color=style.style_for_token(token)['color']
            for part in value.splitlines(keepends=True):
                if color:
                    format=QtGui.QTextCharFormat();format.setForeground(QtGui.QColor('#'+color))
                    self.lines.setdefault(line,[]).append((offset,len(part.rstrip('\n')),format))
                offset+=len(part)
                if part.endswith('\n'):line+=1;offset=0
        self.rehighlight()

    def highlightBlock(self,text):
        for offset,length,format in self.lines.get(self.currentBlock().blockNumber(),[]):self.setFormat(offset,length,format)


class FileListDelegate(QtWidgets.QStyledItemDelegate):
    def paint(self,painter,option,index):
        super().paint(painter,option,index)
        if index.data(QtCore.Qt.UserRole+1) and index.row()>0:
            painter.save()
            painter.setPen(QtGui.QColor('#dedbce'))
            painter.drawLine(option.rect.left()+8,option.rect.top()+1,
                             option.rect.right()-8,option.rect.top()+1)
            painter.restore()


class Folio(QtWidgets.QMainWindow):
    def __init__(self,history_path=None):
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
        self.paste_generation = 0
        self.navigation_generation=0
        self.loading_paths = set()
        self.resolving = False
        self.context_entries = []
        self.context_stack = []
        self.active_batch=None
        self.history=History(history_path)
        self.batches=self.history.recent()
        self.history_complete=False
        self.loading_history=False
        self.history_scroll=0
        self.browsing_folder=False
        self.pending_image=None
        self.pdf_image = None
        self.pdf_zoom = .65
        self.temp = tempfile.TemporaryDirectory(prefix='folio-')
        self.zip_roots={}
        self.dark=False
        self.settings = QtCore.QSettings(str(Path(history_path).with_suffix('.ini')),QtCore.QSettings.IniFormat) if history_path else QtCore.QSettings('Folio', 'Reader')
        self.font_family = str(self.settings.value('font_family', 'JetBrains Mono'))
        self.current_folder = None
        body = QtWidgets.QWidget()
        self.setCentralWidget(body)
        layout = QtWidgets.QVBoxLayout(body)
        layout.setContentsMargins(16, 12, 16, 8)
        self.path_input = PasteInput()
        self.path_input.setPlaceholderText('Paste paths or content')
        self.path_input.setFixedHeight(48)
        header=QtWidgets.QHBoxLayout()
        self.back_button=self.button('← Back',self.go_back)
        self.back_button.setEnabled(False)
        header.addWidget(self.path_input,1)
        header.addStretch()
        self.header=QtWidgets.QWidget()
        self.header.setLayout(header)
        self.date=QtWidgets.QLabel()
        self.date.setObjectName('location')
        self.folder_button=self.button('📁 Folders ▾',self.open_folder_dropdown)
        self.nav=QtWidgets.QHBoxLayout()
        self.nav.addWidget(self.date)
        self.nav.addWidget(self.folder_button)
        self.nav.addStretch()
        self.root_controls=QtWidgets.QWidget()
        root_buttons=QtWidgets.QHBoxLayout(self.root_controls)
        root_buttons.setContentsMargins(0,0,0,0)
        for label,callback in [('Paths',self.show_history),('Recent',self.show_recent),('Downloads',self.show_downloads),('Adhoc',self.show_adhoc),('Projects',self.show_projects)]:
            root_buttons.addWidget(self.button(label,callback))
        self.nav.addWidget(self.root_controls)
        self.pretty=False
        self.prettify_button=self.button('Prettify',self.toggle_prettify)
        self.nav.addWidget(self.prettify_button)
        self.prettify_button.hide()
        self.transpose_button=self.button('Transpose',self.transpose)
        self.transpose_button.hide()
        self.nav.addWidget(self.back_button)
        self.theme_button=self.button('◐',self.toggle_theme)
        self.nav.addWidget(self.theme_button)
        self.top_button=self.button('Top',self.toggle_top,checkable=True)
        self.top_button.setChecked(self.settings.value('top',True,type=bool))
        self.nav.addWidget(self.top_button)
        layout.addLayout(self.nav)
        layout.addWidget(self.header)
        self.clock=QtCore.QTimer(self)
        self.clock.timeout.connect(lambda:self.date.setText(QtCore.QDateTime.currentDateTime().toString('MMMM d · HH:mm:ss')))
        self.clock.start(1000)
        self.date.setText(QtCore.QDateTime.currentDateTime().toString('MMMM d · HH:mm:ss'))
        self.paste_timer = QtCore.QTimer(self)
        self.paste_timer.setSingleShot(True)
        self.paste_timer.setInterval(180)
        self.paste_timer.timeout.connect(self.parse_paste)
        self.path_input.textChanged.connect(self.schedule_paste)
        self.path_input.image_pasted.connect(self.parse_image)
        self.file_list = QtWidgets.QListWidget()
        self.file_list.setItemDelegate(FileListDelegate(self.file_list))
        self.file_list.viewport().setCursor(QtCore.Qt.PointingHandCursor)
        self.file_list.setViewMode(QtWidgets.QListView.ListMode)
        self.file_list.setFlow(QtWidgets.QListView.TopToBottom)
        self.file_list.setMovement(QtWidgets.QListView.Static)
        self.file_list.setResizeMode(QtWidgets.QListView.Adjust)
        self.file_list.setWrapping(False)
        self.file_list.setWordWrap(False)
        self.file_list.setSpacing(0)
        self.file_list.setTextElideMode(QtCore.Qt.ElideMiddle)
        self.file_list.setMinimumHeight(0)
        self.file_list.itemClicked.connect(self.activate_item)
        self.file_list.itemActivated.connect(self.activate_item)
        self.file_list.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.file_list.customContextMenuRequested.connect(self.file_menu)
        layout.addWidget(self.file_list,1)
        self.file_list.hide()
        reading = QtWidgets.QWidget()
        self.reading=reading
        reader = QtWidgets.QVBoxLayout(reading)
        reader.setContentsMargins(0, 0, 0, 0)
        self.location = QtWidgets.QLabel('')
        self.location.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.location.setWordWrap(True)
        self.location.setObjectName('location')
        actions = QtWidgets.QHBoxLayout()
        actions.addWidget(self.location, 1)
        menu = QtWidgets.QMenu(self)
        self.viewer_menu=menu
        menu.addAction('Copy selection',self.copy_selection)
        menu.addAction('Copy content',self.copy_content)
        self.image_copy_action=menu.addAction('Copy image',self.copy_image)
        self.source_toggle = menu.addAction('Source')
        self.source_toggle.setCheckable(True)
        self.source_toggle.triggered.connect(self.refresh_view)
        self.path_button = menu.addAction('Copy path', self.copy_path)
        menu.addAction('Find', self.show_find)
        menu.addSeparator()
        self.font_family_menu = menu.addMenu('Font family')
        for f in AVAILABLE_FONTS:
            self.font_family_menu.addAction(('✓ ' if f == self.font_family else '   ') + f, lambda fam=f: self.set_font_family(fam))
        menu.addAction('Increase font · Ctrl++', lambda:self.zoom(1))
        menu.addAction('Decrease font · Ctrl+-', lambda:self.zoom(-1))
        menu.addAction('Reset font · Ctrl+0', lambda:self.zoom(0))
        menu.addSeparator()
        menu.addAction('Fit PDF width', self.fit_pdf)
        menu.addAction('Previous PDF page',lambda:self.change_page(-1))
        menu.addAction('Next PDF page',lambda:self.change_page(1))
        menu.addSeparator()
        menu.addAction('Close · Esc',self.escape)
        menu.addAction('Open file…', self.choose_files)
        self.copy_button = self.button('Copy', self.copy_visible)
        self.font_down_button = self.button('A-', lambda:self.zoom(-1))
        self.font_up_button = self.button('A+', lambda:self.zoom(1))
        self.font_select_button = self.button('Aa ▾', self.show_font_menu)
        self.nav.insertWidget(self.nav.indexOf(self.back_button),self.copy_button)
        self.nav.removeWidget(self.theme_button)
        self.nav.insertWidget(self.nav.indexOf(self.copy_button),self.font_select_button)
        self.nav.insertWidget(self.nav.indexOf(self.copy_button),self.font_down_button)
        self.nav.insertWidget(self.nav.indexOf(self.copy_button),self.font_up_button)
        self.nav.insertWidget(self.nav.indexOf(self.copy_button),self.theme_button)
        for button,width in [(self.copy_button,64),(self.back_button,80),(self.theme_button,32),(self.top_button,64),(self.font_down_button,36),(self.font_up_button,36),(self.font_select_button,44)]:button.setFixedWidth(width)
        more = QtWidgets.QToolButton()
        more.setText('…')
        more.setMenu(menu)
        more.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        actions.addWidget(more)
        self.toolbar=QtWidgets.QWidget()
        self.toolbar.setLayout(actions)
        reader.addWidget(self.toolbar)
        self.toolbar.hide()
        self.find = QtWidgets.QLineEdit()
        self.find.setPlaceholderText('Find in document')
        self.find.setMaximumWidth(220)
        self.find.returnPressed.connect(self.find_next)
        reader.addWidget(self.find)
        self.find.hide()
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
        controls.addStretch()
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
        self.previous_view_before_zoom = None
        self.bridge.zoom_requested.connect(self.open_image_zoom)
        self.channel.registerObject('folio', self.bridge)
        self.web_page.setWebChannel(self.channel)
        self.source = NumberedText()
        self.source.setReadOnly(True)
        self.source.setFont(QtGui.QFont(self.font_family, 11))
        self.table = CsvTable()
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(False)
        self.table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.table.setSortingEnabled(False)
        self.table.horizontalHeader().setDefaultSectionSize(190)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setDefaultSectionSize(32)
        self.csv_view=QtWidgets.QWidget()
        csv_layout=QtWidgets.QVBoxLayout(self.csv_view)
        csv_layout.setContentsMargins(0,0,0,0)
        csv_controls=QtWidgets.QHBoxLayout()
        csv_controls.addStretch();csv_controls.addWidget(self.transpose_button)
        csv_layout.addLayout(csv_controls);csv_layout.addWidget(self.table)
        self.pdf = PdfScroll()
        self.pdf_label = QtWidgets.QLabel()
        self.pdf_label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.pdf.setWidget(self.pdf_label)
        for widget in (self.web, self.source, self.table, self.pdf):
            self.views.addWidget(self.csv_view if widget is self.table else widget)
            widget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
            widget.customContextMenuRequested.connect(lambda pos,w=widget:self.viewer_menu.exec_(w.mapToGlobal(pos)))
        reader.addWidget(self.views, 1)
        self.stats = QtWidgets.QLabel('Select cells, rows, or columns to see their statistics.')
        self.stats.setWordWrap(True)
        self.stats.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.stats.setObjectName('statistics')
        reader.addWidget(self.stats)
        self.stats.hide()
        layout.addWidget(reading, 1)
        reading.hide()
        self.status = QtWidgets.QLabel('')
        self.status.setObjectName('location')
        layout.addWidget(self.status)
        self.light_style='''
QMainWindow,QWidget {background:#fbf8f1;color:#353b33;font-family:"JetBrains Mono";font-size:13px}
QLabel#location {color:#858879;font-size:11px;padding:4px 0 9px}
QLabel#statistics {color:#3f5143;font-size:12px;padding:6px 0}
QPushButton,QToolButton {background:#f4f1e8;border:1px solid #dddccd;border-radius:5px;padding:5px 10px}
QPushButton:hover {background:#e9eee2;border-color:#a8b8a1}
QPushButton:checked {background:#e1ebdc;color:#315543;border-color:#b0c1a8}
QPushButton:disabled {color:#aaa99e}
QLineEdit,QPlainTextEdit {background:#fffdf8;border:1px solid #dedbcd;border-radius:7px;padding:10px;selection-background-color:#bed5af;selection-color:#17271a}
QListWidget {border:0;background:#fbf8f1;padding:0}
QListWidget::item {background:transparent;border:0;padding:3px 8px}
QListWidget::item:hover {background:#f2f0e7}
QListWidget::item:selected {background:#dfe8d9;color:#2b4d37}
QTableView {background:#fffdf8;alternate-background-color:#f4f1e9;border:1px solid #dedbcd;gridline-color:#e5e2d8;selection-background-color:#bed5af;selection-color:#17271a}
QTableView::item:selected {background:#bed5af;color:#17271a}
QHeaderView::section {background:#eeede3;border:0;padding:8px;color:#435240}
QScrollArea {border:0;background:#eeeae0}
QSplitter::handle {background:#e5e0d4;width:1px}
QTabBar::tab {background:transparent;border:0;border-bottom:2px solid transparent;color:#526b56;padding:7px 12px}
QTabBar::tab:selected {border-bottom-color:#6c8d64;color:#263d2b}
QTabBar::tab:hover {background:#f0eee5}
'''
        self.dark=self.settings.value('dark',False,type=bool)
        self.apply_theme()
        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint,self.top_button.isChecked())
        self.pdf_controls.hide()
        self.source_toggle.setEnabled(False)
        self.copy_button.setEnabled(False)
        self.path_button.setEnabled(False)
        self.source.setPlaceholderText('')
        self.views.setCurrentWidget(self.source)
        for shortcut, callback in [('Ctrl+L', self.ready_for_paste), ('Ctrl+O', self.choose_files), ('Ctrl+F', self.show_find), ('Escape', self.escape),('Ctrl+Shift+C',self.copy_content),('Ctrl++',lambda:self.zoom(1)),('Ctrl+=',lambda:self.zoom(1)),('Ctrl+-',lambda:self.zoom(-1)),('Ctrl+0',lambda:self.zoom(0)),('Alt+Right',lambda:self.change_page(1)),('Alt+Left',lambda:self.change_page(-1))]:
            QtWidgets.QShortcut(QtGui.QKeySequence(shortcut), self, activated=callback)
        QtWidgets.QApplication.instance().installEventFilter(self)
        geometry = self.settings.value('geometry')
        if geometry:
            self.restoreGeometry(geometry)
        self.show_history()
        self.file_list.verticalScrollBar().valueChanged.connect(self.load_more_history)

    def toggle_prettify(self):
        self.pretty=not self.pretty;self.refresh_view()

    def toggle_top(self):
        geometry=self.geometry()
        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint,self.top_button.isChecked())
        self.show();self.setGeometry(geometry)
        self.settings.setValue('top',self.top_button.isChecked())
        self.top_button.setText(('✓ ' if self.top_button.isChecked() else '')+'Top')

    def apply_theme(self):
        style = theme_colors(self.light_style, self.dark)
        style = style.replace('"JetBrains Mono"', f'"{self.font_family}"')
        self.setStyleSheet(style)
        self.source.setProperty('dark', self.dark)
        self.source.setStyleSheet(f'font-family: "{self.font_family}";')
        self.source.gutter.update()
        self.theme_button.setText('☀' if self.dark else '◐')
        self.top_button.setText(('✓ ' if self.top_button.isChecked() else '') + 'Top')

    def toggle_theme(self):
        self.dark = not self.dark
        self.settings.setValue('dark', self.dark)
        self.apply_theme()
        if self.reading.isVisible():
            self.refresh_view()
        else:
            for index in range(self.file_list.count()):
                item = self.file_list.item(index)
                item.setForeground(QtGui.QColor('#a8a8a8' if item.data(QtCore.Qt.UserRole + 1) else '#6cb8e6') if self.dark else QtGui.QColor('#595959' if item.data(QtCore.Qt.UserRole + 1) else '#005a8e'))

    def show_font_menu(self):
        menu = QtWidgets.QMenu(self)
        for f in AVAILABLE_FONTS:
            act = menu.addAction(('✓ ' if f == self.font_family else '   ') + f)
            act.triggered.connect(lambda checked, fam=f: self.set_font_family(fam))
        menu.exec_(self.font_select_button.mapToGlobal(QtCore.QPoint(0, self.font_select_button.height())))

    def set_font_family(self, family):
        self.font_family = family
        self.settings.setValue('font_family', family)
        self.apply_theme()
        font = QtGui.QFont(family, self.source.font().pointSize() or 11)
        self.source.setFont(font)
        self.source.setStyleSheet(f'font-family: "{family}";')
        self.source.gutter.update()
        if hasattr(self, 'font_family_menu'):
            self.font_family_menu.clear()
            for f in AVAILABLE_FONTS:
                self.font_family_menu.addAction(('✓ ' if f == self.font_family else '   ') + f, lambda fam=f: self.set_font_family(fam))
        if self.reading.isVisible():
            self.refresh_view()

    def update_folder_button(self):
        folder = self.current_folder
        if not folder and self.current and self.current.path:
            folder = self.current.path.parent
        if folder and folder.is_dir():
            name = folder.name or str(folder)
            if len(name) > 18:
                name = name[:17] + '…'
            self.folder_button.setText(f'📁 {name} ▾')
        else:
            self.folder_button.setText('📁 Folders ▾')

    def open_folder_dropdown(self):
        folder = self.current_folder
        if not folder or not folder.is_dir():
            if self.current and self.current.path and self.current.path.parent.is_dir():
                folder = self.current.path.parent
            else:
                folder = Path.home()
        menu = QtWidgets.QMenu(self)
        if folder.parent and folder.parent != folder:
            p_name = folder.parent.name or str(folder.parent)
            menu.addAction(f'↑ .. ({p_name})', lambda p=folder.parent: self.open_folder(p))
            menu.addSeparator()
        try:
            entries = sorted(folder.iterdir(), key=lambda p: (not p.is_dir(), p.name.casefold()))
        except OSError as e:
            act = menu.addAction(f'Error: {e}')
            act.setEnabled(False)
            menu.exec_(self.folder_button.mapToGlobal(QtCore.QPoint(0, self.folder_button.height())))
            return
        visible = [p for p in entries if not p.name.startswith('.')] or entries
        dirs = [p for p in visible if p.is_dir()]
        files = [p for p in visible if not p.is_dir()]
        for d in dirs[:30]:
            menu.addAction(f'▸ {d.name}/', lambda p=d: self.open_folder(p))
        if dirs and files:
            menu.addSeparator()
        for f in files[:40]:
            menu.addAction(f'  {f.name}', lambda p=f: self.load_path(p))
        if not dirs and not files:
            act = menu.addAction('(Empty folder)')
            act.setEnabled(False)
        menu.exec_(self.folder_button.mapToGlobal(QtCore.QPoint(0, self.folder_button.height())))

    def show_adhoc(self):
        self.context_stack = []
        self.current_folder = None
        self.update_folder_button()
        self.show_context([repo / 'adhoc' for repo in self.resolver.repos if (repo / 'adhoc').is_dir()])
        self.back_button.setEnabled(True)

    def show_projects(self):
        self.context_stack = []
        self.current_folder = None
        self.update_folder_button()
        home = Path.home()
        priority_names = ['astralane-quant', 'trailblazer', 'whisper-typer', 'grid-grinder', 'personal-finance']
        priority_paths = [home / name for name in priority_names if (home / name).is_dir()]
        other_projects = []
        try:
            for p in sorted(home.iterdir(), key=lambda x: x.name.casefold()):
                if p.name.startswith('.') or not p.is_dir() or p in priority_paths:
                    continue
                if (p / '.git').exists() or any((p / marker).exists() for marker in ('Cargo.toml', 'package.json', 'pyproject.toml', 'requirements.txt', 'Makefile')):
                    other_projects.append(p)
        except OSError:
            pass
        self.show_context(priority_paths + other_projects)
        self.back_button.setEnabled(True)

    def open_zip(self,path):
        if path in self.zip_roots:self.open_folder(self.zip_roots[path]);return
        navigation=self.navigation_generation+1;self.navigation_generation=navigation
        destination=Path(self.temp.name)/('zip-'+uuid.uuid4().hex)/path.stem
        def ready(entries,error):
            if navigation!=self.navigation_generation:return
            if error:self.status.setText(error);return
            self.zip_roots[path]=destination
            self.context_stack.append(list(self.context_entries));self.show_context(entries)
        self.submit(lambda:unzip_contents(path,destination),ready)

    def transpose(self):
        if self.views.currentWidget()==self.csv_view:
            self.transposed=not self.transposed
            rows=self.current.rows
            if self.transposed:
                columns=max(map(len,rows),default=0)
                rows=[[row[c] if c<len(row) else '' for row in rows] for c in range(columns)]
            self.table.setModel(CsvModel(rows,self.dark))
            self.table.selectionModel().selectionChanged.connect(self.selection_changed)
            self.selection_changed()
        elif self.views.currentWidget()==self.web:
            self.web.page().runJavaScript('window.folioTranspose && window.folioTranspose()')

    def button(self, label, callback, checkable=False):
        button = QtWidgets.QPushButton(label)
        button.setCheckable(checkable)
        button.clicked.connect(lambda checked=False:callback())
        return button

    def submit(self, function, callback):
        job = Job(function, lambda result,error:callback(result,error) if not self.closing else None)
        self.pool.start(job)

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.KeyPress and event.matches(QtGui.QKeySequence.Paste):
            focus = QtWidgets.QApplication.focusWidget()
            if focus != self.path_input and not isinstance(focus, QtWidgets.QLineEdit):
                self.path_input.insertFromMimeData(QtWidgets.QApplication.clipboard().mimeData())
                return True
        return super().eventFilter(obj, event)

    def show_find(self):
        self.find.show()
        self.find.setFocus()
        self.find.selectAll()

    def hide_find(self):
        self.find.hide()
        self.path_input.setFocus()

    def ready_for_paste(self):
        self.show_history()
        self.path_input.setFocus()
        self.path_input.selectAll()

    def escape(self):
        if hasattr(self, 'previous_view_before_zoom') and self.previous_view_before_zoom and self.views.currentWidget() == self.pdf:
            self.views.setCurrentWidget(self.previous_view_before_zoom)
            self.previous_view_before_zoom = None
            self.pdf_controls.hide()
            return
        self.find.hide()
        self.paste_generation+=1
        self.generation+=1
        self.pdf_generation+=1
        self.navigation_generation+=1
        self.stats_generation+=1
        self.resolving=False
        self.paste_timer.stop()
        self.path_input.blockSignals(True)
        self.path_input.clear()
        self.path_input.blockSignals(False)
        self.context_stack=[]
        self.pending_image=None
        self.show_history()
        self.path_input.setFocus()

    def show_history(self):
        self.browsing_folder = False
        self.current_folder = None
        self.update_folder_button()
        self.file_list.blockSignals(True)
        self.file_list.clear()
        for batch in self.batches:
            header=QtWidgets.QListWidgetItem(('◩ ' if batch['has_image'] else '≡ ')+batch['stamp'])
            header.setData(QtCore.Qt.UserRole,f"context:{batch['id']}")
            header.setData(QtCore.Qt.UserRole+1,True)
            header.setForeground(QtGui.QColor('#a8a8a8' if self.dark else '#595959'))
            header.setFont(QtGui.QFont(self.font_family,9))
            header.setSizeHint(QtCore.QSize(0,36))
            self.file_list.addItem(header)
            for path in batch['paths']:
                item=self.add_file_item(Path(path))
                item.setData(QtCore.Qt.UserRole+2,batch['id'])
            if not batch['paths']:
                item=QtWidgets.QListWidgetItem('Pasted content')
                item.setData(QtCore.Qt.UserRole,f"dump:{batch['id']}")
                item.setSizeHint(QtCore.QSize(0,36))
                self.file_list.addItem(item)
        self.file_list.blockSignals(False)
        self.reading.hide()
        self.file_list.show()
        self.path_input.show()
        self.header.show()
        self.date.show()
        self.root_controls.show()
        self.prettify_button.hide()
        self.transpose_button.hide()
        self.back_button.setEnabled(False)
        self.status.clear()
        self.copy_button.setEnabled(True)
        self.setWindowTitle('Folio')
        self.file_list.verticalScrollBar().setValue(self.history_scroll)

    def load_more_history(self,value):
        bar=self.file_list.verticalScrollBar()
        if self.loading_history or self.history_complete or self.browsing_folder or self.reading.isVisible() or bar.maximum()==0 or value<bar.maximum()-64:
            return
        self.loading_history=True
        more=self.history.recent(50,len(self.batches))
        if not more:
            self.history_complete=True
        else:
            self.batches.extend(more)
            self.history_scroll=value
            bar.blockSignals(True)
            self.show_history()
            bar.blockSignals(False)
        self.loading_history=False

    def show_recent(self):
        self.context_stack = []
        self.current_folder = None
        self.update_folder_button()
        self.show_context(self.history.recent_files())
        self.back_button.setEnabled(True)

    def show_downloads(self,folder=None):
        folder=Path(folder) if folder else Path.home()/'Downloads'
        self.context_stack = []
        self.current_folder = folder if folder.is_dir() else None
        self.update_folder_button()
        if folder.is_dir():
            generation=self.paste_generation
            def ready(entries,error):
                if generation!=self.paste_generation:
                    return
                if error:
                    self.status.setText(error)
                    return
                self.show_context(entries)
                self.back_button.setEnabled(True)
            self.submit(lambda:sorted(folder.iterdir(),key=lambda p:p.stat().st_mtime,reverse=True),ready)
        else:
            self.show_context([])
            self.status.setText('No Downloads folder')
            self.back_button.setEnabled(True)

    def copy_selection(self):
        if self.views.currentWidget()==self.web:
            QtWidgets.QApplication.clipboard().setText(self.web.page().selectedText())
        elif self.views.currentWidget()==self.source:
            self.source.copy()
        elif self.views.currentWidget()==self.csv_view:
            self.table.keyPressEvent(QtGui.QKeyEvent(QtCore.QEvent.KeyPress,QtCore.Qt.Key_C,QtCore.Qt.ControlModifier))

    def copy_image(self):
        if self.current and self.current.kind=='image':
            QtWidgets.QApplication.clipboard().setImage(QtGui.QImage.fromData(self.current.rows))

    def schedule_paste(self):
        self.pending_image=None
        self.paste_generation += 1
        self.generation += 1
        self.pdf_generation += 1
        self.paste_timer.start()

    def parse_image(self,image):
        self.paste_generation += 1
        self.generation += 1
        self.pdf_generation += 1
        generation=self.paste_generation
        self.paste_timer.stop()
        self.resolving=True
        self.status.setText('Reading image…')
        def read():
            data=QtCore.QByteArray()
            buffer=QtCore.QBuffer(data)
            buffer.open(QtCore.QIODevice.WriteOnly)
            if not image.save(buffer,'PNG'):
                raise ValueError('Could not read clipboard image.')
            png=bytes(data)
            try:return image_text(png),png,None
            except Exception:return '',png,'Image saved; text could not be read.'
        def ready(result,error):
            if generation!=self.paste_generation:
                return
            self.resolving=False
            if error:
                self.status.setText('Image could not be read: '+error)
                return
            text,png,warning=result
            if not text.strip():
                self.batches.insert(0,self.history.add(text,[],png))
                self.pending_image=None
                self.path_input.blockSignals(True);self.path_input.clear();self.path_input.blockSignals(False)
                self.show_history();self.status.setText(warning or 'Image saved; no readable text.')
                return
            self.pending_image=png
            self.path_input.blockSignals(True)
            self.path_input.setPlainText(text)
            self.path_input.blockSignals(False)
            self.paste_timer.stop()
            self.parse_paste()
        self.submit(read,ready)

    def parse_paste(self):
        text = self.path_input.toPlainText()
        if not text.strip():
            return
        generation = self.paste_generation
        image=self.pending_image
        self.resolving = True
        self.status.setText('…')
        def resolve():
            results = []
            names=pasted_paths(text)
            folders=[]
            for name in names:
                try:
                    folders.extend(path for path in self.resolver.resolve(name) if path.is_dir())
                except (OSError,ValueError):
                    pass
            for name in names:
                try:
                    matches=self.resolver.resolve(name)
                    if not matches:
                        matches=sorted(set(p for folder in folders for p in self.resolver.resolve(name,folder)))
                    results.append((name,matches))
                except (OSError, ValueError):
                    results.append((name, []))
            # Folder paths contribute their immediate files to the paste group.
            for folder in sorted(set(folders)):
                try:
                    results.append((str(folder),sorted(folder.iterdir(),key=lambda p:(not p.is_dir(),p.name.casefold()))))
                except OSError:
                    pass
            return results
        def ready(results, error):
            if generation != self.paste_generation:
                return
            self.resolving = False
            if error:
                self.status.setText(error)
                return
            self.status.clear()
            if not results:
                batch=self.history.add(text,[],image)
                self.batches.insert(0,batch)
                self.documents[f"dump:{batch['id']}"]=Document(None,'markdown',text)
                self.select_document_key(f"dump:{batch['id']}")
                return
            seen = set()
            entries=[]
            for name, matches in results:
                for path in matches:
                    if str(path) not in seen:
                        entries.append(path)
                        seen.add(str(path))
            self.batches.insert(0,self.history.add(text,entries,image))
            self.history_scroll=0
            self.pending_image=None
            self.context_stack=[]
            self.path_input.blockSignals(True)
            self.path_input.clear()
            self.path_input.blockSignals(False)
            self.show_history()
            if not entries:
                self.status.setText('No matching paths found')
        self.submit(resolve, ready)

    def add_file_item(self, path):
        path = Path(path)
        try:
            name = str(path.relative_to(Path.home()))
        except ValueError:
            name = str(path)
        for destination in self.zip_roots.values():
            if path.is_relative_to(destination):name=str(path.relative_to(destination));break
        item = QtWidgets.QListWidgetItem(('▸ ' if path.is_dir() else '')+name)
        item.setForeground(QtGui.QColor('#6cb8e6' if self.dark else '#005a8e'))
        item.setData(QtCore.Qt.UserRole, str(path))
        item.setToolTip(str(path))
        item.setSizeHint(QtCore.QSize(0,30))
        self.file_list.addItem(item)
        return item

    def fit_file_list(self):
        self.file_list.setVisible(not self.reading.isVisible())

    def show_context(self,entries=None):
        if entries is not None:
            self.context_entries=list(entries)
        self.file_list.blockSignals(True)
        self.file_list.clear()
        if self.current_folder and self.current_folder.parent and self.current_folder.parent != self.current_folder:
            p_name = self.current_folder.parent.name or str(self.current_folder.parent)
            parent_item = QtWidgets.QListWidgetItem(f'◂ .. ({p_name})')
            parent_item.setData(QtCore.Qt.UserRole, str(self.current_folder.parent))
            parent_item.setForeground(QtGui.QColor('#a8a8a8' if self.dark else '#595959'))
            parent_item.setSizeHint(QtCore.QSize(0, 30))
            self.file_list.addItem(parent_item)
        for path in self.context_entries:
            self.add_file_item(path)
        self.file_list.blockSignals(False)
        self.reading.hide()
        self.file_list.show()
        self.path_input.show()
        self.header.show()
        self.date.show()
        self.root_controls.show()
        self.prettify_button.hide()
        self.transpose_button.hide()
        self.back_button.setEnabled(bool(self.context_stack))
        self.status.clear()
        self.browsing_folder=True
        self.copy_button.setEnabled(True)

    def go_back(self):
        if hasattr(self, 'previous_view_before_zoom') and self.previous_view_before_zoom and self.views.currentWidget() == self.pdf:
            self.views.setCurrentWidget(self.previous_view_before_zoom)
            self.previous_view_before_zoom = None
            self.pdf_controls.hide()
            return
        self.generation+=1
        self.pdf_generation+=1
        if self.reading.isVisible():
            self.show_context() if self.browsing_folder else self.show_history()
        elif self.context_stack:
            item = self.context_stack.pop()
            if isinstance(item, tuple) and len(item) == 2:
                self.current_folder, entries = item
            else:
                self.current_folder = None
                entries = item
            self.update_folder_button()
            self.show_context(entries)
        else:
            self.show_history()

    def open_image_zoom(self, src):
        pixmap = QtGui.QPixmap()
        if src.startswith('data:image/svg+xml;base64,'):
            try:
                from PyQt5.QtSvg import QSvgRenderer
                data = base64.b64decode(src.split(',', 1)[1])
                renderer = QSvgRenderer(data)
                if renderer.isValid():
                    size = renderer.defaultSize()
                    scale = max(1.0, min(4.0, 2400.0 / max(1, size.width())))
                    scaled_size = QtCore.QSize(round(size.width() * scale), round(size.height() * scale))
                    image = QtGui.QImage(scaled_size, QtGui.QImage.Format_ARGB32)
                    image.fill(QtCore.Qt.transparent)
                    painter = QtGui.QPainter(image)
                    renderer.render(painter)
                    painter.end()
                    pixmap = QtGui.QPixmap.fromImage(image)
            except Exception: pass
        elif src.startswith('data:image/'):
            try:
                data = base64.b64decode(src.split(',', 1)[1])
                pixmap.loadFromData(data)
            except Exception: pass
        else:
            if src.startswith('file://'):
                p = Path(unquote(urlparse(src).path))
            elif self.current and self.current.path:
                p = (self.current.path.parent / src).resolve()
            else:
                p = Path(src).resolve()
            if p.exists():
                pixmap.load(str(p))

        if not pixmap.isNull():
            self.previous_view_before_zoom = self.views.currentWidget()
            self.pdf_image = pixmap
            self.views.setCurrentWidget(self.pdf)
            self.fit_pdf()

    def open_folder(self,path):
        path = Path(path).resolve()
        previous = (self.current_folder, list(self.context_entries))
        self.current_folder = path
        self.update_folder_button()
        generation=self.paste_generation
        self.navigation_generation+=1
        navigation=self.navigation_generation
        self.status.setText('…')
        def ready(entries,error):
            if generation!=self.paste_generation or navigation!=self.navigation_generation:
                return
            if error:
                self.status.setText(error)
                return
            self.context_stack.append(previous)
            self.show_context(entries)
            if not entries:
                self.status.setText('Empty folder')
        self.submit(lambda:sorted(path.iterdir(),key=lambda p:(not p.is_dir(),p.name.casefold())),ready)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'file_list'):
            self.fit_file_list()

    def activate_item(self, item, previous=None):
        if not item:
            return
        key = item.data(QtCore.Qt.UserRole)
        if not key:
            return
        self.active_batch=item.data(QtCore.Qt.UserRole+2)
        if not self.browsing_folder and not self.reading.isVisible():
            self.history_scroll=self.file_list.verticalScrollBar().value()
        if key.startswith('context:'):
            batch=next(b for b in self.batches if key==f"context:{b['id']}")
            image=self.history.image(batch['id']) if batch['has_image'] else None
            self.documents[key]=Document(None,'image' if image else 'markdown',batch['source'],rows=image)
            self.select_document_key(key)
            return
        if key.startswith('dump:'):
            if key not in self.documents:
                batch=next(b for b in self.batches if key==f"dump:{b['id']}")
                self.documents[key]=Document(None,'markdown',batch['source'])
            self.select_document_key(key)
            return
        if key and Path(key).is_dir():
            self.open_folder(Path(key))
            return
        if key in self.documents:
            if self.current is not self.documents[key] or not self.reading.isVisible():
                self.select_document(item)
        elif key:
            self.load_path(Path(key))

    def paste_paths(self):
        self.path_input.setPlainText(QtWidgets.QApplication.clipboard().text())
        self.open_paths()

    def choose_files(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, 'Open in Folio', str(Path.home()), 'Documents (*.md *.markdown *.csv *.tsv *.txt *.pdf *.log *.json *.yaml *.toml);;All files (*)')
        if paths:
            self.path_input.setPlainText('\n'.join(paths))

    def open_paths(self, text=None, context=None):
        self.paste_timer.stop()
        paths = pasted_paths(text if text is not None else self.path_input.toPlainText())
        if not paths:
            self.status.setText('Paste a file path first, or use Paste content for Markdown and diagrams.')
            return
        self.resolving = True
        self.status.setText('Resolving paths…')
        self.submit(lambda:[(name, self.resolver.resolve(name, context)) for name in paths], self.paths_ready)

    def paths_ready(self, results, error):
        self.resolving = False
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
        if path.is_dir():
            self.open_folder(path)
            return
        if path.suffix.lower()=='.pdf':
            command=shutil.which('firefox') or shutil.which('xdg-open')
            if command:
                subprocess.Popen([command,path.resolve().as_uri()],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
                self.history.opened(path.resolve())
                self.status.setText('Opened in Firefox')
            else:self.status.setText('Firefox is not installed')
            return
        if path.suffix.lower()=='.zip':
            self.open_zip(path)
            return
        key = str(path.resolve())
        self.navigation_generation+=1
        navigation=self.navigation_generation
        if key in self.documents:
            self.document_ready(self.documents[key], None)
            return
        if key in self.loading_paths:
            return
        self.loading_paths.add(key)
        generation = self.paste_generation
        self.status.setText('Opening ' + path.name + '…')
        def ready(document,error):
            self.loading_paths.discard(key)
            if generation == self.paste_generation and navigation==self.navigation_generation:
                self.document_ready(document,error)
        self.submit(lambda:load_document(path), ready)

    def document_ready(self, document, error):
        if error:
            self.status.setText('Could not open: ' + error)
            return
        key = str(document.path) if document.path else f'paste:{uuid.uuid4().hex}'
        self.documents[key] = document
        if document.path:
            self.history.opened(document.path) if not document.path.is_relative_to(Path(self.temp.name)) else None
        if not document.path:
            self.select_document_key(key)
            return
        item = next((self.file_list.item(i) for i in range(self.file_list.count()) if self.file_list.item(i).data(QtCore.Qt.UserRole) == key),None)
        if item is None:
            item = self.add_file_item(document.path)
        self.fit_file_list()
        self.file_list.setCurrentItem(item)
        self.select_document(item)

    def select_document_key(self,key):
        item=QtWidgets.QListWidgetItem()
        item.setData(QtCore.Qt.UserRole,key)
        self.select_document(item)

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
            self.location.clear()
            return
        self.current = self.documents[item.data(QtCore.Qt.UserRole)]
        if self.current and self.current.path:
            self.current_folder = self.current.path.parent
            self.update_folder_button()
        self.pretty=False
        self.stats_generation+=1
        self.file_list.hide()
        self.path_input.hide()
        self.back_button.setEnabled(True)
        self.root_controls.hide()
        self.reading.show()
        self.image_copy_action.setEnabled(self.current.kind=='image')
        self.header.hide()
        self.date.hide()
        self.find.hide()
        self.status.clear()
        self.setWindowTitle(self.current.path.name if self.current.path else 'Folio')
        self.location.setText(str(self.current.path) if self.current.path else '')
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
        self.pdf_controls.hide()
        self.stats.hide()
        self.transpose_button.hide()
        self.prettify_button.hide()
        self.status.clear()
        if source:
            text=doc.text
            if self.pretty:
                try:text=pretty_json(text)
                except (ValueError,RecursionError) as exc:self.pretty=False;self.status.setText('Cannot prettify: '+str(exc))
            if hasattr(self,'highlighter'):self.highlighter.setDocument(None);self.highlighter.deleteLater();del self.highlighter
            self.source.setPlainText(text)
            if doc.path and doc.path.suffix.lower()=='.json':
                self.prettify_button.setText('Original' if self.pretty else 'Prettify')
                self.prettify_button.show()
            self.highlighter=SourceHighlight(self.source.document(),doc.path,self.dark)
            if not syntax_safe(text):self.status.setText('Raw text · syntax coloring skipped for large files or long lines.')
            self.views.setCurrentWidget(self.source)
        elif doc.kind == 'csv':
            self.transposed=False
            self.table.setModel(CsvModel(doc.rows,self.dark))
            self.transpose_button.show()
            self.table.selectionModel().selectionChanged.connect(self.selection_changed)
            self.selection_changed()
            self.views.setCurrentWidget(self.csv_view)
        elif doc.kind == 'pdf':
            self.views.setCurrentWidget(self.pdf)
            self.render_pdf()
        elif doc.kind == 'image':
            self.views.setCurrentWidget(self.pdf)
            self.pdf_image=QtGui.QPixmap()
            self.pdf_image.loadFromData(doc.rows)
            self.fit_pdf()
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
                self.status.clear()
            self.submit(lambda:document_html(doc.text,self.dark,self.font_family), ready)

    def render_pdf(self, unused=None):
        if not self.current or self.current.kind != 'pdf' or self.source_toggle.isChecked():
            return
        self.pdf_generation += 1
        generation = self.pdf_generation
        doc, page = self.current, self.page_number.value()
        self.pdf_loaded_page=None
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
            self.pdf_loaded_page=page
            self.fit_pdf()
            self.pdf.verticalScrollBar().setValue(0)
            self.pdf.horizontalScrollBar().setValue(0)
            self.status.clear()
        self.submit(lambda:pdf_page(doc.path, page, 144), ready)

    def selection_changed(self, unused=None, previous=None):
        self.stats_generation += 1
        generation = self.stats_generation
        ranges = [(s.top(),s.bottom(),s.left(),s.right()) for s in self.table.selectionModel().selection()]
        if not ranges:
            self.stats.clear()
            self.stats.hide()
            return
        rows = self.table.model().rows
        self.stats.setText('Calculating selection…')
        def calculate():
            return display_summary(selection_summary(rows[r][c] if c<len(rows[r]) else ''
                for top,bottom,left,right in ranges for r in range(top,bottom+1) for c in range(left,right+1)))
        def ready(result, error):
            if generation == self.stats_generation:
                self.stats.setText(result if not error else 'Statistics: '+error)
                self.stats.show()
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
        if direction == 0:
            if self.views.currentWidget() == self.pdf:
                self.pdf_zoom = .65
                self.scale_pdf()
            elif self.views.currentWidget() == self.web:
                self.web.setZoomFactor(1.0)
            else:
                widget = self.views.currentWidget()
                font = widget.font()
                font.setPointSize(11)
                widget.setFont(font)
            return
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
        elif widget == self.csv_view:
            rows = self.table.model().rows
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

    def copy_visible(self):
        if self.reading.isVisible() and self.current:
            if self.current.kind=='image':self.copy_image()
            else:self.copy_content()
        else:
            paths=[str(p) for p in self.context_entries] if self.browsing_folder else [p for batch in self.batches for p in batch['paths']]
            text='\n'.join(dict.fromkeys(paths))
            if not text and not self.browsing_folder and self.batches:text=self.batches[0]['source']
            if text:QtWidgets.QApplication.clipboard().setText(text)

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
        if not item or item.data(QtCore.Qt.UserRole) not in self.documents:
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
        self.paste_timer.stop()
        self.clock.stop()
        QtWidgets.QApplication.instance().removeEventFilter(self)
        self.hide()
        self.pool.clear()
        # Workers hold callbacks into this window. Defer deletion until they
        # finish rather than destroying Qt objects under a running callback.
        self.pool.waitForDone()
        self.temp.cleanup()
        self.history.close()
        super().closeEvent(event)


def main():
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    application = QtWidgets.QApplication(sys.argv)
    application.setApplicationName('FolioQt')
    window = Folio()
    window.show()
    if len(sys.argv)==3 and sys.argv[1]=='--read':
        window.load_path(Path(sys.argv[2]))
    elif len(sys.argv)==3 and sys.argv[1]=='--image':
        window.parse_image(QtGui.QImage(sys.argv[2]))
    elif len(sys.argv) > 1:
        paths = [Path(p).expanduser().resolve() for p in sys.argv[1:] if Path(p).expanduser().exists()]
        if paths:
            window.context_entries = paths
            window.load_path(paths[0])
        else:
            window.path_input.setPlainText('\n'.join(sys.argv[1:]))
    else:
        latest = latest_history_file(window.history, window.batches)
        if latest:
            window.load_path(latest)
    return application.exec_()


if __name__ == '__main__':
    sys.exit(main())
