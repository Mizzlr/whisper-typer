"""Behavioral checks. Use --render under Xvfb for actual offline JS rendering."""
import os
import subprocess
import sys
import tempfile
import time
import unittest
from decimal import Decimal
from pathlib import Path

from files import PathResolver, pasted_paths, clean_path, load_document, Document, pdf_page
from cell_stats import numeric, selection_summary
from rendering import rendered_body

RENDER = '--render' in sys.argv
if RENDER:
    sys.argv.remove('--render')


class FilesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.home = Path(self.temp.name)
        for name in ('repo-a','repo-b'):
            repo = self.home/name
            repo.mkdir()
            subprocess.run(['git','init','-q',str(repo)], check=True)
            (repo/'reports').mkdir()
            (repo/'reports/note.md').write_text('# Hello')
            subprocess.run(['git','-C',str(repo),'add','.'], check=True)
        self.resolver = PathResolver(self.home, self.home/'other')

    def tearDown(self):
        self.temp.cleanup()

    def test_absolute_home_repo_and_line_suffix(self):
        target = self.home/'repo-a/reports/note.md'
        self.assertEqual(self.resolver.resolve(str(target)+':4:2'),[target])
        self.assertEqual(self.resolver.resolve('repo-a/reports/note.md'),[target])
        self.assertEqual(clean_path(target.as_uri()+'#L4'),str(target))

    def test_ambiguous_relative_and_basename(self):
        self.assertEqual(len(self.resolver.resolve('reports/note.md')),2)
        self.assertEqual(len(self.resolver.resolve('note.md')),2)
        self.assertEqual(self.resolver.resolve('missing.md'),[])

    def test_relative_document_link_precedence(self):
        self.assertEqual(self.resolver.resolve('note.md',self.home/'repo-a/reports/current.md'),[self.home/'repo-a/reports/note.md'])

    def test_parse_prose_links_and_multiple_paths(self):
        self.assertEqual(pasted_paths('Report [CSV](repo-a/data.csv) and [note](repo-b/note.md)\nSee `reports/note.md:8`.'),['repo-a/data.csv','repo-b/note.md','reports/note.md:8'])
        self.assertEqual(pasted_paths('Read reports/data.csv and notes/result.md.'),['reports/data.csv','notes/result.md'])

    def test_spaces_and_file_uri(self):
        path = self.home/'with spaces.md';path.write_text('contents')
        self.assertEqual(pasted_paths('"'+str(path)+'"'),[str(path)])
        self.assertEqual(self.resolver.resolve(path.as_uri()),[path])
        with self.assertRaises(ValueError):self.resolver.resolve('file://remote/x.md')

    def test_csv_quoted_multiline_and_ragged(self):
        path = self.home/'data.csv';path.write_text('name,value\n"two\nlines",1.25\nshort\n')
        doc = load_document(path)
        self.assertEqual(doc.rows,[['name','value'],['two\nlines','1.25'],['short']])
        self.assertEqual(doc.text,path.read_text())

    def test_semicolon_tsv_empty_and_encoding(self):
        for suffix, source in [('.csv','a;b\n1;2\n'),('.tsv','a\tb\n1\t2\n')]:
            path=self.home/('data'+suffix);path.write_text(source)
            self.assertEqual(load_document(path).rows[1],['1','2'])
        empty=self.home/'empty.csv';empty.write_text('');self.assertEqual(load_document(empty).rows,[])
        utf=self.home/'utf.txt';utf.write_bytes('Hello α'.encode('utf-16'));self.assertEqual(load_document(utf).text,'Hello α')
        binary=self.home/'binary';binary.write_bytes(b'abc\0def')
        with self.assertRaises(ValueError):load_document(binary)

    def test_wrapped_directories_partial_report_and_fuzzy_recent_untracked(self):
        repo=self.home/'repo-a'
        date=repo/'adhoc/2000-01-02/example_report'
        date.mkdir(parents=True)
        file=date/'RESULTS.csv';file.write_text('value\n1\n')
        resolver=PathResolver(self.home,self.home/'other')
        text='Please read (adhoc/2000-01-\n 02/example_report/RESULTS.csv). Other unrelated prose.'
        self.assertEqual(pasted_paths(text),['adhoc/2000-01-02/example_report/RESULTS.csv'])
        self.assertEqual(resolver.resolve('2/example_report/RESULTS.csv'),[file])
        self.assertEqual(resolver.resolve('RESULT.csv'),[file])
        self.assertEqual(resolver.resolve('example_report'),[date])
        self.assertEqual(pasted_paths('No files here, just ordinary prose.'),[])

    def test_selection_statistics_exact_and_units(self):
        result=selection_summary(['0.1','0.2','1,000','label','','15%','$2','NaN','Inf'])
        self.assertEqual(result['cells'],9);self.assertEqual(result['numeric'],3)
        self.assertEqual(result['sum'],Decimal('1000.3'))
        self.assertEqual(result['minimum'],Decimal('0.1'))
        self.assertIsNone(numeric('1,23'))
        self.assertEqual(selection_summary(['9007199254740993','1'])['sum'],Decimal('9007199254740994'))

    def test_markdown_math_mermaid_and_sanitization(self):
        body=rendered_body('# Heading\n\n$x_i = \\frac{a}{b}$\n\n```mermaid\nflowchart TD\n A --> B\n```\n\n<script>alert(1)</script>\n<img src="x" onerror="alert(1)">')
        self.assertIn('class="mermaid"',body)
        self.assertIn('\\frac{a}{b}',body)
        self.assertNotIn('<script',body);self.assertNotIn('onerror=',body)
        self.assertIn('class="mermaid"',rendered_body('flowchart LR\n A-->B'))
        self.assertIn('$x_i$',rendered_body('`$x_i$`'))


class UiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt5 import QtCore,QtWidgets
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseSoftwareOpenGL)
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
        cls.application=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        from qt_app import Folio
        self.temp=tempfile.TemporaryDirectory()
        self.path=Path(self.temp.name)
        self.window=Folio(self.path/'history.sqlite3')

    def tearDown(self):
        self.window.close();self.window.deleteLater()
        from PyQt5 import QtCore
        QtCore.QCoreApplication.sendPostedEvents(None,QtCore.QEvent.DeferredDelete)
        self.application.processEvents()
        self.temp.cleanup()

    def wait(self, predicate, seconds=20):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            self.application.processEvents()
            if predicate():return
            time.sleep(.02)
        self.fail('Timed out waiting for background rendering')

    def js(self, source):
        result=[]
        self.window.web.page().runJavaScript(source,lambda value:result.append(value))
        self.wait(lambda:bool(result))
        return result[0]

    def test_csv_selection_stats_copy_and_full_source(self):
        from PyQt5 import QtCore,QtGui,QtWidgets
        doc=Document(self.path/'sample.csv','csv','Name,Amount\nA,0.1\nB,0.2\n',[['Name','Amount'],['A','0.1'],['B','0.2']])
        self.window.document_ready(doc,None)
        selection=QtCore.QItemSelection(self.window.table.model().index(1,1),self.window.table.model().index(2,1))
        self.window.table.selectionModel().select(selection,QtCore.QItemSelectionModel.ClearAndSelect)
        self.wait(lambda:'Sum 0.3' in self.window.stats.text())
        event=QtGui.QKeyEvent(QtCore.QEvent.KeyPress,QtCore.Qt.Key_C,QtCore.Qt.ControlModifier)
        self.window.table.keyPressEvent(event)
        self.assertEqual(QtWidgets.QApplication.clipboard().text(),'0.1\n0.2\n')
        self.window.copy_content();self.assertEqual(QtWidgets.QApplication.clipboard().text(),doc.text)
        self.window.source_toggle.setChecked(True);self.window.refresh_view()
        self.assertEqual(self.window.source.toPlainText(),doc.text)
        self.assertFalse(self.window.stats.isVisible())

    def test_pasted_path_background_load_and_reopen(self):
        path=self.path/'a file with spaces.txt'
        source='Original text\n'+('Full contents, not a preview.\n'*1000)
        path.write_text(source)
        self.window.path_input.setPlainText('`'+str(path)+':12`')
        self.window.open_paths()
        self.wait(lambda:self.window.current is not None)
        self.assertEqual(self.window.current.path,path)
        self.window.source_toggle.setChecked(True);self.window.refresh_view()
        self.assertEqual(self.window.source.toPlainText(),source)
        self.window.open_paths(str(path))
        self.wait(lambda:not self.window.resolving)
        self.assertEqual(self.window.file_list.count(),1)
        self.window.copy_path()
        from PyQt5 import QtWidgets
        self.assertEqual(QtWidgets.QApplication.clipboard().text(),str(path))

    def test_text_wrap_and_line_numbers_preserve_original_lines(self):
        from PyQt5 import QtWidgets
        self.window.show()
        text=('A long sentence that should wrap naturally. '*80)+'\n'+('x'*800)+'\nLast line.'
        self.window.document_ready(Document(self.path/'sample.txt','text',text),None)
        self.application.processEvents()
        editor=self.window.source
        self.assertEqual(editor.lineWrapMode(),QtWidgets.QPlainTextEdit.WidgetWidth)
        self.assertEqual(editor.blockCount(),3)
        self.assertGreater(editor.document().firstBlock().layout().lineCount(),1)
        self.assertEqual(editor.horizontalScrollBar().maximum(),0)
        self.assertTrue(editor.gutter.isVisible())
        self.assertEqual(editor.viewportMargins().left(),editor.gutter_width())
        self.assertEqual(editor.toPlainText(),text)
        self.window.copy_content()
        self.assertEqual(QtWidgets.QApplication.clipboard().text(),text)
        width=editor.gutter_width()
        self.window.zoom(1)
        self.assertGreaterEqual(editor.gutter_width(),width)

    def paste(self,text):
        from PyQt5 import QtCore
        mime=QtCore.QMimeData();mime.setText(text)
        self.window.path_input.insertFromMimeData(mime)
        self.wait(lambda:not self.window.resolving and not self.window.paste_timer.isActive())

    def test_auto_paste_group_click_escape_and_old_new_history(self):
        self.window.show()
        file=self.path/'first.txt';file.write_text('Just the file content.')
        second=self.path/'second.csv';second.write_text('name,value\nA,2\n')
        self.paste('Some unrelated prose.\n('+str(file)+')')
        self.assertEqual(len(self.window.batches),1)
        self.assertEqual(self.window.batches[0]['paths'],[str(file)])
        item=next(self.window.file_list.item(i) for i in range(self.window.file_list.count()) if self.window.file_list.item(i).data(256)==str(file))
        self.window.activate_item(item)
        self.wait(lambda:self.window.current is not None)
        self.assertTrue(self.window.reading.isVisible())
        self.assertFalse(self.window.header.isVisible());self.assertFalse(self.window.toolbar.isVisible())
        self.window.source_toggle.setChecked(True);self.window.refresh_view()
        self.assertEqual(self.window.source.toPlainText(),'Just the file content.')
        self.window.escape()
        self.assertTrue(self.window.file_list.isVisible());self.assertEqual(self.window.path_input.toPlainText(),'')
        item=next(self.window.file_list.item(i) for i in range(self.window.file_list.count()) if self.window.file_list.item(i).data(256)==str(file))
        self.window.activate_item(item)
        self.assertTrue(self.window.reading.isVisible())
        self.window.escape()
        self.paste(str(second))
        self.assertEqual(len(self.window.batches),2)
        self.assertEqual(self.window.batches[1]['paths'],[str(file)])
        from history import History
        history=History(self.path/'history.sqlite3')
        self.assertEqual(len(history.recent()),2)
        self.assertEqual((self.path/'history.sqlite3').stat().st_mode&0o777,0o600)
        history.close()

    def test_folder_dump_expands_files_browse_recent_and_downloads(self):
        self.window.show()
        folder=self.path/'reports';folder.mkdir()
        file=folder/'report.txt';file.write_text('A useful report.')
        self.paste(str(folder))
        self.assertIn(str(file),self.window.batches[0]['paths'])
        self.window.open_folder(folder)
        self.wait(lambda:self.window.browsing_folder)
        self.assertEqual(self.window.file_list.count(),1)
        self.window.activate_item(self.window.file_list.item(0))
        self.wait(lambda:self.window.current is not None)
        self.window.escape();self.window.show_recent()
        self.assertEqual(self.window.context_entries,[file])
        self.window.show_downloads(folder)
        self.wait(lambda:self.window.context_entries==[file] and self.window.back_button.isVisible())

    def test_switch_files_above_document_without_returning_to_history(self):
        self.window.show()
        first=self.path/'notes.txt';first.write_text('A short note.')
        second=self.path/'data.csv';second.write_text('name,value\nA,4\n')
        self.paste(str(first)+'\n'+str(second))
        item=next(self.window.file_list.item(i) for i in range(self.window.file_list.count()) if self.window.file_list.item(i).data(256)==str(first))
        self.window.activate_item(item)
        self.wait(lambda:self.window.current is not None)
        self.assertTrue(self.window.file_tabs.isVisible())
        self.assertEqual(self.window.file_tabs.count(),2)
        index=next(i for i in range(2) if self.window.file_tabs.tabData(i)==str(second))
        self.window.file_tabs.setCurrentIndex(index)
        self.wait(lambda:self.window.current.path==second)
        self.assertEqual(self.window.table.model().index(1,1).data(),'4')
        self.assertFalse(self.window.header.isVisible())
        index=next(i for i in range(2) if self.window.file_tabs.tabData(i)==str(first))
        self.window.file_tabs.setCurrentIndex(index)
        self.window.source_toggle.setChecked(True);self.window.refresh_view()
        self.assertEqual(self.window.source.toPlainText(),'A short note.')
        self.window.escape();self.assertTrue(self.window.file_list.isVisible())

    def test_image_paste_keeps_original_and_extracted_text_zoom_and_escape(self):
        from PyQt5 import QtCore,QtGui,QtWidgets
        from unittest.mock import patch
        self.window.show()
        file=self.path/'report.txt';file.write_text('Report content.')
        image=QtGui.QImage(800,400,QtGui.QImage.Format_RGB32);image.fill(QtGui.QColor('#fbf8f1'))
        mime=QtCore.QMimeData();mime.setImageData(image)
        with patch('qt_app.image_text',return_value=str(file)):
            self.window.path_input.insertFromMimeData(mime)
            self.wait(lambda:bool(self.window.batches))
        batch=self.window.batches[0]
        self.assertTrue(batch['has_image']);self.assertEqual(batch['source'],str(file))
        self.assertEqual(batch['paths'],[str(file)])
        self.window.activate_item(self.window.file_list.item(0))
        self.assertEqual(self.window.current.kind,'image')
        width=self.window.pdf_label.width();self.window.zoom(1)
        self.assertGreater(self.window.pdf_label.width(),width)
        self.window.copy_image();self.assertFalse(QtWidgets.QApplication.clipboard().image().isNull())
        self.window.escape();self.assertTrue(self.window.file_list.isVisible())

    def test_copy_and_back_positions_stay_fixed_between_home_and_documents(self):
        self.window.show();self.application.processEvents()
        buttons=[self.window.top_button,self.window.back_button,self.window.copy_button,self.window.theme_button]
        initial=[b.mapToGlobal(b.rect().topLeft()).x() for b in buttons]
        self.assertTrue(self.window.back_button.isVisible());self.assertFalse(self.window.back_button.isEnabled())
        file=self.path/'note.txt';file.write_text('Original file contents.')
        batch=self.window.history.add('original dump',[file]);self.window.batches.insert(0,batch)
        self.window.show_history();self.window.copy_button.click()
        from PyQt5 import QtWidgets
        self.assertEqual(QtWidgets.QApplication.clipboard().text(),str(file))
        self.window.load_path(file);self.wait(lambda:self.window.current is not None)
        self.application.processEvents();self.window.copy_button.click()
        self.assertEqual(QtWidgets.QApplication.clipboard().text(),'Original file contents.')
        self.assertEqual([b.mapToGlobal(b.rect().topLeft()).x() for b in buttons],initial)
        self.window.document_ready(Document(self.path/'data.csv','csv','a,b',[['a','b'],['1','2']]),None)
        self.application.processEvents()
        self.assertEqual([b.mapToGlobal(b.rect().topLeft()).x() for b in buttons],initial)
        self.window.escape();self.application.processEvents()
        self.assertEqual([b.mapToGlobal(b.rect().topLeft()).x() for b in buttons],initial)
        self.window.copy_button.click();self.assertEqual(QtWidgets.QApplication.clipboard().text(),str(file))

    def test_json_prettify_is_reversible_and_preserves_copy_and_file(self):
        from PyQt5 import QtWidgets
        self.window.show()
        file=self.path/'example.json';raw='{"value":1.12345678901234567890123456789,"items":[1,2]}'
        file.write_text(raw);self.window.load_path(file);self.wait(lambda:self.window.current is not None)
        self.application.processEvents();self.assertTrue(self.window.prettify_button.isVisible())
        self.window.prettify_button.click()
        self.assertIn('\n  "value": 1.12345678901234567890123456789',self.window.source.toPlainText())
        self.window.copy_button.click();self.assertEqual(QtWidgets.QApplication.clipboard().text(),raw)
        self.assertEqual(file.read_text(),raw)
        self.window.prettify_button.click();self.assertEqual(self.window.source.toPlainText(),raw)

    @unittest.skipUnless(RENDER,'Actual browser table controls require --render')
    def test_markdown_transpose_buttons_are_per_table(self):
        self.window.show()
        source='| Name | Value |\n| --- | --- |\n| A | 1 |\n| B | 2 |\n\n| Label | Amount |\n| --- | --- |\n| C | 3 |'
        self.window.document_ready(Document(None,'markdown',source),None)
        self.wait(lambda:self.js("document.documentElement.dataset.folioReady==='true'"))
        self.assertEqual(self.js("document.querySelectorAll('.table-controls button').length"),2)
        self.js("document.querySelector('.table-controls button').click()")
        self.assertEqual(self.js("document.querySelectorAll('article table')[0].rows.length"),2)
        self.assertEqual(self.js("document.querySelectorAll('article table')[1].rows[1].cells.length"),3)
        self.assertEqual(self.js("document.querySelectorAll('article table')[1].rows[1].cells[1].textContent"),'C')

    def test_csv_transpose_shift_wheel_and_syntax_theme(self):
        from PyQt5 import QtCore,QtGui
        self.window.show()
        rows=[['label']+[f'c{i}' for i in range(12)],['A']+[str(i) for i in range(12)],['B']+[str(i+1) for i in range(12)]]
        self.window.document_ready(Document(self.path/'data.csv','csv','original',rows),None)
        self.application.processEvents()
        bar=self.window.table.horizontalScrollBar()
        event=QtGui.QWheelEvent(QtCore.QPointF(20,20),QtCore.QPointF(20,20),QtCore.QPoint(0,0),QtCore.QPoint(0,-120),QtCore.Qt.NoButton,QtCore.Qt.ShiftModifier,QtCore.Qt.NoScrollPhase,False)
        self.application.sendEvent(self.window.table.viewport(),event)
        self.assertGreater(bar.value(),0)
        self.window.transpose();self.assertEqual(self.window.table.model().rows[1],['c0','0','1'])
        self.window.transpose();self.assertEqual(self.window.table.model().rows,rows)
        self.window.toggle_theme();self.assertTrue(self.window.dark)
        self.window.document_ready(Document(self.path/'example.py','text','import os\nvalue = 42'),None)
        self.application.processEvents()
        self.assertTrue(self.window.source.document().firstBlock().layout().formats())
        self.window.top_button.setChecked(False);self.window.toggle_top()
        self.assertFalse(self.window.windowFlags() & QtCore.Qt.WindowStaysOnTopHint)

    def test_pdf_firefox_zip_and_adhoc_top_controls(self):
        from unittest.mock import patch
        import zipfile
        self.window.show()
        pdf=self.path/'example.pdf';pdf.write_bytes(b'%PDF synthetic')
        with patch('qt_app.subprocess.Popen') as launch:
            self.window.load_path(pdf)
            self.assertIn('firefox',launch.call_args[0][0][0])
            self.assertEqual(launch.call_args[0][0][1],pdf.as_uri())
        archive=self.path/'example.zip'
        with zipfile.ZipFile(archive,'w') as out:out.writestr('note.md','# From ZIP')
        self.window.load_path(archive)
        self.wait(lambda:bool(self.window.zip_roots))
        self.assertEqual(len(self.window.context_entries),1)
        self.window.load_path(self.window.context_entries[0]);self.wait(lambda:self.window.current is not None)
        self.assertEqual(self.window.current.text,'# From ZIP')
        self.assertTrue(self.window.back_button.isVisible())
        repo=self.path/'repo';(repo/'adhoc').mkdir(parents=True)
        self.window.resolver.repos=[repo];self.window.show_adhoc()
        self.assertEqual(self.window.context_entries,[repo/'adhoc'])

    def test_long_line_and_large_source_skip_tokenization_preserve_copy_and_theme(self):
        from unittest.mock import patch
        self.window.show()
        for source in ('<div data-value="'+'x'*6000+'">text</div>', 'value = 1\n'*20000):
            with patch('pygments.lex',side_effect=AssertionError('Large source must not be tokenized')) as tokenize:
                self.window.document_ready(Document(self.path/'large.html','text',source),None)
                self.application.processEvents();tokenize.assert_not_called()
            self.assertEqual(self.window.source.toPlainText(),source)
            self.assertIn('Raw text',self.window.status.text())
            self.window.copy_content();self.assertEqual(self.application.clipboard().text(),source)
        self.window.document_ready(Document(self.path/'small.py','text','import os\nvalue = 42'),None)
        self.application.processEvents()
        formats=self.window.source.document().firstBlock().layout().formats()
        self.assertEqual(formats[0].format.foreground().color().name(),'#a04900')
        self.window.toggle_theme();self.application.processEvents()
        formats=self.window.source.document().firstBlock().layout().formats()
        self.assertEqual(formats[0].format.foreground().color().name(),'#e9ae7e')
        self.assertIn('background:#000000',self.window.styleSheet())

    def test_encoded_image_with_text_and_ocr_empty_or_failure_stays_in_history(self):
        from unittest.mock import patch
        from PyQt5 import QtCore,QtGui
        image=QtGui.QImage(30,20,QtGui.QImage.Format_RGB32);image.fill(QtGui.QColor('green'))
        encoded=QtCore.QByteArray();buffer=QtCore.QBuffer(encoded);buffer.open(QtCore.QIODevice.WriteOnly);image.save(buffer,'PNG')
        for outcome in ('',RuntimeError('synthetic OCR failure')):
            mime=QtCore.QMimeData();mime.setData('image/png',encoded);mime.setText('alternate clipboard text')
            with patch('qt_app.image_text',side_effect=outcome if isinstance(outcome,Exception) else None,return_value=''):
                self.window.path_input.insertFromMimeData(mime)
                self.wait(lambda:not self.window.resolving)
            batch=self.window.batches[0];self.assertTrue(batch['has_image'])
            self.assertEqual(self.window.history.image(batch['id']),bytes(encoded))
        folder=self.path/'folder';folder.mkdir()
        item=self.window.add_file_item(folder);self.assertTrue(item.text().startswith('▸ '))

    def test_pdf_load_extract_page_navigation_zoom_pan(self):
        from PyQt5 import QtGui,QtCore
        path=self.path/'sample.pdf'
        writer=QtGui.QPdfWriter(str(path));painter=QtGui.QPainter(writer)
        painter.drawText(100,200,'First page');writer.newPage();painter.drawText(100,200,'Second page');painter.end()
        doc=load_document(path)
        self.assertEqual(doc.pages,2);self.assertIn('First page',doc.text);self.assertIn('Second page',doc.text)
        self.window.show();self.window.document_ready(doc,None)
        self.wait(lambda:self.window.pdf_image is not None)
        width=self.window.pdf_label.width()
        self.window.zoom(1);self.assertGreater(self.window.pdf_label.width(),width)
        for _ in range(7):self.window.zoom(1)
        self.assertGreater(self.window.pdf.horizontalScrollBar().maximum(),0)
        self.window.change_page(1)
        self.wait(lambda:self.window.pdf_loaded_page==2)

    @unittest.skipUnless(RENDER,'Actual WebEngine rendering requires --render under Xvfb')
    def test_offline_mermaid_math_markdown_tables_and_code_copy(self):
        self.window.show()
        source='# A clearer view\n\nLocal documents, diagrams and mathematics.\n\n$$E = mc^2$$\n\n```mermaid\nflowchart LR\n Paths --> Folio --> Reading\n```\n\n| Item | Amount |\n| --- | ---: |\n| Tea | 0.1 |\n| Coffee | 0.2 |\n\n```python\nprint("hello")\n```'
        self.window.document_ready(Document(None,'markdown',source),None)
        end=time.monotonic()+30
        while time.monotonic()<end:
            if self.js("Boolean(document.querySelector('.mermaid svg') && document.querySelector('mjx-container') && document.documentElement.dataset.folioReady)"):break
            time.sleep(.1)
        else:self.fail('Offline rendering failed: '+str(self.js("({ready:document.documentElement.dataset.folioReady,mermaid:!!document.querySelector('.mermaid svg'),math:!!document.querySelector('mjx-container'),error:document.querySelector('.render-error')?.textContent,mathjax:typeof MathJax,channel:typeof QWebChannel,body:document.body.innerText.slice(0,800)})")))
        self.assertEqual(self.js("document.querySelectorAll('article table').length"),1)
        self.assertIn('Tea',self.js("document.querySelector('article table').innerText"))
        self.assertIn('Coffee',self.js("document.querySelector('article table').innerText"))
        self.js("document.querySelectorAll('article table tbody tr')[0].cells[2].click()")
        self.wait(lambda:True,.1)
        self.js("document.querySelectorAll('article table tbody tr')[1].cells[2].dispatchEvent(new MouseEvent('click',{ctrlKey:true}))")
        for _ in range(20):
            if 'Sum 0.3' in self.js("document.querySelector('.table-stats').textContent"):break
            self.application.processEvents();time.sleep(.05)
        self.assertIn('Sum 0.3',self.js("document.querySelector('.table-stats').textContent"))
        self.js("document.querySelector('pre button.copy-code').click()")
        from PyQt5 import QtWidgets
        self.wait(lambda:'print("hello")' in QtWidgets.QApplication.clipboard().text())
        artifact=os.environ.get('FOLIO_PREVIEW')
        self.window.transpose()
        self.wait(lambda:self.js("document.querySelector('article table').rows.length") == 2)
        self.assertEqual(self.js("document.querySelector('article table').rows[1].cells[2].textContent"),'0.1')
        self.window.transpose()
        self.wait(lambda:self.js("document.querySelector('article table').rows.length") == 3)

        if artifact:
            from PyQt5 import QtTest
            QtTest.QTest.qWait(500)  # Allow Chromium's compositor to finish painting.
            self.window.grab().save(artifact)


if __name__=='__main__':
    unittest.main()
