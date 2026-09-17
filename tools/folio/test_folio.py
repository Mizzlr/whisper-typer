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
        from app import Folio
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

    def test_image_paste_keeps_original_and_extracted_text_zoom_and_escape(self):
        from PyQt5 import QtCore,QtGui,QtWidgets
        from unittest.mock import patch
        self.window.show()
        file=self.path/'report.txt';file.write_text('Report content.')
        image=QtGui.QImage(800,400,QtGui.QImage.Format_RGB32);image.fill(QtGui.QColor('#fbf8f1'))
        mime=QtCore.QMimeData();mime.setImageData(image)
        with patch('app.image_text',return_value=str(file)):
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
        if artifact:
            from PyQt5 import QtTest
            QtTest.QTest.qWait(500)  # Allow Chromium's compositor to finish painting.
            self.window.grab().save(artifact)


if __name__=='__main__':
    unittest.main()
