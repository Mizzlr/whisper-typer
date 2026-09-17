"""Behavioral checks. Use --render under Xvfb for actual offline JS rendering."""
import os
import json
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

    def test_dark_theme_preserves_literal_document_colors(self):
        from rendering import document_html
        result=document_html('`#fbf8f1` is a literal color.',dark=True)
        self.assertIn('<code>#fbf8f1</code>',result)
        self.assertIn('background:#181d23',result)

    def test_markdown_math_mermaid_and_sanitization(self):
        body=rendered_body('# Heading\n\n$x_i = \\frac{a}{b}$\n\n```mermaid\nflowchart TD\n A --> B\n```\n\n<script>alert(1)</script>\n<img src="x" onerror="alert(1)">')
        self.assertIn('class="mermaid"',body)
        self.assertIn('\\frac{a}{b}',body)
        self.assertNotIn('<script',body);self.assertNotIn('onerror=',body)
        self.assertIn('class="mermaid"',rendered_body('flowchart LR\n A-->B'))
        self.assertIn('$x_i$',rendered_body('`$x_i$`'))


class TkTests(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        from app import Folio
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)
        self.root=tk.Tk(className='Folio');self.window=Folio(self.root,self.path/'history.sqlite3')
        self.root.update()

    def tearDown(self):
        self.window.close();self.temp.cleanup()

    def wait(self,predicate,seconds=35):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            self.root.update()
            if predicate():return
            time.sleep(.02)
        self.fail('Background operation timed out')

    def idle(self):
        self.wait(lambda:not self.window.busy and not self.window.paste_timer)

    def test_paste_history_switcher_back_and_settings(self):
        a=self.path/'note.txt';a.write_text('# A heading\n\nA short note.')
        b=self.path/'data.csv';b.write_text('name,value\nA,4\n')
        self.window.paste(text=f'{a}\n{b}');self.idle()
        self.assertEqual(self.window.batches[0]['paths'],[str(a),str(b)])
        self.window.open_group_path(a,self.window.batches[0]);self.idle()
        self.assertEqual(self.window.current.kind,'markdown')
        self.assertIn('A heading',self.window.html.document.body.textContent)
        self.assertEqual(self.window.tab_paths,[a,b])
        self.assertTrue(self.window.back_button.winfo_ismapped())
        self.window.load_path(b);self.idle();self.assertEqual(self.window.table.rows[1],['A','4'])
        self.window.go_back();self.assertEqual(self.window.view,'list')
        self.window.toggle_top();self.window.toggle_theme()
        settings=json.loads(self.window.settings_path.read_text())
        self.assertFalse(settings['top']);self.assertTrue(settings['dark'])
        self.window.escape();self.assertEqual(self.window.path_input.get('1.0','end-1c'),'')
        self.assertEqual(len(self.window.batches),1)

    def test_table_selection_transpose_and_shift_scroll(self):
        from types import SimpleNamespace
        rows=[['name']+[f'c{i}' for i in range(12)],['A']+[str(i) for i in range(12)],['B']+[str(i+1) for i in range(12)]]
        self.window.select_document(Document(self.path/'values.csv','csv','original',rows));self.root.update()
        table=self.window.table
        table.selected={(1,1),(1,2)};table.update_selection()
        self.idle()
        self.assertIn('Sum 1',table.summary['text']);table.copy()
        self.assertEqual(self.root.clipboard_get(),'0\t1\n')
        before=table.canvas.xview()[0];table.canvas.event_generate('<Button-5>',state=1);self.root.update()
        self.assertGreater(table.canvas.xview()[0],before)
        self.window.transpose();self.assertEqual(table.rows[1],['c0','0','1'])
        self.window.transpose();self.assertEqual(table.rows,rows)
        self.assertEqual(self.window.current.text,'original')

    def test_markdown_table_selection_stats_and_transpose(self):
        self.window.select_document(Document(None,'markdown','| Item | Value |\n| --- | --- |\n| Tea | 0.1 |\n| Coffee | 0.2 |'))
        self.idle();self.root.update()
        self.assertEqual(len(self.window.embedded_tables),1)
        table=self.window.embedded_tables[0];table.selected={(1,1),(2,1)};table.update_selection()
        self.idle()
        self.assertIn('Sum 0.3',table.summary['text'])
        self.window.transpose();self.assertEqual(table.rows[1],['Value','0.1','0.2'])
        self.window.toggle_theme();self.idle()
        self.assertTrue(self.window.dark)

    def test_source_syntax_wrapping_and_original_copy(self):
        source='import os\n# A comment\nvalue = 42\n'+('Long prose should wrap. '*50)
        self.window.select_document(Document(self.path/'example.py','text',source));self.idle()
        self.assertEqual(self.window.source['wrap'],'word')
        self.assertEqual(self.window.source.get('1.0','end-1c'),source)
        self.assertTrue(any(name.startswith('syntax') and self.window.source.tag_ranges(name) for name in self.window.source.tag_names()))
        self.window.copy_content();self.assertEqual(self.root.clipboard_get(),source)

    def test_pdf_opens_firefox_without_pdf_rasterization(self):
        from unittest.mock import patch
        pdf=self.path/'sample.pdf';pdf.write_bytes(b'%PDF synthetic browser fixture')
        with patch('app.subprocess.Popen') as launch:
            self.window.load_path(pdf)
            command=launch.call_args[0][0]
            self.assertIn('firefox',command[0]);self.assertEqual(command[1],pdf.as_uri())
        self.assertIsNone(self.window.current)
        self.assertEqual(self.window.history.recent_files(),[pdf])

    def test_zip_and_folder_navigation(self):
        import zipfile
        archive=self.path/'example.zip'
        with zipfile.ZipFile(archive,'w') as out:
            out.writestr('reports/note.md','# From ZIP');out.writestr('data.csv','a,b\n1,2')
        self.window.load_path(archive);self.idle()
        self.assertEqual(self.window.context_label,'example.zip')
        folder=next(p for p in self.window.context_entries if p.is_dir())
        self.window.load_path(folder);self.idle()
        file=self.window.context_entries[0];self.window.load_path(file);self.idle()
        self.assertIn('From ZIP',self.window.html.document.body.textContent)
        self.window.go_back();self.assertEqual(self.window.context_entries,[file])
        self.window.go_back();self.assertEqual(self.window.context_label,'example.zip')

    def test_ad_hoc_tab_and_recent_downloads(self):
        repo=self.path/'repo';(repo/'adhoc'/'2000-01-01').mkdir(parents=True)
        self.window.resolver.repos=[repo];self.window.show_adhoc()
        self.assertEqual(self.window.context_entries,[repo/'adhoc'])
        self.window.load_path(repo/'adhoc');self.idle()
        self.assertEqual(self.window.context_entries,[repo/'adhoc'/'2000-01-01'])
        downloads=self.path/'Downloads';downloads.mkdir();file=downloads/'note.txt';file.write_text('Hello')
        self.window.show_downloads(downloads);self.idle();self.assertEqual(self.window.context_entries,[file])
        self.window.load_path(file);self.idle();self.window.show_recent();self.assertEqual(self.window.context_entries,[file])

    @unittest.skipUnless(RENDER,'Run --render for local math/diagram worker')
    def test_tk_math_and_mermaid_are_typeset(self):
        from app import typeset
        body=typeset('# Diagram\n\n$x_i = \\frac{a}{b}$\n\n```mermaid\nflowchart LR\n A --> B\n```')
        self.assertEqual(body.count('data:image/png;base64,'),2)
        import re
        sizes=re.findall(r'width="(\d+)" height="(\d+)"',body)
        self.assertLess(int(sizes[0][0]),200)
        self.assertLess(int(sizes[0][1]),100)
        self.window.select_document(Document(None,'markdown','placeholder'))
        self.idle();self.window.show_html(body);self.root.update()
        self.assertIn('Diagram',self.window.html.document.body.textContent)


class ArchiveTests(unittest.TestCase):
    def test_reject_traversal_symlinks_and_oversized_archives_before_writes(self):
        import zipfile,stat
        from files import unzip_contents
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name in ('../escaped.txt','/absolute.txt'):
                archive=root/'unsafe.zip'
                with zipfile.ZipFile(archive,'w') as out:out.writestr(name,'unsafe')
                with self.assertRaises(ValueError):unzip_contents(archive,root/'preview')
                self.assertFalse((root/'preview').exists())
            with zipfile.ZipFile(root/'link.zip','w') as out:
                info=zipfile.ZipInfo('link');info.external_attr=(stat.S_IFLNK|0o777)<<16;out.writestr(info,'../outside')
            with self.assertRaises(ValueError):unzip_contents(root/'link.zip',root/'preview')
            with zipfile.ZipFile(root/'large.zip','w') as out:
                for i in range(5001):out.writestr(str(i),'')
            with self.assertRaises(ValueError):unzip_contents(root/'large.zip',root/'preview')

    def test_embedded_resources_reject_network(self):
        from app import local_resource
        with self.assertRaises(ValueError):local_resource('https://example.com/image.png')
        with self.assertRaises(ValueError):local_resource('file://remote/private.png')


if __name__=='__main__':unittest.main()
