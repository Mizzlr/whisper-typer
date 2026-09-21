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

    def test_median_odd_even_and_exact_json_prettify(self):
        from files import pretty_json
        self.assertEqual(selection_summary(['9','label','1','5'])['median'],Decimal('5'))
        self.assertEqual(selection_summary(['0.1','0.2','20%'])['median'],Decimal('0.15'))
        self.assertNotIn('median',selection_summary(['text']))
        source='{"amount":1.12345678901234567890123456789,"name":"a, b", "empty":{},"items":[1,2],"amount":2}'
        result=pretty_json(source)
        self.assertIn('1.12345678901234567890123456789',result)
        self.assertEqual(result.count('"amount"'),2)
        self.assertIn('"name": "a, b"',result)
        self.assertIn('"empty": {}',result)
        with self.assertRaises(ValueError):pretty_json('{invalid}')
        with self.assertRaises(ValueError):pretty_json('{"value":NaN}')

    def test_dark_theme_preserves_literal_document_colors(self):
        from rendering import document_html
        result=document_html('`#fbf8f1` is a literal color.',dark=True)
        self.assertIn('<code>#fbf8f1</code>',result)
        self.assertIn('background:#000000',result)

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

    def test_copy_and_back_positions_stay_fixed_between_home_and_documents(self):
        self.root.update()
        buttons=[self.window.top_button,self.window.back_button,self.window.copy_button,self.window.theme_button]
        initial=[b.winfo_rootx() for b in buttons]
        self.assertEqual(self.window.back_button['state'],'disabled')
        file=self.path/'note.txt';file.write_text('Original file contents.')
        batch=self.window.history.add('original dump',[file]);self.window.batches.insert(0,batch)
        self.window.show_history();self.window.copy_button.invoke()
        self.assertEqual(self.root.clipboard_get(),str(file))
        self.window.open_group_path(file,batch);self.idle();self.root.update()
        self.window.copy_button.invoke();self.assertEqual(self.root.clipboard_get(),'Original file contents.')
        self.assertEqual([b.winfo_rootx() for b in buttons],initial)
        self.assertEqual(self.window.back_button['state'],'normal')
        self.window.select_document(Document(self.path/'data.csv','csv','a,b',[['a','b'],['1','2']]))
        self.root.update();self.assertEqual([b.winfo_rootx() for b in buttons],initial)
        self.window.escape();self.root.update()
        self.assertEqual([b.winfo_rootx() for b in buttons],initial)
        self.window.copy_button.invoke();self.assertEqual(self.root.clipboard_get(),str(file))

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
        self.assertIn('Sum 0.3',table.summary['text']);self.assertIn('Median 0.15',table.summary['text'])
        self.window.transpose();self.assertEqual(table.rows[1],['Value','0.1','0.2'])
        self.window.toggle_theme();self.idle()
        self.assertTrue(self.window.dark)

    def test_table_local_transpose_and_json_display_only_prettify(self):
        source='| Name | Value |\n| --- | --- |\n| A | 1 |\n| B | 2 |\n\n| Label | Amount |\n| --- | --- |\n| C | 3 |'
        self.window.select_document(Document(None,'markdown',source));self.idle()
        first,second=self.window.embedded_tables
        self.assertEqual(first.transpose_button.master,first.controls)
        original=[list(row) for row in second.rows]
        first.transpose_button.invoke();self.assertEqual(len(first.rows),2);self.assertEqual(second.rows,original)
        file=self.path/'example.json';raw='{"value":1.12345678901234567890123456789,"items":[1,2]}'
        file.write_text(raw);self.window.load_path(file);self.idle()
        self.assertTrue(self.window.prettify_button.winfo_ismapped())
        self.window.prettify_button.invoke();self.idle()
        self.assertIn('\n  "value": 1.12345678901234567890123456789',self.window.source.get('1.0','end-1c'))
        self.window.copy_button.invoke();self.assertEqual(self.root.clipboard_get(),raw)
        self.assertEqual(file.read_text(),raw)
        self.window.prettify_button.invoke();self.idle();self.assertEqual(self.window.source.get('1.0','end-1c'),raw)

    def test_source_syntax_wrapping_and_original_copy(self):
        source='import os\n# A comment\nvalue = 42\n'+('Long prose should wrap. '*50)
        self.window.select_document(Document(self.path/'example.py','text',source));self.idle()
        self.assertEqual(self.window.source['wrap'],'word')
        self.assertEqual(self.window.source.get('1.0','end-1c'),source)
        self.assertTrue(any(name.startswith('syntax') and self.window.source.tag_ranges(name) for name in self.window.source.tag_names()))
        self.window.copy_content();self.assertEqual(self.root.clipboard_get(),source)

    def test_long_line_and_large_source_skip_tokenization_preserve_copy_and_theme(self):
        from unittest.mock import patch
        for source in ('<div data-value="'+'x'*6000+'">text</div>', 'value = 1\n'*20000):
            with patch('pygments.lex',side_effect=AssertionError('Large source must not be tokenized')) as tokenize:
                self.window.select_document(Document(self.path/'large.html','text',source));self.idle()
                tokenize.assert_not_called()
            self.assertEqual(self.window.source.get('1.0','end-1c'),source)
            self.window.copy_content();self.assertEqual(self.root.clipboard_get(),source)
            self.assertIn('Raw text',self.window.status['text'])
        self.window.select_document(Document(self.path/'small.py','text','import os\nvalue = 42'));self.idle()
        self.assertEqual(self.window.source['bg'],'#ffffff')
        self.assertIn('syntaxa04900',self.window.source.tag_names('1.0'))
        self.window.toggle_theme();self.idle()
        self.assertEqual(self.window.source['bg'],'#000000')
        self.assertIn('syntaxe9ae7e',self.window.source.tag_names('1.0'))

    def test_image_ocr_empty_and_failure_preserve_original_and_folder_arrow(self):
        from unittest.mock import patch
        from PIL import Image
        image=Image.new('RGB',(40,30),'green')
        for outcome in ('',RuntimeError('synthetic OCR failure')):
            with patch('app.image_text',side_effect=outcome if isinstance(outcome,Exception) else None,return_value=''):
                self.window.paste(image=image);self.idle()
            batch=self.window.batches[0]
            self.assertTrue(batch['has_image']);self.assertTrue(self.window.history.image(batch['id']))
        with patch('app.image_text',return_value='reports/note.md'), patch.object(self.window.resolver,'resolve',side_effect=RuntimeError('synthetic resolution failure')):
            self.window.paste(image=image);self.idle()
        self.assertTrue(self.window.batches[0]['has_image'])
        folder=self.path/'folder';folder.mkdir()
        self.window.batches.insert(0,self.window.history.add(str(folder),[folder]))
        self.window.show_history()
        self.assertIn('▸ '+str(folder),self.window.list_text.get('1.0','end'))

    def test_real_clipboard_image_with_text_via_ctrl_v_and_virtual_paste(self):
        from unittest.mock import patch
        from PIL import Image
        png=self.path/'synthetic.png';Image.new('RGB',(30,20),'green').save(png)
        # A separate Qt owner offers both text and image on this isolated Xvfb.
        code="from PyQt5 import QtCore,QtGui,QtWidgets; import sys; app=QtWidgets.QApplication([]); data=QtCore.QMimeData(); data.setImageData(QtGui.QImage(sys.argv[1])); data.setText('alternate clipboard text'); app.clipboard().setMimeData(data); print('ready',flush=True); app.exec_()"
        owner=subprocess.Popen(['/usr/bin/python3','-c',code,str(png)],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True)
        try:
            self.assertEqual(owner.stdout.readline().strip(),'ready')
            from clipboard import read_clipboard
            text,image=read_clipboard();self.assertIsNone(text);self.assertEqual(image.size,(30,20))
            with patch('app.image_text',return_value=''):
                self.window.path_input.focus_force();self.root.update()
                self.window.path_input.event_generate('<Control-v>');self.idle()
                self.assertEqual(len(self.window.batches),1);self.assertTrue(self.window.batches[0]['has_image'])
                self.window.path_input.event_generate('<<Paste>>');self.idle()
                self.assertEqual(len(self.window.batches),2);self.assertTrue(self.window.batches[0]['has_image'])
        finally:
            owner.terminate();owner.wait(timeout=5);owner.stdout.close()

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

    def test_diagram_click_opens_image_zoom_and_escape(self):
        body = '<p>Doc with diagram</p><img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg==" width="200" height="200" alt="Test Arch Diagram">'
        self.window.select_document(Document(None, 'markdown', 'placeholder'))
        self.idle(); self.window.show_html(body); self.root.update()
        self.assertIn('0', self.window.zoom_images)
        src, alt = self.window.zoom_images['0']
        self.assertEqual(alt, 'Test Arch Diagram')
        # Simulate user clicking on the diagram
        self.window.open_link('folio-image:0')
        self.root.update()
        self.assertIsNotNone(self.window.zoom_view)
        # Test zoom, pan, fit
        self.window.zoom_view.zoom(1.25)
        self.window.zoom_view.pan('x', 2)
        self.window.zoom_view.fit()
        self.root.update()
        # Escape closes zoom view
        self.window.escape()
        self.root.update()
        self.assertIsNone(self.window.zoom_view)

    def test_table_dynamically_renders_to_full_width(self):
        rows = [
            ['Short A', 'Short B'],
            ['val 1', 'val 2']
        ]
        from tk_widgets import Table, LIGHT
        table = Table(self.root, rows, LIGHT, lambda t: None)
        table.pack(fill='both', expand=True)
        self.root.update_idletasks()
        self.root.update()
        # Verify columns expand to fill the entire available canvas width
        self.assertGreater(len(table.col_widths), 0)
        avail_w = max(100, table.canvas.winfo_width() - 42 - 18)
        self.assertEqual(sum(table.col_widths), avail_w)
        table.destroy()

    def test_font_size_controls_and_persistence(self):
        initial_font = self.window.font_size
        initial_scale = self.window.font_scale
        self.assertEqual(initial_font, 10)
        self.assertEqual(initial_scale, 1.0)
        self.window.font_up_button.invoke()
        self.root.update()
        self.assertEqual(self.window.font_size, 11)
        self.assertAlmostEqual(self.window.font_scale, 1.1)
        self.assertIn('11', str(self.window.source['font']))
        self.assertIn('11', str(self.window.list_text['font']))
        self.window.font_down_button.invoke()
        self.window.font_down_button.invoke()
        self.root.update()
        self.assertEqual(self.window.font_size, 9)
        self.assertAlmostEqual(self.window.font_scale, 0.9)
        self.window.zoom(0)
        self.root.update()
        self.assertEqual(self.window.font_size, 10)
        self.assertEqual(self.window.font_scale, 1.0)
        rows = [['col1', 'col2'], ['val1', 'val2']]
        self.window.select_document(Document(self.path/'test_font.csv', 'csv', 'col1,col2', rows))
        self.root.update()
        self.assertEqual(self.window.table.font_size, 10)
        self.window.font_up_button.invoke()
        self.root.update()
        self.assertEqual(self.window.table.font_size, 11)
        settings = json.loads(self.window.settings_path.read_text())
        self.assertEqual(settings['font_size'], 11)

    def test_adhoc_tab_rename_and_projects_discovery(self):
        self.assertIn('Adhoc', self.window.tab_buttons)
        self.assertNotIn('Ad hoc', self.window.tab_buttons)
        self.assertIn('Projects', self.window.tab_buttons)
        from unittest.mock import patch
        fake_home = self.path / 'fake_home'
        fake_home.mkdir()
        for p in ('whisper-typer', 'astralane-quant', 'personal-finance', 'grid-grinder', 'trailblazer', 'extra-app'):
            (fake_home / p).mkdir()
        (fake_home / 'extra-app' / 'Cargo.toml').write_text('[package]')
        (fake_home / 'non-project').mkdir()
        with patch('pathlib.Path.home', return_value=fake_home):
            self.window.show_projects()
            self.idle()
            expected = [
                fake_home / 'astralane-quant',
                fake_home / 'trailblazer',
                fake_home / 'whisper-typer',
                fake_home / 'grid-grinder',
                fake_home / 'personal-finance',
                fake_home / 'extra-app',
            ]
            self.assertEqual(self.window.context_entries, expected)
            self.assertEqual(self.window.context_label, 'Projects')

    def test_folder_dropdown_and_parent_navigation(self):
        self.assertEqual(self.window.folder_button['text'], '📁 Folders ▾')
        folder = self.path / 'parent_dir' / 'child_dir'
        folder.mkdir(parents=True)
        file_a = folder / 'file_a.txt'
        file_a.write_text('content')
        self.window.open_folder(folder)
        self.idle()
        self.assertEqual(self.window.folder_button['text'], '📁 child_dir ▾')
        list_content = self.window.list_text.get('1.0', 'end')
        self.assertIn('◂ .. (parent_dir)', list_content)
        self.assertIn('file_a.txt', list_content)
        self.window.load_path(folder.parent)
        self.idle()
        self.assertEqual(self.window.folder_button['text'], '📁 parent_dir ▾')
        doc = Document(file_a, 'text', 'content')
        self.window.select_document(doc)
        self.idle()
        self.assertEqual(self.window.folder_button['text'], '📁 child_dir ▾')

    def test_font_family_selection_and_persistence(self):
        self.window.set_font_family('Google Sans Mono')
        self.root.update()
        self.assertEqual(self.window.font_family, 'Google Sans Mono')
        self.assertIn('Google Sans Mono', str(self.window.source['font']))
        self.assertIn('Google Sans Mono', str(self.window.list_text['font']))
        settings = json.loads(self.window.settings_path.read_text())
        self.assertEqual(settings['font_family'], 'Google Sans Mono')





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
