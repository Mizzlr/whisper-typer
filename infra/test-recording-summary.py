#!/usr/bin/python3
"""Check transcript preservation and complete coverage of long meetings."""
import json
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import threading
import unittest
from recording_summary import portions,summarize,split_summary,generate_title,transcript_export


class SummaryTests(unittest.TestCase):
    def test_title_request_is_bounded_and_independent_of_summary(self):
        requests=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                body=json.dumps({'message':{'content':'Title: Release Review Planning'}}).encode()
                self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        try:
            self.assertEqual(generate_title('Start '+('Meeting content. '*1000)+' End',f'http://127.0.0.1:{server.server_port}','fixture'),'Release Review Planning')
            self.assertEqual(len(requests),1)
            source=requests[0]['messages'][1]['content'];self.assertLessEqual(len(source),6003)
            self.assertTrue(source.startswith('Start '));self.assertTrue(source.endswith(' End'))
            self.assertEqual(requests[0]['options']['num_predict'],40)
        finally:server.shutdown();server.server_close()

    def test_topic_heading_is_separate_bounded_and_legacy_summary_is_preserved(self):
        self.assertEqual(split_summary('Title: Release Review Planning\n\nAction: review tomorrow.'),
                         ('Release Review Planning','Action: review tomorrow.'))
        self.assertEqual(split_summary('**Title: Weekly Release Review And Planning Notes**\n\nDecision: approved.'),
                         ('Weekly Release Review And Planning','Decision: approved.'))
        self.assertEqual(split_summary('Decision: review the release.'),('','Decision: review the release.'))

    def test_portions_cover_every_word_with_bounded_requests(self):
        text='\n\n'.join(f'Sentence {n} describes one meeting decision.' for n in range(2000))
        chunks=list(portions(text))
        self.assertGreater(len(chunks),1)
        self.assertTrue(all(len(chunk)<=12000 for chunk in chunks))
        self.assertEqual(' '.join(' '.join(chunks).split()),' '.join(text.split()))

    def test_export_preserves_words_and_timestamps_without_using_summary(self):
        row={'segments':[{'timestamp':'2026-09-17T10:00:00+05:30','text':'Original  words.\nNext sentence.'}],
             'summary':'Different text.'}
        output=transcript_export(row)
        self.assertIn(row['segments'][0]['text'],output)
        self.assertIn(row['segments'][0]['timestamp'],output)
        self.assertNotIn(row['summary'],output)

    def test_long_meeting_maps_all_parts_and_merges_on_local_model(self):
        requests=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                body=json.dumps({'message':{'content':'Decision: review the release.'}}).encode()
                self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        try:
            text=('Discuss the release and review the tests. '*800)+'Final action: review the release.'
            parts=list(portions(text));result=summarize(text,f'http://127.0.0.1:{server.server_port}','fixture')
            self.assertEqual(result,'Decision: review the release.')
            self.assertEqual([request['messages'][1]['content'] for request in requests[:len(parts)]],parts)
            self.assertEqual(len(requests),len(parts)+1)
            self.assertTrue(all(request['keep_alive']==-1 and not request['stream'] for request in requests))
            self.assertTrue(all('3 to 5 words' in request['messages'][0]['content'] for request in requests))
        finally:server.shutdown();server.server_close()


if __name__=='__main__':unittest.main()
