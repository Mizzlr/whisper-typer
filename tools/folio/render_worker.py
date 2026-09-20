"""Isolated offscreen typesetting of math/diagrams for the native Tk reader."""
import base64
import json
import sys
import tempfile
from pathlib import Path
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWebEngineCore import QWebEngineUrlRequestInterceptor
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWebChannel import QWebChannel
from rendering import document_html


class LocalOnly(QWebEngineUrlRequestInterceptor):
    def interceptRequest(self,info):
        if info.requestUrl().scheme() not in ('file','data','qrc','about'):info.block(True)


class Bridge(QtCore.QObject):
    @QtCore.pyqtSlot(str,result=str)
    def stats(self,value):return ''
    @QtCore.pyqtSlot(str)
    def copy(self,value):pass


def main():
    source=sys.stdin.read()
    output_path = None
    if '--output' in sys.argv:
        idx = sys.argv.index('--output')
        if idx + 1 < len(sys.argv):
            output_path = Path(sys.argv[idx + 1])
    app=QtWidgets.QApplication(['folio-typesetter'])
    view=QWebEngineView();view.resize(1400,1800);view.show()
    channel=QWebChannel(view.page());bridge=Bridge(channel)
    channel.registerObject('folio',bridge);view.page().setWebChannel(channel)
    interceptor=LocalOnly(view);view.page().profile().setUrlRequestInterceptor(interceptor)
    with tempfile.TemporaryDirectory(prefix='folio-math-') as folder:
        target=Path(folder)/'view.html'
        target.write_text(document_html(source,dark='--dark' in sys.argv).replace("themeVariables:{","flowchart:{htmlLabels:false},themeVariables:{"))
        view.load(QtCore.QUrl.fromLocalFile(str(target)))
        timer=QtCore.QTimer();timer.setInterval(50)
        pending=[False]
        def result(data):
            pending[0]=False
            if not data:return
            timer.stop()
            state={'body':data['body']}
            def capture(index):
                if index==len(data['items']):
                    payload = json.dumps(state['body'])
                    if output_path:
                        output_path.write_text(payload, encoding='utf-8')
                    else:
                        print('FOLIO_JSON_START' + payload + 'FOLIO_JSON_END')
                    app.quit();return
                item=data['items'][index]
                def positioned(r):
                    def painted():
                        size=QtCore.QSize(max(1,round(r['w'])),max(1,round(r['h'])))
                        image=view.grab(QtCore.QRect(round(r['x']),round(r['y']),size.width(),size.height()))
                        output=QtCore.QBuffer();output.open(QtCore.QIODevice.WriteOnly);image.save(output,'PNG')
                        png=base64.b64encode(bytes(output.data())).decode()
                        state['body']=state['body'].replace(item['outer'],f'<img src="data:image/png;base64,{png}" width="{size.width()}" height="{size.height()}" alt="Typeset diagram or math">',1)
                        capture(index+1)
                    QtCore.QTimer.singleShot(120,painted)
                view.page().runJavaScript("(()=>{const e=document.querySelectorAll('.mermaid,mjx-container')["+str(index)+"];const s=e.querySelector('svg');s.style.maxHeight='1600px';e.scrollIntoView({block:'center'});const r=s.getBoundingClientRect();return {x:r.x,y:r.y,w:r.width,h:r.height};})()",positioned)
            capture(0)
        def check():
            if pending[0]:return
            pending[0]=True
            view.page().runJavaScript("document.documentElement.dataset.folioReady==='true' ? ({body:document.querySelector('article').innerHTML,items:Array.from(document.querySelectorAll('.mermaid,mjx-container')).map(e=>({outer:e.outerHTML,svg:e.querySelector('svg')?.outerHTML})).filter(e=>e.svg)}) : null",result)
        timer.timeout.connect(check);timer.start()
        QtCore.QTimer.singleShot(25000,lambda:app.exit(2))
        status=app.exec_()
        view.close();view.deleteLater();QtCore.QCoreApplication.sendPostedEvents(None,QtCore.QEvent.DeferredDelete)
        return status


if __name__=='__main__':sys.exit(main())
