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
    view=QWebEngineView();view.resize(3200,2400);view.show()
    channel=QWebChannel(view.page());bridge=Bridge(channel)
    channel.registerObject('folio',bridge);view.page().setWebChannel(channel)
    interceptor=LocalOnly(view);view.page().profile().setUrlRequestInterceptor(interceptor)
    with tempfile.TemporaryDirectory(prefix='folio-math-') as folder:
        target=Path(folder)/'view.html'
        target.write_text(document_html(source,dark='--dark' in sys.argv).replace("themeVariables:{","flowchart:{htmlLabels:false},themeVariables:{"))
        view.load(QtCore.QUrl.fromLocalFile(str(target)))
        timer = QtCore.QTimer()
        timer.setInterval(50)
        pending = [False]

        def finish(html):
            payload = json.dumps(html)
            if output_path:
                output_path.write_text(payload, encoding='utf-8')
            else:
                print('FOLIO_JSON_START' + payload + 'FOLIO_JSON_END')
            app.quit()

        def result(data):
            pending[0] = False
            if not data:
                return
            timer.stop()
            total = data.get('count', 0)

            def capture(index):
                if index >= total:
                    view.page().runJavaScript("document.querySelector('article').innerHTML", finish)
                    return

                def positioned(r):
                    if not r or r.get('w', 0) <= 1 or r.get('h', 0) <= 1:
                        capture(index + 1)
                        return

                    def painted():
                        size = QtCore.QSize(max(1, round(r['w'])), max(1, round(r['h'])))
                        image = view.grab(QtCore.QRect(round(r['x']), round(r['y']), size.width(), size.height()))
                        output = QtCore.QBuffer()
                        output.open(QtCore.QIODevice.WriteOnly)
                        image.save(output, 'PNG')
                        png = base64.b64encode(bytes(output.data())).decode()
                        js_replace = (
                            "(()=>{const e=document.querySelector('[data-folio-idx=\""
                            + str(index)
                            + "\"]');if(e){const img=document.createElement('img');img.src='data:image/png;base64,"
                            + png
                            + "';img.width="
                            + str(size.width())
                            + ";img.height="
                            + str(size.height())
                            + ";img.style.maxWidth='100%';img.style.height='auto';img.alt='Typeset diagram or math';e.replaceWith(img);}})()"
                        )
                        view.page().runJavaScript(js_replace, lambda _: capture(index + 1))

                    QtCore.QTimer.singleShot(120, painted)

                js_pos = (
                    "(()=>{const art=document.querySelector('article');if(art){art.style.maxWidth='none';art.style.width='3000px';}const e=document.querySelector('[data-folio-idx=\""
                    + str(index)
                    + "\"]');if(!e)return null;const s=e.querySelector('svg')||e;s.style.maxWidth='none';s.style.maxHeight='none';const vb=s.viewBox?s.viewBox.baseVal:null;if(vb&&vb.width>0){s.style.width=vb.width+'px';s.style.height=vb.height+'px';}s.scrollIntoView({block:'center'});const r=s.getBoundingClientRect();return {x:r.x,y:r.y,w:r.width,h:r.height};})()"
                )
                view.page().runJavaScript(js_pos, positioned)

            capture(0)

        def check():
            if pending[0]:
                return
            pending[0] = True
            js_init = """
            (() => {
                if (document.documentElement.dataset.folioReady !== 'true') return null;
                const candidates = Array.from(document.querySelectorAll('.mermaid,mjx-container'));
                let count = 0;
                candidates.forEach(e => {
                    const s = e.querySelector('svg');
                    if (s && !e.querySelector('.error-icon') && !e.textContent.includes('Syntax error in text')) {
                        e.setAttribute('data-folio-idx', String(count));
                        count++;
                    }
                });
                return {count: count};
            })()
            """
            view.page().runJavaScript(js_init, result)

        timer.timeout.connect(check)
        timer.start()
        QtCore.QTimer.singleShot(25000, lambda: app.exit(2))
        status = app.exec_()
        view.close()
        view.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        return status


if __name__=='__main__':sys.exit(main())
