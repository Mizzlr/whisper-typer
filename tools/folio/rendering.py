"""Offline Markdown, Mermaid and mathematical typesetting."""
import html
import re
from pathlib import Path

import bleach
import markdown


VENDOR = Path.home() / '.local/share/folio/vendor'
CSS = '''
:root {color-scheme:light} * {box-sizing:border-box}
body {margin:0;background:#fbf8f1;color:#302f2b;font:15px/1.75 "JetBrains Mono","DejaVu Sans Mono",monospace}
article {max-width:1000px;margin:auto;padding:38px 48px 90px}
h1,h2,h3 {line-height:1.3;letter-spacing:-.025em;color:#263e35}
h1 {font-size:32px;padding-bottom:16px;border-bottom:1px solid #ddd8ca}
h2 {margin-top:2em} a {color:#356b58;text-decoration:none;border-bottom:1px solid #c4d5c8}
pre,code {font-family:"JetBrains Mono","DejaVu Sans Mono",monospace}
pre {font-size:13px;background:#f0ede4;padding:20px;border:1px solid #e1ddcf;border-radius:10px;overflow:auto;line-height:1.65}
code {font-size:.85em;background:#f0ede4;padding:2px 5px;border-radius:4px} pre code {padding:0}
blockquote {border-left:3px solid #93b19e;margin-left:0;padding:4px 22px;color:#687469}
table {border-collapse:collapse;width:100%;font-size:14px}
th {background:#e8eee4;text-align:left} td,th {padding:10px 16px;border-bottom:1px solid #dfddcf}
tr:nth-child(even) {background:#f4f1e9} img {max-width:100%;height:auto}
hr {border:0;border-top:1px solid #dedacc;margin:32px 0}
.mermaid {background:#fffdf8;border:1px solid #e1ddcf;border-radius:12px;padding:24px;overflow:auto;text-align:center}
.mermaid svg {max-width:100%;height:auto} .render-error {color:#9a4435;font-size:13px}
button.copy-code {float:right;color:#527564;border:1px solid #cad5c8;border-radius:5px;background:#fbf8f1;padding:4px 8px;cursor:pointer;opacity:0}
pre:hover button.copy-code,.table-stats:hover button.copy-code {opacity:1}
td.folio-selected,th.folio-selected {background:#cfe0c7!important;outline:1px solid #95b08a}
table td,table th {cursor:cell;user-select:none} .row-grip {cursor:pointer;color:#89927f;font-size:11px;width:28px}
.table-stats {font-size:12px;color:#6b7b64;line-height:1.6;margin:8px 0 28px}
'''


def rendered_body(source):
    # Store math first, so Markdown cannot consume TeX backslashes/underscores.
    math = []
    code = []
    def protect_code(match):
        code.append(match[0])
        return f'FOLIOCODETOKEN{len(code)-1}END'
    source = re.sub(r'```[^\n]*\n[\s\S]*?```|~~~[^\n]*\n[\s\S]*?~~~|`[^`\n]+`', protect_code, source)
    def protect_math(match):
        math.append(match[0])
        return f'FOLIOMATHTOKEN{len(math)-1}END'
    source = re.sub(r'\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|(?<![\\$])\$(?!\$)(?:[^$\n]|\\\$)+?\$', protect_math, source)
    for i, value in enumerate(code):
        source = source.replace(f'FOLIOCODETOKEN{i}END', value)
    # Bare Mermaid input is accepted as well as fenced mermaid blocks.
    if re.match(r'^\s*(?:graph|flowchart|sequenceDiagram|classDiagram|stateDiagram|erDiagram|gantt|pie|mindmap|timeline|journey)\b', source):
        source = '```mermaid\n' + source + '\n```'
    body = markdown.markdown(source, extensions=['fenced_code', 'tables', 'sane_lists', 'toc'])
    tags = set(bleach.sanitizer.ALLOWED_TAGS) | {'p','pre','div','span','h1','h2','h3','h4','h5','h6','hr','br','table','thead','tbody','tr','th','td','img','del','sup','sub'}
    body = bleach.clean(body, tags=tags,
                        attributes={'*':['class','id'], 'a':['href','title'], 'img':['src','alt','title'], 'th':['align'], 'td':['align']},
                        protocols=['http','https','file','mailto'], strip=True)
    body = re.sub(r'<pre><code class="language-mermaid">([\s\S]*?)</code></pre>', r'<pre class="mermaid">\1</pre>', body)
    for i, value in enumerate(math):
        body = body.replace(f'FOLIOMATHTOKEN{i}END', html.escape(value))
    return body


def document_html(source, dark=False):
    mermaid = (VENDOR/'mermaid/dist/mermaid.min.js').as_uri()
    mathjax = (VENDOR/'mathjax/es5/tex-svg.js').as_uri()
    output = '''<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src file: qrc: 'unsafe-inline' 'unsafe-eval'; style-src 'unsafe-inline'; img-src file: data:; font-src file: data:; connect-src 'none'">
<style>''' + CSS + '''</style>
<script>
// Qt 5 ships an older Chromium. Mermaid's graph/config clones need collections
// and circular references, so a JSON round-trip is not a sufficient fallback.
if(!window.structuredClone)window.structuredClone=function clone(value,seen=new Map()){
 if(value===null||typeof value!=='object')return value;
 if(seen.has(value))return seen.get(value);
 if(value instanceof Date)return new Date(value);
 if(value instanceof RegExp)return new RegExp(value);
 if(value instanceof ArrayBuffer)return value.slice(0);
 if(ArrayBuffer.isView(value))return new value.constructor(value);
 const out=value instanceof Map?new Map():value instanceof Set?new Set():Array.isArray(value)?[]:{};seen.set(value,out);
 if(value instanceof Map){for(const [k,v]of value)out.set(clone(k,seen),clone(v,seen));}
 else if(value instanceof Set){for(const v of value)out.add(clone(v,seen));}
 else for(const k of Object.keys(value))out[k]=clone(value[k],seen);
 return out;
};
if(!Object.hasOwn)Object.hasOwn=(obj,key)=>Object.prototype.hasOwnProperty.call(obj,key);
if(!Array.prototype.at)Array.prototype.at=function(i){i=Math.trunc(i)||0;return this[i<0?this.length+i:i];};
</script>
<script>window.MathJax={tex:{inlineMath:[['$','$'],['\\\\(','\\\\)']],displayMath:[['$$','$$'],['\\\\[','\\\\]']]},options:{enableMenu:false},svg:{fontCache:'local'}};</script>
<script defer src="''' + mathjax + '''"></script>
<script defer src="''' + mermaid + '''"></script></head><body><article>''' + 'FOLIO_DOCUMENT_BODY' + '''</article>
<script src="qrc:///qtwebchannel/qwebchannel.js"></script>
<script>
document.addEventListener('DOMContentLoaded',async()=>{
 mermaid.initialize({startOnLoad:false,securityLevel:'strict',theme:'base',themeVariables:{primaryColor:'#e7eee2',primaryTextColor:'#304439',primaryBorderColor:'#9eb493',lineColor:'#789181',fontFamily:'JetBrains Mono'}});
 const bridge=await new Promise(resolve=>new QWebChannel(qt.webChannelTransport,c=>resolve(c.objects.folio)));
 try {
 await mermaid.run({querySelector:'.mermaid'});
 } catch(e) {let p=document.createElement('p');p.className='render-error';p.textContent='Diagram: '+e.message;document.querySelector('article').appendChild(p);}
 for (const block of document.querySelectorAll('pre:not(.mermaid)')) {
  const b=document.createElement('button');b.className='copy-code';b.textContent='Copy';
  b.onclick=()=>{bridge.copy((block.querySelector('code')||block).textContent);b.textContent='Copied';setTimeout(()=>b.textContent='Copy',1000);};
  block.prepend(b);
 }
 const originals=new WeakMap(),transposed=new WeakMap();
 function initializeTables() {
 for (const table of document.querySelectorAll('article table')) {
  const rows=Array.from(table.rows);let anchor=null;let revision=0;
  const status=document.createElement('div');status.className='table-stats';
  const output=document.createElement('span');status.append(output);
  const copy=document.createElement('button');copy.className='copy-code';copy.textContent='Copy selection';status.prepend(copy);table.after(status);
  const cells=rows.map((row,r)=>Array.from(row.cells).map((cell,c)=>{cell.dataset.row=r;cell.dataset.col=c;return cell;}));
  const selected=new Set();
  const update=()=>{const version=++revision;for(const row of cells)for(const cell of row)cell.classList.toggle('folio-selected',selected.has(cell));bridge.stats(JSON.stringify(Array.from(selected).map(c=>c.textContent)),text=>{if(version===revision){output.textContent=text;}});};
  copy.onclick=()=>{const out=[];for(const row of cells){const chosen=row.filter(c=>selected.has(c));if(chosen.length)out.push(chosen.map(c=>c.textContent).join('\\t'));}bridge.copy(out.join('\\n'));};
  for (let r=0;r<rows.length;r++) {
   const grip=document.createElement(r===0?'th':'td');grip.className='row-grip';grip.textContent=r===0?'#':r;
   rows[r].prepend(grip);
   grip.onclick=e=>{if(!e.ctrlKey&&!e.metaKey)selected.clear();for(const cell of cells[r])selected.add(cell);update();};
   for(const cell of cells[r])cell.onclick=e=>{
    const col=Number(cell.dataset.col);
    if(!e.ctrlKey&&!e.metaKey)selected.clear();
    if(cell.tagName==='TH'){for(const row of cells)if(row[col])selected.add(row[col]);}
    else if(e.shiftKey&&anchor){const [ar,ac]=anchor;for(let rr=Math.min(ar,r);rr<=Math.max(ar,r);rr++)for(let cc=Math.min(ac,col);cc<=Math.max(ac,col);cc++)if(cells[rr][cc])selected.add(cells[rr][cc]);}
    else {if(selected.has(cell))selected.delete(cell);else selected.add(cell);anchor=[r,col];}
    update();
   };
  }
 }
 }
 initializeTables();
 window.folioTranspose=()=>{
  for(const table of document.querySelectorAll('article table')){
   if(!originals.has(table))originals.set(table,Array.from(table.rows).map(row=>Array.from(row.cells).filter(cell=>!cell.classList.contains('row-grip')).map(cell=>cell.textContent)));
   const original=originals.get(table),next=!transposed.get(table);transposed.set(table,next);
   const width=Math.max(0,...original.map(row=>row.length));
   const values=next?Array.from({length:width},(_,c)=>original.map(row=>row[c]||'')):original;
   if(table.nextElementSibling?.classList.contains('table-stats'))table.nextElementSibling.remove();
   table.innerHTML='';
   values.forEach((row,r)=>{const tr=table.insertRow();row.forEach(value=>{const cell=document.createElement(r===0?'th':'td');cell.textContent=value;tr.appendChild(cell);});});
  }
  initializeTables();
 };
 document.documentElement.dataset.folioReady='true';
});
</script></body></html>'''

    return theme_colors(output,dark).replace('FOLIO_DOCUMENT_BODY',rendered_body(source),1)


def theme_colors(text,dark):
    if not dark:return text
    colors={'#fbf8f1':'#181d23','#fffdf8':'#20262e','#353b33':'#e2e7eb','#302f2b':'#e2e7eb','#263e35':'#c1e4cb',
            '#356b58':'#aadbbd','#315443':'#aadbbd','#315543':'#aadbbd','#526b56':'#aadbbd','#263d2b':'#dbf0df',
            '#858879':'#91a0b0','#8e9383':'#91a0b0','#949b8b':'#91a0b0','#f4f1e8':'#252d36','#f0ede4':'#252d36',
            '#f2f0e7':'#252d36','#f0eee5':'#252d36','#f4f1e9':'#252d36','#eeede3':'#252d36','#eeeae0':'#20262e',
            '#f5f2e9':'#252d36','#dedbce':'#343e49','#e1ddcf':'#343e49','#dddccd':'#343e49','#dedbcd':'#343e49',
            '#e5e2d8':'#343e49','#e5e0d4':'#343e49','#687469':'#a8b6aa','#dfddcf':'#343e49',
            '#ddd8ca':'#343e49','#e6ede2':'#2d4437','#e1ebdc':'#2d4437','#e9eee2':'#2d4437','#dfe8d9':'#375645',
            '#bed5af':'#375645','#17271a':'#f0fff4','#2b4d37':'#f0fff4','#3f5143':'#aadbbd','#435240':'#aadbbd',
            '#e8eee4':'#2d4437','#cfe0c7':'#375645','#304439':'#e2e7eb','#e7eee2':'#2d4437'}
    import re
    return re.sub(r'#[0-9a-fA-F]{6}',lambda m:colors.get(m[0].lower(),m[0]),text).replace('color-scheme:light','color-scheme:dark')
