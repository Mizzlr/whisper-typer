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
.mermaid svg {max-width:100%;height:auto;cursor:zoom-in} article img {cursor:zoom-in} .render-error {color:#9a4435;font-size:13px}
button.copy-code {float:right;color:#527564;border:1px solid #cad5c8;border-radius:5px;background:#fbf8f1;padding:4px 8px;cursor:pointer;opacity:0}
pre:hover button.copy-code,.table-stats:hover button.copy-code {opacity:1}
td.folio-selected,th.folio-selected {background:#cfe0c7!important;outline:1px solid #95b08a}
table td,table th {cursor:cell;user-select:none} .row-grip {cursor:pointer;color:#89927f;font-size:11px;width:28px}
.table-controls {text-align:right;margin:8px 0} .table-controls button {font:12px "JetBrains Mono",monospace;color:#527564;border:1px solid #cad5c8;background:#fbf8f1;padding:4px 8px;cursor:pointer}
.table-stats {font-size:12px;color:#6b7b64;line-height:1.6;margin:8px 0 28px}
'''


def sanitize_mermaid_code(code):
    lines = []
    for line in code.splitlines():
        m = re.match(r'^(\s*subgraph\s+)(.+)$', line)
        if m:
            prefix, rest = m.group(1), m.group(2).strip()
            if re.search(r'\[.+\]$', rest) or re.fullmatch(r'[A-Za-z0-9_-]+', rest):
                lines.append(line)
                continue
            clean_title = rest.replace('"', "'")
            safe_id = 'sg_' + re.sub(r'[^A-Za-z0-9_]+', '_', rest).strip('_')[:32]
            lines.append(f'{prefix}{safe_id} ["{clean_title}"]')
            continue
        lines.append(line)
    return '\n'.join(lines)


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
    source = re.sub(r'(^|\n)(```|~~~)mermaid\s*\n([\s\S]*?)\n\2', lambda m: f"{m.group(1)}{m.group(2)}mermaid\n{sanitize_mermaid_code(m.group(3))}\n{m.group(2)}", source, flags=re.IGNORECASE)
    # Bare Mermaid input is accepted as well as fenced mermaid blocks.
    if re.match(r'^\s*(?:graph|flowchart|sequenceDiagram|classDiagram|stateDiagram|erDiagram|gantt|pie|mindmap|timeline|journey)\b', source):
        source = '```mermaid\n' + sanitize_mermaid_code(source) + '\n```'
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
 try {
  if (typeof mermaid !== 'undefined') {
   mermaid.initialize({startOnLoad:false,securityLevel:'strict',theme:'base',themeVariables:{primaryColor:'#e7eee2',primaryTextColor:'#304439',primaryBorderColor:'#9eb493',lineColor:'#789181',fontFamily:'JetBrains Mono'}});
  }
  let bridge = null;
  if (typeof qt !== 'undefined' && qt.webChannelTransport) {
   try {
    bridge = await new Promise(resolve=>new QWebChannel(qt.webChannelTransport,c=>resolve(c.objects.folio)));
   } catch(e) {}
  }
  if (typeof mermaid !== 'undefined') {
   for (const block of Array.from(document.querySelectorAll('.mermaid'))) {
    const raw = block.textContent;
    try {
     await mermaid.run({nodes:[block]});
     if (block.querySelector('.error-icon') || block.textContent.includes('Syntax error in text')) {
      throw new Error('Diagram syntax error');
     }
    } catch(err) {
     block.classList.remove('mermaid');
     block.classList.add('mermaid-error');
     block.innerHTML = '<div style="font-size:11px;color:#9a4435;margin-bottom:6px;font-family:JetBrains Mono,monospace;text-align:left">Diagram (syntax error: ' + (err.message || 'invalid diagram').replace(/</g,'&lt;') + ')</div><pre style="text-align:left;margin:0;padding:12px;background:inherit;border:0;font-size:12px;overflow:auto"><code>' + raw.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;') + '</code></pre>';
    }
   }
  }
  if (bridge) {
   for (const block of document.querySelectorAll('pre:not(.mermaid)')) {
    const b=document.createElement('button');b.className='copy-code';b.textContent='Copy';
    b.onclick=()=>{bridge.copy((block.querySelector('code')||block).textContent);b.textContent='Copied';setTimeout(()=>b.textContent='Copy',1000);};
    block.prepend(b);
   }
   document.addEventListener('click', (e) => {
    if (e.target.closest('button.copy-code, .table-controls, .row-grip, a')) return;
    const img = e.target.closest('img');
    if (img && img.src && typeof bridge.zoom_image === 'function') {
     e.preventDefault();
     bridge.zoom_image(img.src);
     return;
    }
    const mermaidBlock = e.target.closest('.mermaid');
    if (mermaidBlock && typeof bridge.zoom_image === 'function') {
     const svg = mermaidBlock.querySelector('svg');
     if (svg) {
      e.preventDefault();
      const svgData = new XMLSerializer().serializeToString(svg);
      const b64 = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(svgData)));
      bridge.zoom_image(b64);
      return;
     }
    }
   });
  }
  const originals=new WeakMap(),transposed=new WeakMap();
  function initializeTables(tables=document.querySelectorAll('article table')) {
  for (const table of tables) {
   if(table.previousElementSibling?.classList.contains('table-controls'))table.previousElementSibling.remove();
   const controls=document.createElement('div');controls.className='table-controls';
   const transpose=document.createElement('button');transpose.textContent='Transpose';transpose.onclick=()=>window.transposeTable(table);
   controls.append(transpose);table.before(controls);
   const rows=Array.from(table.rows);let anchor=null;let revision=0;
   const status=document.createElement('div');status.className='table-stats';
   const output=document.createElement('span');status.append(output);
   const copy=document.createElement('button');copy.className='copy-code';copy.textContent='Copy selection';status.prepend(copy);table.after(status);
   const cells=rows.map((row,r)=>Array.from(row.cells).map((cell,c)=>{cell.dataset.row=r;cell.dataset.col=c;return cell;}));
   const selected=new Set();
   const update=()=>{const version=++revision;for(const row of cells)for(const cell of row)cell.classList.toggle('folio-selected',selected.has(cell));if(bridge){bridge.stats(JSON.stringify(Array.from(selected).map(c=>c.textContent)),text=>{if(version===revision){output.textContent=text;}});}};
   copy.onclick=()=>{const out=[];for(const row of cells){const chosen=row.filter(c=>selected.has(c));if(chosen.length)out.push(chosen.map(c=>c.textContent).join('\\t'));}if(bridge){bridge.copy(out.join('\\n'));}};
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
  window.transposeTable=table=>{
    if(!originals.has(table))originals.set(table,Array.from(table.rows).map(row=>Array.from(row.cells).filter(cell=>!cell.classList.contains('row-grip')).map(cell=>cell.textContent)));
    const original=originals.get(table),next=!transposed.get(table);transposed.set(table,next);
    const width=Math.max(0,...original.map(row=>row.length));
    const values=next?Array.from({length:width},(_,c)=>original.map(row=>row[c]||'')):original;
    if(table.nextElementSibling?.classList.contains('table-stats'))table.nextElementSibling.remove();
    table.innerHTML='';
    values.forEach((row,r)=>{const tr=table.insertRow();row.forEach(value=>{const cell=document.createElement(r===0?'th':'td');cell.textContent=value;tr.appendChild(cell);});});
   initializeTables([table]);
  };
  window.folioTranspose=()=>{for(const table of document.querySelectorAll('article table'))window.transposeTable(table);};
 } catch(err) {
  console.warn('Folio DOM init:', err);
 } finally {
  document.documentElement.dataset.folioReady='true';
 }
});
</script></body></html>'''

    return theme_colors(output,dark).replace('FOLIO_DOCUMENT_BODY',rendered_body(source),1)


def theme_colors(text,dark):
    from tk_widgets import LIGHT, DARK
    p=DARK if dark else LIGHT
    groups={
        'bg':['#fbf8f1'], 'panel':['#fffdf8','#eeeae0'],
        'fg':['#353b33','#302f2b','#263e35','#263d2b','#304439'],
        'green':['#356b58','#315443','#315543','#526b56','#3f5143','#435240','#527564','#6b7b64','#6c8d64'],
        'muted':['#858879','#8e9383','#949b8b','#687469','#89927f','#aaa99e'],
        'alt':['#f4f1e8','#f0ede4','#f2f0e7','#f0eee5','#f4f1e9','#eeede3','#f5f2e9'],
        'line':['#dedbce','#e1ddcf','#dddccd','#dedbcd','#e5e2d8','#e5e0d4','#dfddcf','#ddd8ca','#dedacc','#c4d5c8','#93b19e','#cad5c8','#95b08a','#a8b8a1','#b0c1a8','#9eb493','#789181'],
        'button':['#e6ede2','#e1ebdc','#e9eee2','#e8eee4','#e7eee2'],
        'selected':['#dfe8d9','#bed5af','#cfe0c7'], 'selected_fg':['#17271a','#2b4d37'],
    }
    colors={color:p[key] for key,values in groups.items() for color in values}
    result=re.sub(r'#[0-9a-fA-F]{6}',lambda m:colors.get(m[0].lower(),m[0]),text)
    return result.replace('color-scheme:light','color-scheme:dark') if dark else result
