"""Small native Tk widgets; table drawing is limited to visible cells."""
import csv
import io
import tkinter as tk
from tkinter import ttk
from cell_stats import selection_summary, display_summary

LIGHT = dict(bg='#ffffff',panel='#ffffff',fg='#111111',muted='#595959',line='#d0d0d0',green='#005a8e',selected='#cbdff5',selected_fg='#111111',button='#eeeeee',alt='#f6f6f6')
DARK = dict(bg='#000000',panel='#000000',fg='#f5f5f5',muted='#a8a8a8',line='#383838',green='#6cb8e6',selected='#193b5c',selected_fg='#ffffff',button='#202020',alt='#101010')


class Table(tk.Frame):
    def __init__(self,parent,rows,palette,on_copy,submit=None):
        super().__init__(parent,bg=palette['bg'])
        self.rows=rows
        self.original_rows=rows
        self.transposed=False
        self.columns=max(map(len,rows),default=0)
        self.palette=palette
        self.on_copy=on_copy
        self.submit=submit
        self.stats_generation=0
        self.selected=set()
        self.anchor=None
        self.col_widths=[]
        self.col_x=[42]
        self.total_width=42
        self.row_height=30
        self.canvas=tk.Canvas(self,bg=palette['panel'],highlightthickness=0)
        self.vertical=ttk.Scrollbar(self,orient='vertical',command=self.yview)
        self.horizontal=ttk.Scrollbar(self,orient='horizontal',command=self.xview)
        self.controls=tk.Frame(self,bg=palette['bg'])
        self.controls.grid(row=0,column=0,columnspan=2,sticky='ew',pady=(0,5))
        self.transpose_button=tk.Button(self.controls,text='Transpose',command=self.transpose,font=('JetBrains Mono',9),
                                       relief='solid',bd=1,padx=7,pady=3,bg=palette['bg'],fg=palette['green'],
                                       activebackground=palette['button'],activeforeground=palette['fg'],takefocus=False)
        self.transpose_button.pack(side='right')
        self.canvas.grid(row=1,column=0,sticky='nsew')
        self.vertical.grid(row=1,column=1,sticky='ns')
        self.horizontal.grid(row=2,column=0,sticky='ew')
        self.summary=tk.Label(self,bg=palette['bg'],fg=palette['green'],font=('JetBrains Mono',9),anchor='w')
        self.summary.grid(row=3,column=0,sticky='ew',pady=5)
        self.rowconfigure(1,weight=1);self.columnconfigure(0,weight=1)
        self.canvas.configure(xscrollcommand=lambda a,b:self.scrollbar(self.horizontal,a,b),yscrollcommand=lambda a,b:self.scrollbar(self.vertical,a,b))
        self.canvas.bind('<Configure>',lambda e:self.draw())
        self.canvas.bind('<Button-1>',self.click)
        self.canvas.bind('<B1-Motion>',self.drag)
        self.canvas.bind('<Control-c>',lambda e:self.copy())
        self.canvas.bind('<Button-4>',lambda e:self.wheel(-1,e))
        self.canvas.bind('<Button-5>',lambda e:self.wheel(1,e))
        self.canvas.bind('<MouseWheel>',lambda e:self.wheel(-1 if e.delta>0 else 1,e))
        self.canvas.bind('<Button-3>',self.menu)

    def scrollbar(self,bar,first,last):
        bar.set(first,last)
        if float(first)<=0 and float(last)>=1:bar.grid_remove()
        else:bar.grid()

    def compute_column_layout(self):
        if self.columns<=0:
            self.col_widths=[];self.col_x=[42];self.total_width=42;return
        c_width=self.canvas.winfo_width()
        avail_w=max(100,c_width-42-18) if c_width>100 else 190*self.columns
        sample_rows=self.rows[:100]
        natural_w=[]
        for col in range(self.columns):
            max_len=0
            for r in sample_rows:
                if col<len(r):
                    val=str(r[col]).replace('\n',' ')
                    if len(val)>max_len:max_len=len(val)
            header_len=len(self.column_label(col))
            char_len=max(max_len,header_len,4)
            needed=char_len*8+16
            natural_w.append(max(120,min(800,needed)))
        total_natural=sum(natural_w)
        if total_natural<=avail_w:
            extra=avail_w-total_natural
            widths=[round(w+extra*(w/total_natural)) for w in natural_w]
            diff=avail_w-sum(widths)
            if widths:widths[-1]+=diff
        else:
            widths=natural_w
        self.col_widths=widths
        self.col_x=[42]
        for w in widths:self.col_x.append(self.col_x[-1]+w)
        self.total_width=self.col_x[-1]

    def transpose(self):
        self.stats_generation+=1
        self.transposed=not self.transposed
        if self.transposed:
            columns=max(map(len,self.original_rows),default=0)
            self.rows=[[row[c] if c<len(row) else '' for row in self.original_rows] for c in range(columns)]
        else:self.rows=self.original_rows
        self.columns=max(map(len,self.rows),default=0)
        self.selected.clear();self.anchor=None;self.summary.configure(text='')
        self.canvas.xview_moveto(0);self.canvas.yview_moveto(0);self.draw()

    @staticmethod
    def column_label(number):
        result='';number+=1
        while number:
            number,digit=divmod(number-1,26);result=chr(65+digit)+result
        return result

    def yview(self,*args):
        self.canvas.yview(*args);self.draw()

    def xview(self,*args):
        self.canvas.xview(*args);self.draw()

    def wheel(self,direction,event):
        if event.state & 1:self.xview('scroll',direction,'units')
        else:self.yview('scroll',direction,'units')
        return 'break'

    def value(self,row,col):
        return self.rows[row][col] if col<len(self.rows[row]) else ''

    def draw(self):
        c=self.canvas;p=self.palette;c.delete('all')
        self.compute_column_layout()
        if not self.col_widths:return
        x0=c.canvasx(0);y0=c.canvasy(0)
        c_w=c.winfo_width();c_h=c.winfo_height()
        region_w=max(c_w,self.total_width)
        region_h=(len(self.rows)+1)*self.row_height
        c.configure(scrollregion=(0,0,region_w,region_h))
        first_row=max(0,int(y0//self.row_height)-1)
        last_row=min(len(self.rows),int((y0+c_h)//self.row_height)+1)
        for row in range(first_row,last_row):
            y=(row+1)*self.row_height
            for col in range(self.columns):
                x=self.col_x[col]
                w=self.col_widths[col]
                if x+w<x0 or x>x0+c_w:continue
                chosen=(row,col) in self.selected
                bg=p['selected'] if chosen else (p['button'] if row==0 else p['alt'] if row%2==0 else p['panel'])
                c.create_rectangle(x,y,x+w,y+self.row_height,fill=bg,outline=p['line'])
                text=str(self.value(row,col)).replace('\n',' ↵ ')
                max_chars=max(3,int((w-14)/7.6))
                if len(text)>max_chars:text=text[:max_chars-1]+'…'
                c.create_text(x+6,y+15,anchor='w',text=text,font=('JetBrains Mono',10),fill=p['selected_fg'] if chosen else p['fg'])
            c.create_rectangle(x0,y,x0+42,y+self.row_height,fill=p['alt'],outline=p['line'])
            c.create_text(x0+21,y+15,text=str(row+1),font=('JetBrains Mono',9),fill=p['muted'])
        for col in range(self.columns):
            x=self.col_x[col]
            w=self.col_widths[col]
            if x+w<x0 or x>x0+c_w:continue
            c.create_rectangle(x,y0,x+w,y0+self.row_height,fill=p['alt'],outline=p['line'])
            c.create_text(x+w/2,y0+15,text=self.column_label(col),font=('JetBrains Mono',10),fill=p['green'])
        c.create_rectangle(x0,y0,x0+42,y0+self.row_height,fill=p['alt'],outline=p['line'])

    def location(self,event):
        x=self.canvas.canvasx(event.x);y=self.canvas.canvasy(event.y)
        row=-1 if event.y<self.row_height else int(y//self.row_height)-1
        if event.x<42 or not self.col_widths:
            col=-1
        else:
            col=-1
            for c in range(len(self.col_widths)):
                if self.col_x[c]<=x<self.col_x[c+1]:
                    col=c;break
        return row,col

    def click(self,event):
        self.canvas.focus_set();row,col=self.location(event)
        if row>=len(self.rows) or col>=self.columns:return 'break'
        if not event.state & 4:self.selected.clear()
        if row==-1 or col==-1:self.anchor=None
        if row==-1 and col==-1:self.selected.update((r,c) for r in range(len(self.rows)) for c in range(self.columns))
        elif row==-1:self.selected.update((r,col) for r in range(len(self.rows)))
        elif col==-1:self.selected.update((row,c) for c in range(self.columns))
        elif event.state & 1 and self.anchor:self.rectangle(self.anchor,(row,col))
        else:
            cell=(row,col)
            if cell in self.selected:self.selected.remove(cell)
            else:self.selected.add(cell)
            self.anchor=cell
        self.update_selection();return 'break'

    def rectangle(self,a,b):
        for row in range(min(a[0],b[0]),max(a[0],b[0])+1):
            for col in range(min(a[1],b[1]),max(a[1],b[1])+1):self.selected.add((row,col))

    def drag(self,event):
        row,col=self.location(event)
        if self.anchor and 0<=row<len(self.rows) and 0<=col<self.columns:
            if not event.state & 4:self.selected.clear()
            self.rectangle(self.anchor,(row,col));self.update_selection()
        return 'break'

    def update_selection(self):
        self.stats_generation+=1;generation=self.stats_generation
        self.draw()
        if not self.selected:self.summary.configure(text='');return
        selected=list(self.selected);rows=self.rows
        def calculate():
            return display_summary(selection_summary(rows[r][c] if c<len(rows[r]) else '' for r,c in selected))
        def ready(value,error):
            if generation==self.stats_generation and self.winfo_exists():self.summary.configure(text=value if not error else 'Statistics unavailable')
        if self.submit:self.submit(calculate,ready)
        else:ready(calculate(),None)

    def copy(self):
        if self.selected:
            rs=[r for r,c in self.selected];cs=[c for r,c in self.selected]
            output=io.StringIO();writer=csv.writer(output,delimiter='\t',lineterminator='\n')
            for r in range(min(rs),max(rs)+1):writer.writerow([self.value(r,c) if (r,c) in self.selected else '' for c in range(min(cs),max(cs)+1)])
            self.on_copy(output.getvalue())
        return 'break'

    def menu(self,event):
        menu=tk.Menu(self,tearoff=False)
        menu.add_command(label='Copy selection',command=self.copy)
        menu.tk_popup(event.x_root,event.y_root)
