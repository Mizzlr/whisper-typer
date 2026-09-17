"""A bounded transcript pane with independent scrolling and tail following."""
from datetime import datetime
import tkinter as tk
from tkinter import font as tkfont


class RecordingView(tk.Frame):
    def __init__(self,parent,row,style,copy_summary):
        super().__init__(parent,bg=style.BG,height=180)
        self.style=style;self.segments=[];self.summary='';self.mode='transcript';self.follow_tail=True;self.layout_timer=None;self.maximum_height=180
        self.grid_propagate(False)
        self.columnconfigure(1,weight=1);self.rowconfigure(1,weight=1)
        self.tabs=tk.Frame(self,bg=style.BG)
        self.tabs.grid(row=0,column=1,columnspan=2,sticky='ew')
        options=dict(font=(style.FONT,9),bg=style.BG,fg=style.METADATA,
                     activebackground=style.PANEL,activeforeground=style.FG,relief='flat',takefocus=False)
        self.transcript_button=tk.Button(self.tabs,text='Transcript',command=lambda:self.show('transcript'),**options)
        self.transcript_button.pack(side='left')
        self.summary_button=tk.Button(self.tabs,text='Summary',command=lambda:self.show('summary'),state='disabled',**options)
        self.summary_button.pack(side='left')
        self.topic=tk.Label(self.tabs,bg=style.BG,fg=style.METADATA,font=(style.FONT,9),anchor='w',width=1)
        self.topic.pack(side='left',fill='x',expand=True,padx=6)
        self.copy_summary=tk.Button(self.tabs,text='Copy summary',command=lambda:copy_summary(self.summary),**options)
        self.note=tk.Label(self.tabs,bg=style.BG,fg=style.METADATA,font=(style.FONT,9))
        self.note.pack(side='left',padx=6)
        rail=tk.Canvas(self,width=10,bg=style.BG,highlightthickness=0,borderwidth=0)
        rail.grid(row=1,column=0,sticky='ns')
        line=rail.create_line(4,0,4,1,fill=style.RAIL)
        rail.bind('<Configure>',lambda event:rail.coords(line,4,0,4,event.height))
        scrollbar=tk.Scrollbar(self)
        scrollbar.grid(row=1,column=2,sticky='ns')
        self.text=tk.Text(self,height=8,width=1,wrap='word',font=(style.FONT,11),bg=style.BG,fg=style.FG,
                          padx=3,pady=2,borderwidth=0,highlightthickness=0,cursor='arrow',yscrollcommand=scrollbar.set,state='disabled')
        self.text.grid(row=1,column=1,sticky='nsew')
        self.text.bind('<Configure>',self.queue_text_layout)
        scrollbar.configure(command=self.scroll)
        self.text.tag_configure('time',foreground=style.METADATA,font=(style.FONT,9),spacing1=5,spacing3=3)
        self.text.tag_configure('body',spacing3=8)
        self.line_height=tkfont.Font(root=self,font=self.text.cget('font')).metrics('linespace')
        self.char_width=max(1,tkfont.Font(root=self,font=self.text.cget('font')).measure('M'))
        self.bind_wheel(self,self.text,scrollbar,rail,self.tabs,self.transcript_button,self.summary_button,self.copy_summary,self.note,self.topic)
        self.update_row(row)

    def bind_wheel(self,*targets):
        # Keep the whole recording card independent of outer history scrolling.
        for target in targets:
            target.bind('<Button-4>',lambda _:self.wheel(-1))
            target.bind('<Button-5>',lambda _:self.wheel(1))
            target.bind('<MouseWheel>',lambda event:self.wheel(-1 if event.delta>0 else 1))

    def queue_text_layout(self, _=None):
        if self.layout_timer is None:self.layout_timer=self.after_idle(self.finish_text_layout)

    def finish_text_layout(self):
        self.layout_timer=None
        if self.text.winfo_width()<=10:return
        self.text.tk.call(self.text._w,'sync')
        source=self.summary if self.mode=='summary' else '\n'.join(segment['text'] for segment in self.segments)
        capacity=max(1,(self.maximum_height-self.tabs.winfo_reqheight())//self.line_height)
        columns=max(1,(self.text.winfo_width()-6)//self.char_width)
        if len(source)>columns*capacity or source.count('\n')>=capacity:
            height=self.maximum_height
        else:
            pixels=self.text.count('1.0','end-1c','ypixels')
            content=(pixels[0] if pixels else 0)+self.line_height+4
            height=min(self.maximum_height,max(48,self.tabs.winfo_reqheight()+content))
        if int(self.cget('height'))!=height:
            self.configure(height=height)
            self.style.queue_header_resize()
        if self.follow_tail and self.mode=='transcript':self.text.see('end')

    def destroy(self):
        if self.layout_timer is not None:self.after_cancel(self.layout_timer)
        super().destroy()

    def resize(self,width,height):
        self.maximum_height=height
        self.configure(width=width)
        self.queue_text_layout()

    def wheel(self,direction):
        top,bottom=self.text.yview()
        if (direction<0 and top<=1e-9) or (direction>0 and bottom>=1-1e-9):
            return self.style.wheel(direction)
        self.text.yview_scroll(direction*3,'units')
        self.track_manual_scroll()
        return 'break'

    def scroll(self,*args):
        self.text.yview(*args)
        self.track_manual_scroll()

    def track_manual_scroll(self):
        if self.mode=='transcript':self.follow_tail=self.text.yview()[1]>=1-1e-9

    def append(self,segments):
        if not segments:return
        follow=self.follow_tail
        self.text.configure(state='normal')
        for segment in segments:
            try:stamp=datetime.fromisoformat(segment['timestamp']).astimezone().strftime('%H:%M:%S')
            except ValueError:stamp=segment['timestamp']
            reason={'pause':'pause','limit':'25 s','stop':'stop'}.get(segment.get('end_reason'),'')
            self.text.insert('end',stamp+(' · '+reason if reason else '')+'\n','time')
            self.text.insert('end',segment['text']+'\n','body')
        self.text.configure(state='disabled')
        if follow:
            self.text.see('end')
            self.winfo_toplevel().after_idle(lambda:self.text.see('end') if self.text.winfo_exists() and self.follow_tail and self.mode=='transcript' else None)

    def show(self,mode):
        self.mode=mode;self.text.configure(state='normal');self.text.delete('1.0','end')
        if mode=='transcript':
            self.copy_summary.pack_forget();self.append(self.segments)
        else:
            self.text.insert('end',self.summary,'body');self.text.configure(state='disabled')
            self.text.yview_moveto(0);self.copy_summary.pack(side='right')
        self.queue_text_layout()

    def update_row(self,row):
        segments=row['segments'];previous=self.segments
        if self.mode=='transcript':
            if segments[:len(previous)]==previous:self.append(segments[len(previous):])
            else:
                self.text.configure(state='normal');self.text.delete('1.0','end');self.append(segments)
        self.segments=list(segments)
        summary=row.get('summary','')
        changed=summary and summary!=self.summary
        self.summary=summary
        self.topic.configure(text=row.get('summary_title') or ('Naming…' if row.get('title_busy') else ''))
        self.summary_button.configure(state='normal' if summary else 'disabled')
        behind=summary and segments and isinstance(row.get('summary_seq'),int) and segments[-1]['seq']>row['summary_seq']
        self.note.configure(text='Summarizing…' if row.get('summary_busy') else row.get('summary_error') or
                            ('Summary is behind' if behind else ''))
        if changed:self.show('summary')
        self.queue_text_layout()
