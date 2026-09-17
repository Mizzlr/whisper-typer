"""Compact image thumbnail with its cached name and description beside it."""
import tkinter as tk


class ImageView(tk.Frame):
    def __init__(self,parent,photo,title,description,style,zoom):
        super().__init__(parent,bg=style.BG,width=400,height=88)
        self.grid_propagate(False);self.columnconfigure(1,weight=1)
        self.photo=photo;self.display_width=400
        self.thumbnail=tk.Label(self,image=photo,bg=style.BG,cursor='hand2',borderwidth=0)
        self.thumbnail.grid(row=0,column=0,rowspan=2,sticky='nw',padx=(0,12))
        self.caption_title=tk.Label(self,bg=style.BG,fg=style.FG,font=(style.FONT,11,'bold'),anchor='w',justify='left',borderwidth=0)
        self.caption_title.grid(row=0,column=1,sticky='nw')
        self.description=tk.Label(self,bg=style.BG,fg=style.FG,font=(style.FONT,10),anchor='w',justify='left',borderwidth=0)
        self.description.grid(row=1,column=1,sticky='nw',pady=(4,0))
        self.bind('<Button-1>',lambda _:zoom())
        self.thumbnail.bind('<Button-1>',lambda _:zoom())
        self.update_caption(title,description)

    def update_caption(self,title,description):
        self.caption_title.configure(text=title);self.description.configure(text=description or '')
        self.resize(self.display_width)

    def resize(self,width):
        signature=(width,self.caption_title.cget('text'),self.description.cget('text'))
        if getattr(self,'layout_signature',None)==signature:return
        self.layout_signature=signature
        self.display_width=width
        text_width=max(100,width-self.photo.width()-12)
        self.caption_title.configure(wraplength=text_width)
        self.description.configure(wraplength=text_width)
        height=max(self.photo.height(),self.caption_title.winfo_reqheight()+self.description.winfo_reqheight()+4)
        self.configure(width=width,height=height)
