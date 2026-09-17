"""Private paste groups; file content is loaded only when opened."""
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path


class History:
    def __init__(self,path=None):
        self.path=Path(path or Path.home()/'.local/share/folio/history.sqlite3')
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.path)
        os.chmod(self.path,0o600)
        self.db.execute('CREATE TABLE IF NOT EXISTS dumps (id INTEGER PRIMARY KEY, stamp TEXT NOT NULL, source TEXT NOT NULL, paths TEXT NOT NULL)')
        if 'image' not in {row[1] for row in self.db.execute('PRAGMA table_info(dumps)')}:
            self.db.execute('ALTER TABLE dumps ADD COLUMN image BLOB')
        self.db.execute('CREATE TABLE IF NOT EXISTS opened (path TEXT PRIMARY KEY, stamp TEXT NOT NULL)')
        self.db.commit()

    def add(self,source,paths,image=None):
        stamp=datetime.now().strftime('%b %d · %H:%M:%S')
        cursor=self.db.execute('INSERT INTO dumps (stamp,source,paths,image) VALUES (?,?,?,?)',(stamp,source,json.dumps([str(p) for p in paths]),image))
        self.db.commit()
        return {'id':cursor.lastrowid,'stamp':stamp,'source':source,'paths':[str(p) for p in paths],'has_image':bool(image)}

    def recent(self,limit=100,offset=0):
        return [{'id':row[0],'stamp':row[1],'source':row[2],'paths':json.loads(row[3]),'has_image':bool(row[4])}
                for row in self.db.execute('SELECT id,stamp,source,paths,image IS NOT NULL FROM dumps ORDER BY id DESC LIMIT ? OFFSET ?',(limit,offset))]

    def image(self,identity):
        return self.db.execute('SELECT image FROM dumps WHERE id=?',(identity,)).fetchone()[0]

    def opened(self,path):
        self.db.execute('INSERT OR REPLACE INTO opened VALUES (?,?)',(str(path),datetime.now().isoformat()))
        self.db.commit()

    def recent_files(self):
        return [Path(row[0]) for row in self.db.execute('SELECT path FROM opened ORDER BY stamp DESC LIMIT 100') if Path(row[0]).is_file()]

    def close(self):
        self.db.close()
