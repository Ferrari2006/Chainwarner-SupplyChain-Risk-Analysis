import sqlite3
from typing import Optional, Dict, Any
import json
import os

DB_PATH = "risk_data.db"

class DatabaseManager:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        conn = self._get_conn()
        cursor = conn.cursor()
        
        # Table for storing raw metrics (Streamed data persistence)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS project_metrics (
                id TEXT PRIMARY KEY,
                data TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Table for storing graph topology (For EasyGraph)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS project_topology (
                source TEXT,
                target TEXT,
                weight REAL,
                UNIQUE(source, target)
            )
        ''')
        
        conn.commit()
        conn.close()

    def upsert_metrics(self, project_name: str, metrics: Dict[str, Any]):
        """Save parsed metrics to SQLite immediately."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO project_metrics (id, data) 
            VALUES (?, ?)
            ON CONFLICT(id) DO UPDATE SET data=excluded.data, updated_at=CURRENT_TIMESTAMP
        ''', (project_name, json.dumps(metrics)))
        conn.commit()
        conn.close()

    def get_metrics(self, project_name: str) -> Optional[Dict[str, Any]]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('SELECT data FROM project_metrics WHERE id = ?', (project_name,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return json.loads(row[0])
        return None

    def save_topology(self, edges: list):
        """Save dependency edges for graph analysis."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.executemany('''
            INSERT OR IGNORE INTO project_topology (source, target, weight)
            VALUES (?, ?, ?)
        ''', [(e['source'], e['target'], e.get('weight', 1.0)) for e in edges])
        conn.commit()
        conn.close()

    def get_all_edges(self):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('SELECT source, target, weight FROM project_topology')
        rows = cursor.fetchall()
        conn.close()
        return [{'source': r[0], 'target': r[1], 'weight': r[2]} for r in rows]

# Global Instance
db = DatabaseManager()