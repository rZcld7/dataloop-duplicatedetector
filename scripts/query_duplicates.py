"""Consulta los grupos de archivos duplicados más grandes:
- Agrupa por file_hash activos.
- Muestra los 20 duplicados con mayor tamaño total.
- No modifica la base de datos, solo consulta y muestra.
"""

import sqlite3
from pathlib import Path

def query_duplicate_groups(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
        SELECT file_hash, COUNT(*) as count, SUM(file_size) as total_size
        FROM files
        WHERE is_active = 1
        GROUP BY file_hash
        HAVING count > 1
        ORDER BY total_size DESC
        LIMIT 20
    ''')
    rows = cursor.fetchall()
    for row in rows:
        print(f"Hash: {row['file_hash']}, Count: {row['count']}, Total Size: {row['total_size']} bytes")
    conn.close()

if __name__ == "__main__":
    db_path = Path(__file__).resolve().parents[1] / "data" / "dataloop.db"
    print(f"Querying duplicates in database: {db_path}")
    query_duplicate_groups(str(db_path))
