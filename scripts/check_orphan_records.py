"""Detecta registros huérfanos en la base de datos:
- Muestra file_groups con file_id inexistente.
- Muestra duplicate_groups sin archivos asociados.
- No borra nada, solo informa con más detalle (formato dict).
"""

import sqlite3
from pathlib import Path

def check_orphan_records(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    print("Registros en file_groups con file_id inexistente en files:")
    cursor.execute('''
        SELECT * FROM file_groups WHERE file_id NOT IN (SELECT id FROM files)
    ''')
    rows = cursor.fetchall()
    for row in rows:
        print(dict(row))
    if not rows:
        print("No se encontraron registros huérfanos en file_groups.")
    
    print("\nGrupos duplicados sin archivos asociados en file_groups:")
    cursor.execute('''
        SELECT * FROM duplicate_groups WHERE id NOT IN (SELECT DISTINCT group_id FROM file_groups)
    ''')
    rows = cursor.fetchall()
    for row in rows:
        print(dict(row))
    if not rows:
        print("No se encontraron grupos duplicados huérfanos.")
    
    conn.close()

if __name__ == "__main__":
    db_path = Path(__file__).resolve().parents[1] / "data" / "dataloop.db"
    print(f"Verificando registros huérfanos en la base de datos: {db_path}")
    check_orphan_records(str(db_path))
