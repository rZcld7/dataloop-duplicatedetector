"""Limpia registros huérfanos en la base de datos:
- Elimina file_groups con file_id inexistente.
- Elimina duplicate_groups sin archivos asociados.
- Modifica la base de datos para mantenerla limpia.
"""

import sqlite3
from pathlib import Path

def cleanup_orphan_duplicates(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Delete file_groups entries where file_id does not exist in files
    cursor.execute('''
        DELETE FROM file_groups
        WHERE file_id NOT IN (SELECT id FROM files)
    ''')
    deleted_file_groups = cursor.rowcount
    print(f"Deleted {deleted_file_groups} orphaned file_groups entries.")
    
    # Delete duplicate_groups entries with no associated file_groups
    cursor.execute('''
        DELETE FROM duplicate_groups
        WHERE id NOT IN (SELECT DISTINCT group_id FROM file_groups)
    ''')
    deleted_duplicate_groups = cursor.rowcount
    print(f"Deleted {deleted_duplicate_groups} orphaned duplicate_groups entries.")
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    db_path = Path(__file__).resolve().parents[1] / "data" / "dataloop.db"
    print(f"Cleaning orphan duplicate records in database: {db_path}")
    cleanup_orphan_duplicates(str(db_path))
