"""Verifica inconsistencias en la base de datos:
- Busca relaciones rotas entre tablas (file_groups y duplicate_groups).
- Muestra los registros huérfanos detectados.
- No realiza cambios, solo muestra resultados.
"""

import sqlite3
from pathlib import Path

def check_db_inconsistencies(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("Verificando inconsistencias en la base de datos...")

    # Archivos en file_groups sin referencia en files
    cursor.execute('''
        SELECT fg.id, fg.file_id FROM file_groups fg
        LEFT JOIN files f ON fg.file_id = f.id
        WHERE f.id IS NULL
    ''')
    orphan_file_groups = cursor.fetchall()
    print(f"file_groups con file_id inexistente: {len(orphan_file_groups)} registros")
    for row in orphan_file_groups:
        print(f"  file_groups.id={row[0]}, file_id={row[1]}")

    # Grupos duplicados sin archivos asociados
    cursor.execute('''
        SELECT dg.id FROM duplicate_groups dg
        LEFT JOIN file_groups fg ON dg.id = fg.group_id
        WHERE fg.group_id IS NULL
    ''')
    orphan_duplicate_groups = cursor.fetchall()
    print(f"duplicate_groups sin archivos asociados: {len(orphan_duplicate_groups)} registros")
    for row in orphan_duplicate_groups:
        print(f"  duplicate_groups.id={row[0]}")

    # Archivos en files sin grupo en file_groups (opcional)
    cursor.execute('''
        SELECT f.id FROM files f
        LEFT JOIN file_groups fg ON f.id = fg.file_id
        WHERE fg.file_id IS NULL
    ''')
    files_without_group = cursor.fetchall()
    print(f"files sin grupo en file_groups: {len(files_without_group)} registros")
    for row in files_without_group:
        print(f"  files.id={row[0]}")

    conn.close()

if __name__ == "__main__":
    db_path = Path(__file__).resolve().parents[1] / "data" / "dataloop.db"
    print(f"Chequeando inconsistencias en la base de datos: {db_path}")
    check_db_inconsistencies(str(db_path))
