"""Resetea completamente los duplicados en la base de datos:
- Elimina todos los registros de file_groups y duplicate_groups.
- Útil para reiniciar el análisis de duplicados desde cero.
"""

import sqlite3
from pathlib import Path

def reset_duplicate_groups(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print(f"Reseteando grupos de duplicados y relaciones en la base de datos: {db_path}")

    # Eliminar todas las relaciones en file_groups
    cursor.execute("DELETE FROM file_groups")
    deleted_file_groups = cursor.rowcount
    print(f"Eliminados {deleted_file_groups} registros de file_groups")

    # Eliminar todos los grupos duplicados
    cursor.execute("DELETE FROM duplicate_groups")
    deleted_duplicate_groups = cursor.rowcount
    print(f"Eliminados {deleted_duplicate_groups} registros de duplicate_groups")

    conn.commit()
    conn.close()

if __name__ == "__main__":
    db_path = Path(__file__).resolve().parents[1] / "data" / "dataloop.db"
    reset_duplicate_groups(str(db_path))
