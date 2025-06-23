"""
Script para forzar la migración de la base de datos
"""

import sqlite3
from pathlib import Path
import sys
import traceback

def force_migrate_database(db_path):
    """Fuerza la migración de la base de datos"""
    
    try:
        # Conectar a la base de datos
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        print(f"🔄 Migrando base de datos: {db_path}")
        
        # Lista de migraciones necesarias
        migrations = [
            # Tabla files
            ("files", "directory", "TEXT"),
            ("files", "access_count", "INTEGER DEFAULT 0"),
            
            # Tabla duplicate_groups  
            ("duplicate_groups", "resolution_method", "TEXT"),
            ("duplicate_groups", "priority_score", "INTEGER DEFAULT 0"),
            
            # Tabla scans
            ("scans", "scan_type", "TEXT DEFAULT 'duplicates'"),
            ("scans", "scan_config", "TEXT"),
            ("scans", "errors_count", "INTEGER DEFAULT 0"),
            
            # Tabla automated_actions
            ("automated_actions", "space_freed", "INTEGER DEFAULT 0"),
            ("automated_actions", "automated", "BOOLEAN DEFAULT 1"),
            
            # Tabla file_groups
            ("file_groups", "keep_reason", "TEXT"),
            ("file_groups", "delete_reason", "TEXT"),
            
            # Tabla settings
            ("settings", "value_type", "TEXT DEFAULT 'string'"),
            ("settings", "description", "TEXT"),
            ("settings", "updated_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
        ]
        
        # Ejecutar cada migración
        for table, column, definition in migrations:
            try:
                # Verificar si la tabla existe
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
                if not cursor.fetchone():
                    print(f"⚠️  Tabla {table} no existe, saltando...")
                    continue
                
                # Verificar si la columna ya existe
                cursor.execute(f"PRAGMA table_info({table})")
                columns = [row["name"] for row in cursor.fetchall()]
                
                if column not in columns:
                    # Agregar la columna
                    cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
                    print(f"✅ Agregada columna: {table}.{column}")
                else:
                    print(f"✓  Ya existe: {table}.{column}")
                    
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e).lower():
                    print(f"✓  Ya existe: {table}.{column}")
                else:
                    print(f"❌ Error en {table}.{column}: {e}")
        
        # Confirmar cambios
        conn.commit()
        print("✅ Migración completada exitosamente")
        
        # Mostrar estructura actual
        print("\n📋 Estructura actual de tablas:")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [row["name"] for row in cursor.fetchall()]
            print(f"  {table}: {len(columns)} columnas")
        
    except Exception as e:
        print(f"❌ Error durante migración: {e}")
        traceback.print_exc()
        raise
    finally:
        try:
            cursor.close()
        except:
            pass
        try:
            conn.close()
        except:
            pass

def main():
    import os
    import sys
    from pathlib import Path

    # Add project root to sys.path to import src.utils.config
    project_root = Path(__file__).parent.parent.resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from src.utils.config import Config
        Config.ensure_directories()
        db_path = Config.DATA_DIR / "dataloop.db"
    except Exception as e:
        print(f"⚠️ No se pudo importar Config o obtener DATA_DIR: {e}")
        db_path = Path("dataloop.db")  # fallback to local file

    print(f"Base de datos: {db_path}")

    if db_path.exists():
        force_migrate_database(str(db_path))
    else:
        print("❌ La base de datos no existe aún. Ejecuta la aplicación primero.")

if __name__ == "__main__":
    main()
