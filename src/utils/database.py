"""
Módulo responsable de gestionar la base de datos de DataLoop.

Proporciona una clase `DatabaseManager` para manejar la conexión SQLite, 
inicializar tablas, ejecutar consultas, registrar escaneos, duplicados, 
acciones automatizadas, y obtener estadísticas del sistema y rendimiento.

Incluye funciones para respaldar, optimizar y mantener limpia la base de datos.
"""

import os
import json
import sqlite3
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from contextlib import contextmanager
from src.core.file_scanner import FileScanner
from src.utils.config import Config


class DatabaseManager:
    """Gestor de base de datos SQLite para DataLoop"""
    
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            # Usar Config para obtener ruta de la base de datos
            Config.ensure_directories()
            db_path = Config.DATA_DIR / "dataloop.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Thread-safe connection pool
        self._local_storage = threading.local()
        
        # Configuración de conexión
        self.connection_timeout = 30.0
        self.busy_timeout = 5000  # 5 segundos en milisegundos
        
        self._initialize_database()
        # Ejecutar migraciones integradas
        self._migrate_schema()
        print(f"✅ Base de datos inicializada y migrada: {self.db_path}")

    def execute_query(self, query: str, params: tuple = ()) -> list:
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def _migrate_schema(self):
        """Verifica y agrega columnas faltantes en la base de datos"""
        with self.get_cursor() as cursor:
            cursor.execute("PRAGMA table_info(files)")
            columns = [row["name"] for row in cursor.fetchall()]

            if "directory" not in columns:
                cursor.execute("ALTER TABLE files ADD COLUMN directory TEXT")
                print("Columna 'directory' agregada a la tabla 'files'")

            if "access_count" not in columns:
                cursor.execute("ALTER TABLE files ADD COLUMN access_count INTEGER DEFAULT 0")
                print("Columna 'access_count' agregada a la tabla 'files'")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Obtiene conexión thread-safe con configuración optimizada"""
        if not hasattr(self._local_storage, 'connection'):
            self._local_storage.connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=self.connection_timeout
            )
            conn = self._local_storage.connection
            conn.row_factory = sqlite3.Row
            
            # Configuraciones de optimización
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")  # Write-Ahead Logging
            conn.execute("PRAGMA synchronous = NORMAL")  # Balance entre velocidad y seguridad
            conn.execute("PRAGMA cache_size = 10000")  # Cache de 10MB
            conn.execute(f"PRAGMA busy_timeout = {self.busy_timeout}")
            conn.execute("PRAGMA temp_store = MEMORY")  # Usar memoria para temporales
        
        return self._local_storage.connection
    
    @contextmanager
    def get_cursor(self):
        """Context manager para cursor con manejo robusto de errores"""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except sqlite3.Error as e:
            conn.rollback()
            print(f"❌ Error en base de datos: {e}")
            raise
        except Exception as e:
            conn.rollback()
            print(f"❌ Error inesperado en BD: {e}")
            raise
        finally:
            cursor.close()
    
    
    def _initialize_database(self):
        """Inicializa las tablas de la base de datos con esquema mejorado"""
        
        with self.get_cursor() as cursor:
            # Crear tablas si no existen
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filepath TEXT NOT NULL UNIQUE,
                    filename TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    file_extension TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    modified_at TIMESTAMP,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    directory TEXT,  -- Nuevo: directorio padre
                    access_count INTEGER DEFAULT 0  -- Nuevo: contador de accesos
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS duplicate_groups (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_hash TEXT NOT NULL UNIQUE,
                    files_count INTEGER NOT NULL,
                    total_size INTEGER NOT NULL,
                    space_wasted INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved BOOLEAN DEFAULT 0,
                    resolution_method TEXT,  -- Cómo se resolvió
                    priority_score INTEGER DEFAULT 0  -- Prioridad para limpieza
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS file_groups (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER,
                    group_id INTEGER,
                    is_original BOOLEAN DEFAULT 0,
                    keep_reason TEXT,  -- Por qué mantener este archivo
                    delete_reason TEXT,  -- Por qué eliminar este archivo
                    FOREIGN KEY (file_id) REFERENCES files (id) ON DELETE CASCADE,
                    FOREIGN KEY (group_id) REFERENCES duplicate_groups (id) ON DELETE CASCADE,
                    UNIQUE(file_id, group_id)
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS scans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    directory TEXT NOT NULL,
                    scan_type TEXT DEFAULT 'duplicates',
                    files_scanned INTEGER NOT NULL,
                    duplicates_found INTEGER NOT NULL,
                    space_analyzed INTEGER NOT NULL,
                    space_wasted INTEGER NOT NULL,
                    scan_duration REAL NOT NULL,
                    scan_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    scan_config TEXT,  -- JSON con configuración usada
                    errors_count INTEGER DEFAULT 0
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS automated_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action_type TEXT NOT NULL,
                    filepath TEXT NOT NULL,
                    original_filepath TEXT,
                    action_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    success BOOLEAN DEFAULT 1,
                    details TEXT,
                    space_freed INTEGER DEFAULT 0,
                    automated BOOLEAN DEFAULT 1  -- Si fue automático o manual
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS settings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL UNIQUE,
                    value TEXT NOT NULL,
                    value_type TEXT DEFAULT 'string',  -- string, int, float, bool, json
                    description TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    operation TEXT NOT NULL,
                    duration REAL NOT NULL,
                    memory_used INTEGER,
                    files_processed INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            # Ejecutar migraciones antes de crear índices
            self._migrate_schema()
            # Crear índices
            indexes = [
                'CREATE INDEX IF NOT EXISTS idx_files_hash ON files(file_hash)',
                'CREATE INDEX IF NOT EXISTS idx_files_path ON files(filepath)',
                'CREATE INDEX IF NOT EXISTS idx_files_active ON files(is_active)',
                'CREATE INDEX IF NOT EXISTS idx_files_directory ON files(directory)',
                'CREATE INDEX IF NOT EXISTS idx_files_extension ON files(file_extension)',
                'CREATE INDEX IF NOT EXISTS idx_files_size ON files(file_size)',
                'CREATE INDEX IF NOT EXISTS idx_duplicate_groups_hash ON duplicate_groups(group_hash)',
                'CREATE INDEX IF NOT EXISTS idx_duplicate_groups_resolved ON duplicate_groups(resolved)',
                'CREATE INDEX IF NOT EXISTS idx_scans_date ON scans(scan_date)',
                'CREATE INDEX IF NOT EXISTS idx_scans_directory ON scans(directory)',
                'CREATE INDEX IF NOT EXISTS idx_file_groups_file ON file_groups(file_id)',
                'CREATE INDEX IF NOT EXISTS idx_file_groups_group ON file_groups(group_id)',
                'CREATE INDEX IF NOT EXISTS idx_actions_date ON automated_actions(action_date)',
                'CREATE INDEX IF NOT EXISTS idx_actions_type ON automated_actions(action_type)'
            ]
            for index_sql in indexes:
                cursor.execute(index_sql)
    
    def add_file(self, filepath: str, file_hash: str, file_size: int, 
                 modified_time: float) -> int:
        """Agrega un archivo a la base de datos con información mejorada"""
        
        path_obj = Path(filepath)
        filename = path_obj.name
        extension = path_obj.suffix.lower()
        directory = str(path_obj.parent)
        modified_at = datetime.fromtimestamp(modified_time)
        
        with self.get_cursor() as cursor:
            cursor.execute('''
                INSERT OR REPLACE INTO files 
                (filepath, filename, file_hash, file_size, file_extension, 
                 directory, modified_at, last_seen, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, 1)
            ''', (filepath, filename, file_hash, file_size, extension, directory, modified_at))
            
            return cursor.lastrowid
    
    def update_file_last_seen(self, filepath: str):
        """Actualiza la última vez que se vio un archivo"""
        with self.get_cursor() as cursor:
            cursor.execute('''
                UPDATE files 
                SET last_seen = CURRENT_TIMESTAMP, access_count = access_count + 1
                WHERE filepath = ?
            ''', (filepath,))
    
    def find_duplicates_by_hash(self, file_hash: str) -> List[Dict]:
        """Encuentra archivos duplicados por hash"""
        
        with self.get_cursor() as cursor:
            cursor.execute('''
                SELECT * FROM files 
                WHERE file_hash = ? AND is_active = 1
                ORDER BY created_at ASC, access_count DESC
            ''', (file_hash,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_all_duplicates(self, min_size: int = 0) -> List[Dict]:
        """Obtiene todos los grupos de duplicados con filtro de tamaño"""
        
        with self.get_cursor() as cursor:
            cursor.execute('''
                SELECT file_hash, COUNT(*) as count, 
                       SUM(file_size) as total_size,
                       MIN(file_size) as min_size,
                       MAX(file_size) as max_size,
                       GROUP_CONCAT(file_extension) as extensions
                FROM files 
                WHERE is_active = 1 AND file_size >= ?
                GROUP BY file_hash 
                HAVING COUNT(*) > 1
                ORDER BY total_size DESC
            ''', (min_size,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def create_duplicate_group(self, group_hash: str, files: List[Dict]) -> int:
        """Crea un grupo de duplicados con lógica mejorada"""
        
        files_count = len(files)
        total_size = sum(f['file_size'] for f in files)
        space_wasted = total_size - files[0]['file_size']  # Espacio que se puede ahorrar
        
        # Calcular score de prioridad (más grande = más importante)
        priority_score = int(space_wasted / (1024 * 1024))  # MB como score base
        
        with self.get_cursor() as cursor:
            cursor.execute('''
                INSERT OR REPLACE INTO duplicate_groups 
                (group_hash, files_count, total_size, space_wasted, priority_score)
                VALUES (?, ?, ?, ?, ?)
            ''', (group_hash, files_count, total_size, space_wasted, priority_score))
            
            group_id = cursor.lastrowid
            
            # Relacionar archivos con el grupo (con lógica de preferencia)
            for i, file_info in enumerate(files):
                is_original = (i == 0)  # El primer archivo (más antiguo) es original
                
                # Determinar razón para mantener
                keep_reason = None
                if is_original:
                    keep_reason = "oldest_file"
                elif file_info.get('access_count', 0) > 0:
                    keep_reason = "recently_accessed"
                
                cursor.execute('''
                    INSERT OR REPLACE INTO file_groups (file_id, group_id, is_original, keep_reason)
                    VALUES (?, ?, ?, ?)
                ''', (file_info['id'], group_id, is_original, keep_reason))
            
            return group_id
    
    def save_scan_results(self, directory: str, files_scanned: int, 
                         duplicates_found: int, space_analyzed: int,
                         space_wasted: int, scan_duration: float, 
                         scan_type: str = 'duplicates', scan_config: dict = None,
                         errors_count: int = 0) -> int:
        """Guarda los resultados de un escaneo con más detalles"""
        
        config_json = json.dumps(scan_config) if scan_config else None
        
        with self.get_cursor() as cursor:
            cursor.execute('''
                INSERT INTO scans 
                (directory, scan_type, files_scanned, duplicates_found, 
                 space_analyzed, space_wasted, scan_duration, scan_config, errors_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (directory, scan_type, files_scanned, duplicates_found, 
                  space_analyzed, space_wasted, scan_duration, config_json, errors_count))
            
            return cursor.lastrowid
    
    def get_scan_history(self, limit: int = 50, scan_type: str = None) -> List[Dict]:
        """Obtiene el historial de escaneos con filtros"""
        
        with self.get_cursor() as cursor:
            if scan_type:
                cursor.execute('''
                    SELECT * FROM scans 
                    WHERE scan_type = ?
                    ORDER BY scan_date DESC 
                    LIMIT ?
                ''', (scan_type, limit))
            else:
                cursor.execute('''
                    SELECT * FROM scans 
                    ORDER BY scan_date DESC 
                    LIMIT ?
                ''', (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_statistics(self) -> Dict:
        """Obtiene estadísticas completas y mejoradas"""
        
        with self.get_cursor() as cursor:
            stats = {}
            
            # Estadísticas básicas de archivos
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_files,
                    SUM(file_size) as total_size,
                    AVG(file_size) as avg_size,
                    MIN(file_size) as min_size,
                    MAX(file_size) as max_size
                FROM files WHERE is_active = 1
            ''')
            file_stats = dict(cursor.fetchone())
            stats.update(file_stats)
            
            # Estadísticas de duplicados
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_duplicates,
                    COUNT(DISTINCT file_hash) as duplicate_groups,
                    SUM(file_size) as duplicates_size
                FROM files 
                WHERE is_active = 1 AND file_hash IN (
                    SELECT file_hash FROM files 
                    WHERE is_active = 1 
                    GROUP BY file_hash 
                    HAVING COUNT(*) > 1
                )
            ''')
            dup_stats = dict(cursor.fetchone())
            stats.update(dup_stats)
            
            # Espacio desperdiciado calculado correctamente
            cursor.execute('''
                SELECT COALESCE(SUM((count - 1) * avg_size), 0) as wasted_space
                FROM (
                    SELECT file_hash, COUNT(*) as count, AVG(file_size) as avg_size
                    FROM files 
                    WHERE is_active = 1
                    GROUP BY file_hash
                    HAVING COUNT(*) > 1
                )
            ''')
            stats['wasted_space'] = int(cursor.fetchone()['wasted_space'])
            
            # Estadísticas de escaneos
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_scans,
                    MAX(scan_date) as last_scan,
                    AVG(scan_duration) as avg_scan_duration,
                    SUM(files_scanned) as total_files_scanned
                FROM scans
            ''')
            scan_stats = dict(cursor.fetchone())
            stats.update(scan_stats)
            
            # Top extensiones
            cursor.execute('''
                SELECT file_extension, COUNT(*) as count, SUM(file_size) as total_size
                FROM files 
                WHERE is_active = 1 AND file_extension IS NOT NULL
                GROUP BY file_extension
                ORDER BY count DESC
                LIMIT 10
            ''')
            stats['top_extensions'] = [dict(row) for row in cursor.fetchall()]
            
            # Directorios con más duplicados
            cursor.execute('''
                SELECT directory, COUNT(*) as duplicates_count
                FROM files 
                WHERE is_active = 1 AND file_hash IN (
                    SELECT file_hash FROM files 
                    WHERE is_active = 1 
                    GROUP BY file_hash 
                    HAVING COUNT(*) > 1
                )
                GROUP BY directory
                ORDER BY duplicates_count DESC
                LIMIT 10
            ''')
            stats['top_duplicate_dirs'] = [dict(row) for row in cursor.fetchall()]
            
            return stats
    
    def log_automated_action(self, action_type: str, filepath: str, 
                           original_filepath: str = None, 
                           success: bool = True, details: str = None,
                           space_freed: int = 0, automated: bool = True):
        """Registra una acción automatizada con más detalles"""
        
        with self.get_cursor() as cursor:
            cursor.execute('''
                INSERT INTO automated_actions 
                (action_type, filepath, original_filepath, success, details, space_freed, automated)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (action_type, filepath, original_filepath, success, details, space_freed, automated))
    
    def log_performance_metric(self, operation: str, duration: float, 
                              memory_used: int = None, files_processed: int = None):
        """Registra métricas de rendimiento"""
        
        with self.get_cursor() as cursor:
            cursor.execute('''
                INSERT INTO performance_metrics 
                (operation, duration, memory_used, files_processed)
                VALUES (?, ?, ?, ?)
            ''', (operation, duration, memory_used, files_processed))
    
    def get_performance_stats(self, hours: int = 24) -> Dict:
        """Obtiene estadísticas de rendimiento"""
        
        since = datetime.now() - timedelta(hours=hours)
        
        with self.get_cursor() as cursor:
            cursor.execute('''
                SELECT 
                    operation,
                    COUNT(*) as count,
                    AVG(duration) as avg_duration,
                    MAX(duration) as max_duration,
                    AVG(memory_used) as avg_memory,
                    SUM(files_processed) as total_files
                FROM performance_metrics 
                WHERE timestamp > ?
                GROUP BY operation
                ORDER BY avg_duration DESC
            ''', (since,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def cleanup_old_records(self, days: int = 90):
        """Limpia registros antiguos con más opciones"""
        
        with self.get_cursor() as cursor:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Marcar archivos inactivos
            cursor.execute('''
                UPDATE files 
                SET is_active = 0 
                WHERE last_seen < ? AND is_active = 1
            ''', (cutoff_date,))
            
            inactive_count = cursor.rowcount
            
            # Limpiar métricas de rendimiento muy antiguas
            old_cutoff = datetime.now() - timedelta(days=30)
            cursor.execute('''
                DELETE FROM performance_metrics 
                WHERE timestamp < ?
            ''', (old_cutoff,))
            
            metrics_deleted = cursor.rowcount
            
            # Limpiar acciones automatizadas muy antiguas
            very_old_cutoff = datetime.now() - timedelta(days=180)
            cursor.execute('''
                DELETE FROM automated_actions 
                WHERE action_date < ?
            ''', (very_old_cutoff,))
            
            actions_deleted = cursor.rowcount
            
            print(f"🧹 Limpieza completada: {inactive_count} archivos marcados inactivos, "
                  f"{metrics_deleted} métricas eliminadas, {actions_deleted} acciones eliminadas")
    
    def vacuum_database(self):
        """Optimiza la base de datos"""
        try:
            conn = self._get_connection()
            # VACUUM debe ejecutarse fuera de transacción
            conn.isolation_level = None
            conn.execute('VACUUM')
            conn.isolation_level = ''
            print("✅ Base de datos optimizada correctamente")
        except Exception as e:
            print(f"❌ Error al optimizar base de datos: {e}")
    
    def get_database_size(self) -> int:
        """Obtiene el tamaño de la base de datos en bytes"""
        
        try:
            return os.path.getsize(self.db_path)
        except OSError:
            return 0
    
    def get_database_info(self) -> Dict:
        """Obtiene información completa de la base de datos"""
        
        with self.get_cursor() as cursor:
            info = {
                'path': str(self.db_path),
                'size_bytes': self.get_database_size(),
                'created': datetime.fromtimestamp(os.path.getctime(self.db_path)).isoformat(),
                'modified': datetime.fromtimestamp(os.path.getmtime(self.db_path)).isoformat()
            }
            
            # Información de tablas
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            info['tables'] = tables
            
            # Conteo de registros por tabla
            table_counts = {}
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                table_counts[table] = cursor.fetchone()[0]
            info['table_counts'] = table_counts
            
            return info
    
    def backup_database(self, backup_path: str = None) -> bool:
        """Crea un respaldo de la base de datos"""
        
        try:
            if backup_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = Config.BACKUPS_DIR / f"dataloop_backup_{timestamp}.db"
            
            backup_path = Path(backup_path)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            source_conn = self._get_connection()
            backup_conn = sqlite3.connect(str(backup_path))
            
            source_conn.backup(backup_conn)
            backup_conn.close()
            
            print(f"✅ Respaldo creado: {backup_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error al crear respaldo: {e}")
            return False
    
    def close(self):
        """Cierra todas las conexiones de forma segura"""
        if hasattr(self._local_storage, 'connection'):
            try:
                self._local_storage.connection.close()
            except Exception as e:
                print(f"Error cerrando conexión: {e}")


# Instancia global con configuración thread-safe
_db_manager = None
_db_lock = threading.Lock()

def get_db_manager() -> DatabaseManager:
    """Obtiene la instancia global del DatabaseManager (Singleton thread-safe)"""
    global _db_manager
    if _db_manager is None:
        with _db_lock:
            if _db_manager is None:
                _db_manager = DatabaseManager()
    return _db_manager

db_manager = get_db_manager()
