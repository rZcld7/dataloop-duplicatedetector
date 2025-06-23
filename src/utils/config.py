"""
Este módulo define la clase Config, que centraliza toda la configuración del proyecto DataLoop.

Incluye rutas importantes, extensiones soportadas, parámetros para logging, rendimiento,
entorno de ejecución, y utilidades para validar, exportar e importar la configuración.

Además, adapta ciertos valores según los recursos del sistema y asegura que
los directorios necesarios estén creados.
"""

import os
import psutil
import platform
from pathlib import Path
from typing import Tuple

class Config:
    # Rutas del proyecto
    @classmethod
    def get_project_root(cls):
        try:
            project_root = Path(__file__).resolve().parent.parent.parent
            print(f"[DEBUG] Resolved PROJECT_ROOT from __file__: {project_root}")
            return project_root
        except NameError:
            import os
            project_root = Path.cwd().parent  # fallback a directorio actual usando Path.cwd()
            print(f"[DEBUG] Resolved PROJECT_ROOT from Path.cwd(): {project_root}")
            return project_root

    PROJECT_ROOT = get_project_root.__func__(None)
    DATA_DIR = PROJECT_ROOT / "data"
    LOGS_DIR = PROJECT_ROOT / "logs"
    TEMP_DIR = DATA_DIR / "temp"
    
    # Configuración de escaneo
    SUPPORTED_EXTENSIONS = {
        'images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'],
        'videos': ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'],
        'documents': ['.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt'],
        'audio': ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma'],
        'archives': ['.zip', '.rar', '.7z', '.tar', '.gz'],
        'others': []
    }
    
    # Configuración de UI
    PAGE_TITLE = "DataLoop v1.0 - Limpiador de Duplicados"
    PAGE_ICON = "🔄"
    
    # Configuración de hash
    CHUNK_SIZE = 8192  # 8KB chunks para archivos grandes
    
    # === CONFIGURACIONES DINÁMICAS BASADAS EN EL SISTEMA ===
    
    # Configuración de logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    # Configuración de rotación de logs
    LOG_MAX_SIZE = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT = 5
    LOG_RETENTION_DAYS = 30
    
    # Configuración de entorno con validación
    @classmethod
    def _get_environment(cls):
        """Determina el entorno de ejecución de forma inteligente"""
        env = os.getenv("ENVIRONMENT", "").lower()
        
        # Si no está definido, inferir del contexto
        if not env:
            # Verificar si PROJECT_ROOT es None o inválido
            if cls.PROJECT_ROOT is None:
                print("[WARNING] Config.PROJECT_ROOT is None, defaulting environment to 'development'")
                env = "development"
            else:
                # Verificar si estamos en desarrollo
                if os.path.exists(cls.PROJECT_ROOT / ".git") or os.path.exists(cls.PROJECT_ROOT / "requirements-dev.txt"):
                    env = "development"
                # Verificar si estamos en testing
                elif "pytest" in os.environ.get("_", "") or "test" in os.environ.get("PWD", "").lower():
                    env = "testing"
                else:
                    env = "production"
        
        return env if env in ["development", "production", "testing"] else "development"
    
    ENVIRONMENT = None  # Inicializar como None, se asignará después de la definición de la clase
    
    # Configuración de base de datos
    DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR}/dataloop.db")
    
    # Configuración de API
    API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))
    API_RETRIES = int(os.getenv("API_RETRIES", "3"))
    
    # === CONFIGURACIÓN DINÁMICA DE PERFORMANCE BASADA EN EL SISTEMA ===
    
    @classmethod
    def _get_optimal_workers(cls):
        """Calcula el número óptimo de workers basado en el sistema"""
        try:
            cpu_count = psutil.cpu_count(logical=False) or os.cpu_count() or 2
            # Para I/O intensivo, usar más workers que CPUs
            optimal = min(cpu_count * 2, 16)  # Máximo 16 workers
            return int(os.getenv("MAX_WORKERS", str(optimal)))
        except:
            return int(os.getenv("MAX_WORKERS", "4"))
    
    @classmethod
    def _get_memory_limit(cls):
        """Calcula el límite de memoria basado en el sistema"""
        try:
            # Obtener memoria total del sistema
            total_memory = psutil.virtual_memory().total
            # Usar máximo 25% de la RAM disponible
            limit_bytes = total_memory * 0.25
            limit_mb = int(limit_bytes / (1024 * 1024))
            # Mínimo 256MB, máximo 2GB
            limit_mb = max(256, min(limit_mb, 2048))
            return int(os.getenv("MEMORY_LIMIT_MB", str(limit_mb)))
        except:
            return int(os.getenv("MEMORY_LIMIT_MB", "512"))
    
    MAX_WORKERS = _get_optimal_workers.__func__(None)
    MEMORY_LIMIT_MB = _get_memory_limit.__func__(None)
    
    # === INFORMACIÓN DEL SISTEMA ===
    
    @classmethod
    def get_system_info(cls):
        """Obtiene información real del sistema"""
        try:
            return {
                'platform': platform.platform(),
                'system': platform.system(),
                'processor': platform.processor(),
                'cpu_count': psutil.cpu_count(logical=False),
                'cpu_count_logical': psutil.cpu_count(logical=True),
                'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'memory_available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
                'disk_usage': {
                    disk.device: {
                        'total_gb': round(psutil.disk_usage(disk.mountpoint).total / (1024**3), 2),
                        'free_gb': round(psutil.disk_usage(disk.mountpoint).free / (1024**3), 2),
                        'used_percent': psutil.disk_usage(disk.mountpoint).percent
                    }
                    for disk in psutil.disk_partitions()
                    if disk.fstype and not disk.device.startswith('/dev/loop')
                },
                'python_version': platform.python_version(),
                'architecture': platform.architecture()[0]
            }
        except Exception as e:
            return {'error': f'No se pudo obtener información del sistema: {e}'}
    
    @classmethod
    def get_default_scan_directories(cls):
        """Obtiene directorios comunes para escanear basados en el sistema operativo"""
        system = platform.system().lower()
        home = Path.home()
        
        directories = []
        
        if system == "windows":
            # Directorios comunes en Windows
            potential_dirs = [
                home / "Documents",
                home / "Downloads", 
                home / "Pictures",
                home / "Videos",
                home / "Music",
                home / "Desktop"
            ]
        elif system == "darwin":  # macOS
            potential_dirs = [
                home / "Documents",
                home / "Downloads",
                home / "Pictures", 
                home / "Movies",
                home / "Music",
                home / "Desktop"
            ]
        else:  # Linux y otros Unix
            potential_dirs = [
                home / "Documents",
                home / "Downloads",
                home / "Pictures",
                home / "Videos", 
                home / "Music",
                home / "Desktop"
            ]
        
        # Filtrar solo directorios que existen
        directories = [str(d) for d in potential_dirs if d.exists() and d.is_dir()]
        
        return directories
    
    @classmethod
    def ensure_directories(cls):
        """Crea directorios necesarios si no existen"""
        try:
            cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
            cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)
            cls.TEMP_DIR.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            print(f"Error creando directorios: {e}")
            return False
    
    @classmethod
    def get(cls, key: str, default=None):
        """
        Método get para compatibilidad con sistemas que esperan un dict-like object
        
        Args:
            key: Nombre del atributo/configuración
            default: Valor por defecto si no existe
            
        Returns:
            Valor de la configuración o default
        """
        return getattr(cls, key, default)
    
    @classmethod
    def get_log_config(cls) -> dict:
        """Retorna configuración específica para el logger"""
        return {
            'logs_dir': cls.LOGS_DIR,
            'max_file_size': cls.LOG_MAX_SIZE,
            'backup_count': cls.LOG_BACKUP_COUNT,
            'console_level': getattr(__import__('logging'), cls.LOG_LEVEL),
            'file_level': __import__('logging').DEBUG,
            'enable_json_logs': cls.ENVIRONMENT == 'production',
            'enable_rotation': True,
            'log_format': cls.LOG_FORMAT,
            'date_format': cls.LOG_DATE_FORMAT
        }
    
    @classmethod
    def get_scheduler_config(cls) -> dict:
        """Configuración optimizada para el scheduler basada en el sistema"""
        system_info = cls.get_system_info()
        
        # Ajustar configuración según recursos del sistema
        if system_info.get('memory_total_gb', 4) < 4:
            # Sistema con poca memoria
            cleanup_time = "03:00"  # Hora menos activa
            deep_scan_day = "sunday"
        else:
            # Sistema con memoria suficiente
            cleanup_time = "02:00"
            deep_scan_day = "saturday"
        
        return {
            'daily_cleanup_time': cleanup_time,
            'weekly_deep_scan_day': deep_scan_day,
            'weekly_deep_scan_time': "03:00",
            'auto_remove_duplicates': False,  # Por seguridad, por defecto False
            'keep_strategy': "oldest",
            'monitored_directories': cls.get_default_scan_directories(),
            'max_workers': cls.MAX_WORKERS,
            'memory_limit_mb': cls.MEMORY_LIMIT_MB
        }
    
    @classmethod
    def is_development(cls) -> bool:
        """Verifica si estamos en entorno de desarrollo"""
        return cls.ENVIRONMENT == 'development'
    
    @classmethod
    def is_production(cls) -> bool:
        """Verifica si estamos en entorno de producción"""
        return cls.ENVIRONMENT == 'production'
    
    @classmethod
    def get_supported_extensions_flat(cls) -> list:
        """Retorna todas las extensiones soportadas en una lista plana"""
        extensions = []
        for ext_list in cls.SUPPORTED_EXTENSIONS.values():
            extensions.extend(ext_list)
        return extensions
    
    @classmethod
    def get_extension_category(cls, extension: str) -> str:
        """
        Obtiene la categoría de una extensión
        
        Args:
            extension: Extensión del archivo (ej: '.jpg')
            
        Returns:
            Categoría del archivo o 'others'
        """
        extension = extension.lower()
        for category, extensions in cls.SUPPORTED_EXTENSIONS.items():
            if extension in extensions:
                return category
        return 'others'
    
    @classmethod
    def validate_config(cls) -> Tuple[bool, list]:
        """
        Valida que la configuración sea correcta
        
        Returns:
            Tuple[bool, list]: (True si válida, lista de errores)
        """
        errors = []
        
        try:
            # Verificar que los directorios se puedan crear
            if not cls.ensure_directories():
                errors.append("No se pudieron crear los directorios necesarios")
            
            # Verificar que los valores numéricos sean válidos
            if cls.CHUNK_SIZE <= 0:
                errors.append("CHUNK_SIZE debe ser mayor a 0")
            
            if cls.LOG_MAX_SIZE <= 0:
                errors.append("LOG_MAX_SIZE debe ser mayor a 0")
                
            if cls.LOG_BACKUP_COUNT <= 0:
                errors.append("LOG_BACKUP_COUNT debe ser mayor a 0")
                
            if cls.MAX_WORKERS <= 0:
                errors.append("MAX_WORKERS debe ser mayor a 0")
            
            # Verificar que el nivel de log sea válido
            valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            if cls.LOG_LEVEL not in valid_levels:
                errors.append(f"LOG_LEVEL debe ser uno de: {valid_levels}")
            
            # Verificar que el entorno sea válido
            valid_envs = ['development', 'production', 'testing']
            if cls.ENVIRONMENT not in valid_envs:
                errors.append(f"ENVIRONMENT debe ser uno de: {valid_envs}")
            
            # Verificar disponibilidad de memoria
            system_memory = psutil.virtual_memory().available / (1024**2)  # MB
            if cls.MEMORY_LIMIT_MB > system_memory * 0.8:
                errors.append(f"MEMORY_LIMIT_MB ({cls.MEMORY_LIMIT_MB}) excede memoria disponible")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"Error en validación de configuración: {e}")
            return False, errors
        
    @classmethod
    def set(cls, key: str, value):
        """
        Método set para establecer valores de configuración
        
        Args:
            key: Nombre del atributo/configuración
            value: Valor a establecer
        """
        setattr(cls, key, value)

    @classmethod
    def export_config(cls) -> str:
        """Exporta la configuración actual como JSON"""
        import json
        config_dict = {}
        
        # Solo exportar configuraciones, no métodos
        for attr in dir(cls):
            if not attr.startswith('_') and not callable(getattr(cls, attr)):
                value = getattr(cls, attr)
                # Convertir Path objects a string
                if isinstance(value, Path):
                    value = str(value)
                config_dict[attr] = value
                
        # Agregar información del sistema
        config_dict['system_info'] = cls.get_system_info()
        
        return json.dumps(config_dict, indent=2, default=str)

    @classmethod
    def import_config(cls, config_json: str):
        """Importa configuración desde JSON"""
        import json
        config_dict = json.loads(config_json)
        
        # Excluir system_info de la importación
        if 'system_info' in config_dict:
            del config_dict['system_info']
            
        for key, value in config_dict.items():
            if hasattr(cls, key) and not key.startswith('_'):
                # Convertir strings de vuelta a Path si es necesario
                if key.endswith('_DIR') and isinstance(value, str):
                    value = Path(value)
                setattr(cls, key, value)


# Inicialización automática
if Config.ensure_directories():
    # Validar configuración
    is_valid, errors = Config.validate_config()
    if not is_valid:
        print("⚠️  Advertencias de configuración:")
        for error in errors:
            print(f"   - {error}")
else:
    print("❌ Error crítico: No se pudieron crear directorios necesarios")
