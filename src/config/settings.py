# src/config/settings.py
import os
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
# Asegúrate de que el archivo .env esté en el directorio raíz del proyecto
load_dotenv()

class BaseConfig:
    """
    Clase base para la configuración general de la aplicación.
    Contiene valores por defecto y comunes a todos los entornos.
    """
    # Clave secreta para Streamlit/sesiones (si alguna vez necesitas un Flask app por detrás o para hashing).
    # ¡Genera una clave larga y aleatoria para producción!
    SECRET_KEY = os.getenv("SECRET_KEY", "fallback_secret_for_dev_if_not_set_very_insecure_do_not_use_in_prod")

    APP_NAME = "Dataloop Duplicate Detector"

    # Ruta base para el escaneo de archivos. CRÍTICA para el funcionamiento de tu app.
    # Asegúrate de que esta ruta exista y tu aplicación tenga permisos para leer/escribir.
    SCAN_BASE_PATH = os.getenv("SCAN_BASE_PATH", "/tmp/dataloop_files") # RUTA DE EJEMPLO

    # Tamaño máximo de archivo permitido para procesamiento (en Megabytes)
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 500))

    # Tipos de archivo permitidos para el escaneo/limpieza
    ALLOWED_FILE_TYPES = [
        "pdf", "doc", "docx", "txt", "csv", "xls", "xlsx",
        "jpg", "jpeg", "png", "gif", "bmp", "tiff",
        "zip", "rar", "7z", "tar", "gz"
    ]

    DEBUG = False # Por defecto, la depuración está desactivada por seguridad

    # URL de la base de datos (ej. para SQLite)
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./instance/app.db")

    # Ruta para logs y reportes (opcional, si se define en .env)
    LOG_PATH = os.getenv("LOG_PATH", "./logs_and_reports")


class DevelopmentConfig(BaseConfig):
    """
    Configuración específica para el entorno de desarrollo.
    """
    DEBUG = os.getenv("DEBUG_MODE", "True").lower() == "true" # Leer de .env o default True
    SCAN_BASE_PATH = os.getenv("SCAN_BASE_PATH", "/tmp/dataloop_dev_files") # Asegúrate de que esta ruta exista
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./instance/dev_limpiador.db")

class ProductionConfig(BaseConfig):
    """
    Configuración específica para el entorno de producción.
    """
    # En producción, SECRET_KEY debe OBLIGATORIAMENTE venir de una variable de entorno
    SECRET_KEY = os.getenv("SECRET_KEY") 
    DEBUG = False
    SCAN_BASE_PATH = os.getenv("SCAN_BASE_PATH", "/var/www/dataloop/uploads") 
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@db_host:5432/production_db")
    LOG_PATH = os.getenv("LOG_PATH", "/var/log/dataloop") # Ruta de logs en producción

class TestingConfig(BaseConfig):
    """
    Configuración específica para el entorno de pruebas.
    """
    TESTING = True # Flag para frameworks de testing
    DEBUG = True
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./instance/test_limpiador.db")
    SCAN_BASE_PATH = "/tmp/dataloop_test_files" # Una ruta aislada para pruebas
    LOG_PATH = "./test_logs"

def get_config():
    """
    Función para cargar la configuración adecuada según el entorno (APP_ENV).
    """
    app_env = os.getenv("APP_ENV", "development").lower()

    if app_env == "production":
        return ProductionConfig()
    elif app_env == "testing":
        return TestingConfig()
    else: # Por defecto o si es 'development'
        return DevelopmentConfig()

# Exporta la configuración que se usará en toda la aplicación
settings = get_config()

# Asegurar que la ruta de escaneo y la ruta de logs existan al iniciar la app
# Esto ayuda a evitar FileNotFoundError al inicio
if not os.path.exists(settings.SCAN_BASE_PATH):
    try:
        os.makedirs(settings.SCAN_BASE_PATH, exist_ok=True)
        print(f"Directorio de escaneo creado: {settings.SCAN_BASE_PATH}")
    except OSError as e:
        print(f"ERROR: No se pudo crear el directorio de escaneo '{settings.SCAN_BASE_PATH}': {e}")
        # Puedes decidir si la app debe fallar o continuar sin esa ruta

if not os.path.exists(settings.LOG_PATH):
    try:
        os.makedirs(settings.LOG_PATH, exist_ok=True)
        print(f"Directorio de logs/reportes creado: {settings.LOG_PATH}")
    except OSError as e:
        print(f"ERROR: No se pudo crear el directorio de logs/reportes '{settings.LOG_PATH}': {e}")
