"""
Sistema de logger: registra eventos, errores, operaciones y métricas del sistema
"""

import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional, Dict, Any, Union
import json
import threading
import traceback
import os
from src.utils.config import Config


class DataLoopLogger:    
    def __init__(self, name: str = "DataLoop", level: Optional[Union[int, str]] = None):
        import logging
        import threading
        from datetime import datetime
        self.name = name
        self.logger = logging.getLogger(name)
        
        # Convertir nivel de string a int si es necesario
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        elif level is None:
            from src.utils.config import Config
            level = getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO)
        
        self.logger.setLevel(level)
        
        # Lock para thread safety
        self._lock = threading.Lock()
        
        # Evitar duplicar handlers
        if not self.logger.handlers:
            self._setup_handlers()
        
        # Estadísticas de logging
        self.stats = {
            'total_logs': 0,
            'errors': 0,
            'warnings': 0,
            'last_error': None,
            'start_time': datetime.now()
        }
    
    def _setup_handlers(self):
        """Configura los handlers de logging usando Config"""
        from src.utils.config import Config
        
        import logging
        from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
        import sys
        
        try:
            # Crear directorio de logs
            Config.ensure_directories()
            
            # Formato base desde Config
            base_formatter = logging.Formatter(
                Config.LOG_FORMAT,
                datefmt=Config.LOG_DATE_FORMAT
            )
            
            # 1. Handler para archivo principal con rotación por tiempo
            log_file = Config.LOGS_DIR / "dataloop.log"
            file_handler = TimedRotatingFileHandler(
                log_file,
                when='midnight',
                interval=1,
                backupCount=Config.LOG_BACKUP_COUNT,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(base_formatter)
            file_handler.suffix = "%Y%m%d"
            
            # 2. Handler para errores críticos (archivo separado)
            error_file = Config.LOGS_DIR / "dataloop_errors.log"
            error_handler = RotatingFileHandler(
                error_file,
                maxBytes=Config.LOG_MAX_SIZE,
                backupCount=Config.LOG_BACKUP_COUNT,
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(base_formatter)
            
            # 3. Handler para consola con colores (si es terminal)
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, Config.LOG_LEVEL.upper()))
            
            # Formato colorizado para consola si es terminal
            if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty() and Config.is_development():
                console_formatter = self._create_colored_formatter()
                console_handler.setFormatter(console_formatter)
            else:
                console_handler.setFormatter(base_formatter)
            
            # 4. Handler JSON para análisis automatizado
            json_file = Config.LOGS_DIR / "dataloop_structured.log"
            json_handler = RotatingFileHandler(
                json_file,
                maxBytes=Config.LOG_MAX_SIZE,
                backupCount=5,
                encoding='utf-8'
            )
            json_handler.setLevel(logging.INFO)
            json_handler.setFormatter(self._create_json_formatter())
            
            # 5. Handler de performance (opcional)
            if Config.is_development():
                perf_file = Config.LOGS_DIR / "dataloop_performance.log"
                perf_handler = RotatingFileHandler(
                    perf_file,
                    maxBytes=10*1024*1024,  # 10MB
                    backupCount=3,
                    encoding='utf-8'
                )
                perf_handler.setLevel(logging.DEBUG)
                perf_handler.addFilter(PerformanceFilter())
                perf_handler.setFormatter(base_formatter)
                self.logger.addHandler(perf_handler)
            
            # Agregar handlers principales
            self.logger.addHandler(file_handler)
            self.logger.addHandler(error_handler)
            self.logger.addHandler(console_handler)
            self.logger.addHandler(json_handler)
            
        except Exception as e:
            # Fallback básico si hay problemas configurando
            print(f"⚠️  Error configurando logger: {e}")
            basic_handler = logging.StreamHandler()
            basic_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(basic_handler)
    
    def _create_colored_formatter(self):
        """Crea un formatter con colores ANSI para terminal"""
        
        class ColoredFormatter(logging.Formatter):
            """Formatter con colores ANSI y emojis"""
            
            COLORS = {
                'DEBUG': '\033[36m',          # Cyan
                'INFO': '\033[32m',           # Green  
                'WARNING': '\033[33m',        # Yellow
                'ERROR': '\033[31m',          # Red
                'CRITICAL': '\033[1;31m',     # Bold Red
            }
            
            EMOJIS = {
                'DEBUG': '🔍',
                'INFO': '📘',
                'WARNING': '⚠️ ',
                'ERROR': '❌',
                'CRITICAL': '🚨',
            }
            
            RESET = '\033[0m'
            
            def format(self, record):
                # Agregar color y emoji
                color = self.COLORS.get(record.levelname, '')
                emoji = self.EMOJIS.get(record.levelname, '')
                
                # Formatear el mensaje base
                formatted = super().format(record)
                
                # Aplicar color y emoji
                return f"{color}{emoji} {formatted}{self.RESET}"
        
        return ColoredFormatter(
            Config.LOG_FORMAT,
            datefmt=Config.LOG_DATE_FORMAT
        )
    
    def _create_json_formatter(self):
        """Crea un formatter JSON para análisis estructurado"""
        
        class JSONFormatter(logging.Formatter):
            """Formatter que convierte logs a JSON"""
            
            def format(self, record):
                log_entry = {
                    'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                    'level': record.levelname,
                    'logger': record.name,
                    'message': record.getMessage(),
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno,
                    'process_id': os.getpid(),
                    'thread_id': threading.get_ident(),
                }
                
                # Agregar información de excepción si existe
                if record.exc_info:
                    log_entry['exception'] = {
                        'type': record.exc_info[0].__name__,
                        'message': str(record.exc_info[1]),
                        'traceback': self.formatException(record.exc_info)
                    }
                
                # Agregar campos extra si existen
                if hasattr(record, 'extra_data'):
                    log_entry['extra'] = record.extra_data
                
                return json.dumps(log_entry, ensure_ascii=False)
        
        return JSONFormatter()
    
    def _update_stats(self, level: str):
        """Actualiza estadísticas de logging"""
        with self._lock:
            self.stats['total_logs'] += 1
            
            if level == 'ERROR':
                self.stats['errors'] += 1
                self.stats['last_error'] = datetime.now()
            elif level == 'WARNING':
                self.stats['warnings'] += 1
    
    # Métodos de logging principales
    def debug(self, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """Log de debug con datos adicionales"""
        if extra_data:
            self.logger.debug(message, extra={'extra_data': extra_data})
        else:
            self.logger.debug(message)
        self._update_stats('DEBUG')
    
    def info(self, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """Log de información con datos adicionales"""
        if extra_data:
            self.logger.info(message, extra={'extra_data': extra_data})
        else:
            self.logger.info(message)
        self._update_stats('INFO')
    
    def warning(self, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """Log de advertencia con datos adicionales"""
        if extra_data:
            self.logger.warning(message, extra={'extra_data': extra_data})
        else:
            self.logger.warning(message)
        self._update_stats('WARNING')
    
    def error(self, message: str, exception: Optional[Exception] = None, extra_data: Optional[Dict[str, Any]] = None):
        """Log de error con manejo de excepciones"""
        if exception:
            self.logger.error(f"{message}: {str(exception)}", 
                            exc_info=exception, 
                            extra={'extra_data': extra_data} if extra_data else None)
        elif extra_data:
            self.logger.error(message, extra={'extra_data': extra_data})
        else:
            self.logger.error(message)
        self._update_stats('ERROR')
    
    def critical(self, message: str, exception: Optional[Exception] = None, extra_data: Optional[Dict[str, Any]] = None):
        """Log crítico con manejo de excepciones"""
        if exception:
            self.logger.critical(f"{message}: {str(exception)}", 
                               exc_info=exception,
                               extra={'extra_data': extra_data} if extra_data else None)
        elif extra_data:
            self.logger.critical(message, extra={'extra_data': extra_data})
        else:
            self.logger.critical(message)
        self._update_stats('CRITICAL')
    
    # Métodos de contexto especializados
    def log_api_call(self, endpoint: str, method: str, status_code: int, response_time: float):
        """Log especializado para llamadas API"""
        extra_data = {
            'endpoint': endpoint,
            'method': method,
            'status_code': status_code,
            'response_time_ms': round(response_time * 1000, 2)
        }
        
        if status_code >= 400:
            self.error(f"API call failed: {method} {endpoint}", extra_data=extra_data)
        else:
            self.info(f"API call: {method} {endpoint}", extra_data=extra_data)
    
    def log_database_operation(self, operation: str, table: str, duration: float, affected_rows: int = 0):
        """Log especializado para operaciones de base de datos"""
        extra_data = {
            'operation': operation,
            'table': table,
            'duration_ms': round(duration * 1000, 2),
            'affected_rows': affected_rows
        }
        
        self.info(f"DB operation: {operation} on {table}", extra_data=extra_data)
    
    def log_file_operation(self, operation: str, file_path: str, file_size: Optional[int] = None):
        """Log especializado para operaciones de archivos"""
        extra_data = {
            'operation': operation,
            'file_path': str(file_path),
            'file_size_bytes': file_size
        }
        
        self.info(f"File operation: {operation}", extra_data=extra_data)
    
    def log_user_action(self, user_id: str, action: str, details: Optional[Dict[str, Any]] = None):
        """Log especializado para acciones de usuario"""
        extra_data = {
            'user_id': user_id,
            'action': action,
            'details': details or {}
        }
        
        self.info(f"User action: {action}", extra_data=extra_data)
    
    def log_performance_metric(self, metric_name: str, value: float, unit: str = 'ms'):
        """Log especializado para métricas de rendimiento"""
        extra_data = {
            'metric_name': metric_name,
            'value': value,
            'unit': unit,
            'timestamp': datetime.now().isoformat()
        }
        
        self.debug(f"Performance metric: {metric_name} = {value} {unit}", extra_data=extra_data)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de logging"""
        with self._lock:
            runtime = datetime.now() - self.stats['start_time']
            return {
                **self.stats,
                'runtime_seconds': runtime.total_seconds(),
                'logs_per_minute': self.stats['total_logs'] / max(runtime.total_seconds() / 60, 1)
            }
    
    def reset_stats(self):
        """Reinicia estadísticas de logging"""
        with self._lock:
            self.stats = {
                'total_logs': 0,
                'errors': 0,
                'warnings': 0,
                'last_error': None,
                'start_time': datetime.now()
            }
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Limpia logs antiguos"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            logs_dir = Config.LOGS_DIR
            
            for log_file in logs_dir.glob("*.log*"):
                if log_file.stat().st_mtime < cutoff_date.timestamp():
                    log_file.unlink()
                    self.info(f"Deleted old log file: {log_file.name}")
        
        except Exception as e:
            self.error("Error cleaning up old logs", exception=e)


class PerformanceFilter(logging.Filter):
    """Filtro para logs de rendimiento"""
    
    def filter(self, record):
        # Solo permitir logs que contengan métricas de rendimiento
        return hasattr(record, 'extra_data') and 'metric_name' in record.extra_data


class ContextLogger:
    """Logger con contexto automático para operaciones"""
    
    def __init__(self, logger: DataLoopLogger, context: Dict[str, Any]):
        self.logger = logger
        self.context = context
    
    def debug(self, message: str, **kwargs):
        extra_data = {**self.context, **kwargs}
        self.logger.debug(message, extra_data=extra_data)
    
    def info(self, message: str, **kwargs):
        extra_data = {**self.context, **kwargs}
        self.logger.info(message, extra_data=extra_data)
    
    def warning(self, message: str, **kwargs):
        extra_data = {**self.context, **kwargs}
        self.logger.warning(message, extra_data=extra_data)
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        extra_data = {**self.context, **kwargs}
        self.logger.error(message, exception=exception, extra_data=extra_data)


# Decoradores útiles
def log_execution_time(logger: DataLoopLogger, metric_name: Optional[str] = None):
    """Decorador para medir y loggear tiempo de ejecución"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            
            try:
                result = func(*args, **kwargs)
                execution_time = (datetime.now() - start_time).total_seconds()
                
                name = metric_name or f"{func.__module__}.{func.__name__}"
                logger.log_performance_metric(name, execution_time * 1000, 'ms')
                
                return result
            
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.error(f"Function {func.__name__} failed after {execution_time:.2f}s", 
                           exception=e)
                raise
        
        return wrapper
    return decorator


def log_function_calls(logger: DataLoopLogger):
    """Decorador para loggear llamadas a funciones"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"Calling function: {func.__name__}", 
                        extra_data={'args_count': len(args), 'kwargs_count': len(kwargs)})
            
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Function {func.__name__} completed successfully")
                return result
            
            except Exception as e:
                logger.error(f"Function {func.__name__} failed", exception=e)
                raise
        
        return wrapper
    return decorator


# Instancia global del logger
logger = DataLoopLogger()

# Funciones de conveniencia
def get_logger(name: str = "DataLoop") -> DataLoopLogger:
    """Obtiene una instancia del logger"""
    return DataLoopLogger(name)

def get_context_logger(context: Dict[str, Any]) -> ContextLogger:
    """Obtiene un logger con contexto"""
    return ContextLogger(logger, context)

def set_log_level(level: Union[str, int]):
    """Establece el nivel de logging globalmente"""
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    logger.logger.setLevel(level)
    
    # Actualizar también los handlers
    for handler in logger.logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(level)


# Funciones de logging directo
def debug(message: str, **kwargs):
    """Log de debug directo"""
    logger.debug(message, extra_data=kwargs if kwargs else None)

def info(message: str, **kwargs):
    """Log de info directo"""
    logger.info(message, extra_data=kwargs if kwargs else None)

def warning(message: str, **kwargs):
    """Log de warning directo"""
    logger.warning(message, extra_data=kwargs if kwargs else None)

def error(message: str, exception: Optional[Exception] = None, **kwargs):
    """Log de error directo"""
    logger.error(message, exception=exception, extra_data=kwargs if kwargs else None)

def critical(message: str, exception: Optional[Exception] = None, **kwargs):
    """Log crítico directo"""
    logger.critical(message, exception=exception, extra_data=kwargs if kwargs else None)


# Manejo de excepciones no capturadas
def handle_exception(exc_type, exc_value, exc_traceback):
    """Handler para excepciones no capturadas"""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    logger.critical("Uncaught exception", 
                   exception=exc_value,
                   extra_data={
                       'exc_type': exc_type.__name__,
                       'traceback': ''.join(traceback.format_tb(exc_traceback))
                   })

# Configurar handler para excepciones no capturadas
sys.excepthook = handle_exception


if __name__ == "__main__":
    # Ejemplo de uso
    test_logger = get_logger("TestLogger")
    
    # Pruebas básicas
    test_logger.info("Sistema de logging iniciado")
    test_logger.debug("Mensaje de debug", extra_data={'test': True})
    test_logger.warning("Mensaje de advertencia")
    
    # Pruebas con contexto
    context_logger = get_context_logger({'module': 'test', 'version': '1.0'})
    context_logger.info("Mensaje con contexto")
    
    # Prueba de métricas
    test_logger.log_performance_metric("test_metric", 150.5, "ms")
    
    # Prueba de API call
    test_logger.log_api_call("/api/test", "GET", 200, 0.125)
    
    # Mostrar estadísticas
    print("Estadísticas de logging:", test_logger.get_stats())
