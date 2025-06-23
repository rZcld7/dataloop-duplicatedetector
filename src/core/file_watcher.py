"""
Monitorea directorios en tiempo real usando Watchdog.
- Detecta nuevos archivos, modificaciones y movimientos.
- Procesa archivos válidos para buscar duplicados.
- Si se detectan duplicados, los reporta y (opcionalmente) ejecuta acciones automáticas.
- Registra eventos, errores y estadísticas del monitoreo.
"""

import os
import time
from pathlib import Path
from typing import Set, Callable, Optional, Dict, Any, List
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from logs.logger import logger
from src.utils.database import db_manager
from src.utils.config import Config
from src.core.file_scanner import FileScanner

class DuplicateDetectionHandler(FileSystemEventHandler):
    """Handler para detectar duplicados en tiempo real"""

    def __init__(self, scanner: FileScanner, 
                 duplicate_callback: Optional[Callable] = None,
                 auto_action: bool = False):
        super().__init__()
        self.scanner = scanner
        self.duplicate_callback = duplicate_callback
        self.auto_action = auto_action
        self.processing_files: Set[str] = set()

        # Solo procesar archivos soportados
        self.supported_extensions = set(Config.get_supported_extensions_flat())

        # Estadísticas del handler
        self.stats = {
            'files_processed': 0,
            'duplicates_found': 0,
            'errors': 0,
            'start_time': time.time()
        }

        logger.info("DuplicateDetectionHandler inicializado")
        logger.debug(f"Auto-acción habilitada: {auto_action}")

    def is_supported_file(self, filepath: str) -> bool:
        """Verifica si el archivo es soportado"""
        try:
            extension = Path(filepath).suffix.lower()
            return extension in self.supported_extensions
        except Exception:
            return False

    def should_process_file(self, filepath: str) -> bool:
        """Determina si un archivo debe ser procesado"""
        try:
            path_obj = Path(filepath)

            # Verificar que existe
            if not path_obj.exists() or not path_obj.is_file():
                return False

            # Verificar extensión soportada
            if not self.is_supported_file(filepath):
                return False

            # Verificar tamaño mínimo
            if path_obj.stat().st_size < 1024:  # Archivos menores a 1KB
                return False

            # Evitar archivos temporales
            name = path_obj.name.lower()
            if name.startswith('.') or name.endswith('.tmp') or '~' in name:
                return False

            return True

        except Exception as e:
            logger.debug(f"Error verificando archivo {filepath}: {e}")
            return False

    def on_created(self, event: FileSystemEvent):
        """Maneja la creación de archivos"""
        if event.is_directory:
            return

        logger.debug(f"Archivo creado: {event.src_path}")
        self.process_file(event.src_path, "created")

    def on_modified(self, event: FileSystemEvent):
        """Maneja la modificación de archivos"""
        if event.is_directory:
            return

        # Evitar procesamiento múltiple del mismo archivo
        if event.src_path not in self.processing_files:
            logger.debug(f"Archivo modificado: {event.src_path}")
            self.process_file(event.src_path, "modified")

    def on_moved(self, event: FileSystemEvent):
        """Maneja archivos movidos"""
        if event.is_directory:
            return

        logger.debug(f"Archivo movido: {event.src_path} -> {event.dest_path}")
        # Procesar el archivo en su nueva ubicación
        self.process_file(event.dest_path, "moved")

    def process_file(self, filepath: str, event_type: str):
        """Procesa un archivo para detectar duplicados"""

        # Verificaciones iniciales
        if not self.should_process_file(filepath):
            return

        # Evitar procesamiento concurrente
        if filepath in self.processing_files:
            logger.debug(f"Archivo ya en procesamiento: {filepath}")
            return

        self.processing_files.add(filepath)

        try:
            logger.debug(f"Procesando archivo {event_type}: {filepath}")
            self.stats['files_processed'] += 1

            # Obtener información del archivo
            file_info = self.scanner.get_file_info(filepath)
            if file_info is None:
                logger.debug(f"No se pudo obtener información del archivo: {filepath}")
                return

            # Buscar duplicados existentes en la base de datos
            existing_duplicates = []
            try:
                existing_duplicates = db_manager.find_duplicates_by_hash(file_info.hash)
            except Exception as e:
                logger.warning(f"Error consultando base de datos para {filepath}: {e}")

            if existing_duplicates:
                # Es un duplicado!
                self.stats['duplicates_found'] += 1

                logger.info(f"🔴 Duplicado detectado en tiempo real!")
                logger.info(f"   Nuevo: {filepath}")
                logger.info(f"   Original: {existing_duplicates[0]['filepath']}")
                logger.info(f"   Tamaño: {self.scanner.format_file_size(file_info.size)}")

                # Callback si está definido
                if self.duplicate_callback:
                    try:
                        self.duplicate_callback(file_info, existing_duplicates, event_type)
                    except Exception as e:
                        logger.error(f"Error en callback de duplicado: {e}")

                # Registrar acción en la base de datos
                try:
                    db_manager.log_automated_action(
                        action_type="duplicate_detected",
                        filepath=filepath,
                        original_filepath=existing_duplicates[0]['filepath'],
                        success=True,
                        details=f"Evento: {event_type}, Hash: {file_info.hash[:16]}, Tamaño: {file_info.size}"
                    )
                except Exception as e:
                    logger.warning(f"Error registrando acción en BD: {e}")

                # Acción automática si está habilitada
                if self.auto_action:
                    self._handle_duplicate_auto_action(file_info, existing_duplicates)

            # Agregar archivo a la base de datos
            try:
                db_manager.add_file(
                    filepath=file_info.path,
                    file_hash=file_info.hash,
                    file_size=file_info.size,
                    modified_time=file_info.modified_time
                )
                logger.debug(f"Archivo agregado a BD: {filepath}")
            except Exception as e:
                logger.warning(f"Error agregando archivo a BD {filepath}: {e}")

        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Error procesando {filepath}: {e}")

            try:
                db_manager.log_automated_action(
                    action_type="processing_error",
                    filepath=filepath,
                    success=False,
                    details=str(e)
                )
            except Exception as db_e:
                logger.error(f"Error registrando error en BD: {db_e}")

        finally:
            # Remover de procesamiento después de un delay
            time.sleep(0.1)
            self.processing_files.discard(filepath)

    def _handle_duplicate_auto_action(self, file_info, existing_duplicates):
        """Maneja acciones automáticas para duplicados"""
        try:
            # Por ahora solo loggeamos, pero aquí se podría implementar
            # eliminación automática, movimiento a carpeta, etc.
            logger.info(f"🤖 Auto-acción requerida para: {file_info.path}")

            # Ejemplo de lógica de auto-acción:
            # - Mantener el archivo más antiguo
            # - Mover duplicados a carpeta específica
            # - Crear enlaces simbólicos, etc.

        except Exception as e:
            logger.error(f"Error en auto-acción: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del handler"""
        uptime = time.time() - self.stats['start_time']
        return {
            **self.stats,
            'uptime_seconds': uptime,
            'files_per_minute': (self.stats['files_processed'] / max(uptime/60, 1)),
            'processing_queue': len(self.processing_files)
        }


class FileWatcher:
    """Monitor de archivos en tiempo real"""

    def __init__(self, scanner: FileScanner):
        self.scanner = scanner
        self.observer = Observer()
        self.watched_paths: Dict[str, Dict] = {}  # path -> config
        self.handlers: Dict[str, DuplicateDetectionHandler] = {}
        self.is_running = False

        # Configuración del watcher
        self.config = {
            'auto_action': False,
            'duplicate_callback': None,
            'max_file_size_mb': Config.MEMORY_LIMIT_MB // 4,  # Limitar archivos grandes
        }

        logger.info("FileWatcher inicializado")

    def set_duplicate_callback(self, callback: Callable):
        """Establece el callback para duplicados detectados"""
        self.config['duplicate_callback'] = callback
        logger.info("Callback de duplicados configurado")

    def enable_auto_action(self, enabled: bool = True):
        """Habilita/deshabilita acciones automáticas"""
        self.config['auto_action'] = enabled
        logger.info(f"Auto-acciones {'habilitadas' if enabled else 'deshabilitadas'}")

    def add_watch_path(self, path: str, recursive: bool = True, 
                      duplicate_callback: Optional[Callable] = None,
                      custom_config: Optional[Dict] = None) -> bool:
        """
        Agrega un directorio al monitoreo

        Args:
            path: Directorio a monitorear
            recursive: Si monitorear subdirectorios
            duplicate_callback: Callback específico para este path
            custom_config: Configuración personalizada
        """

        if not os.path.exists(path) or not os.path.isdir(path):
            logger.error(f"Directorio inválido: {path}")
            return False

        if path in self.watched_paths:
            logger.warning(f"Directorio ya está siendo monitoreado: {path}")
            return False

        try:
            # Configuración para este path
            path_config = {
                'recursive': recursive,
                'duplicate_callback': duplicate_callback or self.config['duplicate_callback'],
                'auto_action': custom_config.get('auto_action', self.config['auto_action']) if custom_config else self.config['auto_action']
            }

            # Crear handler específico para este path
            handler = DuplicateDetectionHandler(
                scanner=self.scanner,
                duplicate_callback=path_config['duplicate_callback'],
                auto_action=path_config['auto_action']
            )

            # Agregar al observer
            watch = self.observer.schedule(handler, path, recursive=recursive)

            # Guardar configuración
            self.watched_paths[path] = {
                **path_config,
                'watch': watch,
                'added_time': time.time()
            }
            self.handlers[path] = handler

            logger.info(f"📁 Directorio agregado al monitoreo: {path}")
            logger.info(f"   Recursivo: {recursive}")
            logger.info(f"   Auto-acción: {path_config['auto_action']}")

            return True

        except Exception as e:
            logger.error(f"Error agregando directorio al monitoreo: {e}")
            return False

    def remove_watch_path(self, path: str) -> bool:
        """Remueve un directorio del monitoreo"""

        if path not in self.watched_paths:
            logger.warning(f"Directorio no está siendo monitoreado: {path}")
            return False

        try:
            # Obtener watch object
            watch = self.watched_paths[path]['watch']

            # Remover del observer
            self.observer.unschedule(watch)

            # Limpiar referencias
            del self.watched_paths[path]
            if path in self.handlers:
                del self.handlers[path]

            logger.info(f"Directorio removido del monitoreo: {path}")
            return True

        except Exception as e:
            logger.error(f"Error removiendo directorio del monitoreo: {e}")
            return False

    def start(self) -> bool:
        """Inicia el monitoreo"""

        if self.is_running:
            logger.warning("FileWatcher ya está ejecutándose")
            return False

        if not self.watched_paths:
            logger.warning("No hay directorios configurados para monitorear")
            return False

        try:
            self.observer.start()
            self.is_running = True

            logger.info("🔍 FileWatcher iniciado exitosamente")
            logger.info(f"   Monitoreando {len(self.watched_paths)} directorios")

            for path, config in self.watched_paths.items():
                logger.info(f"   - {path} (recursivo: {config['recursive']})")

            return True

        except Exception as e:
            logger.error(f"Error iniciando FileWatcher: {e}")
            self.is_running = False
            return False

    def stop(self) -> bool:
        """Detiene el monitoreo"""

        if not self.is_running:
            logger.info("FileWatcher ya está detenido")
            return True

        try:
            logger.info("Deteniendo FileWatcher...")
            self.observer.stop()
            self.observer.join(timeout=5.0)

            if self.observer.is_alive():
                logger.warning("Observer no se detuvo completamente en el timeout")

            self.is_running = False
            logger.info("FileWatcher detenido exitosamente")
            return True

        except Exception as e:
            logger.error(f"Error deteniendo FileWatcher: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Obtiene el estado completo del monitor"""

        # Estadísticas agregadas de todos los handlers
        total_stats = {
            'files_processed': 0,
            'duplicates_found': 0,
            'errors': 0
        }

        handler_stats = {}
        for path, handler in self.handlers.items():
            stats = handler.get_stats()
            handler_stats[path] = stats

            total_stats['files_processed'] += stats['files_processed']
            total_stats['duplicates_found'] += stats['duplicates_found']
            total_stats['errors'] += stats['errors']

        return {
            'is_running': self.is_running,
            'watched_paths_count': len(self.watched_paths),
            'watched_paths': list(self.watched_paths.keys()),
            'observer_alive': self.observer.is_alive() if hasattr(self.observer, 'is_alive') else False,
            'total_stats': total_stats,
            'handler_stats': handler_stats,
            'config': self.config,
            'system_info': {
                'memory_usage': self.scanner.get_memory_usage(),
                'observer_threads': len(self.observer.emitters) if hasattr(self.observer, 'emitters') else 0
            }
        }

    def get_watched_paths_info(self) -> List[Dict]:
        """Obtiene información detallada de todos los directorios monitoreados"""
        
        paths_info = []
        
        for path, config in self.watched_paths.items():
            try:
                path_obj = Path(path)
                
                # Información básica del directorio
                path_info = {
                    'path': path,
                    'exists': path_obj.exists(),
                    'is_directory': path_obj.is_dir() if path_obj.exists() else False,
                    'recursive': config['recursive'],
                    'auto_action': config['auto_action'],
                    'added_time': config['added_time'],
                    'monitoring_duration': time.time() - config['added_time']
                }
                
                # Información del sistema de archivos
                if path_obj.exists():
                    try:
                        stat_info = path_obj.stat()
                        path_info.update({
                            'permissions': oct(stat_info.st_mode)[-3:],
                            'last_modified': stat_info.st_mtime,
                            'accessible': os.access(path, os.R_OK)
                        })
                        
                        # Contar archivos aproximadamente (solo primer nivel)
                        if config['recursive']:
                            file_count = sum(1 for _ in path_obj.rglob('*') if _.is_file())
                        else:
                            file_count = sum(1 for _ in path_obj.iterdir() if _.is_file())
                        
                        path_info['estimated_file_count'] = file_count
                        
                    except Exception as e:
                        path_info.update({
                            'error': str(e),
                            'accessible': False
                        })
                
                # Estadísticas del handler específico
                if path in self.handlers:
                    handler_stats = self.handlers[path].get_stats()
                    path_info['handler_stats'] = handler_stats
                
                paths_info.append(path_info)
                
            except Exception as e:
                paths_info.append({
                    'path': path,
                    'error': f"Error obteniendo información: {str(e)}",
                    'config': config
                })
        
        return paths_info
    
    def cleanup(self):
        """Limpia recursos del FileWatcher"""
        
        logger.info("Iniciando limpieza de FileWatcher...")
        
        try:
            # Detener monitoreo si está activo
            if self.is_running:
                self.stop()
            
            # Limpiar handlers
            for handler in self.handlers.values():
                handler.processing_files.clear()
            
            self.handlers.clear()
            self.watched_paths.clear()
            
            # Limpiar observer
            if hasattr(self.observer, '_emitters'):
                self.observer._emitters.clear()
            
            logger.info("Limpieza de FileWatcher completada")
            
        except Exception as e:
            logger.error(f"Error durante limpieza de FileWatcher: {e}")
    
    def pause_monitoring(self):
        """Pausa temporalmente el monitoreo"""
        
        if not self.is_running:
            logger.info("FileWatcher no está ejecutándose")
            return False
        
        try:
            # Pausar todos los handlers
            for handler in self.handlers.values():
                handler.processing_files.clear()
            
            # Detener observer temporalmente
            self.observer.stop()
            self.is_running = False
            
            logger.info("⏸️ Monitoreo pausado")
            return True
            
        except Exception as e:
            logger.error(f"Error pausando monitoreo: {e}")
            return False
    
    def resume_monitoring(self):
        """Reanuda el monitoreo pausado"""
        
        if self.is_running:
            logger.info("FileWatcher ya está ejecutándose")
            return True
        
        try:
            # Crear nuevo observer (el anterior fue detenido)
            self.observer = Observer()
            
            # Re-agregar todos los watches
            for path, config in self.watched_paths.items():
                if path in self.handlers:
                    handler = self.handlers[path]
                    watch = self.observer.schedule(
                        handler, 
                        path, 
                        recursive=config['recursive']
                    )
                    config['watch'] = watch
            
            # Iniciar observer
            self.observer.start()
            self.is_running = True
            
            logger.info("▶️ Monitoreo reanudado")
            return True
            
        except Exception as e:
            logger.error(f"Error reanudando monitoreo: {e}")
            return False
    
    def update_config(self, new_config: Dict[str, Any]):
        """Actualiza la configuración del FileWatcher"""
        
        old_config = self.config.copy()
        
        try:
            # Actualizar configuración
            self.config.update(new_config)
            
            # Aplicar cambios a handlers existentes si es necesario
            if 'auto_action' in new_config:
                for handler in self.handlers.values():
                    handler.auto_action = new_config['auto_action']
            
            if 'duplicate_callback' in new_config:
                for handler in self.handlers.values():
                    handler.duplicate_callback = new_config['duplicate_callback']
            
            logger.info("Configuración de FileWatcher actualizada")
            logger.debug(f"Cambios: {new_config}")
            
        except Exception as e:
            # Revertir cambios en caso de error
            self.config = old_config
            logger.error(f"Error actualizando configuración: {e}")
            raise
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Obtiene información sobre el uso de memoria del FileWatcher"""
        
        import sys
        
        memory_info = {
            'watched_paths_count': len(self.watched_paths),
            'handlers_count': len(self.handlers),
            'total_processing_files': sum(
                len(handler.processing_files) 
                for handler in self.handlers.values()
            ),
            'estimated_memory_mb': 0
        }
        
        try:
            # Estimar memoria utilizada
            memory_bytes = (
                sys.getsizeof(self.watched_paths) +
                sys.getsizeof(self.handlers) +
                sys.getsizeof(self.config)
            )
            
            # Agregar memoria de handlers
            for handler in self.handlers.values():
                memory_bytes += (
                    sys.getsizeof(handler.processing_files) +
                    sys.getsizeof(handler.stats)
                )
            
            memory_info['estimated_memory_mb'] = memory_bytes / (1024 * 1024)
            
        except Exception as e:
            logger.warning(f"Error calculando uso de memoria: {e}")
        
        return memory_info
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources"""
        self.cleanup()
    
    def __del__(self):
        """Destructor - ensure cleanup"""
        try:
            if hasattr(self, 'is_running') and self.is_running:
                self.stop()
        except Exception:
            pass  # Ignore errors during destruction


# Función factory para obtener una instancia de FileWatcher
def get_file_watcher(scanner: FileScanner = None) -> FileWatcher:
    """Función para obtener una instancia de FileWatcher"""
    if scanner is None:
        raise ValueError("FileScanner es requerido para crear FileWatcher")
    return FileWatcher(scanner)

__all__ = ['get_file_watcher', 'FileWatcher', 'DuplicateDetectionHandler']
