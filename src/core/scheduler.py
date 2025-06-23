"""
Programador automático para tareas de mantenimiento en DataLoop.

Funciones principales:
- Escaneo diario de duplicados y eliminación automática (opcional).
- Mantenimiento mensual del sistema (limpieza de logs, archivos temporales, optimización de BD).
- Verificación del estado del sistema (memoria, espacio, integridad).
- Scheduler basado en hilos y ejecución en segundo plano usando 'schedule'.
- Registro de estadísticas e historial de acciones en la base de datos.
"""

import schedule
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import os
import gc
from pathlib import Path

# Imports internos corregidos para estructura core/
from logs.logger import logger
from src.utils.database import db_manager
from src.core.file_scanner import FileScanner
from src.utils.config import Config

class AutomatedCleanupAction:
    """Acciones automatizadas de limpieza"""
    
    def __init__(self, scanner: FileScanner):
        self.scanner = scanner
        logger.info("AutomatedCleanupAction inicializado")
    
    def safe_duplicate_removal(self, duplicate_groups: List, 
                             keep_strategy: str = "oldest") -> Dict:
        """
        Eliminación segura de duplicados
        
        Args:
            duplicate_groups: Grupos de archivos duplicados
            keep_strategy: 'oldest', 'newest', 'smallest_path', 'largest_size'
        """
        
        results = {
            'processed_groups': 0,
            'files_removed': 0,
            'space_freed': 0,
            'errors': [],
            'kept_files': []
        }
        
        if not duplicate_groups:
            logger.info("No hay grupos de duplicados para procesar")
            return results
        
        logger.info(f"Procesando {len(duplicate_groups)} grupos de duplicados con estrategia '{keep_strategy}'")
        
        for group_idx, group in enumerate(duplicate_groups):
            try:
                if len(group) < 2:
                    logger.warning(f"Grupo {group_idx} tiene menos de 2 archivos, saltando")
                    continue
                
                # Determinar qué archivo mantener según la estrategia
                keep_file = self._select_file_to_keep(group, keep_strategy)
                
                if not keep_file:
                    logger.error(f"No se pudo determinar archivo a mantener en grupo {group_idx}")
                    continue
                
                results['kept_files'].append(keep_file.path)
                logger.info(f"Manteniendo archivo: {keep_file.path}")
                
                # Eliminar los demás archivos del grupo
                for file_info in group:
                    if file_info.path == keep_file.path:
                        continue
                    
                    try:
                        # Verificar que el archivo aún existe y es accesible
                        if not os.path.exists(file_info.path):
                            logger.warning(f"Archivo ya no existe: {file_info.path}")
                            continue
                        
                        if not os.access(file_info.path, os.W_OK):
                            logger.warning(f"Sin permisos de escritura: {file_info.path}")
                            continue
                        
                        # Eliminar archivo
                        os.remove(file_info.path)
                        
                        results['files_removed'] += 1
                        results['space_freed'] += file_info.size
                        
                        # Registrar acción en BD
                        db_manager.log_automated_action(
                            action_type="automated_removal",
                            filepath=file_info.path,
                            original_filepath=keep_file.path,
                            success=True,
                            details=f"Estrategia: {keep_strategy}, Grupo: {group_idx}"
                        )
                        
                        logger.info(f"🗑️ Duplicado eliminado: {file_info.path}")
                        
                    except PermissionError as e:
                        error_msg = f"Sin permisos para eliminar {file_info.path}: {e}"
                        results['errors'].append(error_msg)
                        logger.error(error_msg)
                        
                    except OSError as e:
                        error_msg = f"Error del sistema eliminando {file_info.path}: {e}"
                        results['errors'].append(error_msg)
                        logger.error(error_msg)
                        
                    except Exception as e:
                        error_msg = f"Error inesperado eliminando {file_info.path}: {e}"
                        results['errors'].append(error_msg)
                        logger.error(error_msg)
                        
                        # Registrar error en BD
                        db_manager.log_automated_action(
                            action_type="removal_error",
                            filepath=file_info.path,
                            success=False,
                            details=str(e)
                        )
                
                results['processed_groups'] += 1
                
                # Liberar memoria cada 10 grupos procesados
                if group_idx % 10 == 0:
                    gc.collect()
                
            except Exception as e:
                error_msg = f"Error procesando grupo {group_idx} de duplicados: {e}"
                results['errors'].append(error_msg)
                logger.error(error_msg)
        
        logger.info(f"Eliminación completada: {results['files_removed']} archivos, "
                   f"{results['space_freed'] / (1024*1024):.1f} MB liberados")
        
        return results
    
    def _select_file_to_keep(self, group: List, strategy: str):
        """Selecciona qué archivo mantener según la estrategia"""
        try:
            if strategy == "oldest":
                return min(group, key=lambda f: f.modified_time)
            elif strategy == "newest":
                return max(group, key=lambda f: f.modified_time)
            elif strategy == "smallest_path":
                return min(group, key=lambda f: len(f.path))
            elif strategy == "largest_size":
                return max(group, key=lambda f: f.size)
            else:
                # Por defecto, mantener el primero
                logger.warning(f"Estrategia desconocida '{strategy}', usando primera opción")
                return group[0]
        except Exception as e:
            logger.error(f"Error seleccionando archivo a mantener: {e}")
            return group[0] if group else None

class CleanupScheduler:
    """Programador de limpieza automática"""
    
    def __init__(self, scanner: FileScanner):
        self.scanner = scanner
        self.cleanup_actions = AutomatedCleanupAction(scanner)
        self.is_running = False
        self.scheduler_thread = None
        
        # Obtener configuración del sistema
        self.config = Config.get_scheduler_config()
        
        # Stats del scheduler
        self.stats = {
            'total_runs': 0,
            'successful_runs': 0,
            'last_run': None,
            'last_error': None,
            'total_files_processed': 0,
            'total_space_freed': 0
        }
        
        logger.info("CleanupScheduler inicializado con configuración del sistema")
        logger.info(f"Configuración: {self.config}")
    
    def update_config(self, new_config: Dict):
        """Actualiza la configuración del scheduler"""
        old_config = self.config.copy()
        self.config.update(new_config)
        
        # Validar directorios monitoreados
        valid_dirs = []
        for directory in self.config.get('monitored_directories', []):
            if os.path.exists(directory) and os.path.isdir(directory):
                valid_dirs.append(directory)
            else:
                logger.warning(f"Directorio no válido removido de monitoreo: {directory}")
        
        self.config['monitored_directories'] = valid_dirs
        
        # Re-configurar scheduler si está corriendo
        if self.is_running:
            self.setup_schedule()
        
        logger.info("Configuración del scheduler actualizada")
        logger.info(f"Cambios: {set(new_config.items()) - set(old_config.items())}")
    
    def daily_cleanup_task(self):
        """Tarea de limpieza diaria"""
        start_time = time.time()
        
        try:
            logger.info("🔄 Iniciando limpieza automática diaria")
            
            # Estadísticas antes de la limpieza
            stats_before = db_manager.get_statistics()
            
            # Limpiar registros antiguos de la BD
            cleaned_records = db_manager.cleanup_old_records(days=90)
            logger.info(f"Registros de BD limpiados: {cleaned_records}")
            
            # Procesar directorios monitoreados
            total_duplicates_found = 0
            total_space_freed = 0
            total_files_scanned = 0
            
            monitored_dirs = self.config.get('monitored_directories', [])
            if not monitored_dirs:
                logger.warning("No hay directorios monitoreados configurados")
                return
            
            for directory in monitored_dirs:
                if not os.path.exists(directory):
                    logger.warning(f"Directorio no existe: {directory}")
                    continue
                
                try:
                    logger.info(f"Escaneando directorio: {directory}")
                    
                    # Escaneo rápido (solo nivel superior)
                    results = self.scanner.scan_directory(
                        directory, 
                        include_subdirs=False,
                        max_depth=1
                    )
                    
                    total_files_scanned += results.total_files
                    
                    if results.duplicate_groups and self.config.get('auto_remove_duplicates', False):
                        # Eliminar duplicados automáticamente solo si está habilitado
                        cleanup_results = self.cleanup_actions.safe_duplicate_removal(
                            results.duplicate_groups,
                            keep_strategy=self.config.get('keep_strategy', 'oldest')
                        )
                        
                        total_duplicates_found += cleanup_results['files_removed']
                        total_space_freed += cleanup_results['space_freed']
                    else:
                        # Solo reportar duplicados encontrados
                        total_duplicates_found += results.duplicates_found
                        
                except Exception as e:
                    logger.error(f"Error escaneando {directory}: {e}")
                    continue
            
            # Estadísticas después de la limpieza
            stats_after = db_manager.get_statistics()
            duration = time.time() - start_time
            
            # Registrar el escaneo en BD
            db_manager.save_scan_results(
                directory="automated_daily_cleanup",
                files_scanned=total_files_scanned,
                duplicates_found=total_duplicates_found,
                space_analyzed=stats_after.get('total_size', 0),
                space_wasted=total_space_freed if self.config.get('auto_remove_duplicates', False) else 0,
                scan_duration=duration
            )
            
            # Actualizar estadísticas del scheduler
            self.stats['total_runs'] += 1
            self.stats['successful_runs'] += 1
            self.stats['last_run'] = datetime.now()
            self.stats['total_files_processed'] += total_files_scanned
            self.stats['total_space_freed'] += total_space_freed
            
            logger.info(f"✅ Limpieza diaria completada en {duration:.2f}s")
            logger.info(f"   Archivos escaneados: {total_files_scanned:,}")
            logger.info(f"   Duplicados {'eliminados' if self.config.get('auto_remove_duplicates') else 'encontrados'}: {total_duplicates_found}")
            if total_space_freed > 0:
                logger.info(f"   Espacio liberado: {total_space_freed / (1024*1024):.1f} MB")
            
        except Exception as e:
            self.stats['total_runs'] += 1
            self.stats['last_error'] = str(e)
            logger.error(f"Error en escaneo profundo semanal: {e}")
            raise
    
    def maintenance_task(self):
        """Tarea de mantenimiento del sistema"""
        try:
            logger.info("🛠️ Ejecutando mantenimiento del sistema")
            
            # Limpiar logs antiguos
            self._cleanup_old_logs()
            
            # Optimizar base de datos
            try:
                with db_manager.get_cursor() as cursor:
                    cursor.execute("VACUUM")
                    cursor.execute("ANALYZE")
                logger.info("Base de datos optimizada")
            except Exception as e:
                logger.error(f"Error optimizando BD: {e}")
            
            # Limpiar directorio temporal
            self._cleanup_temp_directory()
            
            # Verificar integridad del sistema
            self._system_health_check()
            
            logger.info("✅ Mantenimiento completado")
            
        except Exception as e:
            logger.error(f"Error en mantenimiento: {e}")
    
    def _cleanup_old_logs(self, days_to_keep: int = 30):
        """Limpia logs antiguos"""
        try:
            logs_dir = Config.LOGS_DIR
            if not logs_dir.exists():
                return
                
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            removed_count = 0
            space_freed = 0
            
            for log_file in logs_dir.glob("*.log*"):
                try:
                    file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if file_mtime < cutoff_date:
                        file_size = log_file.stat().st_size
                        log_file.unlink()
                        removed_count += 1
                        space_freed += file_size
                except Exception as e:
                    logger.error(f"Error eliminando log {log_file}: {e}")
            
            if removed_count > 0:
                logger.info(f"Eliminados {removed_count} archivos de log antiguos "
                           f"({space_freed / (1024*1024):.1f} MB liberados)")
                
        except Exception as e:
            logger.error(f"Error limpiando logs: {e}")
    
    def _cleanup_temp_directory(self):
        """Limpia el directorio temporal"""
        try:
            temp_dir = Config.TEMP_DIR
            if not temp_dir.exists():
                return
            
            removed_count = 0
            space_freed = 0
            
            # Eliminar archivos temporales más antiguos de 24 horas
            cutoff_time = time.time() - (24 * 3600)
            
            for temp_file in temp_dir.iterdir():
                try:
                    if temp_file.is_file() and temp_file.stat().st_mtime < cutoff_time:
                        file_size = temp_file.stat().st_size
                        temp_file.unlink()
                        removed_count += 1
                        space_freed += file_size
                except Exception as e:
                    logger.error(f"Error eliminando archivo temporal {temp_file}: {e}")
            
            if removed_count > 0:
                logger.info(f"Eliminados {removed_count} archivos temporales "
                           f"({space_freed / (1024*1024):.1f} MB liberados)")
        
        except Exception as e:
            logger.error(f"Error limpiando directorio temporal: {e}")
    
    def _system_health_check(self):
        """Verifica la salud del sistema"""
        try:
            import psutil
            
            # Verificar uso de memoria
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                logger.warning(f"Uso de memoria alto: {memory.percent}%")
            
            # Verificar espacio en disco
            for directory in self.config.get('monitored_directories', []):
                try:
                    disk_usage = psutil.disk_usage(directory)
                    if disk_usage.percent > 95:
                        logger.warning(f"Poco espacio en disco para {directory}: {disk_usage.percent}%")
                except Exception:
                    pass
            
            # Verificar que los directorios monitoreados siguen existiendo
            invalid_dirs = []
            for directory in self.config.get('monitored_directories', []):
                if not os.path.exists(directory):
                    invalid_dirs.append(directory)
            
            if invalid_dirs:
                logger.warning(f"Directorios monitoreados no válidos: {invalid_dirs}")
                # Remover directorios inválidos
                self.config['monitored_directories'] = [
                    d for d in self.config['monitored_directories'] 
                    if d not in invalid_dirs
                ]
            
        except Exception as e:
            logger.error(f"Error en verificación de salud del sistema: {e}")
    
    def setup_schedule(self):
        """Configura las tareas programadas"""
        
        # Limpiar schedule previo
        schedule.clear()
        
        try:
            # Tarea diaria
            daily_time = self.config.get('daily_cleanup_time', '02:00')
            schedule.every().day.at(daily_time).do(self.daily_cleanup_task)
            
            # Tarea semanal
            weekly_day = self.config.get('weekly_deep_scan_day', 'sunday').lower()
            weekly_time = self.config.get('weekly_deep_scan_time', '03:00')
            
            # Validar día de la semana
            valid_days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
            if weekly_day not in valid_days:
                logger.warning(f"Día inválido '{weekly_day}', usando 'sunday'")
                weekly_day = 'sunday'
            
            getattr(schedule.every(), weekly_day).at(weekly_time).do(self.weekly_deep_scan_task)
            
            # Mantenimiento mensual (primer domingo de cada mes a las 04:00)
            schedule.every().month.do(self.maintenance_task)
            
            logger.info("Tareas programadas configuradas:")
            logger.info(f"  - Limpieza diaria: {daily_time}")
            logger.info(f"  - Escaneo semanal: {weekly_day} {weekly_time}")
            logger.info(f"  - Mantenimiento: mensual")
            logger.info(f"  - Auto-eliminación de duplicados: {'Habilitada' if self.config.get('auto_remove_duplicates') else 'Deshabilitada'}")
            
        except Exception as e:
            logger.error(f"Error configurando schedule: {e}")
            raise
    
    def _scheduler_loop(self):
        """Loop principal del scheduler"""
        logger.info("Scheduler iniciado en hilo separado")
        
        while self.is_running:
            try:
                # Ejecutar tareas pendientes
                schedule.run_pending()
                
                # Dormir por 60 segundos
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Error en scheduler loop: {e}")
                # En caso de error, esperar más tiempo antes de reintentar
                time.sleep(300)  # 5 minutos
    
    def start(self):
        """Inicia el scheduler"""
        if self.is_running:
            logger.warning("Scheduler ya está ejecutándose")
            return False
        
        try:
            # Validar configuración antes de iniciar
            if not self.config.get('monitored_directories'):
                logger.warning("No hay directorios monitoreados configurados")
            
            # Configurar tareas
            self.setup_schedule()
            
            # Marcar como ejecutándose
            self.is_running = True
            
            # Iniciar en hilo separado
            self.scheduler_thread = threading.Thread(
                target=self._scheduler_loop,
                daemon=True,
                name="CleanupScheduler"
            )
            self.scheduler_thread.start()
            
            logger.info("🕐 Scheduler iniciado correctamente")
            return True
            
        except Exception as e:
            logger.error(f"Error iniciando scheduler: {e}")
            self.is_running = False
            return False
    
    def stop(self):
        """Detiene el scheduler"""
        if not self.is_running:
            logger.info("Scheduler ya está detenido")
            return True
        
        try:
            logger.info("Deteniendo scheduler...")
            self.is_running = False
            schedule.clear()
            
            # Esperar a que termine el hilo
            if self.scheduler_thread and self.scheduler_thread.is_alive():
                self.scheduler_thread.join(timeout=10.0)
                
                if self.scheduler_thread.is_alive():
                    logger.warning("El hilo del scheduler no terminó en el tiempo esperado")
                    return False
            
            logger.info("Scheduler detenido correctamente")
            return True
            
        except Exception as e:
            logger.error(f"Error deteniendo scheduler: {e}")
            return False
    
    def get_next_runs(self) -> List[Dict]:
        """Obtiene las próximas ejecuciones programadas"""
        jobs = []
        
        for job in schedule.jobs:
            try:
                next_run = job.next_run
                jobs.append({
                    'task': str(job.job_func.__name__),
                    'description': self._get_task_description(job.job_func.__name__),
                    'next_run': next_run.strftime("%Y-%m-%d %H:%M:%S") if next_run else "N/A",
                    'next_run_timestamp': next_run.timestamp() if next_run else 0,
                    'interval': str(job.interval) if job.interval else "N/A",
                    'unit': job.unit if hasattr(job, 'unit') else "N/A"
                })
            except Exception as e:
                logger.error(f"Error obteniendo info de job: {e}")
        
        # Ordenar por próxima ejecución
        return sorted(jobs, key=lambda x: x['next_run_timestamp'])
    
    def _get_task_description(self, task_name: str) -> str:
        """Obtiene descripción amigable de la tarea"""
        descriptions = {
            'daily_cleanup_task': 'Limpieza diaria automática',
            'weekly_deep_scan_task': 'Escaneo profundo semanal',
            'maintenance_task': 'Mantenimiento del sistema'
        }
        return descriptions.get(task_name, task_name)
    
    def get_status(self) -> Dict:
        """Obtiene el estado completo del scheduler"""
        return {
            'is_running': self.is_running,
            'thread_alive': self.scheduler_thread.is_alive() if self.scheduler_thread else False,
            'scheduled_jobs': len(schedule.jobs),
            'next_runs': self.get_next_runs(),
            'config': self.config,
            'stats': self.stats,
            'system_info': {
                'monitored_directories_count': len(self.config.get('monitored_directories', [])),
                'valid_directories': [
                    d for d in self.config.get('monitored_directories', []) 
                    if os.path.exists(d)
                ],
                'auto_removal_enabled': self.config.get('auto_remove_duplicates', False),
                'keep_strategy': self.config.get('keep_strategy', 'oldest')
            }
        }
    
    def force_run_task(self, task_name: str) -> bool:
        """Ejecuta una tarea específica inmediatamente"""
        try:
            if task_name == 'daily_cleanup':
                logger.info("Ejecutando limpieza diaria forzada")
                self.daily_cleanup_task()
            elif task_name == 'weekly_deep_scan':
                logger.info("Ejecutando escaneo profundo forzado")
                self.weekly_deep_scan_task()
            elif task_name == 'maintenance':
                logger.info("Ejecutando mantenimiento forzado")
                self.maintenance_task()
            else:
                logger.error(f"Tarea desconocida: {task_name}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error ejecutando tarea {task_name}: {e}")
            return False

# Instancia global del scheduler
_cleanup_scheduler_instance = None

def get_cleanup_scheduler(scanner: FileScanner = None) -> CleanupScheduler:
    """
    Factory function para CleanupScheduler (Singleton)
    
    Args:
        scanner: Instancia del FileScanner. Si es None, se debe pasar en la primera llamada.
        
    Returns:
        CleanupScheduler: Instancia única del scheduler
    """
    global _cleanup_scheduler_instance
    
    if _cleanup_scheduler_instance is None:
        if scanner is None:
            raise ValueError("Se requiere una instancia de FileScanner en la primera llamada")
        _cleanup_scheduler_instance = CleanupScheduler(scanner)
    
    return _cleanup_scheduler_instance

def reset_cleanup_scheduler():
    """Resetea la instancia global del scheduler (útil para testing)"""
    global _cleanup_scheduler_instance
    if _cleanup_scheduler_instance and _cleanup_scheduler_instance.is_running:
        _cleanup_scheduler_instance.stop()
    _cleanup_scheduler_instance = None
