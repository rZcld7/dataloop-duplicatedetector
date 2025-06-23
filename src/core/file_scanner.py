"""Analiza directorios, detecta archivos duplicados por hash,
y genera estadísticas detalladas sin modificar archivos."""

import hashlib
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
from dataclasses import dataclass

from src.utils.config import Config

@dataclass
class FileInfo:
    """Información de un archivo"""
    path: str
    size: int
    hash: str
    modified_time: float
    extension: str

@dataclass
class ScanResults:
    """Resultados del escaneo"""
    total_files: int = 0
    total_size: int = 0
    duplicates_found: int = 0
    space_to_free: int = 0
    scan_time: float = 0.0
    duplicate_groups: List[List[FileInfo]] = None
    
    def __post_init__(self):
        if self.duplicate_groups is None:
            self.duplicate_groups = []

class FileScanner:
    def __init__(self, chunk_size: int = None):
        from logs.logger import logger
        self.logger = logger
        
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.file_hashes: Dict[str, FileInfo] = {}
        self.duplicates: Dict[str, List[FileInfo]] = {}
        self.supported_extensions = set(Config.get_supported_extensions_flat())
        
        self.logger.info("FileScanner inicializado")
        self.logger.debug(f"Chunk size: {self.chunk_size}")
        self.logger.debug(f"Extensiones soportadas: {len(self.supported_extensions)}")
    
    def calculate_file_hash(self, file_path: str) -> Optional[str]:
        """
        Calcula el hash SHA-256 de un archivo
        
        Args:
            file_path: Ruta del archivo
            
        Returns:
            Hash SHA-256 del archivo o None si hay error
        """
        try:
            import hashlib
            hash_sha256 = hashlib.sha256()
            
            with open(file_path, "rb") as f:
                # Leer archivo en chunks para manejar archivos grandes
                while chunk := f.read(self.chunk_size):
                    hash_sha256.update(chunk)
            
            return hash_sha256.hexdigest()
            
        except (IOError, OSError, PermissionError) as e:
            self.logger.warning(f"Error calculando hash para {file_path}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error inesperado calculando hash para {file_path}: {e}")
            return None
    
    def is_supported_file(self, file_path: str) -> bool:
        """Verifica si el archivo es de un tipo soportado"""
        from pathlib import Path
        extension = Path(file_path).suffix.lower()
        return extension in self.supported_extensions
    
    def get_file_info(self, file_path: str) -> Optional[FileInfo]:
        """
        Obtiene información completa de un archivo
        
        Args:
            file_path: Ruta del archivo
            
        Returns:
            FileInfo object o None si hay error
        """
        try:
            from pathlib import Path
            path_obj = Path(file_path)
            
            if not path_obj.exists() or not path_obj.is_file():
                self.logger.debug(f"Archivo no existe o no es archivo: {file_path}")
                return None
            
            # Obtener información básica
            stat_info = path_obj.stat()
            size = stat_info.st_size
            modified_time = stat_info.st_mtime
            extension = path_obj.suffix.lower()
            
            # Skip archivos muy pequeños (probablemente vacíos)
            if size < 1:
                self.logger.debug(f"Archivo muy pequeño, omitiendo: {file_path}")
                return None
            
            # Calcular hash
            file_hash = self.calculate_file_hash(file_path)
            if file_hash is None:
                return None
            
            self.logger.debug(f"Archivo procesado: {file_path} ({self.format_file_size(size)})")
            
            return FileInfo(
                path=str(path_obj.absolute()),
                size=size,
                hash=file_hash,
                modified_time=modified_time,
                extension=extension
            )
            
        except Exception as e:
            self.logger.error(f"Error obteniendo información de {file_path}: {e}")
            return None
    
    def scan_directory(self, directory: str, include_subdirs: bool = True, 
                      min_size_mb: float = 0.0, max_files: int = 10000, 
                      allowed_extensions: list = None) -> ScanResults:
        """
        Escanea un directorio buscando archivos duplicados
        
        Args:
            directory: Directorio a escanear
            include_subdirs: Si incluir subdirectorios
            min_size_mb: Tamaño mínimo de archivo en MB para incluir
            max_files: Número máximo de archivos a procesar
            allowed_extensions: Lista de extensiones permitidas (ejemplo: ['.jpg', '.png'])
            
        Returns:
            ScanResults con los resultados del escaneo
        """
        import time
        from pathlib import Path
        
        start_time = time.time()
        results = ScanResults()
        
        self.logger.info(f"Iniciando escaneo de directorio: {directory}")
        self.logger.info(f"Parámetros: subdirs={include_subdirs}, min_size={min_size_mb}MB, max_files={max_files}")
        
        # Limpiar datos previos
        self.file_hashes.clear()
        self.duplicates.clear()
        
        directory_path = Path(directory)
        
        if not directory_path.exists() or not directory_path.is_dir():
            self.logger.error(f"Error: {directory} no es un directorio válido")
            return results
        
        # Convertir min_size_mb a bytes
        min_size_bytes = int(min_size_mb * 1024 * 1024)
        
        # Obtener lista de archivos
        if include_subdirs:
            file_pattern = "**/*"
        else:
            file_pattern = "*"
        
        # Directorios a excluir del escaneo para evitar errores de rutas largas o inaccesibles
        excluded_dirs = {
            'node_modules', '.git', '__pycache__', '.vscode', '.idea',
            'System Volume Information', '$RECYCLE.BIN', 'Temp', 'temp',
            '.DS_Store', 'Thumbs.db'
        }
        
        try:
            files_to_process = []
            total_found = 0
            
            for f in directory_path.glob(file_pattern):
                total_found += 1
                
                # Excluir directorios problemáticos
                if any(part in excluded_dirs for part in f.parts):
                    continue
                    
                if f.is_file():
                    files_to_process.append(f)
                    
            self.logger.info(f"Archivos encontrados: {total_found}, válidos: {len(files_to_process)}")
            
        except Exception as e:
            self.logger.error(f"Error accediendo al directorio {directory_path}: {e}")
            files_to_process = []
        
        # Filtrar archivos según criterios
        filtered_files = []
        for f in files_to_process:
            try:
                if not f.is_file():
                    continue
                    
                # Filtrar por extensiones permitidas
                if allowed_extensions is not None:
                    if f.suffix.lower() not in allowed_extensions:
                        continue
                
                # Filtrar por tamaño mínimo
                if f.stat().st_size >= min_size_bytes:
                    filtered_files.append(f)
                    
            except Exception as e:
                self.logger.warning(f"Error accediendo a {f}: {e}")
        
        files_to_process = filtered_files
        
        # Limitar número máximo de archivos
        if max_files > 0:
            files_to_process = files_to_process[:max_files]
        
        self.logger.info(f"Procesando {len(files_to_process)} archivos después de filtros")
        
        # Procesar archivos
        processed_count = 0
        for file_path in files_to_process:
            try:
                # Filtrar solo archivos soportados
                if not self.is_supported_file(str(file_path)):
                    continue
                
                # Obtener información del archivo
                file_info = self.get_file_info(str(file_path))
                if file_info is None:
                    continue
                
                results.total_files += 1
                results.total_size += file_info.size
                processed_count += 1
                
                # Log de progreso cada 100 archivos
                if processed_count % 100 == 0:
                    self.logger.info(f"Procesados: {processed_count}/{len(files_to_process)} archivos")
                
                # Verificar si ya existe un archivo con el mismo hash
                if file_info.hash in self.file_hashes:
                    # Es un duplicado
                    if file_info.hash not in self.duplicates:
                        # Primera vez que encontramos este duplicado
                        original_file = self.file_hashes[file_info.hash]
                        self.duplicates[file_info.hash] = [original_file]
                        self.logger.debug(f"Primer duplicado encontrado para hash: {file_info.hash[:16]}...")
                    
                    # Agregar este duplicado al grupo
                    self.duplicates[file_info.hash].append(file_info)
                    results.duplicates_found += 1
                    results.space_to_free += file_info.size
                    
                    self.logger.info(f"🔴 Duplicado: {file_info.path}")
                    self.logger.info(f"   Original: {self.file_hashes[file_info.hash].path}")
                    
                else:
                    # Archivo único (por ahora)
                    self.file_hashes[file_info.hash] = file_info
                    
            except Exception as e:
                self.logger.error(f"Error procesando archivo {file_path}: {e}")
                continue
        
        # Preparar grupos de duplicados para resultados
        results.duplicate_groups = list(self.duplicates.values())
        
        # Calcular tiempo total
        results.scan_time = time.time() - start_time
        
        # Log de resultados finales
        self.logger.info("="*50)
        self.logger.info("RESULTADOS DEL ESCANEO")
        self.logger.info("="*50)
        self.logger.info(f"⏱️  Tiempo total: {results.scan_time:.2f} segundos")
        self.logger.info(f"📁 Archivos procesados: {results.total_files}")
        self.logger.info(f"🔴 Duplicados encontrados: {results.duplicates_found}")
        self.logger.info(f"💾 Espacio total: {self.format_file_size(results.total_size)}")
        self.logger.info(f"🗑️  Espacio a liberar: {self.format_file_size(results.space_to_free)}")
        self.logger.info(f"📊 Grupos de duplicados: {len(results.duplicate_groups)}")
        
        if results.duplicates_found > 0:
            self.logger.info(f"💡 Ahorro potencial: {(results.space_to_free/results.total_size)*100:.1f}% del espacio total")
        
        self.logger.info("="*50)
        
        return results
    
    def get_duplicate_groups(self) -> List[List[FileInfo]]:
        """Retorna los grupos de archivos duplicados"""
        return list(self.duplicates.values())
    
    def format_file_size(self, size_bytes: int) -> str:
        """Formatea el tamaño del archivo en unidades legibles"""
        if size_bytes == 0:
            return "0 B"
            
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"
    
    def get_statistics(self) -> dict:
        """Obtiene estadísticas del último escaneo"""
        total_groups = len(self.duplicates)
        total_duplicates = sum(len(group) - 1 for group in self.duplicates.values())
        
        # Estadísticas por extensión
        extension_stats = {}
        for file_info in self.file_hashes.values():
            ext = file_info.extension or 'sin_extension'
            category = Config.get_extension_category(ext)
            
            if category not in extension_stats:
                extension_stats[category] = {'count': 0, 'size': 0}
            
            extension_stats[category]['count'] += 1
            extension_stats[category]['size'] += file_info.size
        
        return {
            'total_files_processed': len(self.file_hashes),
            'duplicate_groups': total_groups,
            'total_duplicates': total_duplicates,
            'extension_stats': extension_stats,
            'largest_files': sorted(
                self.file_hashes.values(), 
                key=lambda x: x.size, 
                reverse=True
            )[:10]  # Top 10 archivos más grandes
        }
    
    def clear_cache(self):
        """Limpia la caché de archivos procesados"""
        self.logger.info("Limpiando caché del scanner")
        self.file_hashes.clear()
        self.duplicates.clear()
    
    def get_memory_usage(self) -> dict:
        """Obtiene información sobre el uso de memoria del scanner"""
        import sys
        
        return {
            'file_hashes_count': len(self.file_hashes),
            'duplicates_count': len(self.duplicates),
            'estimated_memory_mb': (
                sys.getsizeof(self.file_hashes) + 
                sys.getsizeof(self.duplicates)
            ) / (1024 * 1024)
        } 
