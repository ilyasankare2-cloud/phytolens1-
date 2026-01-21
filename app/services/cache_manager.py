"""
🗄️ Sistema de caché avanzado para predicciones
Soporta: Redis, memoria, disco persistente
"""

import hashlib
import json
import os
import pickle
import time
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


class PredictionCache:
    """
    Sistema de caché multinivel para predicciones
    - L1: Memoria RAM (rápido, limitado)
    - L2: Disco (persistente)
    - L3: Redis (opcional, distribuido)
    """
    
    def __init__(
        self,
        max_memory_items: int = 1000,
        cache_dir: str = "cache",
        ttl_seconds: int = 86400,  # 24 horas
        enable_disk: bool = True,
        enable_redis: bool = False
    ):
        self.max_memory_items = max_memory_items
        self.cache_dir = Path(cache_dir)
        self.ttl_seconds = ttl_seconds
        self.enable_disk = enable_disk
        self.enable_redis = enable_redis
        
        # L1: Caché en memoria con LRU
        self.memory_cache: OrderedDict = OrderedDict()
        self.memory_lock = threading.RLock()
        
        # L2: Caché en disco
        if enable_disk:
            self.cache_dir.mkdir(exist_ok=True)
            self.stats_file = self.cache_dir / "cache_stats.json"
        
        # L3: Redis (opcional)
        self.redis_client = None
        if enable_redis:
            try:
                import redis
                self.redis_client = redis.Redis(
                    host='localhost',
                    port=6379,
                    db=0,
                    decode_responses=False
                )
                self.redis_client.ping()
                logger.info("✓ Redis conectado")
            except (ImportError, Exception) as e:
                logger.warning(f"⚠️ Redis no disponible: {e}")
                self.enable_redis = False
        
        self._load_stats()
        logger.info(f"✓ Cache inicializado: max_items={max_memory_items}, ttl={ttl_seconds}s")
    
    def _hash_input(self, image_bytes: bytes) -> str:
        """Generar hash único para imagen"""
        return hashlib.sha256(image_bytes).hexdigest()
    
    def _get_disk_path(self, key: str) -> Path:
        """Ruta del archivo de caché en disco"""
        return self.cache_dir / f"{key}.pkl"
    
    def _load_stats(self):
        """Cargar estadísticas de caché"""
        if self.enable_disk and self.stats_file.exists():
            try:
                with open(self.stats_file) as f:
                    self.stats = json.load(f)
            except:
                self.stats = {"hits": 0, "misses": 0, "total_size_mb": 0}
        else:
            self.stats = {"hits": 0, "misses": 0, "total_size_mb": 0}
    
    def _save_stats(self):
        """Guardar estadísticas de caché"""
        if self.enable_disk:
            try:
                with open(self.stats_file, 'w') as f:
                    json.dump(self.stats, f, indent=2)
            except:
                pass
    
    def get(self, image_bytes: bytes) -> Optional[Dict[str, Any]]:
        """
        Obtener predicción del caché
        Busca en: L1 (memoria) → L3 (Redis) → L2 (disco)
        """
        key = self._hash_input(image_bytes)
        
        # L1: Memoria
        with self.memory_lock:
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if self._is_valid(entry):
                    self.memory_cache.move_to_end(key)  # LRU
                    self.stats["hits"] += 1
                    logger.debug(f"✓ Cache hit (memory): {key[:8]}...")
                    return entry["result"]
                else:
                    del self.memory_cache[key]
        
        # L3: Redis
        if self.enable_redis:
            try:
                data = self.redis_client.get(f"pred:{key}")
                if data:
                    entry = pickle.loads(data)
                    if self._is_valid(entry):
                        self.stats["hits"] += 1
                        logger.debug(f"✓ Cache hit (Redis): {key[:8]}...")
                        # Llevar a memoria
                        self._add_to_memory(key, entry)
                        return entry["result"]
            except Exception as e:
                logger.debug(f"Redis get error: {e}")
        
        # L2: Disco
        if self.enable_disk:
            disk_path = self._get_disk_path(key)
            if disk_path.exists():
                try:
                    with open(disk_path, 'rb') as f:
                        entry = pickle.load(f)
                    if self._is_valid(entry):
                        self.stats["hits"] += 1
                        logger.debug(f"✓ Cache hit (disk): {key[:8]}...")
                        # Llevar a memoria y Redis
                        self._add_to_memory(key, entry)
                        if self.enable_redis:
                            self.redis_client.setex(
                                f"pred:{key}",
                                self.ttl_seconds,
                                pickle.dumps(entry)
                            )
                        return entry["result"]
                except Exception as e:
                    logger.warning(f"Error reading cache: {e}")
        
        self.stats["misses"] += 1
        self._save_stats()
        return None
    
    def set(
        self,
        image_bytes: bytes,
        result: Dict[str, Any],
        metadata: Optional[Dict] = None
    ):
        """Guardar predicción en caché"""
        key = self._hash_input(image_bytes)
        
        entry = {
            "key": key,
            "result": result,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        # L1: Memoria
        self._add_to_memory(key, entry)
        
        # L3: Redis
        if self.enable_redis:
            try:
                self.redis_client.setex(
                    f"pred:{key}",
                    self.ttl_seconds,
                    pickle.dumps(entry)
                )
            except Exception as e:
                logger.debug(f"Redis set error: {e}")
        
        # L2: Disco
        if self.enable_disk:
            try:
                disk_path = self._get_disk_path(key)
                with open(disk_path, 'wb') as f:
                    pickle.dump(entry, f)
                
                # Actualizar tamaño
                size_mb = disk_path.stat().st_size / (1024 * 1024)
                self.stats["total_size_mb"] = sum(
                    p.stat().st_size for p in self.cache_dir.glob("*.pkl")
                ) / (1024 * 1024)
            except Exception as e:
                logger.warning(f"Error saving to disk: {e}")
        
        self._save_stats()
    
    def _add_to_memory(self, key: str, entry: Dict):
        """Agregar a caché en memoria (LRU)"""
        with self.memory_lock:
            if key in self.memory_cache:
                del self.memory_cache[key]
            
            self.memory_cache[key] = entry
            
            # Evitar LRU si excede tamaño
            while len(self.memory_cache) > self.max_memory_items:
                self.memory_cache.popitem(last=False)
    
    def _is_valid(self, entry: Dict) -> bool:
        """Verificar si entrada es válida (no expirada)"""
        age = time.time() - entry.get("timestamp", 0)
        return age < self.ttl_seconds
    
    def clear_expired(self):
        """Limpiar entradas expiradas"""
        count = 0
        
        # Memoria
        with self.memory_lock:
            expired = [k for k, v in self.memory_cache.items() 
                      if not self._is_valid(v)]
            for k in expired:
                del self.memory_cache[k]
                count += 1
        
        # Disco
        if self.enable_disk:
            for pkl_file in self.cache_dir.glob("*.pkl"):
                try:
                    with open(pkl_file, 'rb') as f:
                        entry = pickle.load(f)
                    if not self._is_valid(entry):
                        pkl_file.unlink()
                        count += 1
                except:
                    pass
        
        logger.info(f"🗑️ Caché limpio: {count} entradas expiradas eliminadas")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de caché"""
        hit_rate = (
            self.stats["hits"] / (self.stats["hits"] + self.stats["misses"])
            if (self.stats["hits"] + self.stats["misses"]) > 0
            else 0
        )
        
        with self.memory_lock:
            memory_items = len(self.memory_cache)
        
        disk_items = len(list(self.cache_dir.glob("*.pkl"))) if self.enable_disk else 0
        
        return {
            "hit_rate": f"{hit_rate:.1%}",
            "total_hits": self.stats["hits"],
            "total_misses": self.stats["misses"],
            "memory_items": memory_items,
            "disk_items": disk_items,
            "disk_size_mb": self.stats["total_size_mb"],
            "redis_enabled": self.enable_redis
        }
    
    def clear_all(self):
        """Limpiar todo el caché"""
        with self.memory_lock:
            self.memory_cache.clear()
        
        if self.enable_disk:
            for f in self.cache_dir.glob("*.pkl"):
                f.unlink()
        
        self.stats = {"hits": 0, "misses": 0, "total_size_mb": 0}
        self._save_stats()
        logger.info("✓ Caché completamente limpio")
