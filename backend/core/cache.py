"""
Simple Cache Management System
Provides a basic caching interface for the application
"""

import asyncio
import json
from typing import Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict

from core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with TTL support"""
    value: Any
    expires_at: datetime
    created_at: datetime
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return datetime.now() > self.expires_at


class CacheManager:
    """Simple in-memory cache with TTL support"""
    
    def __init__(self):
        self._cache = {}
        self._stats = defaultdict(int)
        
        # Start cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
        
    def _start_cleanup_task(self):
        """Start periodic cleanup of expired entries"""
        try:
            loop = asyncio.get_event_loop()
            self._cleanup_task = loop.create_task(self._cleanup_expired())
        except RuntimeError:
            # No event loop running, cleanup will start later
            pass
    
    async def _cleanup_expired(self):
        """Background task to clean up expired entries"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                expired_keys = []
                now = datetime.now()
                
                for key, entry in self._cache.items():
                    if isinstance(entry, CacheEntry) and entry.is_expired():
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self._cache[key]
                
                if expired_keys:
                    logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                    
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            if key not in self._cache:
                self._stats["misses"] += 1
                return None
            
            entry = self._cache[key]
            
            # Handle direct values (for backward compatibility)
            if not isinstance(entry, CacheEntry):
                self._stats["hits"] += 1
                return entry
            
            # Check if entry is expired
            if entry.is_expired():
                del self._cache[key]
                self._stats["misses"] += 1
                return None
            
            self._stats["hits"] += 1
            return entry.value
            
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """Set value in cache with TTL (seconds)"""
        try:
            expires_at = datetime.now() + timedelta(seconds=ttl)
            
            entry = CacheEntry(
                value=value,
                expires_at=expires_at,
                created_at=datetime.now()
            )
            
            self._cache[key] = entry
            self._stats["sets"] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            if key in self._cache:
                del self._cache[key]
                self._stats["deletes"] += 1
                return True
            return False
            
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries"""
        try:
            self._cache.clear()
            self._stats["clears"] += 1
            return True
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        return {
            "entries": len(self._cache),
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "sets": self._stats["sets"],
            "deletes": self._stats["deletes"],
            "clears": self._stats["clears"],
            "hit_rate": (
                self._stats["hits"] / max(self._stats["hits"] + self._stats["misses"], 1)
            ) * 100
        }
    
    def get_keys(self) -> list:
        """Get all cache keys"""
        return list(self._cache.keys())
    
    def __del__(self):
        """Cleanup when cache manager is destroyed"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()


# Global cache instance
_cache_instance = None

def get_cache_manager() -> CacheManager:
    """Get global cache manager instance"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CacheManager()
    return _cache_instance