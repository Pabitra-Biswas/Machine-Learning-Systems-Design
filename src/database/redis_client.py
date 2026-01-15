"""
Redis Caching Layer - Production Grade
Handles async/sync properly, with connection pooling and error recovery
"""

import redis
import redis.asyncio as aioredis
import json
import hashlib
import logging
import asyncio
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class RedisCache:
    """
    Redis cache with proper async/sync handling, connection pooling, 
    and graceful degradation on failures.
    
    IMPORTANT: Use `async with RedisCache(...) as cache:` for proper cleanup!
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        ttl: int = 3600,
        max_connections: int = 10,
        socket_timeout: int = 5,
        retry_on_failure: bool = True
    ):
        """
        Initialize Redis cache
        
        Args:
            redis_url: Redis connection string (redis://host:port/db)
            ttl: Time-to-live for cached items (seconds)
            max_connections: Connection pool size
            socket_timeout: Socket timeout (seconds)
            retry_on_failure: Graceful degradation on Redis failure
        """
        
        self.redis_url = redis_url
        self.ttl = ttl
        self.socket_timeout = socket_timeout
        self.retry_on_failure = retry_on_failure
        self.max_connections = max_connections
        
        # Connection pool for async operations
        self.connection_pool = None
        self.client = None
        self._connected = False
        self._connection_attempts = 0
        self._max_retry_attempts = 3
        
        logger.info(f"🔴 RedisCache initialized (not connected yet)")
        logger.info(f"   Redis URL: {redis_url}")
        logger.info(f"   TTL: {ttl}s, Pool size: {max_connections}")
    
    async def connect(self) -> bool:
        """
        Establish async Redis connection.
        Call this in your FastAPI startup event!
        """
        
        try:
            logger.info("🔄 Connecting to Redis...")
            
            # Create connection pool
            self.connection_pool = aioredis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=self.max_connections,
                socket_connect_timeout=self.socket_timeout,
                socket_keepalive=True,
                socket_keepalive_options={
                    1: 1,  # TCP_KEEPIDLE
                    2: 1,  # TCP_KEEPINTVL
                    3: 3,  # TCP_KEEPCNT
                }
            )
            
            # Create async client
            self.client = aioredis.Redis(connection_pool=self.connection_pool)
            
            # Test connection
            ping = await self.client.ping()
            
            if ping:
                self._connected = True
                self._connection_attempts = 0
                logger.info("✅ Redis connected successfully!")
                return True
            else:
                raise ConnectionError("Redis ping failed")
        
        except Exception as e:
            self._connection_attempts += 1
            self._connected = False
            
            logger.error(f"❌ Redis connection failed: {e}")
            logger.error(f"   Attempt {self._connection_attempts}/{self._max_retry_attempts}")
            
            if self._connection_attempts < self._max_retry_attempts:
                # Retry with exponential backoff
                wait_time = 2 ** self._connection_attempts
                logger.warning(f"   Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
                return await self.connect()
            else:
                logger.critical("❌ Redis connection failed after all retries!")
                return False
    
    @staticmethod
    def _generate_key(text: str) -> str:
        """
        Generate cache key from text hash
        
        Format: pred:<hash[:16]>
        Example: pred:a1b2c3d4e5f6g7h8
        """
        hash_digest = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
        return f"pred:{hash_digest}"
    
    async def get(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Get cached prediction result
        
        Args:
            text: Input text (will be hashed for cache key)
        
        Returns:
            Cached prediction dict or None if not found
        """
        
        # Graceful degradation: return None if Redis unavailable
        if not self._connected or self.client is None:
            logger.debug("⚠️  Redis not connected - cache MISS")
            return None
        
        try:
            key = self._generate_key(text)
            cached_json = await asyncio.wait_for(
                self.client.get(key),
                timeout=self.socket_timeout
            )
            
            if cached_json:
                cached_data = json.loads(cached_json)
                logger.debug(f"✅ Cache HIT: {key[:20]}...")
                return cached_data
            else:
                logger.debug(f"⚠️  Cache MISS: {key[:20]}...")
                return None
        
        except asyncio.TimeoutError:
            logger.warning(f"⏱️  Redis GET timeout: {key[:20]}...")
            return None
        
        except json.JSONDecodeError as e:
            logger.error(f"❌ Cache corruption: {e}")
            try:
                await self.client.delete(key)
                logger.info(f"🧹 Deleted corrupted cache: {key}")
            except:
                pass
            return None
        
        except Exception as e:
            logger.error(f"❌ Cache GET error: {e}")
            return None
    
    async def set(self, text: str, prediction: Dict[str, Any]) -> bool:
        """
        Cache a prediction result
        
        Args:
            text: Input text
            prediction: Prediction dict to cache
        
        Returns:
            True if cached successfully, False otherwise
        """
        
        # Graceful degradation: return True even if Redis fails
        if not self._connected or self.client is None:
            logger.debug("⚠️  Redis not connected - cache SET skipped")
            return True  # ← Important: don't break the request!
        
        try:
            key = self._generate_key(text)
            json_str = json.dumps(prediction)
            
            # Set with TTL
            success = await asyncio.wait_for(
                self.client.setex(
                    key,
                    self.ttl,
                    json_str
                ),
                timeout=self.socket_timeout
            )
            
            if success:
                logger.debug(f"✅ Cache SET: {key[:20]}... (TTL={self.ttl}s)")
                return True
            else:
                logger.warning(f"⚠️  Cache SET returned False: {key[:20]}...")
                return False
        
        except asyncio.TimeoutError:
            logger.warning(f"⏱️  Redis SET timeout: {key[:20]}...")
            return True  # Don't fail the request
        
        except json.JSONDecodeError as e:
            logger.error(f"❌ Cache serialization error: {e}")
            return False
        
        except Exception as e:
            logger.error(f"❌ Cache SET error: {e}")
            return True  # Don't fail the request
    
    async def delete(self, text: str) -> bool:
        """Delete cached prediction"""
        
        if not self._connected or self.client is None:
            return True
        
        try:
            key = self._generate_key(text)
            await self.client.delete(key)
            logger.debug(f"🗑️  Cache deleted: {key[:20]}...")
            return True
        except Exception as e:
            logger.error(f"❌ Cache DELETE error: {e}")
            return True
    
    async def flush_all(self) -> bool:
        """Clear all cache (use with caution!)"""
        
        if not self._connected or self.client is None:
            return False
        
        try:
            await self.client.flushdb()
            logger.warning("🗑️  Cache flushed completely!")
            return True
        except Exception as e:
            logger.error(f"❌ Cache FLUSH error: {e}")
            return False
    
    def is_connected(self) -> bool:
        """
        Check if Redis is connected (SYNCHRONOUS)
        
        Returns:
            True if connected, False otherwise
        """
        return self._connected
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Detailed health check for monitoring
        
        Returns:
            Health status dict
        """
        
        if not self._connected or self.client is None:
            return {
                "status": "disconnected",
                "connected": False,
                "message": "Redis not initialized"
            }
        
        try:
            info = await asyncio.wait_for(
                self.client.info(),
                timeout=self.socket_timeout
            )
            
            return {
                "status": "healthy",
                "connected": True,
                "used_memory_mb": info.get('used_memory', 0) / 1024 / 1024,
                "connected_clients": info.get('connected_clients', 0),
                "keyspace": info.get('db0', {}).get('keys', 0),
                "evicted_keys": info.get('evicted_keys', 0)
            }
        
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e)
            }
    
    async def close(self):
        """
        Properly close Redis connection.
        Call this in your FastAPI shutdown event!
        """
        
        try:
            if self.client:
                await self.client.close()
            
            if self.connection_pool:
                await self.connection_pool.disconnect()
            
            self._connected = False
            logger.info("✅ Redis connection closed")
        
        except Exception as e:
            logger.error(f"❌ Error closing Redis: {e}")
    
    @asynccontextmanager
    async def context(self):
        """
        Context manager for automatic connection/cleanup
        
        Usage:
            async with RedisCache(...).context() as cache:
                result = await cache.get(text)
        """
        await self.connect()
        try:
            yield self
        finally:
            await self.close()


# ============================================
# SYNC WRAPPER (for FastAPI compatibility)
# ============================================

class RedisCacheSync:
    """
    Synchronous wrapper around async RedisCache
    Use this if you MUST use sync code (not recommended!)
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379", ttl: int = 3600):
        self.redis_url = redis_url
        self.ttl = ttl
        self.cache = redis.Redis.from_url(redis_url, decode_responses=True)
        logger.info(f"✅ Sync Redis initialized: {redis_url}")
    
    def get(self, text: str) -> Optional[Dict]:
        """Synchronous get"""
        try:
            from src.database.redis_client import RedisCache
            key = RedisCache._generate_key(text)
            cached = self.cache.get(key)
            
            if cached:
                logger.debug(f"✅ Cache HIT: {key[:20]}...")
                return json.loads(cached)
            
            return None
        
        except Exception as e:
            logger.error(f"❌ Sync cache GET error: {e}")
            return None
    
    def set(self, text: str, prediction: Dict) -> bool:
        """Synchronous set"""
        try:
            from src.database.redis_client import RedisCache
            key = RedisCache._generate_key(text)
            self.cache.setex(key, self.ttl, json.dumps(prediction))
            
            logger.debug(f"✅ Cache SET: {key[:20]}...")
            return True
        
        except Exception as e:
            logger.error(f"❌ Sync cache SET error: {e}")
            return True  # Don't break the request
    
    def is_connected(self) -> bool:
        """Check connection"""
        try:
            self.cache.ping()
            return True
        except:
            return False
    
    def close(self):
        """Close connection"""
        try:
            self.cache.close()
            logger.info("✅ Sync Redis closed")
        except Exception as e:
            logger.error(f"❌ Error closing sync Redis: {e}")