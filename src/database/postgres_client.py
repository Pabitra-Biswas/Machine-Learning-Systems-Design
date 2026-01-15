"""
PostgreSQL Prediction Logger - Production Grade
Async pool management, proper error handling, monitoring metrics
Week 5-6: Production Deployment
"""

import asyncpg
import hashlib
import logging
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from contextlib import asynccontextmanager
import asyncio

logger = logging.getLogger(__name__)


class PostgresLogger:
    """
    Async PostgreSQL logger for prediction tracking & analytics
    
    Features:
    - Connection pooling with auto-retry
    - Prediction logging with hashing
    - Real-time analytics queries
    - Health monitoring
    - Graceful degradation on failures
    """
    
    def __init__(
        self,
        host: str,
        database: str,
        user: str,
        password: str,
        port: int = 5432,
        min_size: int = 2,
        max_size: int = 10,
        timeout: int = 10
    ):
        """
        Initialize PostgreSQL logger
        
        Args:
            host: Database host
            database: Database name
            user: Database user
            password: Database password
            port: Database port (default 5432)
            min_size: Min connection pool size
            max_size: Max connection pool size
            timeout: Connection timeout (seconds)
        """
        
        self.config = {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password,
            "timeout": timeout
        }
        
        self.pool: Optional[asyncpg.Pool] = None
        self.min_size = min_size
        self.max_size = max_size
        self._connected = False
        self._connection_attempts = 0
        self._max_retry_attempts = 3
        
        logger.info(f"🔴 PostgresLogger initialized (not connected yet)")
        logger.info(f"   Database: {host}/{database}")
    
    async def connect(self) -> bool:
        """
        Establish PostgreSQL connection pool
        Call this in your FastAPI startup event!
        """
        
        try:
            logger.info("🔄 Connecting to PostgreSQL...")
            
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                **self.config,
                min_size=self.min_size,
                max_size=self.max_size,
                # Connection validation
                init=self._init_connection,
                # SSL if needed
                ssl=None
            )
            
            # Test connection
            async with self.pool.acquire() as conn:
                result = await conn.fetchval('SELECT NOW()')
                
                if result:
                    self._connected = True
                    self._connection_attempts = 0
                    
                    logger.info("✅ PostgreSQL connected successfully!")
                    
                    # Create tables
                    await self._create_tables()
                    
                    return True
                else:
                    raise ConnectionError("PostgreSQL ping failed")
        
        except Exception as e:
            self._connection_attempts += 1
            self._connected = False
            
            logger.error(f"❌ PostgreSQL connection failed: {e}")
            logger.error(f"   Attempt {self._connection_attempts}/{self._max_retry_attempts}")
            
            if self._connection_attempts < self._max_retry_attempts:
                # Retry with exponential backoff
                wait_time = 2 ** self._connection_attempts
                logger.warning(f"   Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
                return await self.connect()
            else:
                logger.critical("❌ PostgreSQL connection failed after all retries!")
                return False
    
    @staticmethod
    async def _init_connection(conn):
        """Initialize each connection with settings"""
        await conn.execute('SET timezone = "UTC"')
    
    async def _create_tables(self):
        """
        Create prediction log table and indices
        
        Table structure:
        - id: Auto-incrementing primary key
        - timestamp: When prediction was made
        - text_hash: SHA256 hash of input text
        - text_preview: First 200 chars of input
        - predicted_topic: Model output
        - confidence: Prediction confidence (0-1)
        - latency_ms: Inference time
        - ip_address: Client IP (for tracking)
        - user_agent: Client info
        - model_version: Which model version made prediction
        - cached: Whether result was cached
        """
        
        if not self.pool:
            logger.error("❌ Pool not initialized - cannot create tables")
            return
        
        try:
            async with self.pool.acquire() as conn:
                
                # Main predictions table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id BIGSERIAL PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
                        text_hash VARCHAR(64) NOT NULL,
                        text_preview TEXT NOT NULL,
                        predicted_topic VARCHAR(50) NOT NULL,
                        confidence NUMERIC(4,3) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
                        latency_ms NUMERIC(8,2) NOT NULL,
                        ip_address INET,
                        user_agent TEXT,
                        model_version VARCHAR(50) DEFAULT 'unknown',
                        cached BOOLEAN DEFAULT FALSE
                    );
                """)
                
                # Create indices for fast queries
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_predictions_timestamp 
                    ON predictions(timestamp DESC);
                """)
                
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_predictions_topic 
                    ON predictions(predicted_topic);
                """)
                
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_predictions_confidence 
                    ON predictions(confidence DESC);
                """)
                
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_predictions_text_hash 
                    ON predictions(text_hash);
                """)
                
                # Daily stats table (for performance)
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS prediction_stats_daily (
                        date DATE PRIMARY KEY,
                        total_predictions BIGINT,
                        avg_confidence NUMERIC(4,3),
                        avg_latency_ms NUMERIC(8,2),
                        topic_distribution JSONB
                    );
                """)
                
                logger.info("✅ Database tables created")
        
        except Exception as e:
            logger.error(f"❌ Failed to create tables: {e}")
            raise
    
    @staticmethod
    def _hash_text(text: str) -> str:
        """
        Generate SHA256 hash of input text (SYNCHRONOUS)
        Safe to call from async context
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    async def log_prediction(
        self,
        text: str,
        prediction: str,
        confidence: float,
        latency_ms: float,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        model_version: str = "unknown",
        cached: bool = False
    ) -> bool:
        """
        Log prediction to database
        
        Args:
            text: Input article title
            prediction: Predicted topic
            confidence: Confidence score (0-1)
            latency_ms: Inference latency
            ip_address: Client IP (optional)
            user_agent: Client user agent (optional)
            model_version: Model identifier
            cached: Whether result was cached
        
        Returns:
            True if logged successfully, False otherwise
        """
        
        # Graceful degradation: return True even if logging fails
        if not self._connected or self.pool is None:
            logger.debug("⚠️  PostgreSQL not connected - logging skipped")
            return True  # ← Important: don't break the request!
        
        try:
            # Hash text synchronously (fast)
            text_hash = self._hash_text(text)
            text_preview = text[:200] if len(text) > 200 else text
            
            # Insert with timeout
            async with self.pool.acquire() as conn:
                await asyncio.wait_for(
                    conn.execute("""
                        INSERT INTO predictions 
                        (text_hash, text_preview, predicted_topic, confidence, latency_ms, 
                         ip_address, user_agent, model_version, cached)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,
                        text_hash,
                        text_preview,
                        prediction,
                        float(confidence),
                        float(latency_ms),
                        ip_address,
                        user_agent,
                        model_version,
                        cached
                    ),
                    timeout=5.0  # 5 second timeout
                )
            
            logger.debug(f"✅ Logged: {prediction} ({confidence:.1%}) - {latency_ms:.0f}ms")
            return True
        
        except asyncio.TimeoutError:
            logger.warning(f"⏱️  PostgreSQL INSERT timeout")
            return True  # Don't fail the request
        
        except Exception as e:
            logger.error(f"❌ Failed to log prediction: {e}")
            return True  # Don't fail the request
    
    async def get_stats(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get prediction statistics for last N hours
        
        Args:
            hours: Look back window (default 24)
        
        Returns:
            Dict with stats per topic
        """
        
        if not self._connected or self.pool is None:
            logger.warning("⚠️  PostgreSQL not connected - returning empty stats")
            return {}
        
        try:
            async with self.pool.acquire() as conn:
                # FIXED: Use INTERVAL proper syntax
                rows = await conn.fetch("""
                    SELECT 
                        predicted_topic,
                        COUNT(*) as count,
                        AVG(confidence) as avg_confidence,
                        AVG(latency_ms) as avg_latency,
                        MIN(confidence) as min_confidence,
                        MAX(confidence) as max_confidence,
                        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) as p95_latency
                    FROM predictions
                    WHERE timestamp > NOW() - INTERVAL '1 hour' * $1
                    GROUP BY predicted_topic
                    ORDER BY count DESC
                """, hours)
            
            # Convert to readable format
            stats = {}
            total_predictions = 0
            
            for row in rows:
                topic = row['predicted_topic']
                count = row['count']
                total_predictions += count
                
                stats[topic] = {
                    'count': count,
                    'avg_confidence': float(row['avg_confidence']) if row['avg_confidence'] else 0,
                    'avg_latency_ms': float(row['avg_latency']) if row['avg_latency'] else 0,
                    'min_confidence': float(row['min_confidence']) if row['min_confidence'] else 0,
                    'max_confidence': float(row['max_confidence']) if row['max_confidence'] else 0,
                    'p95_latency_ms': float(row['p95_latency']) if row['p95_latency'] else 0,
                }
            
            # Add summary
            stats['_summary'] = {
                'total_predictions': total_predictions,
                'num_topics': len(stats) - 1,  # Exclude _summary
                'window_hours': hours,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info(f"✅ Stats: {total_predictions} predictions in {hours}h")
            return stats
        
        except Exception as e:
            logger.error(f"❌ Failed to get stats: {e}")
            return {}
    
    async def get_low_confidence_predictions(
        self,
        threshold: float = 0.7,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get recent low-confidence predictions (for debugging)
        
        Args:
            threshold: Confidence threshold (default 0.7)
            limit: Max results
        
        Returns:
            List of low-confidence predictions
        """
        
        if not self._connected or self.pool is None:
            return []
        
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT 
                        id, timestamp, text_preview, predicted_topic, 
                        confidence, latency_ms, model_version
                    FROM predictions
                    WHERE confidence < $1
                    ORDER BY timestamp DESC
                    LIMIT $2
                """, threshold, limit)
            
            return [dict(row) for row in rows]
        
        except Exception as e:
            logger.error(f"❌ Failed to get low-confidence predictions: {e}")
            return []
    
    async def get_topic_accuracy(
        self,
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get per-topic accuracy metrics (if you have ground truth labels)
        
        Args:
            hours: Look back window
        
        Returns:
            Accuracy metrics per topic
        """
        
        if not self._connected or self.pool is None:
            return {}
        
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT 
                        predicted_topic,
                        COUNT(*) as total,
                        AVG(confidence) as avg_confidence,
                        COUNT(CASE WHEN confidence > 0.8 THEN 1 END) as high_confidence_count,
                        COUNT(CASE WHEN confidence < 0.7 THEN 1 END) as low_confidence_count
                    FROM predictions
                    WHERE timestamp > NOW() - INTERVAL '1 hour' * $1
                    GROUP BY predicted_topic
                    ORDER BY total DESC
                """, hours)
            
            metrics = {}
            for row in rows:
                topic = row['predicted_topic']
                total = row['total']
                
                metrics[topic] = {
                    'total_predictions': total,
                    'avg_confidence': float(row['avg_confidence']) if row['avg_confidence'] else 0,
                    'high_confidence_pct': (row['high_confidence_count'] / total * 100) if total > 0 else 0,
                    'low_confidence_pct': (row['low_confidence_count'] / total * 100) if total > 0 else 0,
                }
            
            return metrics
        
        except Exception as e:
            logger.error(f"❌ Failed to get topic accuracy: {e}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Detailed health check for monitoring
        
        Returns:
            Health status dict
        """
        
        if not self._connected or self.pool is None:
            return {
                "status": "disconnected",
                "connected": False,
                "message": "PostgreSQL not initialized"
            }
        
        try:
            async with self.pool.acquire() as conn:
                # Test query
                server_time = await conn.fetchval('SELECT NOW()')
                
                # Get pool stats
                pool_size = self.pool.get_size()
                pool_idle = self.pool.get_idle_size()
                
                # Get table sizes
                table_info = await conn.fetchrow("""
                    SELECT 
                        (SELECT COUNT(*) FROM predictions) as total_rows,
                        pg_total_relation_size('predictions') as table_size
                """)
                
                return {
                    "status": "healthy",
                    "connected": True,
                    "server_time": server_time.isoformat(),
                    "pool_size": pool_size,
                    "pool_idle": pool_idle,
                    "total_predictions": table_info['total_rows'],
                    "table_size_mb": table_info['table_size'] / 1024 / 1024 if table_info['table_size'] else 0
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
        Close connection pool gracefully
        Call this in your FastAPI shutdown event!
        """
        
        try:
            if self.pool:
                await self.pool.close()
            
            self._connected = False
            logger.info("✅ PostgreSQL connection pool closed")
        
        except Exception as e:
            logger.error(f"❌ Error closing PostgreSQL: {e}")
    
    @asynccontextmanager
    async def context(self):
        """
        Context manager for automatic connection/cleanup
        
        Usage:
            async with PostgresLogger(...).context() as logger:
                await logger.log_prediction(...)
        """
        await self.connect()
        try:
            yield self
        finally:
            await self.close()