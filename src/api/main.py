"""
FastAPI application - WITH Redis Caching + PostgreSQL Logging
WEEK 5-6: Production-Grade Deployment
"""

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import logging
import time
import os
import asyncio
from datetime import datetime

from src.models.bert_model import BERTNewsClassifier
from src.database.redis_client import RedisCache          # ← NEW
from src.database.postgres_client import PostgresLogger    # ← NEW
from src.config.settings import settings
# from src.api.routes import batch_predict 

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="News Classifier API",
    version="2.0.0",
    description="BERT-based news classification with Redis caching + PostgreSQL logging"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In your app setup (after CORS middleware)
# app.include_router(
#     batch_predict.router,
#     prefix="/predict/batch",
#     tags=["Batch Predictions"]
# )

# ============================================
# GLOBAL INSTANCES (Singleton Pattern)
# ============================================

model = None
redis_cache = None        # ← NEW
postgres_logger = None    # ← NEW

# ============================================
# REQUEST/RESPONSE MODELS
# ============================================

class PredictRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=512, description="Article title")
    use_cache: bool = Field(True, description="Use Redis caching")

class PredictResponse(BaseModel):
    title: str
    topic: str
    confidence: float
    probabilities: dict
    latency_ms: float
    cached: bool = False  # ← NEW: Show if result was cached

class BatchPredictionRequest(BaseModel):
    titles: List[str] = Field(..., max_items=100)

class BatchPredictionResponse(BaseModel):
    count: int
    predictions: List[dict]
    latency_ms: float

# ============================================
# STARTUP/SHUTDOWN EVENTS
# ============================================

@app.on_event("startup")
async def startup():
    """Initialize model, Redis, PostgreSQL on startup"""
    global model, redis_cache, postgres_logger
    
    logger.info("="*70)
    logger.info("🚀 STARTING NEWS CLASSIFIER API")
    logger.info("="*70)
    
    # ============================================
    # 1. LOAD BERT MODEL
    # ============================================
    try:
        kaggle_zip = os.getenv("KAGGLE_MODEL_ZIP", None)
        
        if kaggle_zip and os.path.exists(kaggle_zip):
            logger.info(f"📦 Loading from Kaggle ZIP: {kaggle_zip}")
            model = BERTNewsClassifier(
                kaggle_zip_path=kaggle_zip,
                device=os.getenv("DEVICE", None)
            )
        else:
            logger.info(f"📦 Loading from: {settings.MODEL_PATH}")
            model = BERTNewsClassifier(
                model_path=settings.MODEL_PATH,
                device=os.getenv("DEVICE", None)
            )
        
        logger.info(f"✅ BERT Model loaded")
        # ✅ NEW: Pass model to batch_predict router
        # batch_predict.set_model(model)
        # logger.info("✅ Batch predict model initialized")
        logger.info(f"   Classes: {model.classes}")
        logger.info(f"   Device: {model.device}")
        
    except Exception as e:
        logger.error(f"❌ Failed to load BERT model: {e}")
        raise
    
    # ============================================
    # 2. INITIALIZE REDIS CACHE (NEW!)
    # ============================================
    try:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        cache_ttl = int(os.getenv("CACHE_TTL", "3600"))
        
        redis_cache = RedisCache(
            redis_url=redis_url,
            ttl=cache_ttl,
            max_connections=int(os.getenv("REDIS_POOL_SIZE", "10"))
        )
        
        # Connect to Redis
        connected = await redis_cache.connect()
        
        if connected:
            logger.info(f"✅ Redis cache connected")
            logger.info(f"   URL: {redis_url}")
            logger.info(f"   TTL: {cache_ttl}s")
        else:
            logger.warning(f"⚠️  Redis unavailable - caching disabled")
            redis_cache = None
    
    except Exception as e:
        logger.error(f"❌ Redis initialization failed: {e}")
        logger.warning("⚠️  Continuing without caching...")
        redis_cache = None
    
    # ============================================
    # 3. INITIALIZE POSTGRESQL LOGGER (NEW!)
    # ============================================
    try:
        postgres_logger = PostgresLogger(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            database=os.getenv("POSTGRES_DB", "news_classifier"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres"),
            port=int(os.getenv("POSTGRES_PORT", "5432"))
        )
        
        # Connect to PostgreSQL
        connected = await postgres_logger.connect()
        
        if connected:
            logger.info(f"✅ PostgreSQL logger connected")
            logger.info(f"   Database: {os.getenv('POSTGRES_DB', 'news_classifier')}")
        else:
            logger.warning(f"⚠️  PostgreSQL unavailable - logging disabled")
            postgres_logger = None
    
    except Exception as e:
        logger.error(f"❌ PostgreSQL initialization failed: {e}")
        logger.warning("⚠️  Continuing without database logging...")
        postgres_logger = None
    
    # ============================================
    # STARTUP SUMMARY
    # ============================================
    logger.info("="*70)
    logger.info("✅ API STARTUP COMPLETE")
    logger.info("="*70)
    logger.info(f"Model: {'✅ Loaded' if model else '❌ Failed'}")
    logger.info(f"Redis: {'✅ Connected' if redis_cache else '⚠️  Disabled'}")
    logger.info(f"PostgreSQL: {'✅ Connected' if postgres_logger else '⚠️  Disabled'}")
    logger.info("="*70)

@app.on_event("shutdown")
async def shutdown():
    """Close connections on shutdown"""
    global redis_cache, postgres_logger
    
    logger.info("🛑 Shutting down API...")
    
    # Close Redis
    if redis_cache:
        await redis_cache.close()
        logger.info("✅ Redis connection closed")
    
    # Close PostgreSQL
    if postgres_logger:
        await postgres_logger.close()
        logger.info("✅ PostgreSQL connection closed")
    
    logger.info("✅ API shutdown complete")

# ============================================
# HEALTH CHECK ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "News Classifier API",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "readiness": "/readiness",
            "info": "/info",
            "predict": "/predict",
            "batch": "/predict/batch"
        }
    }

@app.get("/health")
async def health():
    """Health check - basic liveness"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": model is not None
    }

@app.get("/readiness")
async def readiness():
    """Readiness probe for Kubernetes/Docker"""
    
    try:
        # Check if model is loaded
        if not model:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Check Redis connection
        redis_status = False
        if redis_cache:
            try:
                redis_status = redis_cache.is_connected()
            except:
                redis_status = False
        
        # Check PostgreSQL connection
        postgres_status = False
        if postgres_logger:
            try:
                # Option 1: If PostgresLogger has a method to check connection
                postgres_status = hasattr(postgres_logger, 'connection') and postgres_logger.connection is not None
                # OR
                # Option 2: Try a simple query
                # postgres_status = True  # Assume connected if initialized
            except:
                postgres_status = False
        
        all_ready = redis_status and postgres_status
        
        return {
            "ready": all_ready,
            "model_loaded": model is not None,
            "redis_connected": redis_status,
            "postgres_connected": postgres_status
        }
    
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/health/detailed")
async def health_detailed():
    """Detailed health check for all services"""
    
    health_status = {
        "timestamp": datetime.utcnow().isoformat(),
        "api": "healthy",
        "model": {
            "status": "loaded" if model else "not_loaded",
            "classes": model.classes if model else []
        },
        "redis": {
            "status": "connected" if (redis_cache and redis_cache.is_connected()) else "unavailable"
        },
        "postgres": {
            "status": "connected" if postgres_logger else "unavailable"
        }
    }
    
    # Add Redis health details if available
    if redis_cache and redis_cache.is_connected():
        try:
            redis_health = await redis_cache.health_check()
            health_status["redis"].update(redis_health)
        except:
            pass
    
    # Add PostgreSQL health details if available
    if postgres_logger:
        try:
            postgres_health = await postgres_logger.health_check()
            health_status["postgres"].update(postgres_health)
        except:
            pass
    
    return health_status

# ============================================
# INFO ENDPOINT
# ============================================

@app.get("/info")
async def info():
    """Model info endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": "DistilBERT (fine-tuned)",
        "num_classes": model.num_labels,
        "classes": model.classes,
        "version": "2.0.0",
        "features": {
            "caching": redis_cache is not None,
            "logging": postgres_logger is not None,
            "batch_prediction": True
        },
        "supported_endpoints": [
            "POST /predict - Single prediction (cached)",
            "POST /predict/batch - Batch prediction",
            "GET /health - Liveness check",
            "GET /readiness - Readiness check",
            "GET /health/detailed - Full system health",
            "GET /info - This endpoint"
        ]
    }

# ============================================
# PREDICTION ENDPOINTS
# ============================================

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest, x_api_key: Optional[str] = Header(None)):
    """
    Single article prediction with caching & logging
    
    Features:
    - Redis caching (40x speedup on repeated queries)
    - PostgreSQL logging (analytics & monitoring)
    - Graceful degradation (works even if cache/DB down)
    """
    
    # API key validation (optional)
    if x_api_key and x_api_key != os.getenv("API_KEY", ""):
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = time.time()
        cached = False
        
        # ============================================
        # 1. TRY REDIS CACHE (NEW!)
        # ============================================
        if request.use_cache and redis_cache and redis_cache.is_connected():
            try:
                cached_result = await redis_cache.get(request.title)
                
                if cached_result:
                    cached = True
                    latency_ms = (time.time() - start_time) * 1000
                    
                    logger.info(f"💾 Cache HIT: {request.title[:50]}... ({latency_ms:.0f}ms)")
                    
                    return PredictResponse(
                        title=request.title,
                        topic=cached_result['topic'],
                        confidence=cached_result['confidence'],
                        probabilities=cached_result['probabilities'],
                        latency_ms=latency_ms,
                        cached=True
                    )
            except Exception as e:
                logger.warning(f"⚠️  Cache read error: {e}")
                # Continue to model prediction
        
        # ============================================
        # 2. PREDICT WITH MODEL
        # ============================================
        result = model.predict(request.title)
        
        prediction_result = {
            'topic': result['topic'],
            'confidence': result['confidence'],
            'probabilities': result['probabilities']
        }
        
        # ============================================
        # 3. STORE IN REDIS CACHE (NEW! - Fire and forget)
        # ============================================
        if redis_cache and redis_cache.is_connected():
            try:
                # Don't await - fire and forget
                asyncio.create_task(
                    redis_cache.set(request.title, prediction_result)
                )
            except Exception as e:
                logger.warning(f"⚠️  Cache write error: {e}")
                # Don't break the request
        
        # ============================================
        # 4. LOG TO POSTGRESQL (NEW! - Fire and forget)
        # ============================================
        if postgres_logger:
            try:
                asyncio.create_task(
                    postgres_logger.log_prediction(
                        text=request.title,
                        prediction=result['topic'],
                        confidence=result['confidence'],
                        latency_ms=(time.time() - start_time) * 1000,
                        model_version="week4_bert_final",
                        cached=cached
                    )
                )
            except Exception as e:
                logger.warning(f"⚠️  Logging error: {e}")
                # Don't break the request
        
        latency_ms = (time.time() - start_time) * 1000
        
        logger.info(f"✅ Prediction: {request.title[:50]}... → {result['topic']} ({latency_ms:.0f}ms)")
        
        return PredictResponse(
            title=request.title,
            topic=result['topic'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            latency_ms=latency_ms,
            cached=False
        )
    
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """
    Batch prediction (no caching for batch)
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = time.time()
        
        # Batch predict
        predictions = model.predict_batch(request.titles)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Log batch predictions (fire and forget)
        if postgres_logger:
            try:
                for title, pred in zip(request.titles, predictions):
                    asyncio.create_task(
                        postgres_logger.log_prediction(
                            text=title,
                            prediction=pred['topic'],
                            confidence=pred['confidence'],
                            latency_ms=latency_ms / len(request.titles),
                            model_version="week4_bert_final"
                        )
                    )
            except:
                pass  # Don't break batch on logging error
        
        logger.info(f"✅ Batch prediction: {len(predictions)} articles ({latency_ms:.0f}ms)")
        
        return BatchPredictionResponse(
            count=len(predictions),
            predictions=predictions,
            latency_ms=latency_ms
        )
    
    except Exception as e:
        logger.error(f"❌ Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# ANALYTICS ENDPOINTS (NEW!)
# ============================================

@app.get("/stats")
async def stats(hours: int = 24):
    """Get prediction statistics from PostgreSQL"""
    
    if not postgres_logger:
        return {"error": "PostgreSQL not available"}
    
    try:
        stats = await postgres_logger.get_stats(hours=hours)
        return stats
    except Exception as e:
        logger.error(f"❌ Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cache/clear")
async def cache_clear():
    """Clear all cache (admin endpoint)"""
    
    if not redis_cache:
        return {"status": "redis_unavailable"}
    
    try:
        await redis_cache.flush_all()
        logger.warning("🗑️  Cache flushed by admin")
        return {"status": "cleared", "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"❌ Cache clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# RUN SERVER
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        reload=os.getenv("DEBUG", "False").lower() == "true"
    )

# from src.database.redis_client import RedisCache

# # Global cache instance
# redis_cache = None

# @app.on_event("startup")
# async def startup():
#     global redis_cache, model
    
#     logger.info("🚀 Starting News Classifier API...")
    
#     # Load model
#     model = BERTNewsClassifier(...)
#     logger.info("✅ Model loaded")
    
#     # Connect to Redis (NEW!)
#     try:
#         redis_cache = RedisCache(
#             redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
#             ttl=int(os.getenv("CACHE_TTL", "3600"))
#         )
        
#         # IMPORTANT: Await the connection!
#         connected = await redis_cache.connect()
        
#         if connected:
#             logger.info("✅ Redis cache connected")
#         else:
#             logger.warning("⚠️  Redis not available - caching disabled")
#             redis_cache = None
    
#     except Exception as e:
#         logger.error(f"❌ Redis initialization failed: {e}")
#         redis_cache = None  # Graceful degradation

# @app.on_event("shutdown")
# async def shutdown():
#     global redis_cache
    
#     logger.info("🛑 Shutting down...")
    
#     if redis_cache:
#         await redis_cache.close()

# # Health check endpoint
# @app.get("/health")
# async def health():
#     redis_status = "healthy" if redis_cache and redis_cache.is_connected() else "unavailable"
    
#     return {
#         "status": "healthy",
#         "model_loaded": model is not None,
#         "redis": redis_status
#     }

# # Prediction endpoint (UPDATED!)
# @app.post("/predict")
# async def predict(request: PredictionRequest):
#     start_time = time.time()
#     cached = False
    
#     try:
#         # Try cache first
#         if redis_cache and redis_cache.is_connected():
#             cached_result = await redis_cache.get(request.title)
            
#             if cached_result:
#                 cached = True
#                 latency_ms = (time.time() - start_time) * 1000
                
#                 return PredictionResponse(
#                     title=request.title,
#                     topic=cached_result['topic'],
#                     confidence=cached_result['confidence'],
#                     probabilities=cached_result['probabilities'],
#                     cached=True,
#                     latency_ms=latency_ms
#                 )
        
#         # Make prediction
#         prediction = model.predict(request.title)
        
#         result = {
#             'topic': prediction['topic'],
#             'confidence': prediction['confidence'],
#             'probabilities': prediction['probabilities']
#         }
        
#         # Cache result (fire-and-forget, don't wait)
#         if redis_cache and redis_cache.is_connected():
#             asyncio.create_task(redis_cache.set(request.title, result))
        
#         latency_ms = (time.time() - start_time) * 1000
        
#         return PredictionResponse(
#             title=request.title,
#             topic=result['topic'],
#             confidence=result['confidence'],
#             probabilities=result['probabilities'],
#             cached=False,
#             latency_ms=latency_ms
#         )
    
#     except Exception as e:
#         logger.error(f"❌ Prediction error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))
