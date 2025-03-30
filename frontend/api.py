import asyncio
import json
import logging
import os
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import redis.asyncio as aioredis
from dotenv import load_dotenv

# Load environment variables (to get Redis connection details)
load_dotenv("../.env") # Assuming .env is one level up from frontend directory

# Basic Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Redis Connection ---
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_DB = int(os.environ.get("REDIS_DB", "0"))
REDIS_USERNAME = os.environ.get("REDIS_USERNAME", "default")
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")

redis_pool = None
redis_client: Optional[aioredis.Redis] = None

async def get_redis_client() -> Optional[aioredis.Redis]:
    """Gets a Redis client instance, initializing pool if needed."""
    global redis_pool, redis_client
    if redis_client is None:
        try:
            logger.info(f"Connecting to Redis at {REDIS_HOST}:{REDIS_PORT} DB {REDIS_DB}")
            redis_pool = aioredis.ConnectionPool(
                host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
                username=REDIS_USERNAME, password=REDIS_PASSWORD,
                decode_responses=True, # Decode for easier JSON handling
                max_connections=10
            )
            redis_client = aioredis.Redis(connection_pool=redis_pool)
            await redis_client.ping()
            logger.info("Redis connection successful.")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            redis_client = None # Ensure it's None on failure
    return redis_client

# --- FastAPI App ---
app = FastAPI(title="Trading System Monitor API")

# CORS Middleware (allow frontend to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for simplicity, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize Redis client on startup."""
    await get_redis_client()

@app.on_event("shutdown")
async def shutdown_event():
    """Disconnect Redis client on shutdown."""
    global redis_client, redis_pool
    if redis_client:
        await redis_client.close()
        logger.info("Redis client closed.")
    if redis_pool:
        await redis_pool.disconnect()
        logger.info("Redis pool disconnected.")

# --- API Endpoints ---

@app.get("/api/status", response_model=Dict[str, Any])
async def get_system_status():
    """Fetches the latest system status from Redis."""
    client = await get_redis_client()
    if not client:
        raise HTTPException(status_code=503, detail="Redis not available")
    try:
        status_json = await client.get("frontend:system:status")
        if status_json:
            return json.loads(status_json)
        else:
            return {"running": False, "status": "unknown", "message": "No status reported yet."}
    except Exception as e:
        logger.error(f"Error fetching status from Redis: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching status: {e}")

@app.get("/api/notifications", response_model=List[Dict[str, Any]])
async def get_latest_notifications(limit: int = 20):
    """Fetches the latest notifications from Redis list."""
    client = await get_redis_client()
    if not client:
        raise HTTPException(status_code=503, detail="Redis not available")
    try:
        # Fetch latest 'limit' items from the list
        notifications_json = await client.lrange("frontend:notifications", 0, limit - 1)
        notifications = [json.loads(n) for n in notifications_json]
        return notifications
    except Exception as e:
        logger.error(f"Error fetching notifications from Redis: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching notifications: {e}")

# --- Optional: Prometheus Metrics Endpoint ---
# Uncomment if you want to expose metrics from this API itself
# Needs prometheus-client and prometheus-fastapi-instrumentator installed
# from prometheus_fastapi_instrumentator import Instrumentator
# Instrumentator().instrument(app).expose(app)

if __name__ == "__main__":
    import uvicorn
    # Run directly for testing: uvicorn frontend.api:app --reload --port 8001
    uvicorn.run(app, host="0.0.0.0", port=8001)