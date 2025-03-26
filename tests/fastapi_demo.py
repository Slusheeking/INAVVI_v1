from fastapi import FastAPI, WebSocket
from prometheus_fastapi_instrumentator import Instrumentator
import redis
import asyncio
import json
import os
import time
from datetime import datetime

app = FastAPI()
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# WebSocket for Redis updates
@app.websocket("/ws/redis")
async def redis_websocket(websocket: WebSocket):
    await websocket.accept()
    pubsub = r.pubsub()
    # Subscribe synchronously since redis-py doesn't support async subscribe
    pubsub.subscribe('updates', 'file_changes')
    
    try:
        while True:
            message = pubsub.get_message(ignore_subscribe_messages=True)
            if message:
                try:
                    # Try to parse as JSON
                    if isinstance(message['data'], str):
                        data = json.loads(message['data'])
                        await websocket.send_json(data)
                    else:
                        # If not JSON, send as plain text
                        await websocket.send_text(str(message['data']))
                except json.JSONDecodeError:
                    # If JSON parsing fails, send as plain text
                    await websocket.send_text(str(message['data']))
            await asyncio.sleep(0.1)
    except Exception as e:
        print(f"Error in WebSocket: {e}")
    finally:
        pubsub.unsubscribe('updates', 'file_changes')

# File monitoring using Redis
async def monitor_files():
    """Monitor files and publish changes to Redis"""
    file_states = {}
    
    def get_file_state(path):
        """Get file modification time and size"""
        try:
            stat = os.stat(path)
            return {
                'mtime': stat.st_mtime,
                'size': stat.st_size
            }
        except OSError:
            return None

    def scan_files():
        """Scan current directory for files"""
        for root, _, files in os.walk('.'):
            for file in files:
                path = os.path.join(root, file)
                if not path.startswith(('.git', '.env', '__pycache__')):
                    yield path

    while True:
        for filepath in scan_files():
            current_state = get_file_state(filepath)
            if current_state:
                if filepath not in file_states:
                    file_states[filepath] = current_state
                    # New file detected
                    r.publish('file_changes', json.dumps({
                        'type': 'created',
                        'path': filepath,
                        'timestamp': time.time()
                    }))
                elif current_state != file_states[filepath]:
                    # File modified
                    r.publish('file_changes', json.dumps({
                        'type': 'modified',
                        'path': filepath,
                        'timestamp': time.time()
                    }))
                    file_states[filepath] = current_state

        # Check for deleted files
        existing_files = set(scan_files())
        for filepath in list(file_states.keys()):
            if filepath not in existing_files:
                r.publish('file_changes', json.dumps({
                    'type': 'deleted',
                    'path': filepath,
                    'timestamp': time.time()
                }))
                del file_states[filepath]

        await asyncio.sleep(1)  # Check every second

@app.on_event("startup")
async def startup_event():
    """Start the file monitoring task"""
    asyncio.create_task(monitor_files())

@app.get("/")
async def root():
    return {"message": "Hello World"}

# Test endpoints
@app.post("/test-publish")
async def test_publish():
    """Test endpoint to publish a message to Redis"""
    message = {
        "type": "test",
        "message": "Test message",
        "timestamp": datetime.now().isoformat()
    }
    r.publish('updates', json.dumps(message))
    return {"status": "published"}

@app.post("/test-file")
async def test_file():
    """Test endpoint to create a test file"""
    filename = f"test_file_{int(time.time())}.txt"
    with open(filename, 'w') as f:
        f.write(f"Test file created at {datetime.now().isoformat()}")
    return {"status": "created", "filename": filename}

# Set up Prometheus metrics at /metrics
Instrumentator().instrument(app).expose(app)
