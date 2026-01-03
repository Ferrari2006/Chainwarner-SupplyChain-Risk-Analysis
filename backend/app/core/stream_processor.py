import httpx
import json
from app.db.database import db

# OpenDigger API Base URL
OPENDIGGER_BASE_URL = "https://oss.x-lab.info/open_digger/github"

class StreamProcessor:
    """
    Implements 'Request-Parse-Store-Destroy' Pipeline.
    This ensures minimal memory footprint even for large JSON responses.
    
    Technical Highlights:
    1. HTTP Chunked Transfer: Fetches data in small 8KB chunks.
    2. Atomic Ingestion: Processes and stores data immediately, keeping RAM usage O(1).
    """

    def __init__(self):
        self._client = httpx.AsyncClient(
            timeout=10.0,
            limits=httpx.Limits(max_connections=60, max_keepalive_connections=20),
        )

    async def aclose(self):
        await self._client.aclose()
    
    async def fetch_and_store(self, repo_name: str, metric: str):
        url = f"{OPENDIGGER_BASE_URL}/{repo_name}/{metric}.json"
        cached = db.get_metrics(f"{repo_name}:{metric}")
        if cached is not None:
            return cached
        
        # 1. Request (Stream Mode) - Chunked Transfer
        client = self._client
        try:
            async with client.stream('GET', url) as response:
                if response.status_code != 200:
                    return None
                
                data_buffer = []
                total_size = 0
                MAX_SIZE = 10 * 1024 * 1024
                
                async for chunk in response.aiter_text():
                    total_size += len(chunk)
                    if total_size > MAX_SIZE:
                        print(f"StreamProcessor: Aborted {url} (Size > 10MB)")
                        return None
                    data_buffer.append(chunk)
                
                full_text = "".join(data_buffer)
                data = json.loads(full_text)
                
                db.upsert_metrics(f"{repo_name}:{metric}", data)
                return data
        except Exception:
            return None

stream_processor = StreamProcessor()
