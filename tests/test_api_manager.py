import pytest
import asyncio
from api_integration.api_manager import APIManager, APIService

@pytest.mark.asyncio
async def test_youtube_search_simulation():
    async with APIManager() as api:
        result = await api.youtube_search("test", max_results=2)
        assert hasattr(result, "success")
        assert hasattr(result, "status_code")

@pytest.mark.asyncio
async def test_rate_limit_status():
    async with APIManager() as api:
        status = api.get_rate_limit_status()
        assert isinstance(status, dict)

@pytest.mark.asyncio
async def test_batch_request_simulation():
    async with APIManager() as api:
        requests = [
            {"service": "youtube", "endpoint": "search", "params": {"q": "ai"}},
            {"service": "youtube", "endpoint": "videos", "params": {"id": "123"}}
        ]
        results = await api.batch_request(requests)
        assert isinstance(results, list)
        assert all(hasattr(r, "success") for r in results if not isinstance(r, Exception)) 