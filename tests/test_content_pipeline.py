import pytest
import asyncio
from content.content_pipeline import ContentPipeline

@pytest.mark.asyncio
async def test_generate_content_batch():
    pipeline = ContentPipeline()
    result = await pipeline.generate_content_batch(batch_size=2)
    assert isinstance(result, dict)
    assert "content_generated" in result
    assert result["content_generated"] == 2
    assert "platform_breakdown" in result

@pytest.mark.asyncio
async def test_schedule_and_performance():
    pipeline = ContentPipeline()
    await pipeline.generate_content_batch(batch_size=3)
    # Simulate optimization to increase viral_score
    for c in pipeline.content_queue:
        c.viral_score = 0.9
    schedule_result = await pipeline.schedule_content_batch(batch_size=2)
    assert isinstance(schedule_result, dict)
    assert "content_scheduled" in schedule_result
    performance = await pipeline.performance_tracker.update_performance()
    assert isinstance(performance, dict)
    assert "total_views" in performance
    assert performance["total_views"] >= 0 