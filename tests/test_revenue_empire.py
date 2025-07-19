import pytest
import asyncio
from revenue.revenue_empire import RevenueEmpire

@pytest.mark.asyncio
async def test_analyze_all_streams():
    empire = RevenueEmpire()
    analysis = await empire.analyze_all_streams()
    assert isinstance(analysis, dict)
    assert "total_streams_analyzed" in analysis
    assert analysis["total_streams_analyzed"] == 500
    assert "analysis_results" in analysis
    # Check at least one stream has optimization potential
    assert analysis["streams_with_optimization_potential"] > 0

@pytest.mark.asyncio
async def test_optimization_and_compounding():
    empire = RevenueEmpire()
    await empire.analyze_all_streams()
    optimizations = await empire.execute_optimizations(max_optimizations=5)
    assert isinstance(optimizations, dict)
    assert "optimizations_executed" in optimizations
    compounding = await empire.apply_compounding_growth()
    assert isinstance(compounding, dict)
    assert "total_growth" in compounding
    summary = empire.get_revenue_summary()
    assert isinstance(summary, dict)
    assert "total_revenue" in summary 