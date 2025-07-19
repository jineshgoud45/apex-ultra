from config.system_config import CONFIG

def test_agi_config():
    assert CONFIG.agi_brain.reasoning_depth == 5
    assert CONFIG.agi_brain.memory_capacity == 10000
    assert abs(CONFIG.agi_brain.learning_rate - 0.01) < 1e-6

def test_revenue_config():
    assert CONFIG.revenue_empire.max_streams == 500
    assert CONFIG.revenue_empire.optimization_frequency == 3600
    assert abs(CONFIG.revenue_empire.min_profit_threshold - 0.01) < 1e-6

def test_content_config():
    assert "youtube" in CONFIG.content_pipeline.platforms
    assert CONFIG.content_pipeline.generation_interval == 1800
    assert abs(CONFIG.content_pipeline.viral_threshold - 0.7) < 1e-6 