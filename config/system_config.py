"""
Centralized configuration for APEX-ULTRA™ v15.0 AGI COSMOS
Edit this file to change system-wide settings.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Dict
import os

class AGIConfig(BaseSettings):
    reasoning_depth: int = Field(5, description="Depth of AGI reasoning")
    memory_capacity: int = Field(10000, description="Memory capacity for AGI brain")
    learning_rate: float = Field(0.01, description="Learning rate for AGI brain")

class RevenueConfig(BaseSettings):
    optimization_frequency: int = Field(3600, description="How often to optimize revenue streams (seconds)")
    max_streams: int = Field(500, description="Maximum number of revenue streams")
    min_profit_threshold: float = Field(0.01, description="Minimum profit threshold for optimization")

class ContentConfig(BaseSettings):
    generation_interval: int = Field(1800, description="Interval for content generation (seconds)")
    viral_threshold: float = Field(0.7, description="Threshold for viral content selection")
    platforms: List[str] = Field(["youtube", "tiktok", "twitter"], description="Content platforms")

class SystemConfig(BaseSettings):
    agi_brain: AGIConfig = AGIConfig()
    revenue_empire: RevenueConfig = RevenueConfig()
    content_pipeline: ContentConfig = ContentConfig()
    environment: str = Field(os.getenv("APEX_ULTRA_ENV", "development"), description="System environment")

CONFIG = SystemConfig() 