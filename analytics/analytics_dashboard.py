"""
Analytics Dashboard for APEX-ULTRA™
Provides comprehensive metrics tracking, predictive analytics, and interactive visualizations.
"""

import asyncio
import json
import random
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging
from collections import defaultdict
import math

# === Analytics Dashboard Self-Healing, Self-Editing, Watchdog, and AGI/GPT-2.5 Pro Integration ===
import os
import threading
import importlib
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger("apex_ultra.analytics.analytics_dashboard")

ANALYTICS_API_KEY = os.environ.get("ANALYTICS_API_KEY")
if not ANALYTICS_API_KEY:
    logger.warning("ANALYTICS_API_KEY not set. Some features may not work.")

@dataclass
class Metric:
    """Represents a single metric measurement."""
    metric_id: str
    name: str
    value: float
    unit: str
    category: str
    timestamp: datetime
    tags: Dict[str, str]
    metadata: Dict[str, Any]

@dataclass
class Dashboard:
    """Represents an analytics dashboard."""
    dashboard_id: str
    name: str
    description: str
    widgets: List[Dict[str, Any]]
    layout: Dict[str, Any]
    refresh_interval: int
    last_updated: datetime

@dataclass
class Prediction:
    """Represents a prediction from analytics models."""
    prediction_id: str
    model_name: str
    target_metric: str
    predicted_value: float
    confidence: float
    timestamp: datetime
    horizon: str
    factors: List[str]

class MetricsCollector:
    """Collects and stores metrics from various system components."""
    
    def __init__(self):
        self.metrics: List[Metric] = []
        self.metric_definitions = self._load_metric_definitions()
        self.collection_schedules = self._load_collection_schedules()
    
    def _load_metric_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Load definitions for different metric types."""
        return {
            "revenue": {
                "description": "Revenue metrics",
                "unit": "USD",
                "category": "financial",
                "collection_interval": 300,  # 5 minutes
                "aggregation_methods": ["sum", "average", "trend"]
            },
            "audience": {
                "description": "Audience growth and engagement",
                "unit": "count",
                "category": "social",
                "collection_interval": 600,  # 10 minutes
                "aggregation_methods": ["sum", "average", "growth_rate"]
            },
            "content": {
                "description": "Content performance metrics",
                "unit": "various",
                "category": "content",
                "collection_interval": 300,
                "aggregation_methods": ["count", "average", "engagement_rate"]
            },
            "system": {
                "description": "System performance metrics",
                "unit": "various",
                "category": "technical",
                "collection_interval": 60,  # 1 minute
                "aggregation_methods": ["average", "max", "min"]
            },
            "ethics": {
                "description": "Ethical compliance metrics",
                "unit": "score",
                "category": "compliance",
                "collection_interval": 1800,  # 30 minutes
                "aggregation_methods": ["average", "compliance_rate"]
            }
        }
    
    def _load_collection_schedules(self) -> Dict[str, int]:
        """Load collection schedules for different metric categories."""
        return {
            "revenue": 300,  # 5 minutes
            "audience": 600,  # 10 minutes
            "content": 300,  # 5 minutes
            "system": 60,    # 1 minute
            "ethics": 1800   # 30 minutes
        }
    
    async def collect_metrics(self, category: str = "all") -> List[Metric]:
        """Collect metrics for specified category."""
        logger.info(f"Collecting metrics for category: {category}")
        
        collected_metrics = []
        
        if category == "all" or category == "revenue":
            revenue_metrics = await self._collect_revenue_metrics()
            collected_metrics.extend(revenue_metrics)
        
        if category == "all" or category == "audience":
            audience_metrics = await self._collect_audience_metrics()
            collected_metrics.extend(audience_metrics)
        
        if category == "all" or category == "content":
            content_metrics = await self._collect_content_metrics()
            collected_metrics.extend(content_metrics)
        
        if category == "all" or category == "system":
            system_metrics = await self._collect_system_metrics()
            collected_metrics.extend(system_metrics)
        
        if category == "all" or category == "ethics":
            ethics_metrics = await self._collect_ethics_metrics()
            collected_metrics.extend(ethics_metrics)
        
        # Store collected metrics
        self.metrics.extend(collected_metrics)
        
        # Keep only last 10000 metrics
        if len(self.metrics) > 10000:
            self.metrics = self.metrics[-10000:]
        
        logger.info(f"Collected {len(collected_metrics)} metrics")
        return collected_metrics
    
    async def _collect_revenue_metrics(self) -> List[Metric]:
        """Collect revenue-related metrics."""
        metrics = []
        
        # Total revenue
        total_revenue = random.uniform(10000, 100000)
        metrics.append(Metric(
            metric_id=f"revenue_total_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name="total_revenue",
            value=total_revenue,
            unit="USD",
            category="revenue",
            timestamp=datetime.now(),
            tags={"type": "total", "currency": "USD"},
            metadata={"source": "revenue_empire"}
        ))
        
        # Revenue growth rate
        growth_rate = random.uniform(-0.1, 0.3)
        metrics.append(Metric(
            metric_id=f"revenue_growth_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name="revenue_growth_rate",
            value=growth_rate,
            unit="percentage",
            category="revenue",
            timestamp=datetime.now(),
            tags={"type": "growth", "period": "daily"},
            metadata={"source": "revenue_empire"}
        ))
        
        # Active revenue streams
        active_streams = random.randint(400, 500)
        metrics.append(Metric(
            metric_id=f"revenue_streams_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name="active_revenue_streams",
            value=active_streams,
            unit="count",
            category="revenue",
            timestamp=datetime.now(),
            tags={"type": "streams", "status": "active"},
            metadata={"source": "revenue_empire"}
        ))
        
        return metrics
    
    async def _collect_audience_metrics(self) -> List[Metric]:
        """Collect audience-related metrics."""
        metrics = []
        
        # Total audience size
        total_audience = random.randint(100000, 1000000)
        metrics.append(Metric(
            metric_id=f"audience_total_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name="total_audience",
            value=total_audience,
            unit="count",
            category="audience",
            timestamp=datetime.now(),
            tags={"type": "total", "platform": "all"},
            metadata={"source": "audience_builder"}
        ))
        
        # Audience growth rate
        growth_rate = random.uniform(0.01, 0.15)
        metrics.append(Metric(
            metric_id=f"audience_growth_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name="audience_growth_rate",
            value=growth_rate,
            unit="percentage",
            category="audience",
            timestamp=datetime.now(),
            tags={"type": "growth", "period": "daily"},
            metadata={"source": "audience_builder"}
        ))
        
        # Engagement rate
        engagement_rate = random.uniform(0.02, 0.08)
        metrics.append(Metric(
            metric_id=f"audience_engagement_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name="engagement_rate",
            value=engagement_rate,
            unit="percentage",
            category="audience",
            timestamp=datetime.now(),
            tags={"type": "engagement", "period": "daily"},
            metadata={"source": "audience_builder"}
        ))
        
        return metrics
    
    async def _collect_content_metrics(self) -> List[Metric]:
        """Collect content-related metrics."""
        metrics = []
        
        # Content pieces created
        content_created = random.randint(10, 50)
        metrics.append(Metric(
            metric_id=f"content_created_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name="content_pieces_created",
            value=content_created,
            unit="count",
            category="content",
            timestamp=datetime.now(),
            tags={"type": "creation", "period": "daily"},
            metadata={"source": "content_pipeline"}
        ))
        
        # Average viral score
        avg_viral_score = random.uniform(0.3, 0.9)
        metrics.append(Metric(
            metric_id=f"content_viral_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name="average_viral_score",
            value=avg_viral_score,
            unit="score",
            category="content",
            timestamp=datetime.now(),
            tags={"type": "viral", "period": "daily"},
            metadata={"source": "content_pipeline"}
        ))
        
        # Content engagement
        content_engagement = random.uniform(0.05, 0.25)
        metrics.append(Metric(
            metric_id=f"content_engagement_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name="content_engagement_rate",
            value=content_engagement,
            unit="percentage",
            category="content",
            timestamp=datetime.now(),
            tags={"type": "engagement", "period": "daily"},
            metadata={"source": "content_pipeline"}
        ))
        
        return metrics
    
    async def _collect_system_metrics(self) -> List[Metric]:
        """Collect system performance metrics."""
        metrics = []
        
        # CPU usage
        cpu_usage = random.uniform(0.2, 0.8)
        metrics.append(Metric(
            metric_id=f"system_cpu_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name="cpu_usage",
            value=cpu_usage,
            unit="percentage",
            category="system",
            timestamp=datetime.now(),
            tags={"type": "performance", "component": "cpu"},
            metadata={"source": "infrastructure"}
        ))
        
        # Memory usage
        memory_usage = random.uniform(0.3, 0.9)
        metrics.append(Metric(
            metric_id=f"system_memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name="memory_usage",
            value=memory_usage,
            unit="percentage",
            category="system",
            timestamp=datetime.now(),
            tags={"type": "performance", "component": "memory"},
            metadata={"source": "infrastructure"}
        ))
        
        # Response time
        response_time = random.uniform(50, 200)
        metrics.append(Metric(
            metric_id=f"system_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name="average_response_time",
            value=response_time,
            unit="milliseconds",
            category="system",
            timestamp=datetime.now(),
            tags={"type": "performance", "component": "network"},
            metadata={"source": "infrastructure"}
        ))
        
        return metrics
    
    async def _collect_ethics_metrics(self) -> List[Metric]:
        """Collect ethics and compliance metrics."""
        metrics = []
        
        # Ethical compliance score
        compliance_score = random.uniform(0.7, 0.95)
        metrics.append(Metric(
            metric_id=f"ethics_compliance_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name="ethical_compliance_score",
            value=compliance_score,
            unit="score",
            category="ethics",
            timestamp=datetime.now(),
            tags={"type": "compliance", "framework": "general"},
            metadata={"source": "ethics_engine"}
        ))
        
        # Violation count
        violation_count = random.randint(0, 5)
        metrics.append(Metric(
            metric_id=f"ethics_violations_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name="compliance_violations",
            value=violation_count,
            unit="count",
            category="ethics",
            timestamp=datetime.now(),
            tags={"type": "violations", "period": "daily"},
            metadata={"source": "ethics_engine"}
        ))
        
        return metrics
    
    def get_recent_metrics(self, category: str = None, hours: int = 24) -> List[Metric]:
        """Get recent metrics for specified category and time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_metrics = [
            metric for metric in self.metrics
            if metric.timestamp >= cutoff_time
        ]
        
        if category:
            recent_metrics = [
                metric for metric in recent_metrics
                if metric.category == category
            ]
        
        return recent_metrics
    
    def aggregate_metrics(self, metrics: List[Metric], aggregation_method: str) -> Dict[str, float]:
        """Aggregate metrics using specified method."""
        if not metrics:
            return {}
        
        aggregated = {}
        
        # Group by metric name
        by_name = defaultdict(list)
        for metric in metrics:
            by_name[metric.name].append(metric.value)
        
        for name, values in by_name.items():
            if aggregation_method == "sum":
                aggregated[name] = sum(values)
            elif aggregation_method == "average":
                aggregated[name] = sum(values) / len(values)
            elif aggregation_method == "max":
                aggregated[name] = max(values)
            elif aggregation_method == "min":
                aggregated[name] = min(values)
            elif aggregation_method == "count":
                aggregated[name] = len(values)
        
        return aggregated

class PredictiveAnalytics:
    """Provides predictive analytics capabilities."""
    
    def __init__(self):
        self.models = self._load_prediction_models()
        self.historical_data = {}
        self.prediction_cache = {}
    
    def _load_prediction_models(self) -> Dict[str, Dict[str, Any]]:
        """Load prediction model configurations."""
        return {
            "revenue_forecast": {
                "description": "Forecast revenue trends",
                "horizon": "7_days",
                "confidence_threshold": 0.7,
                "update_frequency": 3600  # 1 hour
            },
            "audience_growth": {
                "description": "Predict audience growth",
                "horizon": "30_days",
                "confidence_threshold": 0.6,
                "update_frequency": 7200  # 2 hours
            },
            "content_performance": {
                "description": "Predict content performance",
                "horizon": "24_hours",
                "confidence_threshold": 0.5,
                "update_frequency": 1800  # 30 minutes
            },
            "system_load": {
                "description": "Predict system load",
                "horizon": "1_hour",
                "confidence_threshold": 0.8,
                "update_frequency": 300  # 5 minutes
            }
        }
    
    async def generate_predictions(self, model_name: str, historical_metrics: List[Metric]) -> List[Prediction]:
        """Generate predictions using specified model."""
        logger.info(f"Generating predictions using model: {model_name}")
        
        model_config = self.models.get(model_name)
        if not model_config:
            return []
        
        predictions = []
        
        # Group metrics by name
        metrics_by_name = defaultdict(list)
        for metric in historical_metrics:
            metrics_by_name[metric.name].append(metric)
        
        # Generate predictions for each metric type
        for metric_name, metrics in metrics_by_name.items():
            if len(metrics) < 5:  # Need minimum data points
                continue
            
            # Sort by timestamp
            sorted_metrics = sorted(metrics, key=lambda m: m.timestamp)
            values = [m.value for m in sorted_metrics]
            
            # Generate prediction
            prediction = await self._predict_metric(
                model_name, metric_name, values, model_config
            )
            
            if prediction:
                predictions.append(prediction)
        
        # Cache predictions
        self.prediction_cache[model_name] = {
            "predictions": predictions,
            "timestamp": datetime.now()
        }
        
        logger.info(f"Generated {len(predictions)} predictions for {model_name}")
        return predictions
    
    async def _predict_metric(self, model_name: str, metric_name: str, values: List[float], model_config: Dict[str, Any]) -> Optional[Prediction]:
        """Predict a specific metric."""
        if len(values) < 3:
            return None
        
        # Simple linear regression prediction
        n = len(values)
        x = list(range(n))
        
        # Calculate trend
        if n > 1:
            trend = (values[-1] - values[0]) / max(n - 1, 1)
        else:
            trend = 0
        
        # Predict next value
        predicted_value = values[-1] + trend
        
        # Calculate confidence based on data consistency
        if n > 2:
            variance = sum((v - sum(values) / n) ** 2 for v in values) / n
            confidence = max(0.1, 1.0 - min(variance / max(values[-1], 1), 1.0))
        else:
            confidence = 0.5
        
        # Adjust confidence based on model threshold
        if confidence < model_config["confidence_threshold"]:
            return None
        
        prediction_id = f"{model_name}_{metric_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return Prediction(
            prediction_id=prediction_id,
            model_name=model_name,
            target_metric=metric_name,
            predicted_value=predicted_value,
            confidence=confidence,
            timestamp=datetime.now(),
            horizon=model_config["horizon"],
            factors=["trend_analysis", "historical_patterns"]
        )
    
    def get_cached_predictions(self, model_name: str) -> List[Prediction]:
        """Get cached predictions for a model."""
        cache_entry = self.prediction_cache.get(model_name)
        if cache_entry:
            # Check if cache is still valid
            cache_age = (datetime.now() - cache_entry["timestamp"]).total_seconds()
            model_config = self.models.get(model_name, {})
            update_frequency = model_config.get("update_frequency", 3600)
            
            if cache_age < update_frequency:
                return cache_entry["predictions"]
        
        return []

class DashboardManager:
    """Manages analytics dashboards and visualizations."""
    
    def __init__(self):
        self.dashboards: Dict[str, Dashboard] = {}
        self.widget_templates = self._load_widget_templates()
        self.default_dashboards = self._create_default_dashboards()
    
    def _load_widget_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load widget templates for different visualization types."""
        return {
            "line_chart": {
                "type": "line_chart",
                "description": "Time series line chart",
                "configurable": ["title", "x_axis", "y_axis", "data_source"],
                "default_size": {"width": 6, "height": 4}
            },
            "bar_chart": {
                "type": "bar_chart",
                "description": "Bar chart for comparisons",
                "configurable": ["title", "x_axis", "y_axis", "data_source"],
                "default_size": {"width": 6, "height": 4}
            },
            "gauge": {
                "type": "gauge",
                "description": "Gauge for single value display",
                "configurable": ["title", "min_value", "max_value", "thresholds"],
                "default_size": {"width": 3, "height": 3}
            },
            "metric_card": {
                "type": "metric_card",
                "description": "Simple metric display card",
                "configurable": ["title", "metric", "format", "trend"],
                "default_size": {"width": 3, "height": 2}
            },
            "table": {
                "type": "table",
                "description": "Data table display",
                "configurable": ["title", "columns", "data_source", "sorting"],
                "default_size": {"width": 12, "height": 6}
            }
        }
    
    def _create_default_dashboards(self) -> Dict[str, Dashboard]:
        """Create default dashboards."""
        dashboards = {}
        
        # Overview Dashboard
        overview_widgets = [
            {
                "widget_id": "revenue_gauge",
                "type": "gauge",
                "title": "Total Revenue",
                "config": {"metric": "total_revenue", "unit": "USD"}
            },
            {
                "widget_id": "audience_growth",
                "type": "line_chart",
                "title": "Audience Growth",
                "config": {"metric": "total_audience", "period": "7_days"}
            },
            {
                "widget_id": "content_performance",
                "type": "bar_chart",
                "title": "Content Performance",
                "config": {"metric": "average_viral_score", "period": "24_hours"}
            },
            {
                "widget_id": "system_health",
                "type": "metric_card",
                "title": "System Health",
                "config": {"metric": "cpu_usage", "format": "percentage"}
            }
        ]
        
        dashboards["overview"] = Dashboard(
            dashboard_id="overview",
            name="System Overview",
            description="High-level system metrics and performance",
            widgets=overview_widgets,
            layout={"grid": "responsive", "columns": 12},
            refresh_interval=300,
            last_updated=datetime.now()
        )
        
        # Revenue Dashboard
        revenue_widgets = [
            {
                "widget_id": "revenue_trend",
                "type": "line_chart",
                "title": "Revenue Trend",
                "config": {"metric": "total_revenue", "period": "30_days"}
            },
            {
                "widget_id": "growth_rate",
                "type": "gauge",
                "title": "Growth Rate",
                "config": {"metric": "revenue_growth_rate", "unit": "percentage"}
            },
            {
                "widget_id": "revenue_streams",
                "type": "bar_chart",
                "title": "Active Revenue Streams",
                "config": {"metric": "active_revenue_streams", "period": "7_days"}
            }
        ]
        
        dashboards["revenue"] = Dashboard(
            dashboard_id="revenue",
            name="Revenue Analytics",
            description="Detailed revenue metrics and trends",
            widgets=revenue_widgets,
            layout={"grid": "responsive", "columns": 12},
            refresh_interval=600,
            last_updated=datetime.now()
        )
        
        return dashboards
    
    async def create_dashboard(self, name: str, description: str, widgets: List[Dict[str, Any]]) -> Dashboard:
        """Create a new dashboard."""
        dashboard_id = self._generate_dashboard_id(name)
        
        dashboard = Dashboard(
            dashboard_id=dashboard_id,
            name=name,
            description=description,
            widgets=widgets,
            layout={"grid": "responsive", "columns": 12},
            refresh_interval=300,
            last_updated=datetime.now()
        )
        
        self.dashboards[dashboard_id] = dashboard
        
        logger.info(f"Created dashboard: {dashboard_id}")
        return dashboard
    
    def _generate_dashboard_id(self, name: str) -> str:
        """Generate unique dashboard ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"dashboard_{name.lower().replace(' ', '_')}_{timestamp}"
    
    def get_dashboard_data(self, dashboard_id: str, metrics_collector: MetricsCollector, predictive_analytics: PredictiveAnalytics) -> Dict[str, Any]:
        """Get data for dashboard widgets."""
        dashboard = self.dashboards.get(dashboard_id)
        if not dashboard:
            return {"error": "Dashboard not found"}
        
        widget_data = {}
        
        for widget in dashboard.widgets:
            widget_id = widget["widget_id"]
            widget_type = widget["type"]
            config = widget.get("config", {})
            
            # Get data for widget
            data = self._get_widget_data(widget_type, config, metrics_collector, predictive_analytics)
            widget_data[widget_id] = data
        
        return {
            "dashboard_id": dashboard_id,
            "widget_data": widget_data,
            "last_updated": dashboard.last_updated.isoformat()
        }
    
    def _get_widget_data(self, widget_type: str, config: Dict[str, Any], metrics_collector: MetricsCollector, predictive_analytics: PredictiveAnalytics) -> Dict[str, Any]:
        """Get data for a specific widget type."""
        if widget_type == "gauge":
            return self._get_gauge_data(config, metrics_collector)
        elif widget_type == "line_chart":
            return self._get_line_chart_data(config, metrics_collector)
        elif widget_type == "bar_chart":
            return self._get_bar_chart_data(config, metrics_collector)
        elif widget_type == "metric_card":
            return self._get_metric_card_data(config, metrics_collector)
        elif widget_type == "table":
            return self._get_table_data(config, metrics_collector)
        else:
            return {"error": f"Unknown widget type: {widget_type}"}
    
    def _get_gauge_data(self, config: Dict[str, Any], metrics_collector: MetricsCollector) -> Dict[str, Any]:
        """Get data for gauge widget."""
        metric_name = config.get("metric", "total_revenue")
        recent_metrics = metrics_collector.get_recent_metrics(hours=1)
        
        # Find most recent value for metric
        metric_values = [m.value for m in recent_metrics if m.name == metric_name]
        current_value = metric_values[-1] if metric_values else 0
        
        return {
            "type": "gauge",
            "value": current_value,
            "unit": config.get("unit", ""),
            "min_value": config.get("min_value", 0),
            "max_value": config.get("max_value", current_value * 1.5),
            "thresholds": config.get("thresholds", {"warning": 0.7, "critical": 0.9})
        }
    
    def _get_line_chart_data(self, config: Dict[str, Any], metrics_collector: MetricsCollector) -> Dict[str, Any]:
        """Get data for line chart widget."""
        metric_name = config.get("metric", "total_revenue")
        period = config.get("period", "7_days")
        
        # Calculate hours based on period
        period_hours = {"1_hour": 1, "24_hours": 24, "7_days": 168, "30_days": 720}
        hours = period_hours.get(period, 24)
        
        recent_metrics = metrics_collector.get_recent_metrics(hours=hours)
        metric_metrics = [m for m in recent_metrics if m.name == metric_name]
        
        # Sort by timestamp and extract data
        sorted_metrics = sorted(metric_metrics, key=lambda m: m.timestamp)
        
        data = {
            "type": "line_chart",
            "labels": [m.timestamp.strftime("%Y-%m-%d %H:%M") for m in sorted_metrics],
            "values": [m.value for m in sorted_metrics],
            "unit": config.get("unit", "")
        }
        
        return data
    
    def _get_bar_chart_data(self, config: Dict[str, Any], metrics_collector: MetricsCollector) -> Dict[str, Any]:
        """Get data for bar chart widget."""
        metric_name = config.get("metric", "total_revenue")
        recent_metrics = metrics_collector.get_recent_metrics(hours=24)
        metric_metrics = [m for m in recent_metrics if m.name == metric_name]
        
        # Group by hour
        hourly_data = defaultdict(list)
        for metric in metric_metrics:
            hour = metric.timestamp.strftime("%H:00")
            hourly_data[hour].append(metric.value)
        
        # Calculate averages
        labels = sorted(hourly_data.keys())
        values = [sum(hourly_data[hour]) / len(hourly_data[hour]) for hour in labels]
        
        return {
            "type": "bar_chart",
            "labels": labels,
            "values": values,
            "unit": config.get("unit", "")
        }
    
    def _get_metric_card_data(self, config: Dict[str, Any], metrics_collector: MetricsCollector) -> Dict[str, Any]:
        """Get data for metric card widget."""
        metric_name = config.get("metric", "total_revenue")
        recent_metrics = metrics_collector.get_recent_metrics(hours=1)
        
        metric_values = [m.value for m in recent_metrics if m.name == metric_name]
        current_value = metric_values[-1] if metric_values else 0
        
        # Calculate trend
        if len(metric_values) >= 2:
            trend = "up" if metric_values[-1] > metric_values[0] else "down"
            trend_value = abs(metric_values[-1] - metric_values[0]) / max(metric_values[0], 1)
        else:
            trend = "stable"
            trend_value = 0
        
        return {
            "type": "metric_card",
            "value": current_value,
            "unit": config.get("unit", ""),
            "trend": trend,
            "trend_value": trend_value,
            "format": config.get("format", "number")
        }
    
    def _get_table_data(self, config: Dict[str, Any], metrics_collector: MetricsCollector) -> Dict[str, Any]:
        """Get data for table widget."""
        recent_metrics = metrics_collector.get_recent_metrics(hours=24)
        
        # Group by metric name and get latest values
        latest_metrics = {}
        for metric in recent_metrics:
            if metric.name not in latest_metrics or metric.timestamp > latest_metrics[metric.name].timestamp:
                latest_metrics[metric.name] = metric
        
        # Convert to table format
        rows = []
        for metric_name, metric in latest_metrics.items():
            rows.append({
                "name": metric_name,
                "value": metric.value,
                "unit": metric.unit,
                "category": metric.category,
                "timestamp": metric.timestamp.strftime("%Y-%m-%d %H:%M")
            })
        
        return {
            "type": "table",
            "columns": ["Name", "Value", "Unit", "Category", "Last Updated"],
            "rows": rows
        }

# === AGI/GPT-2.5 Pro Integration Stub ===
class AnalyticsAgiIntegration:
    """Stub for AGI brain and GPT-2.5 Pro integration for analytics/insights."""
    def __init__(self, agi_brain=None):
        self.agi_brain = agi_brain

    async def suggest_analytics_insights(self, context: dict) -> dict:
        if self.agi_brain and hasattr(self.agi_brain, "gpt25pro_reason"):
            prompt = f"Suggest analytics insights for: {context}"
            return await self.agi_brain.gpt25pro_reason(prompt)
        return {"suggestion": "[Stub: Connect AGI brain for LLM-driven analytics insights]"}

# === Production Hardening Hooks ===
def backup_analytics_data(dashboard, backup_path="backups/analytics_backup.json"):
    """Stub: Backup analytics data to a secure location."""
    try:
        with open(backup_path, "w") as f:
            json.dump(dashboard.get_dashboard_status(), f, default=str)
        logger.info(f"Analytics data backed up to {backup_path}")
        return True
    except Exception as e:
        logger.error(f"Backup failed: {e}")
        return False

def report_incident(description, severity="medium"):
    """Stub: Report an incident for compliance and monitoring."""
    logger.warning(f"Incident reported: {description} (Severity: {severity})")
    # In production, send to incident management system
    return True

class AnalyticsDashboardMaintenance:
    """Handles self-healing, self-editing, and watchdog logic for AnalyticsDashboard."""
    def __init__(self, dashboard):
        self.dashboard = dashboard
        self.watchdog_thread = None
        self.watchdog_active = False

    def start_watchdog(self, interval_sec=120):
        if self.watchdog_thread and self.watchdog_thread.is_alive():
            return
        self.watchdog_active = True
        self.watchdog_thread = threading.Thread(target=self._watchdog_loop, args=(interval_sec,), daemon=True)
        self.watchdog_thread.start()

    def stop_watchdog(self):
        self.watchdog_active = False
        if self.watchdog_thread:
            self.watchdog_thread.join(timeout=2)

    def _watchdog_loop(self, interval_sec):
        import time
        while self.watchdog_active:
            try:
                # Health check: can be expanded
                status = self.dashboard.get_dashboard_status()
                if status.get("total_metrics", 0) < 0:
                    self.self_heal(reason="Negative metrics count detected")
            except Exception as e:
                self.self_heal(reason=f"Exception in watchdog: {e}")
            time.sleep(interval_sec)

    def self_edit(self, file_path, new_code, safety_check=True):
        if safety_check:
            allowed = ["analytics/analytics_dashboard.py"]
            if file_path not in allowed:
                raise PermissionError("Self-editing not allowed for this file.")
        with open(file_path, "w") as f:
            f.write(new_code)
        importlib.reload(importlib.import_module(file_path.replace(".py", "").replace("/", ".")))
        return True

    def self_heal(self, reason="Unknown"):
        logger.warning(f"AnalyticsDashboard self-healing triggered: {reason}")
        # Reset some metrics or reload configs as a stub
        self.dashboard._initialize_metrics()
        return True

class AnalyticsDashboard:
    """
    Main analytics dashboard that orchestrates metrics collection, predictions, and visualizations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.metrics_collector = MetricsCollector()
        self.predictive_analytics = PredictiveAnalytics()
        self.dashboard_manager = DashboardManager()
        
        self.analytics_log: List[Dict[str, Any]] = []
        self.performance_history: List[Dict[str, Any]] = []
        self.maintenance = AnalyticsDashboardMaintenance(self)
        self.agi_integration = AnalyticsAgiIntegration()
        self.maintenance.start_watchdog(interval_sec=120)
    
    async def run_analytics_cycle(self) -> Dict[str, Any]:
        """Run a complete analytics cycle."""
        logger.info("Starting analytics cycle")
        
        # 1. Collect metrics
        metrics_collected = await self.metrics_collector.collect_metrics()
        
        # 2. Generate predictions
        predictions_generated = await self._generate_all_predictions()
        
        # 3. Update dashboards
        dashboard_updates = await self._update_dashboards()
        
        # 4. Generate insights
        insights = self._generate_insights(metrics_collected, predictions_generated)
        
        # 5. Update performance tracking
        performance_update = self._update_performance_tracking()
        
        result = {
            "cycle_timestamp": datetime.now().isoformat(),
            "metrics_collected": len(metrics_collected),
            "predictions_generated": len(predictions_generated),
            "dashboard_updates": dashboard_updates,
            "insights": insights,
            "performance": performance_update
        }
        
        # Log analytics cycle
        self.analytics_log.append({
            "timestamp": datetime.now().isoformat(),
            "type": "analytics_cycle",
            "result": result
        })
        
        logger.info("Analytics cycle completed")
        return result
    
    async def _generate_all_predictions(self) -> List[Prediction]:
        """Generate predictions for all models."""
        all_predictions = []
        
        # Get recent metrics for prediction
        recent_metrics = self.metrics_collector.get_recent_metrics(hours=168)  # 7 days
        
        for model_name in self.predictive_analytics.models:
            predictions = await self.predictive_analytics.generate_predictions(
                model_name, recent_metrics
            )
            all_predictions.extend(predictions)
        
        return all_predictions
    
    async def _update_dashboards(self) -> Dict[str, Any]:
        """Update all dashboards with latest data."""
        dashboard_updates = {}
        
        for dashboard_id in self.dashboard_manager.dashboards:
            dashboard_data = self.dashboard_manager.get_dashboard_data(
                dashboard_id, self.metrics_collector, self.predictive_analytics
            )
            dashboard_updates[dashboard_id] = dashboard_data
        
        return dashboard_updates
    
    def _generate_insights(self, metrics: List[Metric], predictions: List[Prediction]) -> List[Dict[str, Any]]:
        """Generate insights from metrics and predictions."""
        insights = []
        
        # Revenue insights
        revenue_metrics = [m for m in metrics if m.category == "revenue"]
        if revenue_metrics:
            revenue_insight = self._analyze_revenue_trends(revenue_metrics)
            if revenue_insight:
                insights.append(revenue_insight)
        
        # Audience insights
        audience_metrics = [m for m in metrics if m.category == "audience"]
        if audience_metrics:
            audience_insight = self._analyze_audience_trends(audience_metrics)
            if audience_insight:
                insights.append(audience_insight)
        
        # Content insights
        content_metrics = [m for m in metrics if m.category == "content"]
        if content_metrics:
            content_insight = self._analyze_content_performance(content_metrics)
            if content_insight:
                insights.append(content_insight)
        
        # System insights
        system_metrics = [m for m in metrics if m.category == "system"]
        if system_metrics:
            system_insight = self._analyze_system_health(system_metrics)
            if system_insight:
                insights.append(system_insight)
        
        return insights
    
    def _analyze_revenue_trends(self, metrics: List[Metric]) -> Optional[Dict[str, Any]]:
        """Analyze revenue trends and generate insights."""
        if not metrics:
            return None
        
        # Get revenue growth rate
        growth_metrics = [m for m in metrics if m.name == "revenue_growth_rate"]
        if growth_metrics:
            latest_growth = growth_metrics[-1].value
            
            if latest_growth > 0.1:
                return {
                    "type": "positive_trend",
                    "category": "revenue",
                    "title": "Strong Revenue Growth",
                    "description": f"Revenue growing at {latest_growth:.1%} rate",
                    "severity": "high" if latest_growth > 0.2 else "medium"
                }
            elif latest_growth < -0.05:
                return {
                    "type": "negative_trend",
                    "category": "revenue",
                    "title": "Revenue Decline Detected",
                    "description": f"Revenue declining at {abs(latest_growth):.1%} rate",
                    "severity": "high"
                }
        
        return None
    
    def _analyze_audience_trends(self, metrics: List[Metric]) -> Optional[Dict[str, Any]]:
        """Analyze audience trends and generate insights."""
        if not metrics:
            return None
        
        # Get audience growth rate
        growth_metrics = [m for m in metrics if m.name == "audience_growth_rate"]
        if growth_metrics:
            latest_growth = growth_metrics[-1].value
            
            if latest_growth > 0.05:
                return {
                    "type": "positive_trend",
                    "category": "audience",
                    "title": "Audience Growing",
                    "description": f"Audience growing at {latest_growth:.1%} rate",
                    "severity": "medium"
                }
        
        return None
    
    def _analyze_content_performance(self, metrics: List[Metric]) -> Optional[Dict[str, Any]]:
        """Analyze content performance and generate insights."""
        if not metrics:
            return None
        
        # Get viral score
        viral_metrics = [m for m in metrics if m.name == "average_viral_score"]
        if viral_metrics:
            latest_viral = viral_metrics[-1].value
            
            if latest_viral > 0.8:
                return {
                    "type": "excellent_performance",
                    "category": "content",
                    "title": "High Viral Content",
                    "description": f"Content achieving {latest_viral:.1%} viral score",
                    "severity": "medium"
                }
            elif latest_viral < 0.3:
                return {
                    "type": "poor_performance",
                    "category": "content",
                    "title": "Low Content Performance",
                    "description": f"Content viral score at {latest_viral:.1%}",
                    "severity": "high"
                }
        
        return None
    
    def _analyze_system_health(self, metrics: List[Metric]) -> Optional[Dict[str, Any]]:
        """Analyze system health and generate insights."""
        if not metrics:
            return None
        
        # Get CPU usage
        cpu_metrics = [m for m in metrics if m.name == "cpu_usage"]
        if cpu_metrics:
            latest_cpu = cpu_metrics[-1].value
            
            if latest_cpu > 0.9:
                return {
                    "type": "system_warning",
                    "category": "system",
                    "title": "High CPU Usage",
                    "description": f"CPU usage at {latest_cpu:.1%}",
                    "severity": "high"
                }
        
        return None
    
    def _update_performance_tracking(self) -> Dict[str, Any]:
        """Update performance tracking metrics."""
        # Calculate analytics performance
        total_metrics = len(self.metrics_collector.metrics)
        total_predictions = len(self.predictive_analytics.prediction_cache)
        total_dashboards = len(self.dashboard_manager.dashboards)
        
        performance_data = {
            "timestamp": datetime.now().isoformat(),
            "total_metrics": total_metrics,
            "total_predictions": total_predictions,
            "total_dashboards": total_dashboards,
            "analytics_performance": "optimal"
        }
        
        self.performance_history.append(performance_data)
        
        # Keep only last 100 entries
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
        
        return performance_data
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary."""
        # Get recent metrics summary
        recent_metrics = self.metrics_collector.get_recent_metrics(hours=24)
        metrics_by_category = defaultdict(int)
        for metric in recent_metrics:
            metrics_by_category[metric.category] += 1
        
        # Get predictions summary
        total_predictions = sum(len(predictions) for predictions in self.predictive_analytics.prediction_cache.values())
        
        # Get dashboard summary
        dashboard_summary = {
            "total_dashboards": len(self.dashboard_manager.dashboards),
            "default_dashboards": len(self.dashboard_manager.default_dashboards)
        }
        
        return {
            "metrics_summary": {
                "total_metrics": len(self.metrics_collector.metrics),
                "recent_metrics": len(recent_metrics),
                "metrics_by_category": dict(metrics_by_category)
            },
            "predictions_summary": {
                "total_predictions": total_predictions,
                "active_models": len(self.predictive_analytics.models)
            },
            "dashboard_summary": dashboard_summary,
            "recent_analytics_log": self.analytics_log[-10:] if self.analytics_log else [],
            "performance_trend": self._calculate_performance_trend()
        }
    
    def _calculate_performance_trend(self) -> str:
        """Calculate performance trend from history."""
        if len(self.performance_history) < 2:
            return "insufficient_data"
        
        recent_performance = self.performance_history[-5:]
        older_performance = self.performance_history[-10:-5]
        
        if not older_performance:
            return "stable"
        
        recent_metrics = sum(p["total_metrics"] for p in recent_performance) / len(recent_performance)
        older_metrics = sum(p["total_metrics"] for p in older_performance) / len(older_performance)
        
        if recent_metrics > older_metrics * 1.1:
            return "improving"
        elif recent_metrics < older_metrics * 0.9:
            return "declining"
        else:
            return "stable" 

    def export_user_data(self, user_id: str) -> dict:
        """Stub: Export all data related to a user for compliance."""
        # In production, gather all data from analytics, audience, etc.
        logger.info(f"Exporting data for user: {user_id}")
        return {"user_id": user_id, "data": "[user data here]"}

    def delete_user_data(self, user_id: str) -> bool:
        """Stub: Delete all data related to a user for compliance."""
        logger.info(f"Deleting data for user: {user_id}")
        # In production, remove user data from all modules
        return True 

    async def agi_suggest_analytics_insights(self, context: dict) -> dict:
        return await self.agi_integration.suggest_analytics_insights(context) 

    def handle_event(self, event_type, payload):
        try:
            if event_type == 'create':
                result = self.create_report(payload)
            elif event_type == 'modify':
                result = self.modify_report(payload)
            elif event_type == 'explain':
                result = self.explain_output(payload)
            elif event_type == 'review':
                result = self.review_report(payload)
            elif event_type == 'approve':
                result = self.approve_report(payload)
            elif event_type == 'reject':
                result = self.reject_report(payload)
            elif event_type == 'feedback':
                result = self.feedback_report(payload)
            else:
                result = {"error": "Unknown event type"}
            log_action(event_type, result)
            return result
        except Exception as e:
            logger.error(f"Error handling event {event_type}: {e}")
            return {"error": str(e)}

    def create_report(self, payload):
        # TODO: Add compliance checks and human review hooks
        result = {"report_id": "ANL123", "status": "created", **payload}
        log_action('create', result)
        return result

    def modify_report(self, payload):
        # Simulate analytics modification
        result = {"analytics_id": payload.get('analytics_id'), "status": "modified", **payload}
        log_action('modify', result)
        return result

    def explain_output(self, result):
        if not result:
            return "No analytics data available."
        explanation = f"Analytics for {result.get('metric', 'N/A')}: value={result.get('value', 'N/A')}, trend={result.get('trend', 'N/A')}. Status: {result.get('status', 'N/A')}."
        if result.get('status') == 'pending_review':
            explanation += " This analytics report is pending human review."
        return explanation

    def review_report(self, payload):
        # Simulate review
        result = {"analytics_id": payload.get('analytics_id'), "status": "under_review"}
        log_action('review', result)
        return result

    def approve_report(self, payload):
        result = {"analytics_id": payload.get('analytics_id'), "status": "approved"}
        log_action('approve', result)
        return result

    def reject_report(self, payload):
        result = {"analytics_id": payload.get('analytics_id'), "status": "rejected"}
        log_action('reject', result)
        return result

    def feedback_report(self, payload):
        result = {"analytics_id": payload.get('analytics_id'), "status": "feedback_received", "feedback": payload.get('feedback')}
        log_action('feedback', result)
        return result

    def log_action(action, details):
        logger.info(f"AnalyticsDashboard action: {action} | {details}") 