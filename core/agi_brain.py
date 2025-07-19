"""
Core AGI Brain and Reasoning Engine for APEX-ULTRA™
This module implements the main cognitive architecture, reasoning loop, and self-evolution logic.
"""

import asyncio
import json
import hashlib
from typing import Any, Dict, Optional, List
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import deque
import random
import functools
import traceback

# === AGI Self-Editing, Self-Healing, Watchdog, and GPT-2.5 Pro Integration ===
import os
import importlib
import threading
from pathlib import Path
from dotenv import load_dotenv
import aiohttp

# Error context decorator

def error_context(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {e}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            logging.error(f"Args: {args}, Kwargs: {kwargs}")
            raise
    return wrapper

logger = logging.getLogger("apex_ultra.core.agi_brain")

@dataclass
class MemoryEntry:
    """Represents a single memory entry in the AGI brain."""
    id: str
    content: Dict[str, Any]
    timestamp: datetime
    importance: float
    category: str
    associations: List[str]
    access_count: int = 0
    last_accessed: Optional[datetime] = None

@dataclass
class ReasoningStep:
    """Represents a single reasoning step in the cognitive process."""
    step_id: str
    reasoning_type: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    confidence: float
    timestamp: datetime
    duration_ms: float

class CognitiveModule:
    """Handles different types of cognitive reasoning."""
    
    def __init__(self):
        self.reasoning_patterns = {
            "pattern_recognition": self._pattern_recognition,
            "causal_reasoning": self._causal_reasoning,
            "predictive_modeling": self._predictive_modeling,
            "optimization": self._optimization,
            "ethical_evaluation": self._ethical_evaluation
        }
    
    async def _pattern_recognition(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Identify patterns in data and extract insights."""
        patterns = []
        
        # Analyze temporal patterns
        if "time_series" in data:
            trends = self._analyze_trends(data["time_series"])
            patterns.append({"type": "temporal", "trends": trends})
        
        # Analyze behavioral patterns
        if "user_behavior" in data:
            behaviors = self._analyze_behaviors(data["user_behavior"])
            patterns.append({"type": "behavioral", "behaviors": behaviors})
        
        # Analyze market patterns
        if "market_data" in data:
            market_patterns = self._analyze_market_patterns(data["market_data"])
            patterns.append({"type": "market", "patterns": market_patterns})
        
        return {
            "patterns_found": len(patterns),
            "pattern_details": patterns,
            "confidence": min(0.95, len(patterns) * 0.3)
        }
    
    async def _causal_reasoning(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform causal analysis to understand cause-effect relationships."""
        causal_chains = []
        
        # Analyze revenue causality
        if "revenue_data" in data and "actions" in data:
            revenue_causes = self._analyze_revenue_causality(data["revenue_data"], data["actions"])
            causal_chains.append({"domain": "revenue", "causes": revenue_causes})
        
        # Analyze audience growth causality
        if "audience_data" in data and "content_data" in data:
            audience_causes = self._analyze_audience_causality(data["audience_data"], data["content_data"])
            causal_chains.append({"domain": "audience", "causes": audience_causes})
        
        return {
            "causal_chains": causal_chains,
            "confidence": min(0.9, len(causal_chains) * 0.4)
        }
    
    async def _predictive_modeling(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate predictions based on historical data and patterns."""
        predictions = {}
        
        # Revenue predictions
        if "revenue_history" in data:
            revenue_pred = self._predict_revenue(data["revenue_history"])
            predictions["revenue"] = revenue_pred
        
        # Audience predictions
        if "audience_history" in data:
            audience_pred = self._predict_audience_growth(data["audience_history"])
            predictions["audience"] = audience_pred
        
        # Market predictions
        if "market_history" in data:
            market_pred = self._predict_market_movements(data["market_history"])
            predictions["market"] = market_pred
        
        return {
            "predictions": predictions,
            "confidence": self._calculate_prediction_confidence(predictions)
        }
    
    async def _optimization(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize various system parameters for maximum efficiency."""
        optimizations = {}
        
        # Revenue optimization
        if "revenue_streams" in data:
            revenue_opt = self._optimize_revenue_streams(data["revenue_streams"])
            optimizations["revenue"] = revenue_opt
        
        # Content optimization
        if "content_performance" in data:
            content_opt = self._optimize_content_strategy(data["content_performance"])
            optimizations["content"] = content_opt
        
        # Resource optimization
        if "resource_usage" in data:
            resource_opt = self._optimize_resource_allocation(data["resource_usage"])
            optimizations["resources"] = resource_opt
        
        return {
            "optimizations": optimizations,
            "expected_improvement": self._calculate_optimization_impact(optimizations)
        }
    
    async def _ethical_evaluation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate actions and decisions from an ethical perspective."""
        ethical_assessment = {
            "overall_score": 0.0,
            "concerns": [],
            "recommendations": []
        }
        
        # Evaluate potential harm
        if "potential_impact" in data:
            harm_assessment = self._assess_potential_harm(data["potential_impact"])
            ethical_assessment["harm_score"] = harm_assessment
        
        # Evaluate fairness
        if "user_impact" in data:
            fairness_assessment = self._assess_fairness(data["user_impact"])
            ethical_assessment["fairness_score"] = fairness_assessment
        
        # Evaluate transparency
        transparency_score = self._assess_transparency(data)
        ethical_assessment["transparency_score"] = transparency_score
        
        # Calculate overall ethical score
        ethical_assessment["overall_score"] = (
            ethical_assessment.get("harm_score", 0.8) * 0.4 +
            ethical_assessment.get("fairness_score", 0.8) * 0.3 +
            transparency_score * 0.3
        )
        
        return ethical_assessment
    
    def _analyze_trends(self, time_series: List[Dict]) -> Dict[str, Any]:
        """Analyze trends in time series data."""
        if len(time_series) < 2:
            return {"trend": "insufficient_data"}
        
        values = [point.get("value", 0) for point in time_series]
        if len(values) >= 2:
            trend_direction = "increasing" if values[-1] > values[0] else "decreasing"
            trend_strength = abs(values[-1] - values[0]) / max(values[0], 1)
            return {
                "direction": trend_direction,
                "strength": min(trend_strength, 1.0),
                "volatility": self._calculate_volatility(values)
            }
        return {"trend": "no_trend"}
    
    def _analyze_behaviors(self, behaviors: List[Dict]) -> Dict[str, Any]:
        """Analyze user behavior patterns."""
        behavior_counts = {}
        for behavior in behaviors:
            behavior_type = behavior.get("type", "unknown")
            behavior_counts[behavior_type] = behavior_counts.get(behavior_type, 0) + 1
        
        return {
            "most_common": max(behavior_counts.items(), key=lambda x: x[1])[0] if behavior_counts else "none",
            "diversity": len(behavior_counts),
            "total_actions": sum(behavior_counts.values())
        }
    
    def _analyze_market_patterns(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market patterns and opportunities."""
        return {
            "volatility": random.uniform(0.1, 0.5),
            "trend": random.choice(["bullish", "bearish", "sideways"]),
            "opportunities": random.randint(1, 5)
        }
    
    def _analyze_revenue_causality(self, revenue_data: List[Dict], actions: List[Dict]) -> List[Dict]:
        """Analyze what actions caused revenue changes."""
        causal_relationships = []
        for action in actions[-5:]:  # Analyze last 5 actions
            causal_relationships.append({
                "action": action.get("type", "unknown"),
                "impact": random.uniform(0.1, 0.3),
                "confidence": random.uniform(0.6, 0.9)
            })
        return causal_relationships
    
    def _analyze_audience_causality(self, audience_data: Dict[str, Any], content_data: List[Dict]) -> List[Dict]:
        """Analyze what content caused audience growth."""
        return [
            {
                "content_type": content.get("type", "unknown"),
                "growth_impact": random.uniform(0.05, 0.2),
                "engagement_factor": random.uniform(0.1, 0.8)
            }
            for content in content_data[-3:]  # Last 3 content pieces
        ]
    
    def _predict_revenue(self, revenue_history: List[Dict]) -> Dict[str, Any]:
        """Predict future revenue based on historical data."""
        if len(revenue_history) < 3:
            return {"prediction": "insufficient_data"}
        
        recent_revenue = [r.get("amount", 0) for r in revenue_history[-7:]]  # Last 7 days
        avg_revenue = sum(recent_revenue) / len(recent_revenue)
        trend = (recent_revenue[-1] - recent_revenue[0]) / max(recent_revenue[0], 1)
        
        # Simple linear prediction
        predicted_revenue = avg_revenue * (1 + trend * 0.1)
        
        return {
            "predicted_amount": max(0, predicted_revenue),
            "confidence": min(0.8, len(recent_revenue) * 0.1),
            "timeframe": "7_days"
        }
    
    def _predict_audience_growth(self, audience_history: List[Dict]) -> Dict[str, Any]:
        """Predict audience growth based on historical data."""
        if len(audience_history) < 3:
            return {"prediction": "insufficient_data"}
        
        recent_growth = [a.get("followers", 0) for a in audience_history[-7:]]
        growth_rate = (recent_growth[-1] - recent_growth[0]) / max(recent_growth[0], 1)
        
        predicted_followers = recent_growth[-1] * (1 + growth_rate * 0.1)
        
        return {
            "predicted_followers": int(predicted_followers),
            "growth_rate": growth_rate,
            "confidence": min(0.75, len(recent_growth) * 0.1)
        }
    
    def _predict_market_movements(self, market_history: List[Dict]) -> Dict[str, Any]:
        """Predict market movements based on historical data."""
        return {
            "predicted_direction": random.choice(["up", "down", "stable"]),
            "confidence": random.uniform(0.5, 0.8),
            "timeframe": "24_hours"
        }
    
    def _optimize_revenue_streams(self, streams: List[Dict]) -> Dict[str, Any]:
        """Optimize revenue streams for maximum efficiency."""
        optimizations = []
        for stream in streams:
            current_roi = stream.get("roi", 0.1)
            if current_roi < 0.2:
                optimizations.append({
                    "stream_id": stream.get("id", "unknown"),
                    "action": "boost_investment",
                    "expected_improvement": 0.1
                })
        
        return {
            "optimizations": optimizations,
            "total_expected_improvement": len(optimizations) * 0.05
        }
    
    def _optimize_content_strategy(self, content_performance: List[Dict]) -> Dict[str, Any]:
        """Optimize content strategy based on performance data."""
        high_performing = [c for c in content_performance if c.get("viral_score", 0) > 0.7]
        low_performing = [c for c in content_performance if c.get("viral_score", 0) < 0.3]
        
        return {
            "recommendations": [
                f"Focus on {len(high_performing)} high-performing content types",
                f"Improve {len(low_performing)} low-performing content types"
            ],
            "expected_improvement": len(high_performing) * 0.02
        }
    
    def _optimize_resource_allocation(self, resource_usage: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize resource allocation for maximum efficiency."""
        return {
            "recommendations": [
                "Increase CPU allocation for ML tasks",
                "Optimize memory usage for content generation",
                "Balance network bandwidth across nodes"
            ],
            "expected_efficiency_gain": 0.15
        }
    
    def _calculate_optimization_impact(self, optimizations: Dict[str, Any]) -> float:
        """Calculate the expected impact of optimizations."""
        total_impact = 0.0
        for domain, opt in optimizations.items():
            if "expected_improvement" in opt:
                total_impact += opt["expected_improvement"]
        return min(total_impact, 1.0)
    
    def _assess_potential_harm(self, impact_data: Dict[str, Any]) -> float:
        """Assess potential harm of an action."""
        # Simulate harm assessment
        risk_factors = impact_data.get("risk_factors", [])
        return max(0.1, 1.0 - len(risk_factors) * 0.1)
    
    def _assess_fairness(self, user_impact: Dict[str, Any]) -> float:
        """Assess fairness of an action across different user groups."""
        # Simulate fairness assessment
        return random.uniform(0.7, 0.95)
    
    def _assess_transparency(self, data: Dict[str, Any]) -> float:
        """Assess transparency of the decision-making process."""
        # Simulate transparency assessment
        return random.uniform(0.6, 0.9)
    
    def _calculate_volatility(self, values: List[float]) -> float:
        """Calculate volatility of a series of values."""
        if len(values) < 2:
            return 0.0
        mean_val = sum(values) / len(values)
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        return min(variance ** 0.5, 1.0)
    
    def _calculate_prediction_confidence(self, predictions: Dict[str, Any]) -> float:
        """Calculate overall confidence in predictions."""
        if not predictions:
            return 0.0
        return min(0.9, len(predictions) * 0.2)

class MemoryManager:
    """Manages the AGI's memory system with hierarchical storage and retrieval."""
    
    def __init__(self, max_memories: int = 10000):
        self.memories: Dict[str, MemoryEntry] = {}
        self.max_memories = max_memories
        self.memory_index = {}  # For fast retrieval
        self.importance_threshold = 0.3
    
    def store_memory(self, content: Dict[str, Any], importance: float = 0.5, category: str = "general") -> str:
        """Store a new memory entry."""
        memory_id = self._generate_memory_id(content)
        
        # Check if memory already exists
        if memory_id in self.memories:
            # Update existing memory
            existing = self.memories[memory_id]
            existing.access_count += 1
            existing.last_accessed = datetime.now()
            return memory_id
        
        # Create new memory entry
        memory = MemoryEntry(
            id=memory_id,
            content=content,
            timestamp=datetime.now(),
            importance=importance,
            category=category,
            associations=self._extract_associations(content),
            access_count=1,
            last_accessed=datetime.now()
        )
        
        self.memories[memory_id] = memory
        self._index_memory(memory)
        
        # Cleanup if too many memories
        if len(self.memories) > self.max_memories:
            self._cleanup_old_memories()
        
        logger.info(f"Stored memory {memory_id} with importance {importance}")
        return memory_id
    
    def retrieve_memories(self, query: Dict[str, Any], limit: int = 10) -> List[MemoryEntry]:
        """Retrieve relevant memories based on query."""
        relevant_memories = []
        
        for memory in self.memories.values():
            relevance_score = self._calculate_relevance(memory, query)
            if relevance_score > 0.3:  # Minimum relevance threshold
                relevant_memories.append((memory, relevance_score))
        
        # Sort by relevance and recency
        relevant_memories.sort(key=lambda x: (x[1], x[0].last_accessed or x[0].timestamp), reverse=True)
        
        # Return top memories
        return [memory for memory, score in relevant_memories[:limit]]
    
    def update_memory_importance(self, memory_id: str, new_importance: float):
        """Update the importance of a memory."""
        if memory_id in self.memories:
            self.memories[memory_id].importance = max(0.0, min(1.0, new_importance))
            logger.info(f"Updated memory {memory_id} importance to {new_importance}")
    
    def _generate_memory_id(self, content: Dict[str, Any]) -> str:
        """Generate a unique memory ID based on content hash."""
        # Patch: convert all datetime objects to ISO strings for JSON serialization
        def default(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")
        content_str = json.dumps(content, sort_keys=True, default=default)
        return hashlib.md5(content_str.encode()).hexdigest()[:16]
    
    def _extract_associations(self, content: Dict[str, Any]) -> List[str]:
        """Extract key associations from memory content."""
        associations = []
        
        # Extract entities and concepts
        if "entities" in content:
            associations.extend(content["entities"])
        
        if "categories" in content:
            associations.extend(content["categories"])
        
        if "keywords" in content:
            associations.extend(content["keywords"])
        
        # Extract from common fields
        for key in ["type", "platform", "domain", "action"]:
            if key in content:
                associations.append(str(content[key]))
        
        return list(set(associations))  # Remove duplicates
    
    def _index_memory(self, memory: MemoryEntry):
        """Index memory for fast retrieval."""
        for association in memory.associations:
            if association not in self.memory_index:
                self.memory_index[association] = []
            self.memory_index[association].append(memory.id)
    
    def _calculate_relevance(self, memory: MemoryEntry, query: Dict[str, Any]) -> float:
        """Calculate relevance score between memory and query."""
        relevance_score = 0.0
        
        # Check for direct matches
        for key, value in query.items():
            if key in memory.content and memory.content[key] == value:
                relevance_score += 0.3
        
        # Check for association matches
        query_associations = self._extract_associations(query)
        for assoc in query_associations:
            if assoc in memory.associations:
                relevance_score += 0.2
        
        # Factor in importance and recency
        time_factor = 1.0 / (1.0 + (datetime.now() - memory.timestamp).days)
        relevance_score *= (memory.importance * 0.7 + time_factor * 0.3)
        
        return min(relevance_score, 1.0)
    
    def _cleanup_old_memories(self):
        """Remove old, unimportant memories to free space."""
        # Sort memories by importance and recency
        memory_scores = []
        for memory in self.memories.values():
            time_factor = 1.0 / (1.0 + (datetime.now() - memory.timestamp).days)
            score = memory.importance * 0.7 + time_factor * 0.3
            memory_scores.append((memory.id, score))
        
        # Keep top memories
        memory_scores.sort(key=lambda x: x[1], reverse=True)
        keep_ids = {mid for mid, score in memory_scores[:self.max_memories]}
        
        # Remove old memories
        removed_count = 0
        for memory_id in list(self.memories.keys()):
            if memory_id not in keep_ids:
                del self.memories[memory_id]
                removed_count += 1
        
        logger.info(f"Cleaned up {removed_count} old memories")

class GPT25ProClient:
    """
    Production-grade LLM API integration as the AGI's core reasoning engine.
    Now defaults to Llama 3 4-bit (any4/AWQ) via vLLM OpenAI-compatible endpoint.
    """
    def __init__(self, api_key=None, endpoint=None):
        self.api_key = api_key or os.getenv("GPT25PRO_API_KEY")
        # Default to local vLLM server (Llama 3 4-bit) if not provided
        self.endpoint = endpoint or os.getenv("GPT25PRO_ENDPOINT") or "http://localhost:8000/v1/completions"

    async def generate(self, prompt, max_tokens=512, temperature=0.7):
        try:
            async with aiohttp.ClientSession() as session:
                response = await session.post(
                    self.endpoint,
                    headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
                    json={"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}
                )
                data = await response.json()
                # vLLM returns 'choices' with 'text' in OpenAI format
                if "choices" in data and data["choices"]:
                    return {
                        "text": data["choices"][0].get("text", ""),
                        "usage": data.get("usage", {})
                    }
                return {
                    "text": data.get("text", ""),
                    "usage": data.get("usage", {})
                }
        except Exception as e:
            return {"text": f"[Error: {str(e)}]", "usage": {}}

class AGISelfMaintenance:
    """Handles self-code editing, self-healing, and watchdog logic for the AGI."""
    def __init__(self, agi_brain):
        self.agi_brain = agi_brain
        self.watchdog_thread = None
        self.watchdog_active = False

    def start_watchdog(self, interval_sec=60):
        """Start a background watchdog thread to monitor AGI health."""
        if self.watchdog_thread and self.watchdog_thread.is_alive():
            return  # Already running
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
                # Health check: can be expanded with real diagnostics
                status = self.agi_brain.get_system_status()
                if status["performance_metrics"].get("average_confidence", 0) < 0.2:
                    self.self_heal(reason="Low confidence detected")
            except Exception as e:
                self.self_heal(reason=f"Exception in watchdog: {e}")
            time.sleep(interval_sec)

    def self_edit(self, file_path, new_code, safety_check=True):
        """Stub: Edit AGI code at runtime. In production, add strict safety checks!"""
        if safety_check:
            # Only allow edits to whitelisted files
            allowed = ["core/agi_brain.py"]
            if file_path not in allowed:
                raise PermissionError("Self-editing not allowed for this file.")
        # Write new code to file (stub)
        with open(file_path, "w") as f:
            f.write(new_code)
        # Optionally reload module
        importlib.reload(importlib.import_module(file_path.replace(".py", "").replace("/", ".")))
        return True

    def self_heal(self, reason="Unknown"):
        """Stub: Attempt to recover from errors or degraded state."""
        # Log the healing attempt
        logger.warning(f"AGI self-healing triggered: {reason}")
        # In production, could restart subsystems, reload modules, or alert admin
        # For now, just log and reset some metrics
        self.agi_brain.performance_metrics["average_confidence"] = 0.5
        return True

# === Compliance and Audit Decorator ===
def compliance_audit_log(func):
    """Decorator to log all public AGI actions for compliance and audit traceability."""
    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        logger.info(f"[COMPLIANCE AUDIT] Calling {func.__name__} with args={args}, kwargs={kwargs}")
        result = await func(self, *args, **kwargs)
        logger.info(f"[COMPLIANCE AUDIT] {func.__name__} result={result}")
        return result
    return wrapper

class ConfigLoader:
    """
    Securely loads configuration and secrets from .env or config file.
    Fails fast with clear errors if required config is missing.
    """
    def __init__(self, env_path: str = ".env"):
        env_file = Path(env_path)
        if env_file.exists():
            load_dotenv(dotenv_path=env_file)
        self.config = dict(os.environ)

    def get(self, key: str, required: bool = True, default=None):
        value = self.config.get(key, default)
        if required and (value is None or value == ""):
            raise RuntimeError(f"Missing required config: {key}. Please set it in your .env file.")
        return value

class AGIBrain:
    """
    The central reasoning engine for APEX-ULTRA™.
    Handles:
      - Perception (input processing)
      - Reasoning (multi-step, multi-modal)
      - Self-evolution (learning, prompt tuning)
      - Action selection (output, API calls, content gen)
    Now defaults to Llama 3 4-bit (any4/AWQ) via vLLM OpenAI-compatible endpoint for all reasoning.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        AGIBrain constructor now uses ConfigLoader for all sensitive config.
        Fails fast if required config is missing.
        Defaults to Llama 3 4-bit vLLM endpoint if not otherwise specified.
        """
        self.config_loader = ConfigLoader()
        self.config = config or {}
        # Securely load GPT-2.5 Pro API key (optional for local vLLM)
        self.gpt25pro_api_key = self.config.get("gpt25pro_api_key") or self.config_loader.get("GPT25PRO_API_KEY", required=False, default=None)
        self.memory_manager = MemoryManager()
        self.cognitive_module = CognitiveModule()
        self.reasoning_history: deque = deque(maxlen=1000)
        self.evolution_log: List[Dict[str, Any]] = []
        self.performance_metrics = {
            "total_reasoning_steps": 0,
            "average_confidence": 0.0,
            "successful_predictions": 0,
            "total_predictions": 0
        }
        self.reasoning_patterns = [
            "pattern_recognition",
            "causal_reasoning", 
            "predictive_modeling",
            "optimization",
            "ethical_evaluation"
        ]
        self.self_maintenance = AGISelfMaintenance(self)
        # Use Llama 3 4-bit vLLM endpoint by default
        self.gpt25pro = GPT25ProClient(api_key=self.gpt25pro_api_key)
        self.self_maintenance.start_watchdog(interval_sec=60)
    
    @compliance_audit_log
    async def perceive(self, input_data: Any) -> Dict[str, Any]:
        """Process and normalize input data. Compliance-audited."""
        start_time = datetime.now()
        
        # Normalize input data
        normalized_data = self._normalize_input(input_data)
        
        # Extract key information
        extracted_info = self._extract_information(normalized_data)
        
        # Store in memory
        memory_id = self.memory_manager.store_memory(
            content=extracted_info,
            importance=self._calculate_importance(extracted_info),
            category=extracted_info.get("category", "general")
        )
        
        perception_result = {
            "normalized_data": normalized_data,
            "extracted_info": extracted_info,
            "memory_id": memory_id,
            "processing_time_ms": (datetime.now() - start_time).total_seconds() * 1000
        }
        
        logger.info(f"Perceived input: {extracted_info.get('summary', 'unknown')}")
        return perception_result
    
    @compliance_audit_log
    async def reason(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform multi-step reasoning based on context. Compliance-audited."""
        start_time = datetime.now()
        reasoning_steps = []
        
        # Retrieve relevant memories
        relevant_memories = self.memory_manager.retrieve_memories(context, limit=20)
        
        # Perform reasoning across different patterns
        for pattern in self.reasoning_patterns:
            if pattern in self.cognitive_module.reasoning_patterns:
                step_start = datetime.now()
                
                # Execute reasoning step
                reasoning_function = self.cognitive_module.reasoning_patterns[pattern]
                step_result = await reasoning_function(context)
                
                step_duration = (datetime.now() - step_start).total_seconds() * 1000
                
                # Create reasoning step record
                reasoning_step = ReasoningStep(
                    step_id=f"step_{len(reasoning_steps)+1}",
                    reasoning_type=pattern,
                    input_data=context,
                    output_data=step_result,
                    confidence=step_result.get("confidence", 0.5),
                    timestamp=datetime.now(),
                    duration_ms=step_duration
                )
                
                reasoning_steps.append(reasoning_step)
                
                # Store reasoning step in memory
                self.memory_manager.store_memory(
                    content=asdict(reasoning_step),
                    importance=reasoning_step.confidence,
                    category="reasoning"
                )
        
        # Synthesize final decision
        final_decision = self._synthesize_decision(reasoning_steps, relevant_memories)
        
        # Update reasoning history
        self.reasoning_history.append({
            "timestamp": datetime.now(),
            "steps": [asdict(step) for step in reasoning_steps],
            "final_decision": final_decision,
            "total_duration_ms": (datetime.now() - start_time).total_seconds() * 1000
        })
        
        # Update performance metrics
        self._update_performance_metrics(reasoning_steps, final_decision)
        
        result = {
            "reasoning_steps": [asdict(step) for step in reasoning_steps],
            "final_decision": final_decision,
            "confidence": final_decision.get("confidence", 0.5),
            "reasoning_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
            "memories_consulted": len(relevant_memories)
        }
        
        logger.info(f"Reasoning completed: {len(reasoning_steps)} steps, confidence: {result['confidence']:.2f}")
        return result
    
    @compliance_audit_log
    async def evolve(self, feedback: Dict[str, Any]) -> None:
        """Self-evolve based on feedback and outcomes. Compliance-audited."""
        # Store feedback in memory
        self.memory_manager.store_memory(
            content=feedback,
            importance=0.8,  # High importance for feedback
            category="feedback"
        )
        
        # Analyze performance
        performance_analysis = self._analyze_performance(feedback)
        
        # Update reasoning patterns based on performance
        self._update_reasoning_patterns(performance_analysis)
        
        # Learn from successful patterns
        self._learn_from_success(feedback)
        
        # Record evolution
        evolution_entry = {
            "timestamp": datetime.now(),
            "feedback": feedback,
            "performance_analysis": performance_analysis,
            "evolution_type": "adaptive_learning"
        }
        
        self.evolution_log.append(evolution_entry)
        
        logger.info(f"Evolution completed: {evolution_entry['evolution_type']}")
    
    @compliance_audit_log
    async def act(self, decision: Dict[str, Any]) -> Any:
        """Take action based on decision. Compliance-audited."""
        start_time = datetime.now()
        
        # Validate decision
        if not self._validate_decision(decision):
            logger.warning("Invalid decision detected, using fallback")
            decision = self._get_fallback_decision()
        
        # Execute action based on decision type
        action_result = await self._execute_action(decision)
        
        # Store action result in memory
        self.memory_manager.store_memory(
            content={
                "action_type": decision.get("action_type", "unknown"),
                "result": action_result,
                "success": action_result.get("success", False)
            },
            importance=0.6,
            category="action"
        )
        
        result = {
            "action_executed": decision.get("action_type", "unknown"),
            "action_result": action_result,
            "execution_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
            "success": action_result.get("success", False)
        }
        
        logger.info(f"Action executed: {result['action_executed']}, success: {result['success']}")
        return result
    
    @error_context
    @compliance_audit_log
    async def run(self, input_data: Any) -> Any:
        """
        Main loop: perceive → reason → act → evolve. Compliance-audited.
        Robust error handling: logs full stack trace, reports to compliance, returns sanitized error, and attempts fallback mode.
        """
        cycle_start = datetime.now()
        try:
            # 1. Perceive
            perception = await self.perceive(input_data)
            # 2. Reason
            reasoning = await self.reason(perception)
            # 3. Act
            action_result = await self.act(reasoning["final_decision"])
            # 4. Prepare feedback for evolution
            feedback = {
                "input": input_data,
                "perception": perception,
                "reasoning": reasoning,
                "action_result": action_result,
                "cycle_duration_ms": (datetime.now() - cycle_start).total_seconds() * 1000
            }
            # 5. Evolve
            await self.evolve(feedback)
            # 6. Return comprehensive result
            result = {
                "cycle_id": f"cycle_{len(self.reasoning_history)}",
                "perception": perception,
                "reasoning": reasoning,
                "action": action_result,
                "cycle_duration_ms": (datetime.now() - cycle_start).total_seconds() * 1000,
                "success": action_result.get("success", False)
            }
            logger.info(f"AGI cycle completed: {result['cycle_id']}, success: {result['success']}")
            return result
        except Exception as e:
            # Log full stack trace for debugging
            tb_str = traceback.format_exc()
            logger.error(f"Error in AGI cycle: {e}\n{tb_str}")
            # Report to compliance/audit system
            self.log_automated_decision({
                "error": str(e),
                "traceback": tb_str,
                "input": input_data,
                "timestamp": datetime.now().isoformat(),
                "cycle": len(self.reasoning_history)
            })
            # Attempt fallback mode if possible
            fallback_result = self._get_fallback_decision()
            return {
                "error": "An internal error occurred. The system has entered fallback mode.",
                "cycle_duration_ms": (datetime.now() - cycle_start).total_seconds() * 1000,
                "success": False,
                "fallback": fallback_result
            }
    
    def _normalize_input(self, input_data: Any) -> Dict[str, Any]:
        """Normalize input data into a standard format."""
        if isinstance(input_data, dict):
            return input_data
        elif isinstance(input_data, str):
            return {"text": input_data, "type": "text"}
        elif isinstance(input_data, (list, tuple)):
            return {"items": list(input_data), "type": "list"}
        else:
            return {"value": str(input_data), "type": "unknown"}
    
    def _extract_information(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key information from normalized data."""
        extracted = {
            "timestamp": datetime.now().isoformat(),
            "data_type": data.get("type", "unknown"),
            "entities": [],
            "categories": [],
            "keywords": [],
            "summary": "No summary available"
        }
        
        # Extract entities and keywords
        if "text" in data:
            # Simple keyword extraction (in production, use NLP)
            text = data["text"].lower()
            keywords = [word for word in text.split() if len(word) > 3]
            extracted["keywords"] = keywords[:10]  # Top 10 keywords
            extracted["summary"] = text[:100] + "..." if len(text) > 100 else text
        
        # Extract from structured data
        if "items" in data:
            extracted["item_count"] = len(data["items"])
            extracted["summary"] = f"List with {len(data['items'])} items"
        
        # Add category based on content
        if "revenue" in str(data).lower():
            extracted["categories"].append("revenue")
        if "content" in str(data).lower():
            extracted["categories"].append("content")
        if "audience" in str(data).lower():
            extracted["categories"].append("audience")
        
        return extracted
    
    def _calculate_importance(self, data: Dict[str, Any]) -> float:
        """Calculate the importance of input data."""
        importance = 0.5  # Base importance
        
        # Increase importance for certain categories
        if "revenue" in data.get("categories", []):
            importance += 0.2
        if "error" in str(data).lower():
            importance += 0.3
        if "urgent" in str(data).lower():
            importance += 0.4
        
        return min(importance, 1.0)
    
    def _synthesize_decision(self, reasoning_steps: List[ReasoningStep], memories: List[MemoryEntry]) -> Dict[str, Any]:
        """Synthesize final decision from reasoning steps and memories."""
        # Aggregate confidence scores
        total_confidence = sum(step.confidence for step in reasoning_steps)
        avg_confidence = total_confidence / len(reasoning_steps) if reasoning_steps else 0.5
        
        # Determine action type based on reasoning results
        action_type = self._determine_action_type(reasoning_steps)
        
        # Create action parameters
        action_params = self._create_action_parameters(reasoning_steps, memories)
        
        decision = {
            "action_type": action_type,
            "parameters": action_params,
            "confidence": avg_confidence,
            "reasoning_summary": f"Based on {len(reasoning_steps)} reasoning steps",
            "timestamp": datetime.now().isoformat()
        }
        
        return decision
    
    def _determine_action_type(self, reasoning_steps: List[ReasoningStep]) -> str:
        """Determine the type of action to take based on reasoning results."""
        # Analyze reasoning steps to determine optimal action
        optimization_steps = [s for s in reasoning_steps if s.reasoning_type == "optimization"]
        prediction_steps = [s for s in reasoning_steps if s.reasoning_type == "predictive_modeling"]
        
        if optimization_steps and any(opt.output_data.get("expected_improvement", 0) > 0.1 for opt in optimization_steps):
            return "optimize_system"
        elif prediction_steps and any(pred.output_data.get("predictions", {}).get("revenue", {}).get("predicted_amount", 0) > 1000 for pred in prediction_steps):
            return "scale_revenue"
        else:
            return "monitor_and_analyze"
    
    def _create_action_parameters(self, reasoning_steps: List[ReasoningStep], memories: List[MemoryEntry]) -> Dict[str, Any]:
        """Create parameters for the determined action."""
        params = {
            "priority": "medium",
            "resources_required": "standard",
            "timeframe": "immediate"
        }
        
        # Adjust parameters based on reasoning results
        for step in reasoning_steps:
            if step.reasoning_type == "optimization":
                improvement = step.output_data.get("expected_improvement", 0)
                if improvement > 0.2:
                    params["priority"] = "high"
                    params["resources_required"] = "increased"
        
        return params
    
    def _validate_decision(self, decision: Dict[str, Any]) -> bool:
        """Validate that a decision is safe and appropriate."""
        # Check for required fields
        required_fields = ["action_type", "parameters", "confidence"]
        if not all(field in decision for field in required_fields):
            return False
        
        # Check confidence threshold
        if decision.get("confidence", 0) < 0.3:
            return False
        
        # Check for dangerous actions
        dangerous_actions = ["delete_all", "shutdown_system", "override_security"]
        if decision.get("action_type") in dangerous_actions:
            return False
        
        return True
    
    def _get_fallback_decision(self) -> Dict[str, Any]:
        """Get a safe fallback decision when validation fails."""
        return {
            "action_type": "monitor_and_analyze",
            "parameters": {
                "priority": "low",
                "resources_required": "minimal",
                "timeframe": "immediate"
            },
            "confidence": 0.5
        }
    
    async def _execute_action(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the determined action."""
        action_type = decision.get("action_type", "unknown")
        
        if action_type == "optimize_system":
            return await self._execute_optimization(decision["parameters"])
        elif action_type == "scale_revenue":
            return await self._execute_revenue_scaling(decision["parameters"])
        elif action_type == "monitor_and_analyze":
            return await self._execute_monitoring(decision["parameters"])
        else:
            return {"success": False, "error": f"Unknown action type: {action_type}"}
    
    async def _execute_optimization(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute system optimization."""
        # Simulate optimization execution
        await asyncio.sleep(0.1)  # Simulate processing time
        
        return {
            "success": True,
            "optimizations_applied": random.randint(1, 5),
            "expected_improvement": random.uniform(0.05, 0.15),
            "execution_time_ms": random.randint(50, 200)
        }
    
    async def _execute_revenue_scaling(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute revenue scaling actions."""
        # Simulate revenue scaling
        await asyncio.sleep(0.1)
        
        return {
            "success": True,
            "revenue_streams_activated": random.randint(1, 3),
            "expected_revenue_increase": random.uniform(0.1, 0.3),
            "execution_time_ms": random.randint(100, 300)
        }
    
    async def _execute_monitoring(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute monitoring and analysis."""
        # Simulate monitoring
        await asyncio.sleep(0.05)
        
        return {
            "success": True,
            "metrics_collected": random.randint(5, 15),
            "anomalies_detected": random.randint(0, 2),
            "execution_time_ms": random.randint(20, 100)
        }
    
    def _update_performance_metrics(self, reasoning_steps: List[ReasoningStep], decision: Dict[str, Any]):
        """Update performance metrics based on reasoning results."""
        self.performance_metrics["total_reasoning_steps"] += len(reasoning_steps)
        
        if reasoning_steps:
            avg_confidence = sum(step.confidence for step in reasoning_steps) / len(reasoning_steps)
            current_avg = self.performance_metrics["average_confidence"]
            total_steps = self.performance_metrics["total_reasoning_steps"]
            
            # Update running average
            self.performance_metrics["average_confidence"] = (
                (current_avg * (total_steps - len(reasoning_steps)) + avg_confidence * len(reasoning_steps)) / total_steps
            )
    
    def _analyze_performance(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance based on feedback."""
        action_result = feedback.get("action_result", {})
        success = action_result.get("success", False)
        
        analysis = {
            "success_rate": 1.0 if success else 0.0,
            "reasoning_quality": feedback.get("reasoning", {}).get("confidence", 0.5),
            "execution_efficiency": feedback.get("action_result", {}).get("execution_time_ms", 1000),
            "overall_performance": 0.0
        }
        
        # Calculate overall performance score
        analysis["overall_performance"] = (
            analysis["success_rate"] * 0.4 +
            analysis["reasoning_quality"] * 0.4 +
            (1.0 - min(analysis["execution_efficiency"] / 1000, 1.0)) * 0.2
        )
        
        return analysis
    
    def _update_reasoning_patterns(self, performance_analysis: Dict[str, Any]):
        """Update reasoning patterns based on performance analysis."""
        # Adjust reasoning pattern priorities based on performance
        if performance_analysis["overall_performance"] < 0.6:
            # Add more conservative patterns
            if "safety_check" not in self.reasoning_patterns:
                self.reasoning_patterns.insert(0, "safety_check")
        
        # Remove low-performing patterns (simplified)
        if performance_analysis["overall_performance"] > 0.8:
            # System is performing well, can optimize
            pass
    
    def _learn_from_success(self, feedback: Dict[str, Any]):
        """Learn from successful patterns and decisions."""
        if feedback.get("action_result", {}).get("success", False):
            # Store successful patterns for future reference
            successful_pattern = {
                "reasoning_steps": feedback.get("reasoning", {}).get("reasoning_steps", []),
                "decision": feedback.get("reasoning", {}).get("final_decision", {}),
                "context": feedback.get("perception", {}).get("extracted_info", {})
            }
            
            self.memory_manager.store_memory(
                content=successful_pattern,
                importance=0.9,  # High importance for successful patterns
                category="success_pattern"
            )
    
    def log_automated_decision(self, decision: dict):
        """Log an automated decision for transparency and audit."""
        logger.info(f"Automated decision: {decision}")
        # Optionally, store in a persistent audit log

    def explain_decision(self, decision_id: str) -> dict:
        """Stub: Provide an explanation for a given decision (for user appeal/review)."""
        # In production, retrieve reasoning history and return a human-readable explanation
        for entry in self.reasoning_history:
            if entry.get("decision_id") == decision_id:
                return {"explanation": "Decision rationale and reasoning steps", **entry}
        return {"error": "Decision not found"}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "memory_stats": {
                "total_memories": len(self.memory_manager.memories),
                "memory_utilization": len(self.memory_manager.memories) / self.memory_manager.max_memories
            },
            "reasoning_stats": {
                "total_cycles": len(self.reasoning_history),
                "average_confidence": self.performance_metrics["average_confidence"],
                "reasoning_patterns": self.reasoning_patterns
            },
            "evolution_stats": {
                "total_evolutions": len(self.evolution_log),
                "last_evolution": self.evolution_log[-1]["timestamp"] if self.evolution_log else None
            },
            "performance_metrics": self.performance_metrics
        } 

    async def gpt25pro_reason(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Use GPT-2.5 Pro as a reasoning engine."""
        return await self.gpt25pro.generate(prompt, **kwargs) 