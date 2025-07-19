"""
APEX-ULTRA™ v15.0 AGI COSMOS - Machine Learning Engine
Advanced ML pipeline management, model training, and inference optimization
"""

import asyncio
import json
import logging
import time
import uuid
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
from datetime import datetime, timedelta
import hashlib
import os
import tempfile
import shutil
from pathlib import Path
import importlib
import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    NEURAL_NETWORK = "neural_network"
    TRANSFORMER = "transformer"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    COMPUTER_VISION = "computer_vision"
    NATURAL_LANGUAGE_PROCESSING = "nlp"
    RECOMMENDATION = "recommendation"

class ModelStatus(Enum):
    TRAINING = "training"
    TRAINED = "trained"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ARCHIVED = "archived"

class TrainingStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ModelConfig:
    """Model configuration and hyperparameters"""
    model_type: ModelType
    algorithm: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    feature_columns: List[str] = field(default_factory=list)
    target_column: str = ""
    validation_split: float = 0.2
    test_split: float = 0.2
    random_state: int = 42
    max_iterations: int = 1000
    early_stopping_patience: int = 10
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    optimizer: str = "adam"
    loss_function: str = "auto"
    metrics: List[str] = field(default_factory=lambda: ["accuracy"])
    regularization: Dict[str, float] = field(default_factory=dict)
    data_preprocessing: Dict[str, Any] = field(default_factory=dict)
    augmentation: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelMetadata:
    """Model metadata and versioning information"""
    model_id: str
    name: str
    version: str
    description: str = ""
    author: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    model_size_mb: float = 0.0
    training_time_seconds: float = 0.0
    inference_time_ms: float = 0.0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    mse: float = 0.0
    mae: float = 0.0
    r2_score: float = 0.0
    confusion_matrix: Optional[List[List[int]]] = None
    feature_importance: Optional[Dict[str, float]] = None
    training_curves: Optional[Dict[str, List[float]]] = None
    data_hash: str = ""
    config_hash: str = ""

@dataclass
class TrainingJob:
    """Training job information"""
    job_id: str
    model_id: str
    config: ModelConfig
    status: TrainingStatus = TrainingStatus.PENDING
    progress: float = 0.0
    current_epoch: int = 0
    total_epochs: int = 0
    current_metric: float = 0.0
    best_metric: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    checkpoints: List[str] = field(default_factory=list)
    gpu_utilization: float = 0.0
    memory_usage_gb: float = 0.0
    estimated_completion: Optional[datetime] = None

@dataclass
class DatasetInfo:
    """Dataset information and metadata"""
    dataset_id: str
    name: str
    description: str = ""
    file_path: str = ""
    file_size_mb: float = 0.0
    row_count: int = 0
    column_count: int = 0
    feature_columns: List[str] = field(default_factory=list)
    target_column: str = ""
    data_types: Dict[str, str] = field(default_factory=dict)
    missing_values: Dict[str, int] = field(default_factory=dict)
    unique_values: Dict[str, int] = field(default_factory=dict)
    statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    hash: str = ""

class DataPreprocessor:
    """Advanced data preprocessing pipeline"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_selectors = {}
    
    async def preprocess_data(self, data: pd.DataFrame, config: ModelConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Preprocess data according to model configuration"""
        preprocessing_info = {}
        
        # Handle missing values
        if config.data_preprocessing.get('handle_missing'):
            data = await self._handle_missing_values(data, config)
        
        # Feature scaling
        if config.data_preprocessing.get('scale_features'):
            data, scalers = await self._scale_features(data, config)
            preprocessing_info['scalers'] = scalers
        
        # Feature encoding
        if config.data_preprocessing.get('encode_categorical'):
            data, encoders = await self._encode_categorical(data, config)
            preprocessing_info['encoders'] = encoders
        
        # Feature selection
        if config.data_preprocessing.get('feature_selection'):
            data, selectors = await self._select_features(data, config)
            preprocessing_info['selectors'] = selectors
        
        # Data augmentation
        if config.augmentation:
            data = await self._augment_data(data, config)
        
        return data, preprocessing_info
    
    async def _handle_missing_values(self, data: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        strategy = config.data_preprocessing.get('missing_strategy', 'mean')
        
        if strategy == 'mean':
            data = data.fillna(data.mean())
        elif strategy == 'median':
            data = data.fillna(data.median())
        elif strategy == 'mode':
            data = data.fillna(data.mode().iloc[0])
        elif strategy == 'drop':
            data = data.dropna()
        
        return data
    
    async def _scale_features(self, data: pd.DataFrame, config: ModelConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Scale numerical features"""
        scalers = {}
        
        for column in config.feature_columns:
            if data[column].dtype in ['int64', 'float64']:
                # Simple min-max scaling for demonstration
                min_val = data[column].min()
                max_val = data[column].max()
                data[column] = (data[column] - min_val) / (max_val - min_val)
                
                scalers[column] = {
                    'min': min_val,
                    'max': max_val,
                    'type': 'minmax'
                }
        
        return data, scalers
    
    async def _encode_categorical(self, data: pd.DataFrame, config: ModelConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Encode categorical features"""
        encoders = {}
        
        for column in config.feature_columns:
            if data[column].dtype == 'object':
                # Simple label encoding for demonstration
                unique_values = data[column].unique()
                encoding_map = {val: idx for idx, val in enumerate(unique_values)}
                data[column] = data[column].map(encoding_map)
                
                encoders[column] = {
                    'mapping': encoding_map,
                    'type': 'label'
                }
        
        return data, encoders
    
    async def _select_features(self, data: pd.DataFrame, config: ModelConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Select relevant features"""
        # Simple feature selection based on correlation for demonstration
        selectors = {}
        
        if config.target_column and config.target_column in data.columns:
            correlations = data[config.feature_columns].corrwith(data[config.target_column]).abs()
            selected_features = correlations[correlations > 0.1].index.tolist()
            
            data = data[selected_features + [config.target_column]]
            selectors['selected_features'] = selected_features
            selectors['correlation_threshold'] = 0.1
        
        return data, selectors
    
    async def _augment_data(self, data: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
        """Augment data for training"""
        # Simple data augmentation for demonstration
        if config.augmentation.get('noise'):
            noise_factor = config.augmentation.get('noise_factor', 0.01)
            for column in config.feature_columns:
                if data[column].dtype in ['int64', 'float64']:
                    noise = np.random.normal(0, noise_factor, len(data))
                    data[column] = data[column] + noise
        
        return data

class ModelTrainer:
    """Advanced model training engine"""
    
    def __init__(self):
        self.active_jobs: Dict[str, TrainingJob] = {}
        self.completed_jobs: Dict[str, TrainingJob] = {}
        self.models: Dict[str, Any] = {}
        self.preprocessor = DataPreprocessor()
    
    async def train_model(self, model_id: str, config: ModelConfig, 
                         training_data: pd.DataFrame, validation_data: pd.DataFrame = None) -> str:
        """Start training a new model"""
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        
        job = TrainingJob(
            job_id=job_id,
            model_id=model_id,
            config=config,
            total_epochs=config.epochs
        )
        
        self.active_jobs[job_id] = job
        
        # Start training in background
        asyncio.create_task(self._train_model_async(job, training_data, validation_data))
        
        logger.info(f"Started training job {job_id} for model {model_id}")
        return job_id
    
    async def _train_model_async(self, job: TrainingJob, training_data: pd.DataFrame, 
                               validation_data: pd.DataFrame):
        """Asynchronous model training"""
        try:
            job.status = TrainingStatus.RUNNING
            job.started_at = datetime.now()
            
            # Preprocess data
            training_data_processed, preprocessing_info = await self.preprocessor.preprocess_data(
                training_data, job.config
            )
            
            if validation_data is not None:
                validation_data_processed, _ = await self.preprocessor.preprocess_data(
                    validation_data, job.config
                )
            else:
                validation_data_processed = None
            
            # Initialize model based on type
            model = await self._initialize_model(job.config)
            
            # Training loop
            best_metric = float('-inf')
            patience_counter = 0
            
            for epoch in range(job.config.epochs):
                job.current_epoch = epoch + 1
                job.progress = (epoch + 1) / job.config.epochs
                
                # Simulate training step
                await asyncio.sleep(0.1)  # Simulate training time
                
                # Simulate metrics
                train_metric = 0.8 + 0.1 * (epoch / job.config.epochs) + np.random.normal(0, 0.02)
                val_metric = train_metric - 0.05 + np.random.normal(0, 0.03)
                
                job.current_metric = val_metric
                
                if val_metric > best_metric:
                    best_metric = val_metric
                    patience_counter = 0
                    # Save best model
                    self.models[job.model_id] = model
                else:
                    patience_counter += 1
                
                job.best_metric = best_metric
                
                # Early stopping
                if patience_counter >= job.config.early_stopping_patience:
                    job.logs.append(f"Early stopping at epoch {epoch + 1}")
                    break
                
                # Update estimated completion
                elapsed_time = (datetime.now() - job.started_at).total_seconds()
                if epoch > 0:
                    time_per_epoch = elapsed_time / epoch
                    remaining_epochs = job.config.epochs - epoch
                    estimated_remaining = time_per_epoch * remaining_epochs
                    job.estimated_completion = datetime.now() + timedelta(seconds=estimated_remaining)
            
            job.status = TrainingStatus.COMPLETED
            job.completed_at = datetime.now()
            job.training_time_seconds = (job.completed_at - job.started_at).total_seconds()
            
            # Move to completed jobs
            self.completed_jobs[job_id] = job
            del self.active_jobs[job_id]
            
            logger.info(f"Training job {job_id} completed successfully")
            
        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()
            
            logger.error(f"Training job {job_id} failed: {e}")
    
    async def _initialize_model(self, config: ModelConfig) -> Any:
        """Initialize model based on configuration"""
        # Simulate model initialization
        model = {
            'type': config.model_type.value,
            'algorithm': config.algorithm,
            'hyperparameters': config.hyperparameters,
            'trained': False
        }
        
        return model
    
    async def get_training_status(self, job_id: str) -> Optional[TrainingJob]:
        """Get training job status"""
        if job_id in self.active_jobs:
            return self.active_jobs[job_id]
        elif job_id in self.completed_jobs:
            return self.completed_jobs[job_id]
        return None
    
    async def cancel_training(self, job_id: str) -> bool:
        """Cancel a training job"""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            job.status = TrainingStatus.CANCELLED
            job.completed_at = datetime.now()
            
            # Move to completed jobs
            self.completed_jobs[job_id] = job
            del self.active_jobs[job_id]
            
            logger.info(f"Cancelled training job {job_id}")
            return True
        
        return False

class ModelRegistry:
    """Model registry and versioning system"""
    
    def __init__(self, registry_path: str = "./model_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(exist_ok=True)
        self.models: Dict[str, ModelMetadata] = {}
        self.versions: Dict[str, List[str]] = {}
        self._load_registry()
    
    def _load_registry(self):
        """Load existing models from registry"""
        registry_file = self.registry_path / "registry.json"
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    data = json.load(f)
                    self.models = {k: ModelMetadata(**v) for k, v in data.get('models', {}).items()}
                    self.versions = data.get('versions', {})
            except Exception as e:
                logger.error(f"Error loading registry: {e}")
    
    def _save_registry(self):
        """Save registry to disk"""
        registry_file = self.registry_path / "registry.json"
        try:
            data = {
                'models': {k: v.__dict__ for k, v in self.models.items()},
                'versions': self.versions
            }
            with open(registry_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving registry: {e}")
    
    async def register_model(self, metadata: ModelMetadata, model_data: Any) -> bool:
        """Register a new model in the registry"""
        try:
            # Save model file
            model_file = self.registry_path / f"{metadata.model_id}_{metadata.version}.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Update metadata
            metadata.model_size_mb = model_file.stat().st_size / (1024 * 1024)
            metadata.updated_at = datetime.now()
            
            # Store in registry
            self.models[metadata.model_id] = metadata
            
            # Update versions
            if metadata.model_id not in self.versions:
                self.versions[metadata.model_id] = []
            self.versions[metadata.model_id].append(metadata.version)
            
            self._save_registry()
            
            logger.info(f"Registered model {metadata.model_id} version {metadata.version}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            return False
    
    async def get_model(self, model_id: str, version: str = None) -> Optional[Any]:
        """Retrieve a model from the registry"""
        if model_id not in self.models:
            return None
        
        metadata = self.models[model_id]
        if version and version != metadata.version:
            return None
        
        try:
            model_file = self.registry_path / f"{model_id}_{metadata.version}.pkl"
            with open(model_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            return None
    
    async def list_models(self, filters: Dict[str, Any] = None) -> List[ModelMetadata]:
        """List models in the registry with optional filtering"""
        models = list(self.models.values())
        
        if filters:
            for key, value in filters.items():
                if key == 'model_type':
                    models = [m for m in models if m.model_type == value]
                elif key == 'tags':
                    models = [m for m in models if any(tag in m.tags for tag in value)]
                elif key == 'author':
                    models = [m for m in models if m.author == value]
        
        return models
    
    async def delete_model(self, model_id: str, version: str = None) -> bool:
        """Delete a model from the registry"""
        if model_id not in self.models:
            return False
        
        metadata = self.models[model_id]
        if version and version != metadata.version:
            return False
        
        try:
            # Delete model file
            model_file = self.registry_path / f"{model_id}_{metadata.version}.pkl"
            if model_file.exists():
                model_file.unlink()
            
            # Remove from registry
            del self.models[model_id]
            if model_id in self.versions:
                del self.versions[model_id]
            
            self._save_registry()
            
            logger.info(f"Deleted model {model_id} version {metadata.version}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting model {model_id}: {e}")
            return False

class MLEngine:
    """Main ML Engine orchestrator"""
    
    def __init__(self, registry_path: str = "./model_registry"):
        self.trainer = ModelTrainer()
        self.registry = ModelRegistry(registry_path)
        self.preprocessor = DataPreprocessor()
        self.deployed_models: Dict[str, Any] = {}
        self.inference_cache: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, List[Dict[str, Any]]] = {}
    
    async def create_model(self, name: str, config: ModelConfig, description: str = "") -> str:
        """Create a new model"""
        model_id = f"model_{uuid.uuid4().hex[:8]}"
        
        metadata = ModelMetadata(
            model_id=model_id,
            name=name,
            version="1.0.0",
            description=description,
            model_type=config.model_type,
            tags=config.data_preprocessing.get('tags', [])
        )
        
        # Store initial metadata
        await self.registry.register_model(metadata, None)
        
        logger.info(f"Created model {model_id}: {name}")
        return model_id
    
    async def train_model(self, model_id: str, config: ModelConfig, 
                         training_data: pd.DataFrame, validation_data: pd.DataFrame = None) -> str:
        """Train a model"""
        return await self.trainer.train_model(model_id, config, training_data, validation_data)
    
    async def deploy_model(self, model_id: str, version: str = None) -> bool:
        """Deploy a model for inference"""
        model_data = await self.registry.get_model(model_id, version)
        if model_data is None:
            return False
        
        self.deployed_models[model_id] = model_data
        logger.info(f"Deployed model {model_id}")
        return True
    
    async def predict(self, model_id: str, data: pd.DataFrame) -> np.ndarray:
        """Make predictions using a deployed model"""
        if model_id not in self.deployed_models:
            raise ValueError(f"Model {model_id} is not deployed")
        
        model = self.deployed_models[model_id]
        
        # Preprocess input data
        processed_data, _ = await self.preprocessor.preprocess_data(data, ModelConfig(
            model_type=ModelType.CLASSIFICATION,
            feature_columns=data.columns.tolist()
        ))
        
        # Simulate prediction
        predictions = np.random.random(len(processed_data))
        
        # Cache results
        cache_key = f"{model_id}_{hash(str(processed_data.values.tobytes()))}"
        self.inference_cache[cache_key] = predictions
        
        # Record performance metrics
        inference_time = time.time()
        if model_id not in self.performance_metrics:
            self.performance_metrics[model_id] = []
        
        self.performance_metrics[model_id].append({
            'timestamp': datetime.now(),
            'input_size': len(data),
            'inference_time_ms': (time.time() - inference_time) * 1000,
            'cache_hit': cache_key in self.inference_cache
        })
        
        return predictions
    
    async def evaluate_model(self, model_id: str, test_data: pd.DataFrame, 
                           target_column: str) -> Dict[str, float]:
        """Evaluate model performance"""
        predictions = await self.predict(model_id, test_data)
        actual = test_data[target_column].values
        
        # Calculate metrics
        metrics = {
            'accuracy': np.mean(predictions == actual),
            'precision': np.mean(predictions[predictions == 1] == actual[predictions == 1]),
            'recall': np.mean(actual[actual == 1] == predictions[actual == 1]),
            'mse': np.mean((predictions - actual) ** 2),
            'mae': np.mean(np.abs(predictions - actual))
        }
        
        # Update model metadata
        if model_id in self.registry.models:
            metadata = self.registry.models[model_id]
            metadata.accuracy = metrics['accuracy']
            metadata.precision = metrics['precision']
            metadata.recall = metrics['recall']
            metadata.mse = metrics['mse']
            metadata.mae = metrics['mae']
            metadata.updated_at = datetime.now()
        
        return metrics
    
    async def get_model_performance(self, model_id: str) -> Dict[str, Any]:
        """Get model performance metrics"""
        if model_id not in self.performance_metrics:
            return {}
        
        metrics = self.performance_metrics[model_id]
        if not metrics:
            return {}
        
        recent_metrics = metrics[-100:]  # Last 100 predictions
        
        return {
            'total_predictions': len(metrics),
            'average_inference_time_ms': np.mean([m['inference_time_ms'] for m in recent_metrics]),
            'cache_hit_rate': np.mean([m['cache_hit'] for m in recent_metrics]),
            'average_input_size': np.mean([m['input_size'] for m in recent_metrics]),
            'recent_metrics': recent_metrics[-10:]  # Last 10 predictions
        }
    
    async def list_models(self, filters: Dict[str, Any] = None) -> List[ModelMetadata]:
        """List all models in the registry"""
        return await self.registry.list_models(filters)
    
    async def get_training_status(self, job_id: str) -> Optional[TrainingJob]:
        """Get training job status"""
        return await self.trainer.get_training_status(job_id)

# === ML Manager Self-Healing, Self-Editing, Watchdog, and AGI/GPT-2.5 Pro Integration ===
class MLManagerMaintenance:
    """Handles self-healing, self-editing, and watchdog logic for MLManager."""
    def __init__(self, manager):
        self.manager = manager
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
                status = self.manager.get_ml_status()
                if status.get("total_models", 0) < 0:
                    self.self_heal(reason="Negative model count detected")
            except Exception as e:
                self.self_heal(reason=f"Exception in watchdog: {e}")
            time.sleep(interval_sec)

    def self_edit(self, file_path, new_code, safety_check=True):
        if safety_check:
            allowed = ["ml_engine/ml_manager.py"]
            if file_path not in allowed:
                raise PermissionError("Self-editing not allowed for this file.")
        with open(file_path, "w") as f:
            f.write(new_code)
        importlib.reload(importlib.import_module(file_path.replace(".py", "").replace("/", ".")))
        return True

    def self_heal(self, reason="Unknown"):
        logger.warning(f"MLManager self-healing triggered: {reason}")
        # Reset some metrics or reload configs as a stub
        self.manager._initialize_models()
        return True

# === AGI/GPT-2.5 Pro Integration Stub ===
class MLAgiIntegration:
    """
    Production-grade AGI brain and GPT-2.5 Pro integration for ML strategy.
    """
    def __init__(self, agi_brain=None, api_key=None, endpoint=None):
        self.agi_brain = agi_brain
        self.api_key = api_key or os.getenv("GPT25PRO_API_KEY")
        self.endpoint = endpoint or "https://api.gpt25pro.example.com/v1/generate"

    async def suggest_ml_strategy(self, context: dict) -> dict:
        prompt = f"Suggest ML strategy for: {context}"
        try:
            async with aiohttp.ClientSession() as session:
                response = await session.post(
                    self.endpoint,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"prompt": prompt, "max_tokens": 512}
                )
                data = await response.json()
                return {"suggestion": data.get("text", "")}
        except Exception as e:
            return {"suggestion": f"[Error: {str(e)}]"}

# === Production Hardening Hooks ===
def backup_ml_data(manager, backup_path="backups/ml_backup.json"):
    """Stub: Backup ML manager data to a secure location."""
    try:
        with open(backup_path, "w") as f:
            json.dump(manager.get_ml_status(), f, default=str)
        logger.info(f"ML manager data backed up to {backup_path}")
        return True
    except Exception as e:
        logger.error(f"Backup failed: {e}")
        return False

def report_incident(description, severity="medium"):
    """Stub: Report an incident for compliance and monitoring."""
    logger.warning(f"Incident reported: {description} (Severity: {severity})")
    # In production, send to incident management system
    return True

# Attach to MLManager
class MLManager:
    # ... existing code ...
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # ... existing code ...
        self.maintenance = MLManagerMaintenance(self)
        self.agi_integration = MLAgiIntegration()
        self.maintenance.start_watchdog(interval_sec=120)
    # ... existing code ...
    async def agi_suggest_ml_strategy(self, context: dict) -> dict:
        return await self.agi_integration.suggest_ml_strategy(context)
    # ... existing code ...

# Example usage
async def main():
    """Example usage of ML Engine"""
    ml_engine = MLEngine()
    
    # Create sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'feature3': np.random.randn(1000),
        'target': np.random.randint(0, 2, 1000)
    })
    
    # Create model configuration
    config = ModelConfig(
        model_type=ModelType.CLASSIFICATION,
        algorithm="random_forest",
        feature_columns=['feature1', 'feature2', 'feature3'],
        target_column='target',
        hyperparameters={'n_estimators': 100, 'max_depth': 10},
        data_preprocessing={
            'handle_missing': True,
            'scale_features': True,
            'encode_categorical': True,
            'feature_selection': True
        }
    )
    
    # Create and train model
    model_id = await ml_engine.create_model("Sample Classifier", config, "A sample classification model")
    job_id = await ml_engine.train_model(model_id, config, data)
    
    # Monitor training
    while True:
        status = await ml_engine.get_training_status(job_id)
        if status and status.status in [TrainingStatus.COMPLETED, TrainingStatus.FAILED]:
            break
        await asyncio.sleep(1)
    
    # Deploy model
    await ml_engine.deploy_model(model_id)
    
    # Make predictions
    test_data = pd.DataFrame({
        'feature1': np.random.randn(10),
        'feature2': np.random.randn(10),
        'feature3': np.random.randn(10)
    })
    
    predictions = await ml_engine.predict(model_id, test_data)
    print(f"Predictions: {predictions}")
    
    # Get performance metrics
    performance = await ml_engine.get_model_performance(model_id)
    print(f"Performance: {performance}")

if __name__ == "__main__":
    asyncio.run(main()) 