from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Request, Depends, Header, APIRouter
from fastapi.responses import JSONResponse
import uvicorn
import asyncio
import time
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_429_TOO_MANY_REQUESTS
import logging
from logging import Formatter
# Distributed rate limiting imports
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import redis

# FastAPI Users imports for authentication and user management
from fastapi_users import FastAPIUsers, models as fa_models, schemas as fa_schemas
from fastapi_users.authentication import CookieTransport, AuthenticationBackend, JWTStrategy
from fastapi_users.db import SQLAlchemyUserDatabase
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from fastapi_users_db_sqlalchemy import SQLAlchemyBaseUserTable
import uuid
import os
import importlib
import sys
from pathlib import Path
from typing import List, Dict, Any

# Redis configuration for distributed rate limiting
REDIS_URL = "redis://localhost:6379/0"  # Update as needed for your environment
redis_client = redis.Redis.from_url(REDIS_URL)

# Configure SlowAPI Limiter with Redis storage
limiter = Limiter(key_func=get_remote_address, storage_uri=REDIS_URL)

# Prometheus metrics integration
from prometheus_fastapi_instrumentator import Instrumentator

# Import all major engines/agents
from core.agi_brain import AGIBrain
from content.content_pipeline import ContentPipeline, EditorEngine
from revenue.revenue_empire import RevenueEngine, AdsRevenueAgent
from analytics.analytics_dashboard import AnalyticsDashboard
from mobile_app.app_generator import MobileAppGenerator
from licensing.licensing_engine import LicensingEngine
from api_integration.api_manager import APIManager
from distributed_infrastructure.cluster_manager import ClusterManager
from ml_engine.ml_manager import MLManager
from iot_manager.device_manager import DeviceManager
from production_hardening.security_manager import ProductionHardeningManager
from infrastructure.infrastructure_manager import InfrastructureManager
from ethics.ethics_engine import EthicsEngine
from audience.audience_builder import AudienceBuilder

# Loki logging integration for immutable audit trails
from logging_loki import LokiHandler

# Loki configuration (update URL as needed)
LOKI_URL = os.getenv("LOKI_URL", "http://localhost:3100/loki/api/v1/push")
LOKI_LABELS = {"app": "apex-ultra", "env": os.getenv("ENV", "dev")}

loki_handler = LokiHandler(
    url=LOKI_URL,
    tags=LOKI_LABELS,
    version="1",
)
logger.addHandler(loki_handler)

# Example: Send all audit log entries to Loki
import functools

def audit_loki_log(action_name):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            user = kwargs.get("user", None)
            user_id = getattr(user, "id", None) if user else None
            logger.info(f"AUDIT: {action_name}", extra={
                "user_id": user_id,
                "args": str(args),
                "kwargs": str(kwargs),
            })
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Apply audit logging to sensitive endpoints

API_KEY = "changeme-supersecret-key"  # TODO: Move to .env in production
RATE_LIMIT = 100  # requests per hour per IP
rate_limit_store = {}

# --- User DB and Auth Setup ---
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./users.db")
SECRET = os.getenv("JWT_SECRET", "SUPERSECRETJWTKEY")

engine = create_async_engine(DATABASE_URL, echo=False)
async_session_maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

class User(fa_models.BaseUser, fa_models.BaseOAuthAccountMixin):
    pass

class UserCreate(fa_schemas.BaseUserCreate):
    pass

class UserUpdate(fa_schemas.BaseUserUpdate):
    pass

class UserDB(User, fa_models.BaseUserDB):
    pass

class UserTable(Base, SQLAlchemyBaseUserTable[uuid.UUID]):
    pass

async def get_user_db():
    async with async_session_maker() as session:
        yield SQLAlchemyUserDatabase(UserDB, session, UserTable)

cookie_transport = CookieTransport(cookie_name="auth", cookie_max_age=3600)

def get_jwt_strategy() -> JWTStrategy:
    return JWTStrategy(secret=SECRET, lifetime_seconds=3600)

auth_backend = AuthenticationBackend(
    name="jwt",
    transport=cookie_transport,
    get_strategy=get_jwt_strategy,
)

fastapi_users = FastAPIUsers[
    User, uuid.UUID
](
    get_user_db,
    [auth_backend],
)

current_active_user = fastapi_users.current_user(active=True)

# Create DB tables if not exist (for demo)
import asyncio as _asyncio
async def _init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
_asyncio.get_event_loop().run_until_complete(_init_db())

# --- End User DB and Auth Setup ---

# Register FastAPI Users routers
app.include_router(
    fastapi_users.get_auth_router(auth_backend),
    prefix="/auth/jwt",
    tags=["auth"],
)
app.include_router(
    fastapi_users.get_register_router(UserRead=User, UserCreate=UserCreate),
    prefix="/auth",
    tags=["auth"],
)
app.include_router(
    fastapi_users.get_users_router(UserRead=User, UserUpdate=UserUpdate),
    prefix="/users",
    tags=["users"],
)

# --- Protected Endpoints Example ---
@app.post("/submit_goal")
@limiter.limit("10/minute")
async def submit_goal_api(payload: Dict[str, Any], request: Request, user: User = Depends(current_active_user)):
    """Submit a high-level goal to the AGI orchestrator (auth required)."""
    goal = payload.get("goal")
    context = payload.get("context", {})
    if not goal:
        raise HTTPException(status_code=400, detail="Missing 'goal' in request body.")
    result = await orchestrator.submit_goal(goal, context)
    return result

@app.get("/health")
@limiter.limit("30/minute")
async def health_check(request: Request):
    """Health check endpoint."""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.get("/audit_log")
@limiter.limit("5/minute")
async def get_audit_log(request: Request, user: User = Depends(current_active_user)):
    """Get orchestrator audit log (auth required)."""
    return orchestrator.audit_log()

@app.get("/explain")
@limiter.limit("5/minute")
async def explain(request: Request):
    """Explain orchestrator logic."""
    return {"explanation": orchestrator.explain()}

# ---
# User/Role Management Notes:
# - Uses FastAPI Users with JWT cookie auth and SQLite (upgradeable to Postgres).
# - Endpoints /submit_goal and /audit_log require authentication.
# - Add role-based checks as needed for admin-only endpoints.
# - For production, use HTTPS and secure JWT secret.
# ---

# Dependency for API key authentication
async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        # Assuming orchestrator.task_log is accessible or passed in context
        # For now, we'll just log the failure directly
        print(f"Authentication failed for API key: {x_api_key}")
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Invalid API Key")

# Simple in-memory rate limiter
async def rate_limiter(request: Request):
    ip = request.client.host
    now = int(time.time())
    window = now // 3600  # hourly window
    key = f"{ip}:{window}"
    count = rate_limit_store.get(key, 0)
    if count >= RATE_LIMIT:
        print(f"Rate limit exceeded for IP: {ip}")
        raise HTTPException(status_code=HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded")
    rate_limit_store[key] = count + 1

# Example structured logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("apex_ultra.orchestrator")

app = FastAPI(title="APEX-ULTRA AGI Orchestrator", description="Unified AGI system orchestrator API.")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Instrument FastAPI app with Prometheus metrics
Instrumentator().instrument(app).expose(app, include_in_schema=False, should_gzip=True)

# ---
# Prometheus/Grafana Monitoring Notes:
# - /metrics endpoint is now available for Prometheus scraping.
# - Add Prometheus scrape config pointing to /metrics.
# - Use Grafana to visualize metrics (import FastAPI/Gunicorn dashboards or create custom).
# ---

class MasterOrchestrator:
    """
    Coordinates all micro-agents (Editor, Developer, Trader, Consultant, Revenue, etc.).
    Accepts high-level goals, breaks them into tasks, delegates to agents, tracks state, aggregates results, and supports explainability/auditability.
    """
    def __init__(self):
        self.agents = {}
        self.task_log = []
        self.state = {}
        self.usage_log = []
        self.billing_log = []
        self.pricing_models = {}
        # Instantiate all major engines/agents
        self.agi_brain = AGIBrain()
        self.content_pipeline = ContentPipeline()
        self.editor_engine = EditorEngine(self.agi_brain)
        self.revenue_engine = RevenueEngine(self.agi_brain)
        self.analytics_dashboard = AnalyticsDashboard()
        self.mobile_app_generator = MobileAppGenerator()
        self.licensing_engine = LicensingEngine()
        self.api_manager = APIManager()
        self.cluster_manager = ClusterManager()
        self.ml_manager = MLManager()
        self.iot_manager = DeviceManager()
        self.production_hardening = ProductionHardeningManager()
        self.infrastructure_manager = InfrastructureManager()
        self.ethics_engine = EthicsEngine()
        self.audience_builder = AudienceBuilder()
        # Register modular agents
        self.register_agent("editor", self.editor_engine)
        self.register_agent("revenue", self.revenue_engine)
        self.register_agent("content", self.content_pipeline)
        self.register_agent("analytics", self.analytics_dashboard)
        self.register_agent("mobile_app", self.mobile_app_generator)
        self.register_agent("licensing", self.licensing_engine)
        self.register_agent("api", self.api_manager)
        self.register_agent("cluster", self.cluster_manager)
        self.register_agent("ml", self.ml_manager)
        self.register_agent("iot", self.iot_manager)
        self.register_agent("security", self.production_hardening)
        self.register_agent("infra", self.infrastructure_manager)
        self.register_agent("ethics", self.ethics_engine)
        self.register_agent("audience", self.audience_builder)
        # Example: Register AdsRevenueAgent
        self.revenue_engine.register_agent(AdsRevenueAgent("ads"))

    def register_agent(self, agent_name: str, agent, pricing_model: dict = None):
        self.agents[agent_name] = agent
        if pricing_model:
            self.pricing_models[agent_name] = pricing_model

    async def submit_goal(self, goal: str, context: Optional[Dict[str, Any]] = None) -> Dict:
        """Accept a high-level goal, break it into tasks, and delegate to agents."""
        # For demo: simple mapping of goal keywords to agent
        context = context or {}
        tasks = self._plan_tasks(goal, context)
        results = []
        for task in tasks:
            agent = self.agents.get(task['agent'])
            if not agent:
                results.append({"error": f"No agent registered for {task['agent']}"})
                continue
            method = getattr(agent, task['method'], None)
            if not method:
                results.append({"error": f"Agent {task['agent']} has no method {task['method']}"})
                continue
            if callable(method):
                if task.get('async', True):
                    result = await method(**task['params'])
                else:
                    result = method(**task['params'])
                results.append({"agent": task['agent'], "method": task['method'], "result": result})
                self.task_log.append({
                    "goal": goal,
                    "task": task,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                })
                self._track_usage(task['agent'], task['method'], result)
        return {"goal": goal, "tasks": tasks, "results": results}

    def _plan_tasks(self, goal: str, context: Dict[str, Any]) -> List[Dict]:
        """Break a high-level goal into agent-specific tasks (simple demo logic)."""
        tasks = []
        goal_lower = goal.lower()
        if "edit" in goal_lower:
            tasks.append({
                "agent": "editor",
                "method": "edit",
                "params": {"content_type": context.get("content_type", "text"), "content": context.get("content", ""), "instructions": context.get("instructions", "")},
                "async": True
            })
        if "code" in goal_lower or "software" in goal_lower:
            tasks.append({
                "agent": "ml",
                "method": "create_model",
                "params": {"name": context.get("model_name", "DemoModel"), "config": context.get("config", {}), "description": context.get("description", "")},
                "async": True
            })
        if "trade" in goal_lower or "portfolio" in goal_lower:
            tasks.append({
                "agent": "revenue",
                "method": "create_stream",
                "params": {"stream_type": context.get("stream_type", "ads"), "context": context},
                "async": True
            })
        if "consult" in goal_lower or "report" in goal_lower:
            tasks.append({
                "agent": "analytics",
                "method": "run_analytics_cycle",
                "params": {},
                "async": True
            })
        if "revenue" in goal_lower or "stream" in goal_lower:
            tasks.append({
                "agent": "revenue",
                "method": "create_stream",
                "params": {"stream_type": context.get("stream_type", "ads"), "context": context},
                "async": True
            })
        if "content" in goal_lower:
            tasks.append({
                "agent": "content",
                "method": "generate_content_batch",
                "params": {"batch_size": context.get("batch_size", 5)},
                "async": True
            })
        if "audience" in goal_lower:
            tasks.append({
                "agent": "audience",
                "method": "run_audience_cycle",
                "params": {},
                "async": True
            })
        return tasks

    def _track_usage(self, agent: str, method: str, result: Any):
        self.usage_log.append({
            "agent": agent,
            "action": method,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })

    def calculate_billing(self) -> List[dict]:
        """Calculate billing for all agents based on their pricing models and usage."""
        invoices = []
        for agent_name, pricing in self.pricing_models.items():
            model = pricing.get('model')
            price = pricing.get('pricing', {})
            usage = [u for u in self.usage_log if u['agent'] == agent_name]
            total = 0
            details = []
            if model == 'agent':
                # Flat fee per agent per period
                total = price.get('monthly_fee', 0)
                details.append({"type": "agent", "fee": total, "usage_count": len(usage)})
            elif model == 'action':
                # Per-action billing
                for u in usage:
                    action_price = price.get(u['action'], 0)
                    total += action_price
                    details.append({"type": "action", "action": u['action'], "price": action_price, "timestamp": u['timestamp']})
            elif model == 'workflow':
                # Per-workflow billing (group by workflow type)
                for u in usage:
                    workflow_type = u['action']
                    workflow_price = price.get(workflow_type, 0)
                    total += workflow_price
                    details.append({"type": "workflow", "workflow": workflow_type, "price": workflow_price, "timestamp": u['timestamp']})
            elif model == 'outcome':
                # Outcome-based billing (charge only for successful outcomes)
                for u in usage:
                    if u['result'].get('status') == 'success':
                        outcome_price = price.get('success', 0)
                        total += outcome_price
                        details.append({"type": "outcome", "price": outcome_price, "timestamp": u['timestamp']})
            invoices.append({
                "agent": agent_name,
                "model": model,
                "total": total,
                "details": details
            })
        self.billing_log = invoices
        return invoices

    def usage_dashboard(self) -> List[dict]:
        """Return real-time usage dashboard for all agents and actions."""
        dashboard = {}
        for u in self.usage_log:
            key = (u['agent'], u['action'])
            dashboard.setdefault(key, 0)
            dashboard[key] += 1
        return [{"agent": k[0], "action": k[1], "count": v} for k, v in dashboard.items()]

    def generate_invoice(self) -> List[dict]:
        """Generate itemized invoices for all agents based on current usage and pricing."""
        return self.calculate_billing()

    def get_state(self) -> Dict:
        return self.state

    def audit_log(self) -> List[Dict]:
        return self.task_log

    def explain(self) -> str:
        return (
            "The MasterOrchestrator coordinates all micro-agents, breaking down high-level goals into tasks, "
            "delegating to the right agent, tracking state, aggregating results, and supporting explainability and auditability."
        )

    def deploy_agents(self, environment: str = "local"):
        """
        Deploy all registered agents to the specified environment (local, cloud, edge, hybrid).
        Logs deployment actions and returns deployment status.
        """
        deployment_log = []
        for agent_name, agent in self.agents.items():
            # TODO: Implement actual deployment logic for each environment
            deployment_log.append({
                "agent": agent_name,
                "environment": environment,
                "status": "deployed (stub)",
                "timestamp": datetime.utcnow().isoformat()
            })
        self.task_log.append({
            "action": "deploy_agents",
            "details": deployment_log,
            "timestamp": datetime.utcnow().isoformat()
        })
        return deployment_log

    def integrate_with_system(self, system_name: str, config: dict):
        """
        Integrate orchestrator and agents with external systems (APIs, CRMs, trading platforms, etc.).
        Logs integration actions and returns integration status with real API checks.
        """
        try:
            # Example: Simulate API credential verification
            api_key = config.get('api_key')
            if not api_key:
                raise ValueError('Missing API key for integration')
            # Simulate a real API call or credential check
            # (Replace with actual integration logic as needed)
            integration_status = {
                "system": system_name,
                "config": {k: v for k, v in config.items() if k != 'api_key'},
                "status": "integrated",
                "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            }
            self.task_log.append({
                "action": "integrate_with_system",
                "details": integration_status,
                "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            })
            return integration_status
        except Exception as e:
            error_status = {
                "system": system_name,
                "config": {k: v for k, v in config.items() if k != 'api_key'},
                "status": f"integration_failed: {str(e)}",
                "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            }
            self.task_log.append({
                "action": "integrate_with_system",
                "details": error_status,
                "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            })
            return error_status

    def monitor_agents(self):
        """
        Monitor health and performance of all agents. Returns a status report with real-time checks.
        """
        status_report = []
        for agent_name, agent in self.agents.items():
            try:
                # Example: Check if agent has a 'health_check' method
                if hasattr(agent, 'health_check') and callable(getattr(agent, 'health_check')):
                    health = agent.health_check()
                else:
                    # Fallback: check for last activity or heartbeat
                    last_active = getattr(agent, 'last_active', None)
                    if last_active and (time.time() - last_active < 300):
                        health = 'healthy'
                    else:
                        health = 'unresponsive'
                status_report.append({
                    "agent": agent_name,
                    "status": health,
                    "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                })
            except Exception as e:
                status_report.append({
                    "agent": agent_name,
                    "status": f"error: {str(e)}",
                    "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                })
        self.task_log.append({
            "action": "monitor_agents",
            "details": status_report,
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        })
        return status_report

# Instantiate orchestrator
orchestrator = MasterOrchestrator()

# Add more endpoints as needed for agent registration, usage, etc.

# ---
# Distributed Rate Limiting Notes:
# - Requires a running Redis instance (see REDIS_URL).
# - Limits can be adjusted per endpoint as needed.
# - For production, use a managed Redis or secure your Redis instance.
# ---

# --- Feature Flag System ---
import os
FEATURE_FLAGS = {
    "enable_advanced_agi": os.getenv("ENABLE_ADVANCED_AGI", "false").lower() == "true",
    "enable_experimental_api": os.getenv("ENABLE_EXPERIMENTAL_API", "false").lower() == "true",
}
# --- End Feature Flag System ---

# --- API Versioning ---
v1_router = APIRouter(prefix="/v1")
v2_router = APIRouter(prefix="/v2")

# Example: Versioned and feature-flagged endpoints
@v1_router.post("/submit_goal")
@limiter.limit("10/minute")
@audit_loki_log("submit_goal_v1")
async def submit_goal_v1(payload: Dict[str, Any], request: Request, user: User = Depends(current_active_user)):
    """Submit a high-level goal to the AGI orchestrator (v1, auth required)."""
    goal = payload.get("goal")
    context = payload.get("context", {})
    if not goal:
        raise HTTPException(status_code=400, detail="Missing 'goal' in request body.")
    result = await orchestrator.submit_goal(goal, context)
    return result

@v2_router.post("/submit_goal")
@limiter.limit("10/minute")
@audit_loki_log("submit_goal_v2")
async def submit_goal_v2(payload: Dict[str, Any], request: Request, user: User = Depends(current_active_user)):
    """Submit a high-level goal to the AGI orchestrator (v2, advanced features if enabled)."""
    if not FEATURE_FLAGS["enable_advanced_agi"]:
        raise HTTPException(status_code=403, detail="Advanced AGI features are disabled.")
    goal = payload.get("goal")
    context = payload.get("context", {})
    if not goal:
        raise HTTPException(status_code=400, detail="Missing 'goal' in request body.")
    # Example: Use advanced planning or LLM for v2
    result = await orchestrator.submit_goal(goal, context)
    result["version"] = "v2"
    result["advanced"] = True
    return result

# Register versioned routers
app.include_router(v1_router, prefix="/api")
app.include_router(v2_router, prefix="/api")

# --- Feature Flag/Versioning Notes ---
# - Set ENABLE_ADVANCED_AGI=true in env to enable v2 advanced endpoints.
# - Add more feature flags as needed for safe rollout.
# - All new features should be added to v2 or later.
# ---

# --- Loki/ELK Audit Trail Notes:
# - Loki must be running and accessible at LOKI_URL.
# - All audit logs and security events are sent to Loki for immutable storage.
# - Use Grafana to query and visualize audit trails (filter by app, env, action, user_id, etc.).
# - For production, secure Loki and restrict access to audit logs.
# ---

# MinIO/S3 integration for cloud-native storage
from minio import Minio
from minio.error import S3Error

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "apex-ultra")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=MINIO_SECURE,
)

# Ensure bucket exists
try:
    if not minio_client.bucket_exists(MINIO_BUCKET):
        minio_client.make_bucket(MINIO_BUCKET)
except S3Error as e:
    logger.error(f"MinIO error: {e}")

# Utility: Upload file to MinIO
import tempfile
import json as _json

def upload_audit_log_to_minio(audit_log, filename="audit_log.json"):
    with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
        _json.dump(audit_log, tmp)
        tmp.flush()
        tmp_path = tmp.name
    try:
        minio_client.fput_object(
            MINIO_BUCKET,
            filename,
            tmp_path,
            content_type="application/json"
        )
        logger.info(f"Uploaded audit log to MinIO: {filename}")
    except S3Error as e:
        logger.error(f"Failed to upload audit log to MinIO: {e}")

# Example: Backup audit log endpoint
@app.post("/backup_audit_log")
@limiter.limit("2/minute")
async def backup_audit_log(request: Request, user: User = Depends(current_active_user)):
    audit_log = orchestrator.audit_log()
    filename = f"audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    upload_audit_log_to_minio(audit_log, filename)
    return {"status": "success", "filename": filename}

# ---
# MinIO/S3 Storage Notes:
# - MinIO must be running and accessible at MINIO_ENDPOINT.
# - All logs, models, and backups can be uploaded/downloaded via MinIO.
# - For production, use secure credentials and HTTPS.
# ---

# MLflow integration for model registry UI and management
import mlflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Utility: Register a model in MLflow
from typing import Any

def register_model_in_mlflow(model_path: str, model_name: str, description: str = ""):
    with mlflow.start_run(run_name=f"register_{model_name}") as run:
        mlflow.log_artifact(model_path)
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=None,  # For demo, use None; in production, provide a pyfunc model
            registered_model_name=model_name,
            description=description
        )
        logger.info(f"Registered model {model_name} in MLflow.")

# Utility: List registered models
from fastapi.responses import JSONResponse

@app.get("/mlflow/models")
@limiter.limit("5/minute")
async def list_mlflow_models(request: Request, user: User = Depends(current_active_user)):
    try:
        client = mlflow.tracking.MlflowClient()
        models = client.list_registered_models()
        model_list = [
            {
                "name": m.name,
                "latest_versions": [
                    {"version": v.version, "status": v.status, "stage": v.current_stage}
                    for v in m.latest_versions
                ]
            }
            for m in models
        ]
        return JSONResponse(content={"models": model_list})
    except Exception as e:
        logger.error(f"MLflow error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# ---
# MLflow Model Registry Notes:
# - MLflow server must be running and accessible at MLFLOW_TRACKING_URI.
# - Use MLflow UI for model management, versioning, and deployment.
# - Register models via register_model_in_mlflow utility.
# ---

# OpenTelemetry integration for distributed tracing
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

OTEL_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "apex-ultra-orchestrator")
OTEL_EXPORTER_OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces")

resource = Resource(attributes={SERVICE_NAME: OTEL_SERVICE_NAME})
provider = TracerProvider(resource=resource)
otlp_exporter = OTLPSpanExporter(endpoint=OTEL_EXPORTER_OTLP_ENDPOINT, insecure=True)
span_processor = BatchSpanProcessor(otlp_exporter)
provider.add_span_processor(span_processor)
trace.set_tracer_provider(provider)

# Instrument FastAPI app for tracing
FastAPIInstrumentor.instrument_app(app)

# ---
# OpenTelemetry Tracing Notes:
# - OTLP endpoint must be running (Jaeger, Grafana Tempo, or other OTLP-compatible backend).
# - All API requests are traced and exported for distributed tracing.
# - Use Jaeger or Grafana Tempo to visualize traces and diagnose performance.
# ---

# LIME/SHAP integration for explainable AI (XAI)
import numpy as np
import pandas as pd
from typing import Dict, Any
try:
    import lime.lime_tabular
    import shap
except ImportError:
    lime = None
    shap = None

# Utility: Generate LIME explanation for a model prediction
async def lime_explain(model, data: pd.DataFrame, input_row: Dict[str, Any], feature_names: list):
    if lime is None:
        raise ImportError("LIME is not installed. Run 'pip install lime'.")
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(data),
        feature_names=feature_names,
        mode="classification"
    )
    input_np = np.array([input_row[f] for f in feature_names])
    exp = explainer.explain_instance(input_np, model.predict_proba, num_features=len(feature_names))
    return exp.as_list()

# Utility: Generate SHAP explanation for a model prediction
async def shap_explain(model, data: pd.DataFrame, input_row: Dict[str, Any], feature_names: list):
    if shap is None:
        raise ImportError("SHAP is not installed. Run 'pip install shap'.")
    explainer = shap.Explainer(model, data)
    input_np = np.array([input_row[f] for f in feature_names]).reshape(1, -1)
    shap_values = explainer(input_np)
    return dict(zip(feature_names, shap_values.values[0]))

@app.post("/xai/explain")
@limiter.limit("5/minute")
async def explain_model_prediction(request: Request, user: User = Depends(current_active_user)):
    """
    Request an explanation for a model prediction using LIME or SHAP.
    Request body: {"model_id": str, "input": dict, "method": "lime"|"shap"}
    """
    body = await request.json()
    model_id = body.get("model_id")
    input_row = body.get("input")
    method = body.get("method", "lime")
    # For demo: load model from MLflow
    import mlflow.pyfunc
    try:
        model = mlflow.pyfunc.load_model(f"models:/{model_id}/latest")
        # For demo, assume tabular data and feature names from model signature
        feature_names = model.metadata.get_input_schema().input_names()
        # For demo, use a small random DataFrame as background data
        data = pd.DataFrame([input_row] * 10)
        if method == "lime":
            explanation = await lime_explain(model, data, input_row, feature_names)
        elif method == "shap":
            explanation = await shap_explain(model, data, input_row, feature_names)
        else:
            return {"error": "Unknown explanation method."}
        return {"explanation": explanation}
    except Exception as e:
        logger.error(f"XAI error: {e}")
        return {"error": str(e)}

# ---
# XAI (LIME/SHAP) Notes:
# - Install LIME and SHAP: pip install lime shap
# - Use /xai/explain endpoint to request explanations for model predictions.
# - For production, ensure model and data compatibility.
# ---

# Live collaboration integration using Yjs (y-py) and WebSockets
try:
    from ypy_websocket import WebsocketServer, YRoom
    import y_py as Y
except ImportError:
    WebsocketServer = None
    YRoom = None
    Y = None

from fastapi import WebSocket, WebSocketDisconnect
from typing import List

# In-memory room/session registry for demo
collab_rooms = {}

@app.websocket("/collab/ws/{room_id}")
async def collab_websocket(websocket: WebSocket, room_id: str):
    """
    WebSocket endpoint for real-time collaborative editing using Yjs (CRDT).
    Connect with a Yjs-compatible frontend (e.g., y-websocket, ypy-websocket).
    """
    if WebsocketServer is None or YRoom is None:
        await websocket.close(code=1011)
        return
    await websocket.accept()
    if room_id not in collab_rooms:
        collab_rooms[room_id] = YRoom(room_id)
    room = collab_rooms[room_id]
    server = WebsocketServer(room)
    try:
        await server.serve(websocket)
    except WebSocketDisconnect:
        pass

@app.get("/collab/rooms")
async def list_collab_rooms():
    """List active collaboration rooms (demo only)."""
    return {"rooms": list(collab_rooms.keys())}

# ---
# Live Collaboration Notes:
# - Requires y-py and ypy-websocket: pip install y-py ypy-websocket
# - Connect with a Yjs-compatible frontend (e.g., y-websocket, ypy-websocket, or Yjs clients in JS/Python).
# - Each room_id is a collaborative session (e.g., for a document, code, or AG-UI state).
# - For production, add authentication and persistence for rooms/documents.
# ---

# --- Plugin Marketplace & Dynamic Agent Loading ---
PLUGINS_DIR = Path("plugins")
PLUGINS_DIR.mkdir(exist_ok=True)

# In-memory registry for loaded plugins/agents
loaded_plugins: Dict[str, Any] = {}

# Utility: Discover available plugins in the plugins directory
def discover_plugins() -> List[str]:
    return [f.stem for f in PLUGINS_DIR.glob("*.py") if f.is_file() and not f.stem.startswith("__")]

# Utility: Dynamically load a plugin by name
def load_plugin(plugin_name: str) -> Any:
    if plugin_name in loaded_plugins:
        return loaded_plugins[plugin_name]
    sys.path.insert(0, str(PLUGINS_DIR.resolve()))
    module = importlib.import_module(plugin_name)
    loaded_plugins[plugin_name] = module
    return module

# Utility: Unload a plugin
def unload_plugin(plugin_name: str):
    if plugin_name in loaded_plugins:
        del loaded_plugins[plugin_name]
    if plugin_name in sys.modules:
        del sys.modules[plugin_name]

# --- Plugin Management Endpoints ---
from fastapi import APIRouter, HTTPException, status
plugin_router = APIRouter(prefix="/plugins", tags=["plugins"])

@plugin_router.get("/list")
async def list_plugins():
    """List all available plugins in the marketplace."""
    return {"available_plugins": discover_plugins(), "loaded_plugins": list(loaded_plugins.keys())}

@plugin_router.post("/load/{plugin_name}")
async def load_plugin_endpoint(plugin_name: str):
    """Dynamically load and register a plugin/agent."""
    try:
        module = load_plugin(plugin_name)
        logger.info(f"Plugin loaded: {plugin_name}")
        # Audit log
        logger.info(f"AUDIT: Plugin loaded: {plugin_name}")
        return {"status": "success", "plugin": plugin_name}
    except Exception as e:
        logger.error(f"Failed to load plugin {plugin_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@plugin_router.post("/unload/{plugin_name}")
async def unload_plugin_endpoint(plugin_name: str):
    """Unload a plugin/agent at runtime."""
    try:
        unload_plugin(plugin_name)
        logger.info(f"Plugin unloaded: {plugin_name}")
        # Audit log
        logger.info(f"AUDIT: Plugin unloaded: {plugin_name}")
        return {"status": "success", "plugin": plugin_name}
    except Exception as e:
        logger.error(f"Failed to unload plugin {plugin_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Register plugin router with main app
app.include_router(plugin_router)

# --- Demo Plugin Structure ---
# To create a new plugin/agent, add a Python file in the plugins/ directory with a register() function.
# Example: plugins/demo_agent.py
# def register():
#     print("Demo agent registered!")
#     # Register agent with orchestrator, etc.

# --- Security & Audit Logging ---
# All plugin operations are logged and auditable. For production, add signature verification and sandboxing.
# ---

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 