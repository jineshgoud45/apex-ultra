"""
APEX-ULTRA™ v15.0 AGI COSMOS - API Integration Manager
Comprehensive API management, rate limiting, and multi-service integration
"""

import asyncio
import aiohttp
import time
import json
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import hmac
import base64
from datetime import datetime, timedelta
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIService(Enum):
    YOUTUBE = "youtube"
    REDDIT = "reddit"
    TWITTER = "twitter"
    INSTAGRAM = "instagram"
    LINKEDIN = "linkedin"
    FACEBOOK = "facebook"
    STRIPE = "stripe"
    PAYPAL = "paypal"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE_AI = "google_ai"
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    TWILIO = "twilio"
    SENDGRID = "sendgrid"
    MAILCHIMP = "mailchimp"
    SHOPIFY = "shopify"
    WORDPRESS = "wordpress"

@dataclass
class APIEndpoint:
    """API endpoint configuration"""
    url: str
    method: str = "GET"
    headers: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    body: Optional[Dict[str, Any]] = None
    timeout: int = 30
    retry_count: int = 3
    rate_limit: Optional[int] = None
    rate_limit_window: int = 60

@dataclass
class APIResponse:
    """Standardized API response"""
    success: bool
    data: Any
    status_code: int
    headers: Dict[str, str]
    error_message: Optional[str] = None
    rate_limit_remaining: Optional[int] = None
    rate_limit_reset: Optional[datetime] = None
    request_id: Optional[str] = None

@dataclass
class RateLimitInfo:
    """Rate limiting information"""
    service: APIService
    requests_made: int = 0
    window_start: datetime = field(default_factory=datetime.now)
    limit: Optional[int] = None
    reset_time: Optional[datetime] = None

# === API Manager Self-Healing, Self-Editing, Watchdog, and AGI/GPT-2.5 Pro Integration ===
import os
import threading
import importlib

class APIManagerMaintenance:
    """Handles self-healing, self-editing, and watchdog logic for APIManager."""
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
                status = self.manager.get_api_status()
                if status.get("total_integrations", 0) < 0:
                    self.self_heal(reason="Negative integration count detected")
            except Exception as e:
                self.self_heal(reason=f"Exception in watchdog: {e}")
            time.sleep(interval_sec)

    def self_edit(self, file_path, new_code, safety_check=True):
        if safety_check:
            allowed = ["api_integration/api_manager.py"]
            if file_path not in allowed:
                raise PermissionError("Self-editing not allowed for this file.")
        with open(file_path, "w") as f:
            f.write(new_code)
        importlib.reload(importlib.import_module(file_path.replace(".py", "").replace("/", ".")))
        return True

    def self_heal(self, reason="Unknown"):
        logger.warning(f"APIManager self-healing triggered: {reason}")
        # Reset some metrics or reload configs as a stub
        self.manager._initialize_integrations()
        return True

# === AGI/GPT-2.5 Pro Integration Stub ===
import aiohttp

class APIAgiIntegration:
    """
    Production-grade AGI brain and GPT-2.5 Pro integration for API strategy.
    """
    def __init__(self, agi_brain=None, api_key=None, endpoint=None):
        self.agi_brain = agi_brain
        self.api_key = api_key or os.getenv("GPT25PRO_API_KEY")
        self.endpoint = endpoint or "https://api.gpt25pro.example.com/v1/generate"

    async def suggest_api_strategy(self, context: dict) -> dict:
        prompt = f"Suggest API integration strategy for: {context}"
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
def backup_api_data(manager, backup_path="backups/api_backup.json"):
    """Stub: Backup API integration data to a secure location."""
    try:
        with open(backup_path, "w") as f:
            json.dump(manager.get_api_status(), f, default=str)
        logger.info(f"API integration data backed up to {backup_path}")
        return True
    except Exception as e:
        logger.error(f"Backup failed: {e}")
        return False

def report_incident(description, severity="medium"):
    """Stub: Report an incident for compliance and monitoring."""
    logger.warning(f"Incident reported: {description} (Severity: {severity})")
    # In production, send to incident management system
    return True

class APIManager:
    """Comprehensive API integration manager with rate limiting and error handling"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limits: Dict[APIService, RateLimitInfo] = {}
        self.api_keys: Dict[APIService, str] = {}
        self.endpoints: Dict[APIService, Dict[str, APIEndpoint]] = {}
        self.cache: Dict[str, Any] = {}
        self.cache_ttl: Dict[str, datetime] = {}
        self._setup_endpoints()
        self._load_api_keys()
        self.maintenance = APIManagerMaintenance(self)
        self.agi_integration = APIAgiIntegration()
        self.maintenance.start_watchdog(interval_sec=120)
    
    def _setup_endpoints(self):
        """Setup API endpoints for all services"""
        # YouTube API endpoints
        self.endpoints[APIService.YOUTUBE] = {
            "search": APIEndpoint(
                url="https://www.googleapis.com/youtube/v3/search",
                params={"part": "snippet", "maxResults": 50},
                rate_limit=10000,
                rate_limit_window=86400
            ),
            "videos": APIEndpoint(
                url="https://www.googleapis.com/youtube/v3/videos",
                params={"part": "snippet,statistics,contentDetails"},
                rate_limit=10000,
                rate_limit_window=86400
            ),
            "channels": APIEndpoint(
                url="https://www.googleapis.com/youtube/v3/channels",
                params={"part": "snippet,statistics,contentDetails"},
                rate_limit=10000,
                rate_limit_window=86400
            ),
            "playlists": APIEndpoint(
                url="https://www.googleapis.com/youtube/v3/playlists",
                params={"part": "snippet,contentDetails"},
                rate_limit=10000,
                rate_limit_window=86400
            )
        }
        
        # Reddit API endpoints
        self.endpoints[APIService.REDDIT] = {
            "subreddit_posts": APIEndpoint(
                url="https://oauth.reddit.com/r/{subreddit}/hot",
                rate_limit=60,
                rate_limit_window=60
            ),
            "submit_post": APIEndpoint(
                url="https://oauth.reddit.com/api/submit",
                method="POST",
                rate_limit=30,
                rate_limit_window=600
            ),
            "comment": APIEndpoint(
                url="https://oauth.reddit.com/api/comment",
                method="POST",
                rate_limit=30,
                rate_limit_window=600
            )
        }
        
        # Stripe API endpoints
        self.endpoints[APIService.STRIPE] = {
            "create_payment_intent": APIEndpoint(
                url="https://api.stripe.com/v1/payment_intents",
                method="POST",
                rate_limit=100,
                rate_limit_window=60
            ),
            "create_customer": APIEndpoint(
                url="https://api.stripe.com/v1/customers",
                method="POST",
                rate_limit=100,
                rate_limit_window=60
            ),
            "list_charges": APIEndpoint(
                url="https://api.stripe.com/v1/charges",
                rate_limit=100,
                rate_limit_window=60
            )
        }
        
        # OpenAI API endpoints
        self.endpoints[APIService.OPENAI] = {
            "chat_completion": APIEndpoint(
                url="https://api.openai.com/v1/chat/completions",
                method="POST",
                rate_limit=3500,
                rate_limit_window=60
            ),
            "text_completion": APIEndpoint(
                url="https://api.openai.com/v1/completions",
                method="POST",
                rate_limit=3500,
                rate_limit_window=60
            ),
            "embeddings": APIEndpoint(
                url="https://api.openai.com/v1/embeddings",
                method="POST",
                rate_limit=3500,
                rate_limit_window=60
            )
        }
    
    def _load_api_keys(self):
        """Load API keys from environment variables"""
        api_key_mapping = {
            APIService.YOUTUBE: "YOUTUBE_API_KEY",
            APIService.REDDIT: "REDDIT_CLIENT_ID",
            APIService.TWITTER: "TWITTER_BEARER_TOKEN",
            APIService.INSTAGRAM: "INSTAGRAM_ACCESS_TOKEN",
            APIService.LINKEDIN: "LINKEDIN_ACCESS_TOKEN",
            APIService.FACEBOOK: "FACEBOOK_ACCESS_TOKEN",
            APIService.STRIPE: "STRIPE_SECRET_KEY",
            APIService.PAYPAL: "PAYPAL_CLIENT_ID",
            APIService.OPENAI: "OPENAI_API_KEY",
            APIService.ANTHROPIC: "ANTHROPIC_API_KEY",
            APIService.GOOGLE_AI: "GOOGLE_AI_API_KEY",
            APIService.AWS: "AWS_ACCESS_KEY_ID",
            APIService.AZURE: "AZURE_SUBSCRIPTION_KEY",
            APIService.GCP: "GOOGLE_CLOUD_API_KEY",
            APIService.TWILIO: "TWILIO_ACCOUNT_SID",
            APIService.SENDGRID: "SENDGRID_API_KEY",
            APIService.MAILCHIMP: "MAILCHIMP_API_KEY",
            APIService.SHOPIFY: "SHOPIFY_ACCESS_TOKEN",
            APIService.WORDPRESS: "WORDPRESS_ACCESS_TOKEN"
        }
        
        for service, env_var in api_key_mapping.items():
            api_key = os.getenv(env_var)
            if api_key:
                self.api_keys[service] = api_key
                logger.info(f"Loaded API key for {service.value}")
            else:
                logger.warning(f"Missing API key for {service.value} ({env_var})")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _check_rate_limit(self, service: APIService) -> bool:
        """Check if rate limit allows request"""
        if service not in self.rate_limits:
            self.rate_limits[service] = RateLimitInfo(service=service)
        
        rate_info = self.rate_limits[service]
        now = datetime.now()
        
        # Reset window if needed
        if now - rate_info.window_start > timedelta(seconds=60):
            rate_info.requests_made = 0
            rate_info.window_start = now
        
        # Check if we're at the limit
        if rate_info.limit and rate_info.requests_made >= rate_info.limit:
            return False
        
        return True
    
    async def _update_rate_limit(self, service: APIService, response_headers: Dict[str, str]):
        """Update rate limit information from response headers"""
        if service not in self.rate_limits:
            self.rate_limits[service] = RateLimitInfo(service=service)
        
        rate_info = self.rate_limits[service]
        rate_info.requests_made += 1
        
        # Parse rate limit headers
        remaining = response_headers.get('X-RateLimit-Remaining')
        reset = response_headers.get('X-RateLimit-Reset')
        
        if remaining:
            try:
                rate_info.limit = int(remaining)
            except ValueError:
                pass
        
        if reset:
            try:
                reset_timestamp = int(reset)
                rate_info.reset_time = datetime.fromtimestamp(reset_timestamp)
            except ValueError:
                pass
    
    async def _make_request(self, service: APIService, endpoint_name: str, 
                          params: Optional[Dict[str, Any]] = None,
                          body: Optional[Dict[str, Any]] = None,
                          headers: Optional[Dict[str, str]] = None) -> APIResponse:
        """Make API request with rate limiting and error handling"""
        
        # Check rate limit
        if not await self._check_rate_limit(service):
            return APIResponse(
                success=False,
                data=None,
                status_code=429,
                headers={},
                error_message="Rate limit exceeded"
            )
        
        # Get endpoint configuration
        if service not in self.endpoints or endpoint_name not in self.endpoints[service]:
            return APIResponse(
                success=False,
                data=None,
                status_code=400,
                headers={},
                error_message=f"Unknown endpoint: {service.value}/{endpoint_name}"
            )
        
        endpoint = self.endpoints[service][endpoint_name]
        
        # Prepare request
        url = endpoint.url
        method = endpoint.method
        request_headers = endpoint.headers.copy()
        request_params = endpoint.params.copy()
        request_body = endpoint.body or body
        
        # Add service-specific headers
        if service == APIService.YOUTUBE:
            if service in self.api_keys:
                request_params['key'] = self.api_keys[service]
        elif service == APIService.REDDIT:
            if service in self.api_keys:
                request_headers['Authorization'] = f"Bearer {self.api_keys[service]}"
        elif service == APIService.STRIPE:
            if service in self.api_keys:
                request_headers['Authorization'] = f"Bearer {self.api_keys[service]}"
        elif service == APIService.OPENAI:
            if service in self.api_keys:
                request_headers['Authorization'] = f"Bearer {self.api_keys[service]}"
                request_headers['Content-Type'] = 'application/json'
        
        # Merge custom parameters
        if params:
            request_params.update(params)
        if headers:
            request_headers.update(headers)
        
        # Format URL with parameters
        if '{' in url and '}' in url:
            for key, value in request_params.items():
                if f'{{{key}}}' in url:
                    url = url.replace(f'{{{key}}}', str(value))
                    del request_params[key]
        
        try:
            # Make request
            async with self.session.request(
                method=method,
                url=url,
                params=request_params,
                json=request_body,
                headers=request_headers,
                timeout=aiohttp.ClientTimeout(total=endpoint.timeout)
            ) as response:
                
                response_data = await response.json() if response.content_type == 'application/json' else await response.text()
                
                # Update rate limit info
                await self._update_rate_limit(service, dict(response.headers))
                
                return APIResponse(
                    success=200 <= response.status < 300,
                    data=response_data,
                    status_code=response.status,
                    headers=dict(response.headers),
                    error_message=None if response.status < 400 else str(response_data),
                    rate_limit_remaining=self.rate_limits[service].limit,
                    rate_limit_reset=self.rate_limits[service].reset_time,
                    request_id=response.headers.get('X-Request-ID')
                )
                
        except asyncio.TimeoutError:
            return APIResponse(
                success=False,
                data=None,
                status_code=408,
                headers={},
                error_message="Request timeout"
            )
        except Exception as e:
            logger.error(f"API request failed: {e}")
            return APIResponse(
                success=False,
                data=None,
                status_code=500,
                headers={},
                error_message=str(e)
            )
    
    async def youtube_search(self, query: str, max_results: int = 10) -> APIResponse:
        """Search YouTube videos"""
        params = {
            'q': query,
            'maxResults': min(max_results, 50),
            'type': 'video',
            'part': 'snippet'
        }
        return await self._make_request(APIService.YOUTUBE, "search", params=params)
    
    async def youtube_get_video(self, video_id: str) -> APIResponse:
        """Get YouTube video details"""
        params = {'id': video_id}
        return await self._make_request(APIService.YOUTUBE, "videos", params=params)
    
    async def reddit_get_posts(self, subreddit: str, limit: int = 25) -> APIResponse:
        """Get Reddit posts from subreddit"""
        params = {'limit': limit}
        return await self._make_request(APIService.REDDIT, "subreddit_posts", params=params)
    
    async def stripe_create_payment_intent(self, amount: int, currency: str = 'usd') -> APIResponse:
        """Create Stripe payment intent"""
        body = {
            'amount': amount,
            'currency': currency,
            'automatic_payment_methods': {'enabled': True}
        }
        return await self._make_request(APIService.STRIPE, "create_payment_intent", body=body)
    
    async def openai_chat_completion(self, messages: List[Dict[str, str]], 
                                   model: str = "gpt-4") -> APIResponse:
        """Create OpenAI chat completion"""
        body = {
            'model': model,
            'messages': messages,
            'max_tokens': 1000,
            'temperature': 0.7
        }
        return await self._make_request(APIService.OPENAI, "chat_completion", body=body)
    
    async def batch_request(self, requests: List[Dict[str, Any]]) -> List[APIResponse]:
        """Execute multiple API requests in parallel"""
        tasks = []
        for req in requests:
            service = APIService(req['service'])
            endpoint = req['endpoint']
            params = req.get('params')
            body = req.get('body')
            headers = req.get('headers')
            
            task = self._make_request(service, endpoint, params, body, headers)
            tasks.append(task)
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status for all services"""
        status = {}
        for service, rate_info in self.rate_limits.items():
            status[service.value] = {
                'requests_made': rate_info.requests_made,
                'limit': rate_info.limit,
                'window_start': rate_info.window_start.isoformat(),
                'reset_time': rate_info.reset_time.isoformat() if rate_info.reset_time else None
            }
        return status
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all API services"""
        health_status = {}
        
        # Simple health checks for each service
        health_checks = [
            (APIService.YOUTUBE, "search", {'q': 'test', 'maxResults': 1}),
            (APIService.OPENAI, "chat_completion", {
                'model': 'gpt-3.5-turbo',
                'messages': [{'role': 'user', 'content': 'Hello'}],
                'max_tokens': 5
            })
        ]
        
        for service, endpoint, params in health_checks:
            try:
                response = await self._make_request(service, endpoint, params=params)
                health_status[service.value] = response.success
            except Exception as e:
                health_status[service.value] = False
                logger.error(f"Health check failed for {service.value}: {e}")
        
        return health_status

    async def agi_suggest_api_strategy(self, context: dict) -> dict:
        return await self.agi_integration.suggest_api_strategy(context)

    def handle_event(self, event_type, payload):
        if event_type == 'create':
            return self.create_integration(payload)
        elif event_type == 'modify':
            return self.modify_integration(payload)
        elif event_type == 'explain':
            return self.explain_output(payload)
        elif event_type == 'review':
            return self.review_integration(payload)
        elif event_type == 'approve':
            return self.approve_integration(payload)
        elif event_type == 'reject':
            return self.reject_integration(payload)
        elif event_type == 'feedback':
            return self.feedback_integration(payload)
        else:
            return {"error": "Unknown event type"}

    def create_integration(self, payload):
        result = {"integration_id": "API123", "status": "created", **payload}
        self.log_action('create', result)
        return result

    def modify_integration(self, payload):
        result = {"integration_id": payload.get('integration_id'), "status": "modified", **payload}
        self.log_action('modify', result)
        return result

    def explain_output(self, result):
        if not result:
            return "No integration data available."
        explanation = f"Integration '{result.get('integration_id', 'N/A')}' for service {result.get('service', 'N/A')}, status: {result.get('status', 'N/A')}."
        if result.get('status') == 'pending_review':
            explanation += " This integration is pending human review."
        return explanation

    def review_integration(self, payload):
        result = {"integration_id": payload.get('integration_id'), "status": "under_review"}
        self.log_action('review', result)
        return result

    def approve_integration(self, payload):
        result = {"integration_id": payload.get('integration_id'), "status": "approved"}
        self.log_action('approve', result)
        return result

    def reject_integration(self, payload):
        result = {"integration_id": payload.get('integration_id'), "status": "rejected"}
        self.log_action('reject', result)
        return result

    def feedback_integration(self, payload):
        result = {"integration_id": payload.get('integration_id'), "status": "feedback_received", "feedback": payload.get('feedback')}
        self.log_action('feedback', result)
        return result

    def log_action(self, action, details):
        if not hasattr(self, 'audit_log'):
            self.audit_log = []
        self.audit_log.append({"action": action, "details": details})

# Example usage
async def main():
    """Example usage of API Manager"""
    async with APIManager() as api_manager:
        # Search YouTube
        youtube_result = await api_manager.youtube_search("artificial intelligence", max_results=5)
        print(f"YouTube search result: {youtube_result.success}")
        
        # OpenAI chat completion
        openai_result = await api_manager.openai_chat_completion([
            {"role": "user", "content": "What is AGI?"}
        ])
        print(f"OpenAI result: {openai_result.success}")
        
        # Check rate limits
        rate_status = api_manager.get_rate_limit_status()
        print(f"Rate limit status: {rate_status}")
        
        # Health check
        health = await api_manager.health_check()
        print(f"Health status: {health}")

if __name__ == "__main__":
    asyncio.run(main()) 