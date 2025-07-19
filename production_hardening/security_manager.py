"""
APEX-ULTRA™ v15.0 AGI COSMOS - Production Hardening & Security Manager
Advanced security, monitoring, backup, and disaster recovery systems
"""

import asyncio
import json
import logging
import time
import uuid
import hashlib
import hmac
import base64
import secrets
import ssl
import socket
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import threading
from datetime import datetime, timedelta
import os
import shutil
import tempfile
import zipfile
import tarfile
from pathlib import Path
import psutil
import subprocess
import sqlite3
import pickle

# === Security Manager Self-Healing, Self-Editing, Watchdog, and AGI/GPT-2.5 Pro Integration ===
import os
import threading
import importlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    SECURITY = "security"

class BackupType(Enum):
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    SNAPSHOT = "snapshot"

class MonitoringMetric(Enum):
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_USAGE = "disk_usage"
    NETWORK_TRAFFIC = "network_traffic"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    SECURITY_EVENTS = "security_events"
    BACKUP_STATUS = "backup_status"

@dataclass
class SecurityEvent:
    """Security event information"""
    event_id: str
    timestamp: datetime
    event_type: str
    severity: SecurityLevel
    source_ip: str = ""
    user_id: str = ""
    action: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    blocked: bool = False
    resolved: bool = False
    resolution_notes: str = ""

@dataclass
class SystemAlert:
    """System alert information"""
    alert_id: str
    timestamp: datetime
    alert_type: AlertType
    title: str
    message: str
    source: str = ""
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    acknowledged: bool = False
    resolved: bool = False
    assigned_to: str = ""
    resolution_notes: str = ""

@dataclass
class BackupJob:
    """Backup job information"""
    job_id: str
    backup_type: BackupType
    source_paths: List[str]
    destination_path: str
    compression: bool = True
    encryption: bool = True
    retention_days: int = 30
    status: str = "pending"
    progress: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    size_bytes: int = 0
    error_message: Optional[str] = None
    checksum: str = ""

@dataclass
class MonitoringThreshold:
    """Monitoring threshold configuration"""
    metric: MonitoringMetric
    warning_threshold: float
    critical_threshold: float
    enabled: bool = True
    check_interval_seconds: int = 60
    notification_channels: List[str] = field(default_factory=list)

# === AGI/GPT-2.5 Pro Integration Stub ===
class SecurityAgiIntegration:
    """Stub for AGI brain and GPT-2.5 Pro integration for security/strategy."""
    def __init__(self, agi_brain=None):
        self.agi_brain = agi_brain

    async def suggest_security_strategy(self, context: dict) -> dict:
        if self.agi_brain and hasattr(self.agi_brain, "gpt25pro_reason"):
            prompt = f"Suggest security strategy for: {context}"
            return await self.agi_brain.gpt25pro_reason(prompt)
        return {"suggestion": "[Stub: Connect AGI brain for LLM-driven security strategy]"}

# === Production Hardening Hooks ===
def backup_security_data(manager, backup_path="backups/security_backup.json"):
    """Stub: Backup security manager data to a secure location."""
    try:
        with open(backup_path, "w") as f:
            json.dump(manager.get_security_status(), f, default=str)
        logger.info(f"Security manager data backed up to {backup_path}")
        return True
    except Exception as e:
        logger.error(f"Backup failed: {e}")
        return False

def report_incident(description, severity="medium"):
    """Stub: Report an incident for compliance and monitoring."""
    logger.warning(f"Incident reported: {description} (Severity: {severity})")
    # In production, send to incident management system
    return True

class SecurityManager:
    """Advanced security management system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.security_events: List[SecurityEvent] = []
        self.blocked_ips: Set[str] = set()
        self.allowed_ips: Set[str] = set()
        self.user_sessions: Dict[str, Dict[str, Any]] = {}
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        self.encryption_keys: Dict[str, bytes] = {}
        self.security_policies: Dict[str, Dict[str, Any]] = {}
        self.rate_limits: Dict[str, Dict[str, Any]] = {}
        self.audit_log: List[Dict[str, Any]] = []
        
        # Security configurations
        self.max_login_attempts = 5
        self.session_timeout_minutes = 30
        self.password_min_length = 12
        self.require_2fa = True
        self.encrypt_sensitive_data = True
        
        # Initialize security policies
        self._setup_default_policies()
        self.maintenance = SecurityManagerMaintenance(self)
        self.agi_integration = SecurityAgiIntegration()
        self.maintenance.start_watchdog(interval_sec=120)
    
    def _setup_default_policies(self):
        """Setup default security policies"""
        self.security_policies = {
            'password_policy': {
                'min_length': 12,
                'require_uppercase': True,
                'require_lowercase': True,
                'require_numbers': True,
                'require_special_chars': True,
                'max_age_days': 90,
                'prevent_reuse': 5
            },
            'session_policy': {
                'timeout_minutes': 30,
                'max_concurrent_sessions': 3,
                'require_reauth_for_sensitive_actions': True
            },
            'network_policy': {
                'max_connections_per_ip': 100,
                'rate_limit_requests_per_minute': 60,
                'block_suspicious_ips': True,
                'require_ssl': True
            },
            'data_policy': {
                'encrypt_at_rest': True,
                'encrypt_in_transit': True,
                'backup_encryption': True,
                'data_retention_days': 2555  # 7 years
            }
        }
    
    async def log_security_event(self, event_type: str, severity: SecurityLevel, 
                               source_ip: str = "", user_id: str = "", 
                               action: str = "", details: Dict[str, Any] = None) -> str:
        """Log a security event"""
        event_id = f"sec_{uuid.uuid4().hex[:8]}"
        
        event = SecurityEvent(
            event_id=event_id,
            timestamp=datetime.now(),
            event_type=event_type,
            severity=severity,
            source_ip=source_ip,
            user_id=user_id,
            action=action,
            details=details or {}
        )
        
        self.security_events.append(event)
        
        # Check if IP should be blocked
        if severity in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
            await self._evaluate_ip_blocking(source_ip, event)
        
        # Trigger alerts for critical events
        if severity == SecurityLevel.CRITICAL:
            await self._trigger_security_alert(event)
        
        logger.warning(f"Security event: {event_type} from {source_ip} - {severity.value}")
        return event_id
    
    async def _evaluate_ip_blocking(self, source_ip: str, event: SecurityEvent):
        """Evaluate if an IP should be blocked"""
        if not source_ip:
            return
        
        # Count recent events from this IP
        recent_events = [
            e for e in self.security_events[-1000:]  # Last 1000 events
            if e.source_ip == source_ip and 
            e.timestamp > datetime.now() - timedelta(hours=1) and
            e.severity in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]
        ]
        
        if len(recent_events) >= 5:  # Block after 5 high/critical events in 1 hour
            self.blocked_ips.add(source_ip)
            logger.warning(f"Blocked IP {source_ip} due to multiple security events")
    
    async def _trigger_security_alert(self, event: SecurityEvent):
        """Trigger security alert for critical events"""
        alert = SystemAlert(
            alert_id=f"sec_alert_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now(),
            alert_type=AlertType.SECURITY,
            title=f"Critical Security Event: {event.event_type}",
            message=f"Critical security event detected from {event.source_ip}",
            source="security_manager",
            metric_value=1.0,
            threshold=0.0
        )
        
        # In a real implementation, this would send the alert to monitoring system
        logger.critical(f"SECURITY ALERT: {alert.title}")
    
    async def authenticate_user(self, username: str, password: str, 
                              source_ip: str = "") -> Tuple[bool, str]:
        """Authenticate a user"""
        # Simulate authentication
        if username == "admin" and password == "secure_password_123!":
            session_id = self._create_user_session(username, source_ip)
            return True, session_id
        
        # Log failed attempt
        await self.log_security_event(
            event_type="failed_login",
            severity=SecurityLevel.MEDIUM,
            source_ip=source_ip,
            user_id=username,
            action="login_attempt"
        )
        
        return False, ""
    
    def _create_user_session(self, username: str, source_ip: str) -> str:
        """Create a new user session"""
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        
        self.user_sessions[session_id] = {
            'username': username,
            'source_ip': source_ip,
            'created_at': datetime.now(),
            'last_activity': datetime.now(),
            'active': True
        }
        
        return session_id
    
    async def validate_session(self, session_id: str, source_ip: str = "") -> bool:
        """Validate a user session"""
        if session_id not in self.user_sessions:
            return False
        
        session = self.user_sessions[session_id]
        
        # Check if session is still active
        if not session['active']:
            return False
        
        # Check session timeout
        timeout_minutes = self.security_policies['session_policy']['timeout_minutes']
        if (datetime.now() - session['last_activity']).total_seconds() > timeout_minutes * 60:
            session['active'] = False
            return False
        
        # Check IP address (optional)
        if source_ip and session['source_ip'] != source_ip:
            await self.log_security_event(
                event_type="session_ip_mismatch",
                severity=SecurityLevel.MEDIUM,
                source_ip=source_ip,
                user_id=session['username'],
                action="session_validation"
            )
            return False
        
        # Update last activity
        session['last_activity'] = datetime.now()
        return True
    
    async def generate_api_key(self, user_id: str, permissions: List[str]) -> str:
        """Generate a new API key"""
        api_key = f"ak_{secrets.token_urlsafe(32)}"
        
        self.api_keys[api_key] = {
            'user_id': user_id,
            'permissions': permissions,
            'created_at': datetime.now(),
            'last_used': None,
            'active': True
        }
        
        return api_key
    
    async def validate_api_key(self, api_key: str, required_permission: str = None) -> bool:
        """Validate an API key"""
        if api_key not in self.api_keys:
            return False
        
        key_info = self.api_keys[api_key]
        
        if not key_info['active']:
            return False
        
        # Check permissions
        if required_permission and required_permission not in key_info['permissions']:
            return False
        
        # Update last used
        key_info['last_used'] = datetime.now()
        return True
    
    async def encrypt_data(self, data: str, key_id: str = "default") -> str:
        """Encrypt sensitive data"""
        if key_id not in self.encryption_keys:
            # Generate new key
            self.encryption_keys[key_id] = secrets.token_bytes(32)
        
        # Simple encryption (in production, use proper encryption libraries)
        key = self.encryption_keys[key_id]
        encoded_data = data.encode('utf-8')
        
        # XOR encryption (simplified)
        encrypted = bytes(a ^ b for a, b in zip(encoded_data, key * (len(encoded_data) // len(key) + 1)))
        
        return base64.b64encode(encrypted).decode('utf-8')
    
    async def decrypt_data(self, encrypted_data: str, key_id: str = "default") -> str:
        """Decrypt sensitive data"""
        if key_id not in self.encryption_keys:
            raise ValueError(f"Encryption key {key_id} not found")
        
        key = self.encryption_keys[key_id]
        encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
        
        # XOR decryption (simplified)
        decrypted = bytes(a ^ b for a, b in zip(encrypted_bytes, key * (len(encrypted_bytes) // len(key) + 1)))
        
        return decrypted.decode('utf-8')
    
    async def check_rate_limit(self, identifier: str, limit: int, window_seconds: int = 60) -> bool:
        """Check rate limiting"""
        now = datetime.now()
        
        if identifier not in self.rate_limits:
            self.rate_limits[identifier] = {
                'requests': [],
                'limit': limit,
                'window': window_seconds
            }
        
        rate_info = self.rate_limits[identifier]
        
        # Remove old requests
        cutoff_time = now - timedelta(seconds=window_seconds)
        rate_info['requests'] = [req for req in rate_info['requests'] if req > cutoff_time]
        
        # Check if limit exceeded
        if len(rate_info['requests']) >= limit:
            return False
        
        # Add current request
        rate_info['requests'].append(now)
        return True
    
    async def get_security_report(self) -> Dict[str, Any]:
        """Generate security report"""
        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        
        recent_events = [e for e in self.security_events if e.timestamp > last_24h]
        
        return {
            'total_events_24h': len(recent_events),
            'critical_events': len([e for e in recent_events if e.severity == SecurityLevel.CRITICAL]),
            'high_events': len([e for e in recent_events if e.severity == SecurityLevel.HIGH]),
            'blocked_ips': len(self.blocked_ips),
            'active_sessions': len([s for s in self.user_sessions.values() if s['active']]),
            'active_api_keys': len([k for k in self.api_keys.values() if k['active']]),
            'security_score': self._calculate_security_score()
        }
    
    def _calculate_security_score(self) -> float:
        """Calculate overall security score (0-100)"""
        score = 100.0
        
        # Deduct points for recent security events
        recent_critical = len([e for e in self.security_events[-100:] if e.severity == SecurityLevel.CRITICAL])
        score -= recent_critical * 10
        
        # Deduct points for blocked IPs
        score -= len(self.blocked_ips) * 2
        
        return max(0.0, score)

    def report_incident_to_cert_in(self, incident: dict):
        """Stub: Report a security incident to CERT-In as required by Indian law."""
        logger.critical(f"Reporting incident to CERT-In: {incident}")
        # In production, send an email or API call to CERT-In
        # https://www.cert-in.org.in/

    async def log_security_incident(self, incident_type: str, details: dict):
        """Log and report a security incident."""
        incident = {
            "type": incident_type,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        logger.error(f"Security incident: {incident}")
        self.report_incident_to_cert_in(incident)

    async def agi_suggest_security_strategy(self, context: dict) -> dict:
        return await self.agi_integration.suggest_security_strategy(context)

class MonitoringManager:
    """Advanced system monitoring and alerting"""
    
    def __init__(self):
        self.metrics: Dict[str, List[Tuple[datetime, float]]] = {}
        self.alerts: List[SystemAlert] = []
        self.thresholds: Dict[MonitoringMetric, MonitoringThreshold] = {}
        self.alert_handlers: Dict[AlertType, List[Callable]] = {}
        self.monitoring_interval = 30  # seconds
        self.metrics_retention_hours = 24
        
        # Setup default thresholds
        self._setup_default_thresholds()
        self._setup_alert_handlers()
    
    def _setup_default_thresholds(self):
        """Setup default monitoring thresholds"""
        self.thresholds = {
            MonitoringMetric.CPU_USAGE: MonitoringThreshold(
                metric=MonitoringMetric.CPU_USAGE,
                warning_threshold=70.0,
                critical_threshold=90.0,
                check_interval_seconds=60
            ),
            MonitoringMetric.MEMORY_USAGE: MonitoringThreshold(
                metric=MonitoringMetric.MEMORY_USAGE,
                warning_threshold=80.0,
                critical_threshold=95.0,
                check_interval_seconds=60
            ),
            MonitoringMetric.DISK_USAGE: MonitoringThreshold(
                metric=MonitoringMetric.DISK_USAGE,
                warning_threshold=85.0,
                critical_threshold=95.0,
                check_interval_seconds=300
            ),
            MonitoringMetric.ERROR_RATE: MonitoringThreshold(
                metric=MonitoringMetric.ERROR_RATE,
                warning_threshold=5.0,
                critical_threshold=15.0,
                check_interval_seconds=60
            )
        }
    
    def _setup_alert_handlers(self):
        """Setup alert handlers"""
        self.alert_handlers = {
            AlertType.INFO: [self._log_alert],
            AlertType.WARNING: [self._log_alert, self._send_email_alert],
            AlertType.ERROR: [self._log_alert, self._send_email_alert, self._send_sms_alert],
            AlertType.CRITICAL: [self._log_alert, self._send_email_alert, self._send_sms_alert, self._page_oncall],
            AlertType.SECURITY: [self._log_alert, self._send_email_alert, self._send_sms_alert, self._page_oncall]
        }
    
    async def start_monitoring(self):
        """Start the monitoring system"""
        logger.info("Starting monitoring system")
        
        # Start background monitoring tasks
        asyncio.create_task(self._collect_metrics())
        asyncio.create_task(self._check_thresholds())
        asyncio.create_task(self._cleanup_old_metrics())
    
    async def _collect_metrics(self):
        """Collect system metrics"""
        while True:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                await self.record_metric(MonitoringMetric.CPU_USAGE, cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                await self.record_metric(MonitoringMetric.MEMORY_USAGE, memory.percent)
                
                # Disk usage
                disk = psutil.disk_usage('/')
                await self.record_metric(MonitoringMetric.DISK_USAGE, (disk.used / disk.total) * 100)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(5)
    
    async def record_metric(self, metric: MonitoringMetric, value: float):
        """Record a metric value"""
        metric_key = metric.value
        
        if metric_key not in self.metrics:
            self.metrics[metric_key] = []
        
        self.metrics[metric_key].append((datetime.now(), value))
        
        # Keep only recent metrics
        cutoff_time = datetime.now() - timedelta(hours=self.metrics_retention_hours)
        self.metrics[metric_key] = [
            (timestamp, value) for timestamp, value in self.metrics[metric_key]
            if timestamp > cutoff_time
        ]
    
    async def _check_thresholds(self):
        """Check metrics against thresholds"""
        while True:
            try:
                for metric, threshold in self.thresholds.items():
                    if not threshold.enabled:
                        continue
                    
                    current_value = await self.get_current_metric_value(metric)
                    if current_value is None:
                        continue
                    
                    # Check critical threshold
                    if current_value >= threshold.critical_threshold:
                        await self._create_alert(
                            alert_type=AlertType.CRITICAL,
                            title=f"Critical {metric.value}",
                            message=f"{metric.value} is at {current_value:.1f}% (threshold: {threshold.critical_threshold}%)",
                            source="monitoring",
                            metric_value=current_value,
                            threshold=threshold.critical_threshold
                        )
                    # Check warning threshold
                    elif current_value >= threshold.warning_threshold:
                        await self._create_alert(
                            alert_type=AlertType.WARNING,
                            title=f"Warning {metric.value}",
                            message=f"{metric.value} is at {current_value:.1f}% (threshold: {threshold.warning_threshold}%)",
                            source="monitoring",
                            metric_value=current_value,
                            threshold=threshold.warning_threshold
                        )
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error checking thresholds: {e}")
                await asyncio.sleep(10)
    
    async def get_current_metric_value(self, metric: MonitoringMetric) -> Optional[float]:
        """Get current value for a metric"""
        metric_key = metric.value
        
        if metric_key not in self.metrics or not self.metrics[metric_key]:
            return None
        
        # Return the most recent value
        return self.metrics[metric_key][-1][1]
    
    async def _create_alert(self, alert_type: AlertType, title: str, message: str,
                          source: str = "", metric_value: Optional[float] = None,
                          threshold: Optional[float] = None):
        """Create a new alert"""
        alert = SystemAlert(
            alert_id=f"alert_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now(),
            alert_type=alert_type,
            title=title,
            message=message,
            source=source,
            metric_value=metric_value,
            threshold=threshold
        )
        
        self.alerts.append(alert)
        
        # Trigger alert handlers
        if alert_type in self.alert_handlers:
            for handler in self.alert_handlers[alert_type]:
                try:
                    await handler(alert)
                except Exception as e:
                    logger.error(f"Error in alert handler: {e}")
        
        logger.warning(f"Alert: {title} - {message}")
    
    async def _log_alert(self, alert: SystemAlert):
        """Log alert to system logs"""
        logger.warning(f"ALERT [{alert.alert_type.value.upper()}]: {alert.title} - {alert.message}")
    
    async def _send_email_alert(self, alert: SystemAlert):
        """Send email alert"""
        # In a real implementation, this would send an email
        logger.info(f"Email alert sent: {alert.title}")
    
    async def _send_sms_alert(self, alert: SystemAlert):
        """Send SMS alert"""
        # In a real implementation, this would send an SMS
        logger.info(f"SMS alert sent: {alert.title}")
    
    async def _page_oncall(self, alert: SystemAlert):
        """Page on-call engineer"""
        # In a real implementation, this would page the on-call engineer
        logger.critical(f"PAGING ON-CALL: {alert.title}")
    
    async def _cleanup_old_metrics(self):
        """Clean up old metrics data"""
        while True:
            try:
                cutoff_time = datetime.now() - timedelta(hours=self.metrics_retention_hours)
                
                for metric_key in self.metrics:
                    self.metrics[metric_key] = [
                        (timestamp, value) for timestamp, value in self.metrics[metric_key]
                        if timestamp > cutoff_time
                    ]
                
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                logger.error(f"Error cleaning up metrics: {e}")
                await asyncio.sleep(300)
    
    async def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get monitoring dashboard data"""
        dashboard_data = {
            'current_metrics': {},
            'recent_alerts': [],
            'system_health': 'healthy'
        }
        
        # Current metrics
        for metric in MonitoringMetric:
            value = await self.get_current_metric_value(metric)
            if value is not None:
                dashboard_data['current_metrics'][metric.value] = value
        
        # Recent alerts (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        recent_alerts = [a for a in self.alerts if a.timestamp > cutoff_time]
        dashboard_data['recent_alerts'] = [
            {
                'id': a.alert_id,
                'type': a.alert_type.value,
                'title': a.title,
                'message': a.message,
                'timestamp': a.timestamp.isoformat(),
                'acknowledged': a.acknowledged,
                'resolved': a.resolved
            }
            for a in recent_alerts[-10:]  # Last 10 alerts
        ]
        
        # System health
        critical_alerts = [a for a in recent_alerts if a.alert_type == AlertType.CRITICAL and not a.resolved]
        if critical_alerts:
            dashboard_data['system_health'] = 'critical'
        elif any(a.alert_type == AlertType.ERROR for a in recent_alerts):
            dashboard_data['system_health'] = 'warning'
        
        return dashboard_data

class BackupManager:
    """Advanced backup and disaster recovery system"""
    
    def __init__(self, backup_root: str = "./backups"):
        self.backup_root = Path(backup_root)
        self.backup_root.mkdir(exist_ok=True)
        self.backup_jobs: Dict[str, BackupJob] = {}
        self.backup_schedules: Dict[str, Dict[str, Any]] = {}
        self.encryption_enabled = True
        self.compression_enabled = True
        self.retention_policy = {
            'daily': 7,
            'weekly': 4,
            'monthly': 12,
            'yearly': 5
        }
    
    async def create_backup(self, backup_type: BackupType, source_paths: List[str],
                          destination_path: str = None, compression: bool = True,
                          encryption: bool = True) -> str:
        """Create a new backup job"""
        job_id = f"backup_{uuid.uuid4().hex[:8]}"
        
        if destination_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            destination_path = str(self.backup_root / f"{backup_type.value}_{timestamp}.tar.gz")
        
        job = BackupJob(
            job_id=job_id,
            backup_type=backup_type,
            source_paths=source_paths,
            destination_path=destination_path,
            compression=compression,
            encryption=encryption
        )
        
        self.backup_jobs[job_id] = job
        
        # Start backup in background
        asyncio.create_task(self._execute_backup(job))
        
        logger.info(f"Started backup job {job_id}: {backup_type.value}")
        return job_id
    
    async def _execute_backup(self, job: BackupJob):
        """Execute a backup job"""
        try:
            job.status = "running"
            job.started_at = datetime.now()
            
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Copy files to temporary directory
                total_files = 0
                for source_path in job.source_paths:
                    if os.path.exists(source_path):
                        if os.path.isfile(source_path):
                            shutil.copy2(source_path, temp_path)
                            total_files += 1
                        elif os.path.isdir(source_path):
                            shutil.copytree(source_path, temp_path / Path(source_path).name)
                            total_files += len(list(Path(source_path).rglob('*')))
                
                job.progress = 0.5
                
                # Create archive
                if job.compression:
                    with tarfile.open(job.destination_path, 'w:gz') as tar:
                        tar.add(temp_path, arcname='')
                else:
                    with tarfile.open(job.destination_path, 'w') as tar:
                        tar.add(temp_path, arcname='')
                
                job.progress = 0.8
                
                # Encrypt if needed
                if job.encryption:
                    await self._encrypt_backup(job.destination_path)
                
                job.progress = 1.0
                job.status = "completed"
                job.completed_at = datetime.now()
                
                # Calculate size and checksum
                job.size_bytes = os.path.getsize(job.destination_path)
                job.checksum = await self._calculate_checksum(job.destination_path)
                
                logger.info(f"Backup job {job.job_id} completed successfully")
                
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.now()
            logger.error(f"Backup job {job.job_id} failed: {e}")
    
    async def _encrypt_backup(self, file_path: str):
        """Encrypt backup file"""
        # In a real implementation, this would encrypt the file
        # For now, we'll just simulate encryption
        logger.info(f"Encrypting backup: {file_path}")
    
    async def _calculate_checksum(self, file_path: str) -> str:
        """Calculate file checksum"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    async def restore_backup(self, backup_path: str, destination_path: str) -> bool:
        """Restore from backup"""
        try:
            # Decrypt if needed
            if self.encryption_enabled:
                await self._decrypt_backup(backup_path)
            
            # Extract archive
            with tarfile.open(backup_path, 'r:*') as tar:
                tar.extractall(destination_path)
            
            logger.info(f"Restored backup from {backup_path} to {destination_path}")
            return True
            
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False
    
    async def _decrypt_backup(self, file_path: str):
        """Decrypt backup file"""
        # In a real implementation, this would decrypt the file
        logger.info(f"Decrypting backup: {file_path}")
    
    async def schedule_backup(self, schedule_id: str, backup_type: BackupType,
                            source_paths: List[str], schedule: Dict[str, Any]) -> bool:
        """Schedule recurring backup"""
        self.backup_schedules[schedule_id] = {
            'backup_type': backup_type,
            'source_paths': source_paths,
            'schedule': schedule,
            'last_run': None,
            'next_run': self._calculate_next_run(schedule),
            'enabled': True
        }
        
        logger.info(f"Scheduled backup {schedule_id}")
        return True
    
    def _calculate_next_run(self, schedule: Dict[str, Any]) -> datetime:
        """Calculate next run time based on schedule"""
        now = datetime.now()
        
        if schedule.get('frequency') == 'daily':
            return now + timedelta(days=1)
        elif schedule.get('frequency') == 'weekly':
            return now + timedelta(weeks=1)
        elif schedule.get('frequency') == 'monthly':
            return now + timedelta(days=30)
        
        return now + timedelta(hours=1)
    
    async def get_backup_status(self, job_id: str = None) -> Dict[str, Any]:
        """Get backup job status"""
        if job_id:
            if job_id in self.backup_jobs:
                job = self.backup_jobs[job_id]
                return {
                    'job_id': job.job_id,
                    'status': job.status,
                    'progress': job.progress,
                    'started_at': job.started_at.isoformat() if job.started_at else None,
                    'completed_at': job.completed_at.isoformat() if job.completed_at else None,
                    'size_bytes': job.size_bytes,
                    'error_message': job.error_message
                }
            return {}
        
        # Return all jobs
        return {
            'jobs': [
                {
                    'job_id': job.job_id,
                    'status': job.status,
                    'progress': job.progress,
                    'backup_type': job.backup_type.value
                }
                for job in self.backup_jobs.values()
            ],
            'schedules': [
                {
                    'schedule_id': schedule_id,
                    'enabled': schedule_info['enabled'],
                    'next_run': schedule_info['next_run'].isoformat() if schedule_info['next_run'] else None
                }
                for schedule_id, schedule_info in self.backup_schedules.items()
            ]
        }

class ProductionHardeningManager:
    """Main production hardening orchestrator"""
    
    def __init__(self):
        self.security_manager = SecurityManager()
        self.monitoring_manager = MonitoringManager()
        self.backup_manager = BackupManager()
        self.running = False
        
        # System health
        self.system_health = "healthy"
        self.last_health_check = datetime.now()
        self.health_check_interval = 300  # 5 minutes
    
    async def start(self):
        """Start the production hardening system"""
        self.running = True
        logger.info("Starting Production Hardening Manager")
        
        # Start all subsystems
        await self.monitoring_manager.start_monitoring()
        
        # Start background tasks
        asyncio.create_task(self._health_monitor())
        asyncio.create_task(self._security_monitor())
        asyncio.create_task(self._backup_scheduler())
        
        logger.info("Production Hardening Manager started successfully")
    
    async def stop(self):
        """Stop the production hardening system"""
        self.running = False
        logger.info("Stopping Production Hardening Manager...")
    
    async def _health_monitor(self):
        """Monitor overall system health"""
        while self.running:
            try:
                # Get monitoring dashboard
                dashboard = await self.monitoring_manager.get_monitoring_dashboard()
                
                # Update system health
                self.system_health = dashboard['system_health']
                self.last_health_check = datetime.now()
                
                # Log health status
                if self.system_health != "healthy":
                    logger.warning(f"System health: {self.system_health}")
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(60)
    
    async def _security_monitor(self):
        """Monitor security events"""
        while self.running:
            try:
                # Get security report
                security_report = await self.security_manager.get_security_report()
                
                # Check for critical security issues
                if security_report['critical_events'] > 0:
                    logger.critical(f"Critical security events detected: {security_report['critical_events']}")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in security monitor: {e}")
                await asyncio.sleep(60)
    
    async def _backup_scheduler(self):
        """Process backup schedules"""
        while self.running:
            try:
                now = datetime.now()
                
                for schedule_id, schedule_info in self.backup_manager.backup_schedules.items():
                    if not schedule_info['enabled']:
                        continue
                    
                    if schedule_info['next_run'] and now >= schedule_info['next_run']:
                        # Execute scheduled backup
                        await self.backup_manager.create_backup(
                            backup_type=schedule_info['backup_type'],
                            source_paths=schedule_info['source_paths']
                        )
                        
                        # Update schedule
                        schedule_info['last_run'] = now
                        schedule_info['next_run'] = self.backup_manager._calculate_next_run(schedule_info['schedule'])
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in backup scheduler: {e}")
                await asyncio.sleep(300)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        return {
            'system_health': self.system_health,
            'last_health_check': self.last_health_check.isoformat(),
            'security': await self.security_manager.get_security_report(),
            'monitoring': await self.monitoring_manager.get_monitoring_dashboard(),
            'backups': await self.backup_manager.get_backup_status()
        }
    
    async def create_emergency_backup(self) -> str:
        """Create emergency backup"""
        return await self.backup_manager.create_backup(
            backup_type=BackupType.FULL,
            source_paths=['./data', './config', './logs'],
            compression=True,
            encryption=True
        )
    
    async def initiate_disaster_recovery(self, backup_path: str) -> bool:
        """Initiate disaster recovery process"""
        try:
            logger.critical("Initiating disaster recovery process")
            
            # Stop all services
            # In a real implementation, this would stop all running services
            
            # Restore from backup
            success = await self.backup_manager.restore_backup(backup_path, "./restore")
            
            if success:
                logger.info("Disaster recovery completed successfully")
                # In a real implementation, this would restart services
            else:
                logger.error("Disaster recovery failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Disaster recovery error: {e}")
            return False

# Example usage
async def main():
    """Example usage of Production Hardening Manager"""
    hardening_manager = ProductionHardeningManager()
    await hardening_manager.start()
    
    # Create a backup
    backup_job_id = await hardening_manager.backup_manager.create_backup(
        backup_type=BackupType.FULL,
        source_paths=['./config', './logs'],
        compression=True,
        encryption=True
    )
    
    # Schedule recurring backup
    await hardening_manager.backup_manager.schedule_backup(
        schedule_id="daily_backup",
        backup_type=BackupType.INCREMENTAL,
        source_paths=['./data'],
        schedule={'frequency': 'daily', 'time': '02:00'}
    )
    
    # Simulate security event
    await hardening_manager.security_manager.log_security_event(
        event_type="failed_login",
        severity=SecurityLevel.MEDIUM,
        source_ip="192.168.1.100",
        user_id="unknown_user"
    )
    
    # Get system status
    status = await hardening_manager.get_system_status()
    print(f"System status: {status['system_health']}")
    
    # Wait for monitoring to collect some data
    await asyncio.sleep(10)
    
    # Get monitoring dashboard
    dashboard = await hardening_manager.monitoring_manager.get_monitoring_dashboard()
    print(f"Current CPU usage: {dashboard['current_metrics'].get('cpu_usage', 'N/A')}%")
    
    await hardening_manager.stop()

if __name__ == "__main__":
    asyncio.run(main()) 


class SecurityManagerMaintenance:
    """
    Production-grade maintenance and watchdog for SecurityManager.
    Periodically checks security events, logs stats, and performs automated incident cleanup.
    """
    def __init__(self, manager):
        self.manager = manager
        self._watchdog_thread = None
        self._stop_event = threading.Event()

    def start_watchdog(self, interval_sec=120):
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            return  # Already running
        self._stop_event.clear()
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, args=(interval_sec,), daemon=True)
        self._watchdog_thread.start()

    def stop_watchdog(self):
        self._stop_event.set()
        if self._watchdog_thread:
            self._watchdog_thread.join()

    def _watchdog_loop(self, interval_sec):
        while not self._stop_event.is_set():
            self._perform_maintenance()
            time.sleep(interval_sec)

    def _perform_maintenance(self):
        # Log security event stats
        event_count = len(self.manager.security_events)
        blocked_ips = len(self.manager.blocked_ips)
        logger.info(f"[Watchdog] Security events: {event_count}, Blocked IPs: {blocked_ips}")
        # Example: Remove resolved events older than 30 days
        cutoff = datetime.now() - timedelta(days=30)
        before = len(self.manager.security_events)
        self.manager.security_events = [e for e in self.manager.security_events if not (e.resolved and e.timestamp < cutoff)]
        after = len(self.manager.security_events)
        if before != after:
            logger.info(f"[Watchdog] Cleaned up {before - after} old resolved security events.") 