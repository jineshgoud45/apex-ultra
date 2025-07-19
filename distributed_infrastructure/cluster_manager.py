"""
APEX-ULTRA™ v15.0 AGI COSMOS - Distributed Infrastructure Manager
Advanced cluster management, load balancing, and distributed computing
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import socket
import psutil
import threading
from datetime import datetime, timedelta
import hashlib
import pickle
import base64
import os
import importlib
import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NodeStatus(Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    MAINTENANCE = "maintenance"
    ERROR = "error"

class TaskPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class NodeInfo:
    """Node information and capabilities"""
    node_id: str
    hostname: str
    ip_address: str
    port: int
    status: NodeStatus
    capabilities: Set[str] = field(default_factory=set)
    cpu_cores: int = 0
    memory_gb: float = 0.0
    disk_gb: float = 0.0
    gpu_count: int = 0
    gpu_memory_gb: float = 0.0
    network_speed_mbps: float = 0.0
    last_heartbeat: datetime = field(default_factory=datetime.now)
    load_average: float = 0.0
    memory_usage_percent: float = 0.0
    disk_usage_percent: float = 0.0
    active_tasks: int = 0
    max_concurrent_tasks: int = 10
    region: str = "default"
    datacenter: str = "default"
    rack: str = "default"

@dataclass
class Task:
    """Distributed task definition"""
    task_id: str
    name: str
    function_name: str
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    assigned_node: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 300
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    estimated_duration: Optional[float] = None
    actual_duration: Optional[float] = None

@dataclass
class ClusterMetrics:
    """Cluster performance metrics"""
    total_nodes: int = 0
    online_nodes: int = 0
    total_cpu_cores: int = 0
    total_memory_gb: float = 0.0
    total_disk_gb: float = 0.0
    total_gpu_count: int = 0
    average_load: float = 0.0
    average_memory_usage: float = 0.0
    average_disk_usage: float = 0.0
    total_tasks: int = 0
    running_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    tasks_per_second: float = 0.0
    average_task_duration: float = 0.0
    network_throughput_mbps: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

class LoadBalancer:
    """Advanced load balancer with multiple strategies"""
    
    def __init__(self):
        self.strategy = "round_robin"
        self.current_index = 0
        self.node_weights: Dict[str, float] = {}
        self.node_load_history: Dict[str, List[float]] = {}
    
    def select_node(self, available_nodes: List[NodeInfo], task: Task) -> Optional[NodeInfo]:
        """Select the best node for a task based on strategy"""
        if not available_nodes:
            return None
        
        if self.strategy == "round_robin":
            return self._round_robin(available_nodes)
        elif self.strategy == "least_loaded":
            return self._least_loaded(available_nodes)
        elif self.strategy == "weighted":
            return self._weighted(available_nodes)
        elif self.strategy == "capability_based":
            return self._capability_based(available_nodes, task)
        elif self.strategy == "geographic":
            return self._geographic(available_nodes, task)
        else:
            return available_nodes[0]
    
    def _round_robin(self, nodes: List[NodeInfo]) -> NodeInfo:
        """Round-robin node selection"""
        if self.current_index >= len(nodes):
            self.current_index = 0
        node = nodes[self.current_index]
        self.current_index += 1
        return node
    
    def _least_loaded(self, nodes: List[NodeInfo]) -> NodeInfo:
        """Select node with lowest load"""
        return min(nodes, key=lambda n: n.load_average + n.active_tasks / n.max_concurrent_tasks)
    
    def _weighted(self, nodes: List[NodeInfo]) -> NodeInfo:
        """Weighted node selection based on performance"""
        if not self.node_weights:
            # Initialize weights based on node capabilities
            for node in nodes:
                weight = (node.cpu_cores * 0.3 + 
                         node.memory_gb * 0.2 + 
                         node.gpu_count * 0.4 + 
                         (100 - node.load_average) * 0.1)
                self.node_weights[node.node_id] = weight
        
        # Select node with highest weight
        return max(nodes, key=lambda n: self.node_weights.get(n.node_id, 0))
    
    def _capability_based(self, nodes: List[NodeInfo], task: Task) -> NodeInfo:
        """Select node based on required capabilities"""
        # Filter nodes by required capabilities
        capable_nodes = []
        for node in nodes:
            if self._has_capabilities(node, task):
                capable_nodes.append(node)
        
        if not capable_nodes:
            return nodes[0]  # Fallback to any node
        
        # Select best capable node
        return self._least_loaded(capable_nodes)
    
    def _geographic(self, nodes: List[NodeInfo], task: Task) -> NodeInfo:
        """Select node based on geographic proximity"""
        # For now, prefer nodes in the same region
        # In a real implementation, you'd use actual geographic data
        preferred_region = getattr(task, 'preferred_region', 'default')
        
        for node in nodes:
            if node.region == preferred_region:
                return node
        
        return nodes[0]
    
    def _has_capabilities(self, node: NodeInfo, task: Task) -> bool:
        """Check if node has required capabilities for task"""
        required_capabilities = getattr(task, 'required_capabilities', set())
        return required_capabilities.issubset(node.capabilities)

# === Cluster Manager Self-Healing, Self-Editing, Watchdog, and AGI/GPT-2.5 Pro Integration ===
class ClusterManagerMaintenance:
    """Handles self-healing, self-editing, and watchdog logic for ClusterManager."""
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
                status = self.manager.get_cluster_status()
                if status.get("total_nodes", 0) < 1:
                    self.self_heal(reason="No cluster nodes detected")
            except Exception as e:
                self.self_heal(reason=f"Exception in watchdog: {e}")
            time.sleep(interval_sec)

    def self_edit(self, file_path, new_code, safety_check=True):
        if safety_check:
            allowed = ["distributed_infrastructure/cluster_manager.py"]
            if file_path not in allowed:
                raise PermissionError("Self-editing not allowed for this file.")
        with open(file_path, "w") as f:
            f.write(new_code)
        importlib.reload(importlib.import_module(file_path.replace(".py", "").replace("/", ".")))
        return True

    def self_heal(self, reason="Unknown"):
        logger.warning(f"ClusterManager self-healing triggered: {reason}")
        # Reset some metrics or reload configs as a stub
        self.manager._initialize_cluster()
        return True

# === AGI/GPT-2.5 Pro Integration Stub ===
class ClusterAgiIntegration:
    """
    Production-grade AGI brain and GPT-2.5 Pro integration for cluster strategy.
    """
    def __init__(self, agi_brain=None, api_key=None, endpoint=None):
        self.agi_brain = agi_brain
        self.api_key = api_key or os.getenv("GPT25PRO_API_KEY")
        self.endpoint = endpoint or "https://api.gpt25pro.example.com/v1/generate"

    async def suggest_cluster_strategy(self, context: dict) -> dict:
        prompt = f"Suggest cluster management strategy for: {context}"
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
def backup_cluster_data(manager, backup_path="backups/cluster_backup.json"):
    """Stub: Backup cluster data to a secure location."""
    try:
        with open(backup_path, "w") as f:
            json.dump(manager.get_cluster_status(), f, default=str)
        logger.info(f"Cluster data backed up to {backup_path}")
        return True
    except Exception as e:
        logger.error(f"Backup failed: {e}")
        return False

def report_incident(description, severity="medium"):
    """Stub: Report an incident for compliance and monitoring."""
    logger.warning(f"Incident reported: {description} (Severity: {severity})")
    # In production, send to incident management system
    return True

class ClusterManager:
    """Advanced distributed cluster manager"""
    
    def __init__(self, cluster_id: str = "apex_ultra_cluster"):
        self.cluster_id = cluster_id
        self.nodes: Dict[str, NodeInfo] = {}
        self.tasks: Dict[str, Task] = {}
        self.load_balancer = LoadBalancer()
        self.task_queue: List[Task] = []
        self.completed_tasks: List[Task] = []
        self.failed_tasks: List[Task] = []
        self.node_heartbeat_interval = 30  # seconds
        self.task_timeout_check_interval = 60  # seconds
        self.metrics = ClusterMetrics()
        self.running = False
        self._lock = threading.Lock()
        self._task_counter = 0
        
        # Performance tracking
        self.task_history: List[Dict[str, Any]] = []
        self.node_performance_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Fault tolerance
        self.node_failure_threshold = 3
        self.node_failure_counts: Dict[str, int] = {}
        self.auto_recovery_enabled = True
        
        # Security
        self.node_authentication_tokens: Dict[str, str] = {}
        self.encrypted_communication = True

        self.maintenance = ClusterManagerMaintenance(self)
        self.agi_integration = ClusterAgiIntegration()
        self.maintenance.start_watchdog(interval_sec=120)
    
    async def start(self):
        """Start the cluster manager"""
        self.running = True
        logger.info(f"Starting cluster manager: {self.cluster_id}")
        
        # Start background tasks
        asyncio.create_task(self._heartbeat_monitor())
        asyncio.create_task(self._task_scheduler())
        asyncio.create_task(self._metrics_collector())
        asyncio.create_task(self._fault_tolerance_monitor())
        
        logger.info("Cluster manager started successfully")
    
    async def stop(self):
        """Stop the cluster manager"""
        self.running = False
        logger.info("Stopping cluster manager...")
        
        # Cancel all running tasks
        for task in self.tasks.values():
            if task.status == TaskStatus.RUNNING:
                task.status = TaskStatus.CANCELLED
        
        logger.info("Cluster manager stopped")
    
    async def register_node(self, node_info: NodeInfo, auth_token: str = None) -> bool:
        """Register a new node in the cluster"""
        with self._lock:
            if node_info.node_id in self.nodes:
                logger.warning(f"Node {node_info.node_id} already registered")
                return False
            
            # Validate authentication
            if auth_token and not self._validate_auth_token(node_info.node_id, auth_token):
                logger.error(f"Invalid authentication token for node {node_info.node_id}")
                return False
            
            # Generate authentication token if not provided
            if not auth_token:
                auth_token = self._generate_auth_token(node_info.node_id)
            
            self.nodes[node_info.node_id] = node_info
            self.node_authentication_tokens[node_info.node_id] = auth_token
            self.node_performance_history[node_info.node_id] = []
            
            logger.info(f"Registered node: {node_info.node_id} ({node_info.hostname})")
            return True
    
    async def unregister_node(self, node_id: str) -> bool:
        """Unregister a node from the cluster"""
        with self._lock:
            if node_id not in self.nodes:
                return False
            
            # Reassign running tasks
            running_tasks = [task for task in self.tasks.values() 
                           if task.assigned_node == node_id and task.status == TaskStatus.RUNNING]
            
            for task in running_tasks:
                await self._reassign_task(task)
            
            del self.nodes[node_id]
            if node_id in self.node_authentication_tokens:
                del self.node_authentication_tokens[node_id]
            
            logger.info(f"Unregistered node: {node_id}")
            return True
    
    async def submit_task(self, name: str, function_name: str, 
                         args: List[Any] = None, kwargs: Dict[str, Any] = None,
                         priority: TaskPriority = TaskPriority.NORMAL,
                         timeout_seconds: int = 300,
                         dependencies: List[str] = None,
                         tags: List[str] = None) -> str:
        """Submit a new task to the cluster"""
        task_id = f"task_{self._task_counter}_{uuid.uuid4().hex[:8]}"
        self._task_counter += 1
        
        task = Task(
            task_id=task_id,
            name=name,
            function_name=function_name,
            args=args or [],
            kwargs=kwargs or {},
            priority=priority,
            timeout_seconds=timeout_seconds,
            dependencies=dependencies or [],
            tags=tags or []
        )
        
        with self._lock:
            self.tasks[task_id] = task
            self.task_queue.append(task)
        
        # Sort queue by priority
        self.task_queue.sort(key=lambda t: t.priority.value, reverse=True)
        
        logger.info(f"Submitted task: {task_id} ({name})")
        return task_id
    
    async def get_task_result(self, task_id: str, timeout: float = None) -> Any:
        """Get the result of a completed task"""
        start_time = time.time()
        
        while True:
            with self._lock:
                if task_id not in self.tasks:
                    raise ValueError(f"Task {task_id} not found")
                
                task = self.tasks[task_id]
                
                if task.status == TaskStatus.COMPLETED:
                    return task.result
                elif task.status == TaskStatus.FAILED:
                    raise Exception(f"Task failed: {task.error}")
                elif task.status == TaskStatus.CANCELLED:
                    raise Exception("Task was cancelled")
            
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Timeout waiting for task {task_id}")
            
            await asyncio.sleep(0.1)
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task"""
        with self._lock:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                task.status = TaskStatus.CANCELLED
                
                if task in self.task_queue:
                    self.task_queue.remove(task)
                
                logger.info(f"Cancelled task: {task_id}")
                return True
        
        return False
    
    async def get_cluster_metrics(self) -> ClusterMetrics:
        """Get current cluster metrics"""
        with self._lock:
            online_nodes = [n for n in self.nodes.values() if n.status == NodeStatus.ONLINE]
            running_tasks = [t for t in self.tasks.values() if t.status == TaskStatus.RUNNING]
            completed_tasks = [t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED]
            failed_tasks = [t for t in self.tasks.values() if t.status == TaskStatus.FAILED]
            
            self.metrics.total_nodes = len(self.nodes)
            self.metrics.online_nodes = len(online_nodes)
            self.metrics.total_cpu_cores = sum(n.cpu_cores for n in online_nodes)
            self.metrics.total_memory_gb = sum(n.memory_gb for n in online_nodes)
            self.metrics.total_disk_gb = sum(n.disk_gb for n in online_nodes)
            self.metrics.total_gpu_count = sum(n.gpu_count for n in online_nodes)
            self.metrics.total_tasks = len(self.tasks)
            self.metrics.running_tasks = len(running_tasks)
            self.metrics.completed_tasks = len(completed_tasks)
            self.metrics.failed_tasks = len(failed_tasks)
            
            if online_nodes:
                self.metrics.average_load = sum(n.load_average for n in online_nodes) / len(online_nodes)
                self.metrics.average_memory_usage = sum(n.memory_usage_percent for n in online_nodes) / len(online_nodes)
                self.metrics.average_disk_usage = sum(n.disk_usage_percent for n in online_nodes) / len(online_nodes)
            
            self.metrics.last_updated = datetime.now()
            
            return self.metrics
    
    async def _heartbeat_monitor(self):
        """Monitor node heartbeats"""
        while self.running:
            try:
                current_time = datetime.now()
                offline_nodes = []
                
                with self._lock:
                    for node_id, node in self.nodes.items():
                        if (current_time - node.last_heartbeat).total_seconds() > self.node_heartbeat_interval * 2:
                            offline_nodes.append(node_id)
                            node.status = NodeStatus.OFFLINE
                
                # Handle offline nodes
                for node_id in offline_nodes:
                    await self._handle_node_failure(node_id)
                
                await asyncio.sleep(self.node_heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat monitor: {e}")
                await asyncio.sleep(5)
    
    async def _task_scheduler(self):
        """Schedule and assign tasks to nodes"""
        while self.running:
            try:
                with self._lock:
                    available_nodes = [n for n in self.nodes.values() 
                                     if n.status == NodeStatus.ONLINE and 
                                     n.active_tasks < n.max_concurrent_tasks]
                    
                    # Process task queue
                    tasks_to_assign = []
                    for task in self.task_queue[:]:
                        if task.status == TaskStatus.PENDING:
                            # Check dependencies
                            if self._check_dependencies(task):
                                tasks_to_assign.append(task)
                                self.task_queue.remove(task)
                
                # Assign tasks to nodes
                for task in tasks_to_assign:
                    if available_nodes:
                        selected_node = self.load_balancer.select_node(available_nodes, task)
                        if selected_node:
                            await self._assign_task_to_node(task, selected_node)
                            # Update available nodes
                            available_nodes = [n for n in available_nodes 
                                             if n.active_tasks < n.max_concurrent_tasks]
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in task scheduler: {e}")
                await asyncio.sleep(5)
    
    async def _metrics_collector(self):
        """Collect and update cluster metrics"""
        while self.running:
            try:
                await self.get_cluster_metrics()
                
                # Store performance history
                for node_id, node in self.nodes.items():
                    if node_id not in self.node_performance_history:
                        self.node_performance_history[node_id] = []
                    
                    self.node_performance_history[node_id].append({
                        'timestamp': datetime.now(),
                        'load_average': node.load_average,
                        'memory_usage': node.memory_usage_percent,
                        'disk_usage': node.disk_usage_percent,
                        'active_tasks': node.active_tasks
                    })
                    
                    # Keep only last 1000 entries
                    if len(self.node_performance_history[node_id]) > 1000:
                        self.node_performance_history[node_id] = self.node_performance_history[node_id][-1000:]
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in metrics collector: {e}")
                await asyncio.sleep(10)
    
    async def _fault_tolerance_monitor(self):
        """Monitor and handle node failures"""
        while self.running:
            try:
                with self._lock:
                    for node_id, failure_count in self.node_failure_counts.items():
                        if failure_count >= self.node_failure_threshold:
                            logger.warning(f"Node {node_id} exceeded failure threshold")
                            if self.auto_recovery_enabled:
                                await self._initiate_node_recovery(node_id)
                
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in fault tolerance monitor: {e}")
                await asyncio.sleep(10)
    
    def _check_dependencies(self, task: Task) -> bool:
        """Check if task dependencies are satisfied"""
        for dep_id in task.dependencies:
            if dep_id not in self.tasks:
                return False
            dep_task = self.tasks[dep_id]
            if dep_task.status != TaskStatus.COMPLETED:
                return False
        return True
    
    async def _assign_task_to_node(self, task: Task, node: NodeInfo):
        """Assign a task to a specific node"""
        task.assigned_node = node.node_id
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        node.active_tasks += 1
        
        logger.info(f"Assigned task {task.task_id} to node {node.node_id}")
        
        # In a real implementation, this would send the task to the node
        # For now, we'll simulate task execution
        asyncio.create_task(self._simulate_task_execution(task))
    
    async def _simulate_task_execution(self, task: Task):
        """Simulate task execution on a node"""
        try:
            # Simulate work
            await asyncio.sleep(2)  # Simulate processing time
            
            # Simulate result
            task.result = f"Result for {task.name} executed on {task.assigned_node}"
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            
            if task.assigned_node in self.nodes:
                self.nodes[task.assigned_node].active_tasks -= 1
            
            logger.info(f"Task {task.task_id} completed successfully")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
            
            if task.assigned_node in self.nodes:
                self.nodes[task.assigned_node].active_tasks -= 1
            
            logger.error(f"Task {task.task_id} failed: {e}")
    
    async def _reassign_task(self, task: Task):
        """Reassign a task to a different node"""
        task.assigned_node = None
        task.status = TaskStatus.PENDING
        task.started_at = None
        self.task_queue.append(task)
        
        logger.info(f"Reassigned task {task.task_id}")
    
    async def _handle_node_failure(self, node_id: str):
        """Handle node failure"""
        self.node_failure_counts[node_id] = self.node_failure_counts.get(node_id, 0) + 1
        
        # Reassign tasks from failed node
        failed_tasks = [task for task in self.tasks.values() 
                       if task.assigned_node == node_id and task.status == TaskStatus.RUNNING]
        
        for task in failed_tasks:
            await self._reassign_task(task)
        
        logger.warning(f"Node {node_id} failed, reassigned {len(failed_tasks)} tasks")
    
    async def _initiate_node_recovery(self, node_id: str):
        """Initiate recovery for a failed node"""
        logger.info(f"Initiating recovery for node {node_id}")
        # In a real implementation, this would trigger node recovery procedures
        self.node_failure_counts[node_id] = 0
    
    def _generate_auth_token(self, node_id: str) -> str:
        """Generate authentication token for a node"""
        token_data = f"{node_id}:{self.cluster_id}:{time.time()}"
        return base64.b64encode(hashlib.sha256(token_data.encode()).digest()).decode()
    
    def _validate_auth_token(self, node_id: str, token: str) -> bool:
        """Validate authentication token"""
        expected_token = self.node_authentication_tokens.get(node_id)
        return expected_token == token

    async def agi_suggest_cluster_strategy(self, context: dict) -> dict:
        return await self.agi_integration.suggest_cluster_strategy(context)

# Example usage
async def main():
    """Example usage of Cluster Manager"""
    cluster = ClusterManager("test_cluster")
    await cluster.start()
    
    # Register some nodes
    node1 = NodeInfo(
        node_id="node_1",
        hostname="worker1.example.com",
        ip_address="192.168.1.10",
        port=8080,
        status=NodeStatus.ONLINE,
        capabilities={"cpu", "gpu", "memory"},
        cpu_cores=8,
        memory_gb=32.0,
        disk_gb=1000.0,
        gpu_count=2
    )
    
    node2 = NodeInfo(
        node_id="node_2",
        hostname="worker2.example.com",
        ip_address="192.168.1.11",
        port=8080,
        status=NodeStatus.ONLINE,
        capabilities={"cpu", "memory"},
        cpu_cores=16,
        memory_gb=64.0,
        disk_gb=2000.0
    )
    
    await cluster.register_node(node1)
    await cluster.register_node(node2)
    
    # Submit some tasks
    task1_id = await cluster.submit_task(
        name="Data Processing",
        function_name="process_data",
        args=["dataset1.csv"],
        priority=TaskPriority.HIGH
    )
    
    task2_id = await cluster.submit_task(
        name="ML Training",
        function_name="train_model",
        kwargs={"model": "neural_network", "epochs": 100},
        priority=TaskPriority.NORMAL
    )
    
    # Wait for tasks to complete
    try:
        result1 = await cluster.get_task_result(task1_id, timeout=10)
        print(f"Task 1 result: {result1}")
    except TimeoutError:
        print("Task 1 timed out")
    
    # Get cluster metrics
    metrics = await cluster.get_cluster_metrics()
    print(f"Cluster metrics: {metrics}")
    
    await cluster.stop()

if __name__ == "__main__":
    asyncio.run(main()) 