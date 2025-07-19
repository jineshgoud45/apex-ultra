"""
APEX-ULTRA™ v15.0 AGI COSMOS - IoT Device Manager
Advanced IoT device management, sensor data processing, and edge computing
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
from datetime import datetime, timedelta
import hashlib
import os
import math
import statistics
from collections import deque
import numpy as np

# === Device Manager Self-Healing, Self-Editing, Watchdog, and AGI/GPT-2.5 Pro Integration ===
import os
import threading
import importlib
import aiohttp

class DeviceManagerMaintenance:
    """Handles self-healing, self-editing, and watchdog logic for DeviceManager."""
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
                status = self.manager.get_device_status()
                if status.get("total_devices", 0) < 0:
                    self.self_heal(reason="Negative device count detected")
            except Exception as e:
                self.self_heal(reason=f"Exception in watchdog: {e}")
            time.sleep(interval_sec)

    def self_edit(self, file_path, new_code, safety_check=True):
        if safety_check:
            allowed = ["iot_manager/device_manager.py"]
            if file_path not in allowed:
                raise PermissionError("Self-editing not allowed for this file.")
        with open(file_path, "w") as f:
            f.write(new_code)
        importlib.reload(importlib.import_module(file_path.replace(".py", "").replace("/", ".")))
        return True

    def self_heal(self, reason="Unknown"):
        logger.warning(f"DeviceManager self-healing triggered: {reason}")
        # Reset some metrics or reload configs as a stub
        self.manager._initialize_devices()
        return True

# === AGI/GPT-2.5 Pro Integration Stub ===
class DeviceAgiIntegration:
    """
    Production-grade AGI brain and GPT-2.5 Pro integration for device/IoT strategy.
    """
    def __init__(self, agi_brain=None, api_key=None, endpoint=None):
        self.agi_brain = agi_brain
        self.api_key = api_key or os.getenv("GPT25PRO_API_KEY")
        self.endpoint = endpoint or "https://api.gpt25pro.example.com/v1/generate"

    async def suggest_device_strategy(self, context: dict) -> dict:
        prompt = f"Suggest IoT/device management strategy for: {context}"
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
def backup_device_data(manager, backup_path="backups/device_backup.json"):
    """Stub: Backup device manager data to a secure location."""
    try:
        with open(backup_path, "w") as f:
            json.dump(manager.get_device_status(), f, default=str)
        logger.info(f"Device manager data backed up to {backup_path}")
        return True
    except Exception as e:
        logger.error(f"Backup failed: {e}")
        return False

def report_incident(description, severity="medium"):
    """Stub: Report an incident for compliance and monitoring."""
    logger.warning(f"Incident reported: {description} (Severity: {severity})")
    # In production, send to incident management system
    return True

# Attach to DeviceManager
class DeviceManager:
    """Main IoT device manager orchestrator"""
    
    def __init__(self):
        self.devices: Dict[str, DeviceInfo] = {}
        self.sensor_data: Dict[str, List[SensorData]] = {}
        self.commands: Dict[str, DeviceCommand] = {}
        self.data_processor = SensorDataProcessor()
        self.automation_engine = AutomationEngine()
        self.device_heartbeat_interval = 30  # seconds
        self.data_retention_days = 30
        self.running = False
        
        # Performance tracking
        self.device_performance: Dict[str, Dict[str, Any]] = {}
        self.data_throughput: Dict[str, int] = {}
        
        # Security
        self.device_tokens: Dict[str, str] = {}
        self.encrypted_communication = True
        self.maintenance = DeviceManagerMaintenance(self)
        self.agi_integration = DeviceAgiIntegration()
        self.maintenance.start_watchdog(interval_sec=120)
    
    async def start(self):
        """Start the IoT device manager"""
        self.running = True
        logger.info("Starting IoT Device Manager")
        
        # Start background tasks
        asyncio.create_task(self._device_monitor())
        asyncio.create_task(self._data_cleanup())
        asyncio.create_task(self._automation_processor())
        
        logger.info("IoT Device Manager started successfully")
    
    async def stop(self):
        """Stop the IoT device manager"""
        self.running = False
        logger.info("Stopping IoT Device Manager...")
    
    async def register_device(self, device_info: DeviceInfo, auth_token: str = None) -> bool:
        """Register a new IoT device"""
        if device_info.device_id in self.devices:
            logger.warning(f"Device {device_info.device_id} already registered")
            return False
        
        # Generate authentication token if not provided
        if not auth_token:
            auth_token = self._generate_auth_token(device_info.device_id)
        
        device_info.status = DeviceStatus.ONLINE
        device_info.last_seen = datetime.now()
        device_info.updated_at = datetime.now()
        
        self.devices[device_info.device_id] = device_info
        self.device_tokens[device_info.device_id] = auth_token
        self.sensor_data[device_info.device_id] = []
        self.device_performance[device_info.device_id] = {
            'data_points': 0,
            'last_data': None,
            'uptime': 0,
            'errors': 0
        }
        
        logger.info(f"Registered device: {device_info.name} ({device_info.device_id})")
        return True
    
    async def unregister_device(self, device_id: str) -> bool:
        """Unregister a device"""
        if device_id not in self.devices:
            return False
        
        del self.devices[device_id]
        if device_id in self.device_tokens:
            del self.device_tokens[device_id]
        if device_id in self.sensor_data:
            del self.sensor_data[device_id]
        if device_id in self.device_performance:
            del self.device_performance[device_id]
        
        logger.info(f"Unregistered device: {device_id}")
        return True
    
    async def receive_sensor_data(self, device_id: str, sensor_data: SensorData) -> Dict[str, Any]:
        """Receive and process sensor data from a device"""
        if device_id not in self.devices:
            raise ValueError(f"Device {device_id} not registered")
        
        # Update device status
        self.devices[device_id].last_seen = datetime.now()
        self.devices[device_id].status = DeviceStatus.ONLINE
        
        # Store data
        self.sensor_data[device_id].append(sensor_data)
        
        # Update performance metrics
        self.device_performance[device_id]['data_points'] += 1
        self.device_performance[device_id]['last_data'] = datetime.now()
        self.data_throughput[device_id] = self.data_throughput.get(device_id, 0) + 1
        
        # Process data
        processed_data = await self.data_processor.process_sensor_data(sensor_data)
        
        # Trigger automation rules
        context = {
            'device_id': device_id,
            'sensor_data': {sensor_data.sensor_id: sensor_data.value},
            'processed_data': processed_data,
            'timestamp': datetime.now()
        }
        
        triggered_rules = await self.automation_engine.evaluate_rules(context)
        
        # Execute triggered rules
        for rule in triggered_rules:
            await self.automation_engine.execute_rule(rule, context)
        
        logger.debug(f"Processed sensor data from {device_id}: {sensor_data.sensor_id}")
        return processed_data
    
    async def send_command(self, device_id: str, command_type: str, 
                          parameters: Dict[str, Any] = None, priority: int = 1) -> str:
        """Send a command to a device"""
        if device_id not in self.devices:
            raise ValueError(f"Device {device_id} not registered")
        
        command_id = f"cmd_{uuid.uuid4().hex[:8]}"
        
        command = DeviceCommand(
            command_id=command_id,
            device_id=device_id,
            command_type=command_type,
            parameters=parameters or {},
            priority=priority
        )
        
        self.commands[command_id] = command
        
        # In a real implementation, this would send the command to the device
        logger.info(f"Sent command {command_type} to device {device_id}")
        
        return command_id
    
    async def get_device_status(self, device_id: str) -> Optional[DeviceInfo]:
        """Get device status"""
        return self.devices.get(device_id)
    
    async def get_device_data(self, device_id: str, 
                            start_time: datetime = None, 
                            end_time: datetime = None,
                            sensor_id: str = None) -> List[SensorData]:
        """Get device sensor data"""
        if device_id not in self.sensor_data:
            return []
        
        data = self.sensor_data[device_id]
        
        # Filter by time range
        if start_time:
            data = [d for d in data if d.timestamp >= start_time]
        if end_time:
            data = [d for d in data if d.timestamp <= end_time]
        
        # Filter by sensor
        if sensor_id:
            data = [d for d in data if d.sensor_id == sensor_id]
        
        return data
    
    async def create_automation_rule(self, name: str, trigger_type: AutomationTrigger,
                                   trigger_conditions: Dict[str, Any],
                                   actions: List[Dict[str, Any]]) -> str:
        """Create a new automation rule"""
        rule_id = f"rule_{uuid.uuid4().hex[:8]}"
        
        rule = AutomationRule(
            rule_id=rule_id,
            name=name,
            trigger_type=trigger_type,
            trigger_conditions=trigger_conditions,
            actions=actions
        )
        
        await self.automation_engine.add_rule(rule)
        return rule_id
    
    async def get_automation_rules(self) -> List[AutomationRule]:
        """Get all automation rules"""
        return list(self.automation_engine.rules.values())
    
    async def get_device_performance(self, device_id: str) -> Dict[str, Any]:
        """Get device performance metrics"""
        if device_id not in self.device_performance:
            return {}
        
        performance = self.device_performance[device_id].copy()
        performance['data_throughput'] = self.data_throughput.get(device_id, 0)
        
        return performance
    
    async def _device_monitor(self):
        """Monitor device health and status"""
        while self.running:
            try:
                current_time = datetime.now()
                offline_devices = []
                
                for device_id, device in self.devices.items():
                    # Check if device is offline
                    if (current_time - device.last_seen).total_seconds() > self.device_heartbeat_interval * 2:
                        if device.status != DeviceStatus.OFFLINE:
                            device.status = DeviceStatus.OFFLINE
                            offline_devices.append(device_id)
                    
                    # Update uptime
                    if device.status == DeviceStatus.ONLINE:
                        self.device_performance[device_id]['uptime'] += 1
                
                if offline_devices:
                    logger.warning(f"Devices went offline: {offline_devices}")
                
                await asyncio.sleep(self.device_heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in device monitor: {e}")
                await asyncio.sleep(5)
    
    async def _data_cleanup(self):
        """Clean up old sensor data"""
        while self.running:
            try:
                cutoff_time = datetime.now() - timedelta(days=self.data_retention_days)
                
                for device_id, data_list in self.sensor_data.items():
                    # Remove old data
                    self.sensor_data[device_id] = [
                        data for data in data_list 
                        if data.timestamp > cutoff_time
                    ]
                
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                logger.error(f"Error in data cleanup: {e}")
                await asyncio.sleep(300)
    
    async def _automation_processor(self):
        """Process automation rules"""
        while self.running:
            try:
                # This would process any pending automation tasks
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in automation processor: {e}")
                await asyncio.sleep(5)
    
    def _generate_auth_token(self, device_id: str) -> str:
        """Generate authentication token for a device"""
        token_data = f"{device_id}:{time.time()}"
        return hashlib.sha256(token_data.encode()).hexdigest()

    async def agi_suggest_device_strategy(self, context: dict) -> dict:
        return await self.agi_integration.suggest_device_strategy(context)

# Example usage
async def main():
    """Example usage of IoT Device Manager"""
    iot_manager = IoTDeviceManager()
    await iot_manager.start()
    
    # Register a temperature sensor
    temp_sensor = DeviceInfo(
        device_id="temp_sensor_001",
        name="Living Room Temperature Sensor",
        device_type=DeviceType.SENSOR,
        model="TEMP-2000",
        manufacturer="SensorCorp",
        firmware_version="1.2.3",
        sensors=["temperature", "humidity"],
        location={"lat": 40.7128, "lon": -74.0060}
    )
    
    await iot_manager.register_device(temp_sensor)
    
    # Create automation rule
    rule_id = await iot_manager.create_automation_rule(
        name="High Temperature Alert",
        trigger_type=AutomationTrigger.THRESHOLD,
        trigger_conditions={
            'sensor_id': 'temperature',
            'threshold': 25.0,
            'operator': '>'
        },
        actions=[
            {
                'type': 'send_notification',
                'message': 'Temperature is too high!',
                'channel': 'email'
            }
        ]
    )
    
    # Simulate sensor data
    sensor_data = SensorData(
        device_id="temp_sensor_001",
        sensor_id="temperature",
        data_type=DataType.TEMPERATURE,
        value=26.5,
        unit="°C"
    )
    
    processed_data = await iot_manager.receive_sensor_data("temp_sensor_001", sensor_data)
    print(f"Processed data: {processed_data}")
    
    # Get device status
    status = await iot_manager.get_device_status("temp_sensor_001")
    print(f"Device status: {status.status}")
    
    # Get performance metrics
    performance = await iot_manager.get_device_performance("temp_sensor_001")
    print(f"Performance: {performance}")
    
    await iot_manager.stop()

if __name__ == "__main__":
    asyncio.run(main()) 