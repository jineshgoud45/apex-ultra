import pytest
import asyncio
from production_hardening.security_manager import ProductionHardeningManager, SecurityLevel, BackupType

@pytest.mark.asyncio
async def test_security_event_logging():
    manager = ProductionHardeningManager()
    event_id = await manager.security_manager.log_security_event(
        event_type="test_event",
        severity=SecurityLevel.MEDIUM,
        source_ip="127.0.0.1",
        user_id="test_user"
    )
    assert isinstance(event_id, str)
    report = await manager.security_manager.get_security_report()
    assert "total_events_24h" in report

@pytest.mark.asyncio
async def test_monitoring_and_alerts():
    manager = ProductionHardeningManager()
    await manager.monitoring_manager.start_monitoring()
    dashboard = await manager.monitoring_manager.get_monitoring_dashboard()
    assert "current_metrics" in dashboard
    assert "system_health" in dashboard

@pytest.mark.asyncio
async def test_backup_and_restore():
    manager = ProductionHardeningManager()
    backup_id = await manager.backup_manager.create_backup(
        backup_type=BackupType.FULL,
        source_paths=["./"],
        compression=False,
        encryption=False
    )
    status = await manager.backup_manager.get_backup_status(backup_id)
    assert "job_id" in status 