import pytest
from unittest.mock import patch, MagicMock
from core.agi_brain import AGIBrain, ConfigLoader
import os
import asyncio

@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("GPT25PRO_API_KEY", "test-key")
    yield
    monkeypatch.delenv("GPT25PRO_API_KEY", raising=False)

def test_config_loader_success():
    loader = ConfigLoader()
    assert loader.get("GPT25PRO_API_KEY") == "test-key"

def test_config_loader_missing(monkeypatch):
    monkeypatch.delenv("GPT25PRO_API_KEY", raising=False)
    loader = ConfigLoader()
    with pytest.raises(RuntimeError):
        loader.get("GPT25PRO_API_KEY")

@pytest.mark.asyncio
def test_agi_brain_run_normal(monkeypatch):
    agi = AGIBrain()
    # Patch methods to return simple stubs
    agi.perceive = MagicMock(return_value=asyncio.Future())
    agi.perceive.return_value.set_result({"perceived": True})
    agi.reason = MagicMock(return_value=asyncio.Future())
    agi.reason.return_value.set_result({"final_decision": {"decision": True}})
    agi.act = MagicMock(return_value=asyncio.Future())
    agi.act.return_value.set_result({"success": True})
    agi.evolve = MagicMock(return_value=asyncio.Future())
    agi.evolve.return_value.set_result(None)
    result = asyncio.get_event_loop().run_until_complete(agi.run({"input": "test"}))
    assert result["success"] is True
    assert "cycle_id" in result

@pytest.mark.asyncio
def test_agi_brain_run_error(monkeypatch):
    agi = AGIBrain()
    agi.perceive = MagicMock(side_effect=Exception("fail"))
    with patch.object(agi, "log_automated_decision") as mock_log:
        result = asyncio.get_event_loop().run_until_complete(agi.run({"input": "test"}))
        assert result["success"] is False
        assert "fallback" in result
        assert "error" in result
        mock_log.assert_called()

@pytest.mark.asyncio
def test_compliance_audit_log(monkeypatch):
    agi = AGIBrain()
    # Patch logger to capture audit logs
    with patch("core.agi_brain.logger.info") as mock_info:
        agi.perceive = MagicMock(return_value=asyncio.Future())
        agi.perceive.return_value.set_result({"perceived": True})
        agi.reason = MagicMock(return_value=asyncio.Future())
        agi.reason.return_value.set_result({"final_decision": {"decision": True}})
        agi.act = MagicMock(return_value=asyncio.Future())
        agi.act.return_value.set_result({"success": True})
        agi.evolve = MagicMock(return_value=asyncio.Future())
        agi.evolve.return_value.set_result(None)
        asyncio.get_event_loop().run_until_complete(agi.run({"input": "test"}))
        # Check that compliance audit logs were written
        audit_logs = [call for call in mock_info.call_args_list if "[COMPLIANCE AUDIT]" in str(call)]
        assert audit_logs 