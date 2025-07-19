"""
Health Monitor Agent Plugin
--------------------------
This is a sample health monitor agent for the APEX-ULTRA™ plugin marketplace.
It can be dynamically loaded/unloaded at runtime via the orchestrator API.
When registered, it checks the health/status of all loaded agents/plugins and prints a report.
"""

import sys

def register():
    """Register the health monitor agent and check the health of all loaded agents/plugins."""
    print("Health monitor agent registered! Checking health of loaded agents...")
    # Discover all loaded plugin modules (excluding self)
    loaded_plugins = [m for n, m in sys.modules.items() if n.startswith("plugins.") and n != __name__]
    health_report = {}
    for plugin in loaded_plugins:
        name = getattr(plugin, "__name__", str(plugin))
        if hasattr(plugin, "register"):
            health_report[name] = "OK (register() present)"
        elif hasattr(plugin, "health_check"):
            try:
                status = plugin.health_check()
                health_report[name] = f"Custom health: {status}"
            except Exception as e:
                health_report[name] = f"Health check error: {e}"
        else:
            health_report[name] = "No health method found"
    print("Agent Health Report:")
    for name, status in health_report.items():
        print(f"- {name}: {status}")
    print(f"Health check complete. {len(loaded_plugins)} agents checked.")

# Optionally, add more advanced health checks or reporting hooks here. 