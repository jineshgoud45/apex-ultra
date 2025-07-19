"""
Benchmark Agent Plugin
---------------------
This is a sample benchmark agent for the APEX-ULTRA™ plugin marketplace.
It can be dynamically loaded/unloaded at runtime via the orchestrator API.
When registered, it runs a simple benchmark and prints the result.
"""

import time

def register():
    """Register the benchmark agent and run a sample benchmark."""
    print("Benchmark agent registered! Running benchmark...")
    start = time.perf_counter()
    # Example: sum of squares benchmark
    result = sum(i * i for i in range(10**6))
    elapsed = time.perf_counter() - start
    print(f"Benchmark complete: sum of squares (0..1M) = {result} in {elapsed:.4f} seconds.")

# Optionally, add more benchmarking logic or reporting hooks here. 