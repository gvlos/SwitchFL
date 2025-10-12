#!/usr/bin/env python3
"""Test if non-determinism is consistent or random."""

import os
import sys

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from switchfl.utils.build_env import build_standard_async_env

def test_multiple_runs():
    """Test if non-determinism is consistent across multiple runs."""
    seed = 202122
    
    results = []
    
    for run_num in range(5):
        print(f"\n=== Run {run_num + 1} ===")
        
        # Force complete randomness reset
        import numpy as np
        import random
        import os
        
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # Create environment
        env = build_standard_async_env(
            height=25,
            width=25,
            max_num_cities=2,
            num_trains=2,
            grid_mode=True,
            seed=seed
        )
        
        # Reset environment
        env.reset(seed=seed)
        
        # Collect info about first few agents
        agents_info = []
        for i, agent in enumerate(env.rail_env.agents):
            agents_info.append({
                'handle': agent.handle,
                'pos': agent.initial_position,
                'dir': agent.initial_direction,
                'target': agent.target
            })
        
        # Get first active agent
        try:
            first_active = next(env.agent_iter())
        except StopIteration:
            first_active = None
        
        result = {
            'agents': agents_info,
            'first_active': first_active
        }
        results.append(result)
        
        print(f"  First active: {first_active}")
        for i, info in enumerate(agents_info):
            print(f"  Agent {i}: handle={info['handle']}, pos={info['pos']}, dir={info['dir']}")
    
    # Check if all runs are identical
    print(f"\n=== Analysis ===")
    all_identical = True
    for i in range(1, len(results)):
        if results[i] != results[0]:
            print(f"Run {i+1} differs from Run 1")
            all_identical = False
        
    if all_identical:
        print("✓ All runs are identical - non-determinism is due to environment interaction")
    else:
        print("✗ Runs differ - non-determinism is in environment creation itself")

if __name__ == "__main__":
    test_multiple_runs()