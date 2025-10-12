#!/usr/bin/env python3
"""Debug train direction consistency issues."""

import os
import sys

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from switchfl.utils.build_env import build_standard_async_env

def debug_train_directions():
    """Debug train direction consistency."""
    seed = 202122
    
    print("Creating two environments with identical parameters...")
    
    # Force complete randomness reset before each environment creation
    import numpy as np
    import random
    import os
    
    def reset_all_random_state(seed):
        """Completely reset all random state."""
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        # Also reset any cached random state
        np.random.get_state()  # Force state reset
    
    # Create first environment with complete state reset
    reset_all_random_state(seed)
    env1 = build_standard_async_env(
        height=25,
        width=25,
        max_num_cities=2,
        num_trains=2,
        grid_mode=True,
        seed=seed
    )
    
    # Create second environment with identical state reset
    reset_all_random_state(seed)
    env2 = build_standard_async_env(
        height=25,
        width=25,
        max_num_cities=2,
        num_trains=2,
        grid_mode=True,
        seed=seed
    )
    
    # Reset environments
    print("\nResetting environments...")
    env1.reset(seed=seed)
    env2.reset(seed=seed)
    
    print("\n=== Environment 1 ===")
    for i, agent in enumerate(env1.rail_env.agents):
        print(f"  Agent {i}: handle={agent.handle}, pos={agent.initial_position}, dir={agent.initial_direction}, target={agent.target}")
    
    print("\n=== Environment 2 ===")
    for i, agent in enumerate(env2.rail_env.agents):
        print(f"  Agent {i}: handle={agent.handle}, pos={agent.initial_position}, dir={agent.initial_direction}, target={agent.target}")
    
    # Check consistency
    print("\n=== Consistency Check ===")
    positions_match = all(
        a1.initial_position == a2.initial_position 
        for a1, a2 in zip(env1.rail_env.agents, env2.rail_env.agents)
    )
    
    directions_match = all(
        a1.initial_direction == a2.initial_direction 
        for a1, a2 in zip(env1.rail_env.agents, env2.rail_env.agents)
    )
    
    handles_match = all(
        a1.handle == a2.handle 
        for a1, a2 in zip(env1.rail_env.agents, env2.rail_env.agents)
    )
    
    targets_match = all(
        a1.target == a2.target 
        for a1, a2 in zip(env1.rail_env.agents, env2.rail_env.agents)
    )
    
    print(f"Positions match: {positions_match}")
    print(f"Directions match: {directions_match}")
    print(f"Handles match: {handles_match}")
    print(f"Targets match: {targets_match}")
    
    # Check which agent becomes active
    try:
        agent1 = next(env1.agent_iter())
        agent2 = next(env2.agent_iter())
        print(f"\nFirst active agent:")
        print(f"  Environment 1: {agent1}")
        print(f"  Environment 2: {agent2}")
        print(f"  Match: {agent1 == agent2}")
    except StopIteration:
        print("\nNo active agents in environments")

if __name__ == "__main__":
    debug_train_directions()