#!/usr/bin/env python3
"""Debug agent sorting and initial positions."""

import os
import sys

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from switchfl.utils.build_env import build_standard_async_env

def debug_agent_sorting():
    """Debug the agent sorting mechanism."""
    seed = 202122
    
    print("Creating two environments with identical parameters...")
    
    # Create first environment
    env1 = build_standard_async_env(
        height=25,
        width=25,
        max_num_cities=2,
        num_trains=2,
        grid_mode=True,
        seed=seed
    )
    
    # Create second environment  
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
    print(f"Agents: {env1.rail_env.agents}")
    print(f"Number of agents: {len(env1.rail_env.agents)}")
    for i, agent in enumerate(env1.rail_env.agents):
        print(f"  Agent {i}: handle={agent.handle}, initial_position={agent.initial_position}, target={agent.target}")
    
    try:
        active_switches_1 = []
        agent_iter_1 = env1.agent_iter()
        while True:
            try:
                active_switches_1.append(next(agent_iter_1))
            except StopIteration:
                break
        print(f"Active switches: {active_switches_1}")
    except Exception as e:
        print(f"Error getting active switches: {e}")
    
    print("\n=== Environment 2 ===")
    print(f"Agents: {env2.rail_env.agents}")
    print(f"Number of agents: {len(env2.rail_env.agents)}")
    for i, agent in enumerate(env2.rail_env.agents):
        print(f"  Agent {i}: handle={agent.handle}, initial_position={agent.initial_position}, target={agent.target}")
        
    try:
        active_switches_2 = []
        agent_iter_2 = env2.agent_iter()
        while True:
            try:
                active_switches_2.append(next(agent_iter_2))
            except StopIteration:
                break
        print(f"Active switches: {active_switches_2}")
    except Exception as e:
        print(f"Error getting active switches: {e}")
    
    # Check if agents are actually sorted the same way
    print("\n=== Sorting comparison ===")
    def sort_key(agent):
        return agent.initial_position or (0, 0)
    
    env1_sorted = sorted(env1.rail_env.agents, key=sort_key)
    env2_sorted = sorted(env2.rail_env.agents, key=sort_key)
    
    print("Environment 1 sorted agents:")
    for i, agent in enumerate(env1_sorted):
        print(f"  Agent {i}: handle={agent.handle}, initial_position={agent.initial_position}")
        
    print("Environment 2 sorted agents:")
    for i, agent in enumerate(env2_sorted):
        print(f"  Agent {i}: handle={agent.handle}, initial_position={agent.initial_position}")
        
    # Check if they match
    positions_match = all(
        a1.initial_position == a2.initial_position 
        for a1, a2 in zip(env1_sorted, env2_sorted)
    )
    
    handles_match = all(
        a1.handle == a2.handle 
        for a1, a2 in zip(env1_sorted, env2_sorted)
    )
    
    print(f"\nPositions match: {positions_match}")
    print(f"Handles match: {handles_match}")

if __name__ == "__main__":
    debug_agent_sorting()