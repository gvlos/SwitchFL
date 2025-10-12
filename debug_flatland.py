#!/usr/bin/env python3
"""
Deep debug script to understand Flatland's train assignment behavior.
"""

import numpy as np
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator

def debug_flatland_determinism():
    """Debug the Flatland RailEnv determinism at the lowest level"""
    seed = 999999
    
    print("=== Testing Flatland RailEnv Determinism ===")
    
    # Create two identical rail environments
    def create_rail_env(seed):
        return RailEnv(
            width=20,
            height=20,
            rail_generator=sparse_rail_generator(
                max_num_cities=2,
                grid_mode=True,
                max_rails_between_cities=1,
                max_rail_pairs_in_city=1,
                seed=seed,
            ),
            line_generator=sparse_line_generator(seed=seed),
            number_of_agents=2,
        )
    
    # Test multiple resets of the same environment
    print("\n--- Testing single environment multiple resets ---")
    env = create_rail_env(seed)
    
    for reset_num in range(3):
        # Set all seeds before reset
        np.random.seed(seed)
        import random
        random.seed(seed)
        
        obs, info = env.reset()
        
        print(f"\nReset {reset_num + 1}:")
        for i, agent in enumerate(env.agents):
            print(f"  Agent {i}: pos={agent.position}, initial_pos={agent.initial_position}, "
                  f"target={agent.target}, state={agent.state}")
    
    # Test two separate environments
    print("\n--- Testing two separate identical environments ---")
    
    # Set seeds before creating environments
    np.random.seed(seed)
    import random
    random.seed(seed)
    env1 = create_rail_env(seed)
    
    np.random.seed(seed)
    random.seed(seed)
    env2 = create_rail_env(seed)
    
    # Reset both with same seed
    np.random.seed(seed)
    random.seed(seed)
    obs1, info1 = env1.reset()
    
    np.random.seed(seed)
    random.seed(seed)
    obs2, info2 = env2.reset()
    
    print(f"\nEnvironment 1:")
    for i, agent in enumerate(env1.agents):
        print(f"  Agent {i}: pos={agent.position}, initial_pos={agent.initial_position}, "
              f"target={agent.target}, state={agent.state}")
    
    print(f"\nEnvironment 2:")
    for i, agent in enumerate(env2.agents):
        print(f"  Agent {i}: pos={agent.position}, initial_pos={agent.initial_position}, "
              f"target={agent.target}, state={agent.state}")
    
    # Check if rail grids are identical
    print(f"\nRail grids identical: {np.array_equal(env1.rail.grid, env2.rail.grid)}")
    print(f"Number of agents identical: {len(env1.agents) == len(env2.agents)}")
    
    # Check specific differences
    if len(env1.agents) == len(env2.agents):
        positions_match = all(
            a1.initial_position == a2.initial_position and a1.target == a2.target
            for a1, a2 in zip(env1.agents, env2.agents)
        )
        print(f"All train positions and targets match: {positions_match}")
        
        if not positions_match:
            print("Detailed comparison:")
            for i, (a1, a2) in enumerate(zip(env1.agents, env2.agents)):
                if a1.initial_position != a2.initial_position or a1.target != a2.target:
                    print(f"  Agent {i}: Env1 {a1.initial_position}→{a1.target} vs "
                          f"Env2 {a2.initial_position}→{a2.target}")

if __name__ == "__main__":
    debug_flatland_determinism()