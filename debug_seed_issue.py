#!/usr/bin/env python3
"""
Debug script to isolate the seed consistency issue.
"""

import numpy as np
from flatland.envs.rail_env import RailEnvActions
from switchfl.utils.build_env import build_standard_async_env

def debug_train_movements():
    """Debug train movements between identical environments"""
    seed = 202122
    
    # Create two identical environments
    env1 = build_standard_async_env(
        height=25,
        width=25,
        max_num_cities=2,
        num_trains=2,
        grid_mode=True,
        seed=seed
    )
    
    env2 = build_standard_async_env(
        height=25,
        width=25,
        max_num_cities=2,
        num_trains=2,
        grid_mode=True,
        seed=seed
    )
    
    # Reset both environments
    env1.reset(seed=seed)
    env2.reset(seed=seed)
    
    print("=== Initial State Comparison ===")
    print(f"Env1 agents: {env1.agents}")
    print(f"Env2 agents: {env2.agents}")
    
    print(f"Env1 active_switch_agents: {env1.active_switch_agents}")
    print(f"Env2 active_switch_agents: {env2.active_switch_agents}")
    
    # Check train positions
    def get_train_info(env, label):
        print(f"\n{label} train info:")
        for i, train in enumerate(env.rail_env.agents):
            print(f"  Train {i}: pos={train.position}, dir={train.direction}, "
                  f"initial_pos={train.initial_position}, target={train.target}, "
                  f"state={train.state}")
    
    get_train_info(env1, "Env1")
    get_train_info(env2, "Env2")
    
    # Check if rail grids are identical
    print(f"\nRail grids identical: {np.array_equal(env1.rail_env.rail.grid, env2.rail_env.rail.grid)}")
    
    # Check train action plans
    print(f"\nEnv1 train_action_plan: {env1.train_action_plan}")
    print(f"Env2 train_action_plan: {env2.train_action_plan}")
    
    # Now let's manually step through the _move_trains process
    print("\n=== Stepping through train movements ===")
    
    def debug_move_trains(env, label):
        print(f"\n{label} - Before _move_trains:")
        for i, train in enumerate(env.rail_env.agents):
            print(f"  Train {i}: pos={train.position}, dir={train.direction}, state={train.state}")
        
        # Call _move_trains
        env._move_trains()
        
        print(f"{label} - After _move_trains:")
        for i, train in enumerate(env.rail_env.agents):
            print(f"  Train {i}: pos={train.position}, dir={train.direction}, state={train.state}")
        
        # Call _check_active_switch
        env.active_switch_agents.clear()  # Clear to see what gets added
        env._check_active_switch()
        print(f"{label} - Active switches: {env.active_switch_agents}")
    
    debug_move_trains(env1, "Env1")
    debug_move_trains(env2, "Env2")

if __name__ == "__main__":
    debug_train_movements()