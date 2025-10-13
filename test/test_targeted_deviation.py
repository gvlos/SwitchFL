#!/usr/bin/env python3
"""
Targeted test to provoke the specific train deviation error.

This test directly manipulates the conditions that trigger the error in _check_action_execution.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


from switchfl.utils.build_env import build_standard_async_env


def test_provoke_deviation_error():
    """Test that directly provokes the train deviation error."""
    print("Creating environment to provoke train deviation error...")
    
    # Create environment
    env = build_standard_async_env(
        height=20,
        width=20,
        max_num_cities=2,
        num_trains=2,
        max_rails_between_cities=1,
        max_rail_pairs_in_city=1,
        grid_mode=True,
        seed=42
    )
    env.reset(seed=42)
    
    print(f"Environment created with {len(env.rail_env.agents)} trains")
    
    # Capture all logger calls
    captured_logs = []
    original_error = env.logger.error
    
    def capture_error(msg, *args, **kwargs):
        captured_logs.append(msg)
        print(f"🔴 CAPTURED ERROR: {msg}")
        return original_error(msg, *args, **kwargs)
    
    env.logger.error = capture_error
    
    # Find trains and manipulate their state to trigger the error
    success = False
    
    try:
        # Move trains to get them to switches
        for step in range(10):
            env._move_trains_to_switch()
            if len(env.active_switch_agents) > 0:
                break
        
        # Now examine each train and try to create deviation scenario
        for train in env.rail_env.agents:
            if train.position is None:
                continue
                
            print(f"\n--- Testing train {train.handle} at position {train.position} ---")
            
            try:
                # Get the train's ports
                next_port = env.rail_network.get_trains_next_port(train)
                prev_port = env.rail_network.get_trains_prev_port(train)
                
                print(f"Train {train.handle}: next_port={next_port}, prev_port={prev_port}")
                
                if next_port is not None and prev_port is not None:
                    # Get the switch nodes
                    source_switch = env.rail_network.get_switch_on_port(prev_port)
                    target_switch = env.rail_network.get_switch_on_port(next_port)
                    
                    if source_switch and target_switch:
                        print(f"Source switch: {source_switch.id}, Target switch: {target_switch.id}")
                        
                        # Get rail pieces between the switches
                        rail_pieces = env.rail_network.get_rail_pieces_between_ports(
                            source_switch.id, target_switch.id
                        )
                        
                        print(f"Rail pieces: {rail_pieces}")
                        valid_positions = [source_switch.id, target_switch.id]
                        if rail_pieces:
                            valid_positions.extend(rail_pieces)
                        
                        print(f"Valid positions for train: {valid_positions}")
                        print(f"Current train position: {train.position}")
                        
                        # Store original position
                        original_position = train.position
                        
                        # Now move the train to an INVALID position
                        # Try several positions that are definitely not on the path
                        invalid_positions = [(0, 0), (19, 19), (10, 10), (1, 1)]
                        
                        for invalid_pos in invalid_positions:
                            if invalid_pos not in valid_positions:
                                print(f"Testing invalid position: {invalid_pos}")
                                
                                # Move train to invalid position
                                train.position = invalid_pos
                                
                                # Call the method that checks for deviations
                                env._check_action_execution()
                                
                                # Check if we captured the error
                                if captured_logs:
                                    print(f"✅ SUCCESS! Captured error: {captured_logs[-1]}")
                                    success = True
                                    break
                        
                        # Restore original position
                        train.position = original_position
                        
                        if success:
                            break
                            
            except Exception as e:
                print(f"Exception testing train {train.handle}: {e}")
                continue
        
        if not success:
            print("❌ Could not provoke deviation error with current trains")
            
            # Try a more direct approach - manually create the exact error condition
            print("\nTrying manual approach...")
            
            # Find any train
            train = None
            for t in env.rail_env.agents:
                if t.position is not None:
                    train = t
                    break
            
            if train:
                print(f"Using train {train.handle} for manual test")
                
                # Create fake ports to simulate the error condition
                class FakePort:
                    def __init__(self, coord):
                        self.coord = coord
                
                # Mock the get_trains_next_port and get_trains_prev_port methods
                def mock_get_next_port(t):
                    if t.handle == train.handle:
                        return FakePort((10, 10))
                    return None
                
                def mock_get_prev_port(t):
                    if t.handle == train.handle:
                        return FakePort((5, 5))
                    return None
                
                # Mock the get_switch_on_port method
                class FakeSwitch:
                    def __init__(self, id_val):
                        self.id = id_val
                
                def mock_get_switch_on_port(port):
                    if port.coord == (10, 10):
                        return FakeSwitch((10, 10))
                    elif port.coord == (5, 5):
                        return FakeSwitch((5, 5))
                    return None
                
                # Mock the get_rail_pieces_between_ports method
                def mock_get_rail_pieces(source_id, target_id):
                    return [(6, 6), (7, 7), (8, 8), (9, 9)]  # Some path between switches
                
                # Apply mocks
                env.rail_network.get_trains_next_port = mock_get_next_port
                env.rail_network.get_trains_prev_port = mock_get_prev_port
                env.rail_network.get_switch_on_port = mock_get_switch_on_port
                env.rail_network.get_rail_pieces_between_ports = mock_get_rail_pieces
                
                # Store original position
                original_position = train.position
                
                # Move train to position NOT in the valid path
                train.position = (0, 0)  # This should NOT be in [(6,6), (7,7), (8,8), (9,9), (10,10), (5,5)]
                
                print("Set train position to (0, 0) - should trigger deviation error")
                
                # Call the check method
                env._check_action_execution()
                
                # Restore position
                train.position = original_position
                
                if captured_logs:
                    print(f"✅ SUCCESS! Manual approach captured error: {captured_logs[-1]}")
                    success = True
        
    except Exception as e:
        print(f"❌ Exception during test: {e}")
        import traceback
        traceback.print_exc()
    
    # Report results
    print(f"\n{'='*60}")
    print("RESULTS:")
    if captured_logs:
        print(f"✅ Successfully captured {len(captured_logs)} error log(s):")
        for i, log in enumerate(captured_logs, 1):
            print(f"  {i}. {log}")
            if "deviated from planned path" in log:
                print("     ✅ This is the target error message!")
    else:
        print("❌ No error logs captured")
    
    return success


def test_examine_error_conditions():
    """Examine the actual conditions that would trigger the error."""
    print("\n" + "="*60)
    print("EXAMINING ERROR CONDITIONS")
    print("="*60)
    
    # Create environment
    env = build_standard_async_env(
        height=20,
        width=20,
        max_num_cities=2,
        num_trains=2,
        max_rails_between_cities=1,
        max_rail_pairs_in_city=1,
        grid_mode=True,
        seed=123
    )
    env.reset(seed=123)
    
    # Move trains to activate them
    for step in range(10):
        env._move_trains_to_switch()
        if len(env.active_switch_agents) > 0:
            break
    
    print(f"Active switch agents: {env.active_switch_agents}")
    
    # Examine each train's situation
    for train in env.rail_env.agents:
        if train.position is None:
            continue
            
        print(f"\n--- TRAIN {train.handle} ANALYSIS ---")
        print(f"Position: {train.position}")
        
        try:
            next_port = env.rail_network.get_trains_next_port(train)
            prev_port = env.rail_network.get_trains_prev_port(train)
            
            print(f"Next port: {next_port}")
            print(f"Prev port: {prev_port}")
            
            if next_port is not None and prev_port is not None:
                source_switch = env.rail_network.get_switch_on_port(prev_port)
                target_switch = env.rail_network.get_switch_on_port(next_port)
                
                print(f"Source switch: {source_switch.id if source_switch else None}")
                print(f"Target switch: {target_switch.id if target_switch else None}")
                
                if source_switch and target_switch:
                    rail_pieces = env.rail_network.get_rail_pieces_between_ports(
                        source_switch.id, target_switch.id
                    )
                    
                    print(f"Rail pieces between switches: {rail_pieces}")
                    
                    # The error condition is: train.position not in [*rail_pieces, source_node, target_node]
                    valid_positions = [source_switch.id, target_switch.id]
                    if rail_pieces:
                        valid_positions.extend(rail_pieces)
                    
                    print(f"Valid positions: {valid_positions}")
                    print(f"Train position {train.position} is valid: {train.position in valid_positions}")
                    
                    if train.position not in valid_positions:
                        print("🔴 THIS TRAIN WOULD TRIGGER THE ERROR!")
                    else:
                        print("✅ This train is on a valid path")
            else:
                print("❌ Cannot determine path - missing port information")
                
        except Exception as e:
            print(f"❌ Exception analyzing train {train.handle}: {e}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    print("Running targeted train deviation test...")
    
    try:
        success = test_provoke_deviation_error()
        test_examine_error_conditions()
        
        if success:
            print("\n🎉 TEST SUITE SUCCESSFUL - Error message provoked!")
        else:
            print("\n❌ TEST SUITE FAILED - Could not provoke error message")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()