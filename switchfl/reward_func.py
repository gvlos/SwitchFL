from abc import ABC, abstractmethod
from typing import Any
from switchfl.observer import compute_delay
from flatland.envs.rail_env import RailEnvActions


class _RewardFunction(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self, observation: Any, action: Any) -> float:
        raise NotImplementedError


class StandardRewardFunction(_RewardFunction):
    def __init__(self, rail_env):
        super().__init__()
        self.rail_env = rail_env
        self.destination_bonus = 200

    def __call__(self, train, train_actions, train_to_last_node, port_blocked):
        """
        Updates the rewards based on the current state of the environment
        """
        # Compute rewards for trains that have arrived

        new_position = train.position
        new_direction = train.direction

        for action in train_actions:
            if action != RailEnvActions.STOP_MOVING:
                (
                    _,
                    (new_position,
                    new_direction),
                    _,
                    _,
                ) = self.rail_env.rail.check_action_on_agent(
                    action,
                    ((new_position),
                    new_direction)
                )

        curr_delay = compute_delay(self.rail_env, train, new_position, new_direction)
        if new_position == train.target:
            return self.destination_bonus, curr_delay

        if port_blocked:
            return 0, curr_delay
        else:
            _, last_delay = train_to_last_node[train.handle]
                    
            delay_diff = last_delay - curr_delay
        
            return delay_diff, curr_delay

