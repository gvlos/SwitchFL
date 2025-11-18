from abc import ABC, abstractmethod
from typing import Any
from switchfl.observer import compute_delay


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

    def __call__(self, train, train_actions, train_to_last_node):
        """
        Updates the rewards based on the current state of the environment
        """
        # Compute rewards for trains that have arrived

        new_position = train.position
        new_direction = train.direction

        for action in train_actions:
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

        if new_position == train.target:
            return self.destination_bonus, 0

        curr_delay = compute_delay(self.rail_env, train)
            
        _, last_delay = train_to_last_node[train.handle]
                
        reward = last_delay - curr_delay
            
        return reward, curr_delay

