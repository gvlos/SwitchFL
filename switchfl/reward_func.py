from abc import ABC, abstractmethod
from typing import Any


class _RewardFunction(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self, observation: Any, action: Any) -> float:
        raise NotImplementedError


class StandardRewardFunction(_RewardFunction):
    def __init__(self):
        super().__init__()

    def __call__(self, observation, action):
        return super().__call__(observation, action)
