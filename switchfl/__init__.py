from typing import Tuple


NodeId = Tuple[int, int]
PortId = Tuple[float, float]
SwitchId = str  # switch_(x, y)
SwitchPosition = Tuple[int, int]
TrainAgentHandle = int


from switchfl.switch_env import ASyncSwitchEnv
