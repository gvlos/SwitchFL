import numpy as np
from gymnasium.spaces import MultiDiscrete, Space



class DiscreteSwitchObsSpace(Space):
    def __init__(
        self, n_gaits: int, n_stations: int, n_delay_levels: int = 3, seed=None
    ):
        n_dims = n_gaits + (n_stations + 1) + (n_delay_levels + 1)
        shape = (n_dims,)
        dtype = int
        super().__init__(shape, dtype, seed)

        self.n_gaits = n_gaits  # how many rails are connected to a node -> Binary variable: is one occupied or not
        self.n_stations = (
            n_stations  # where a train would like to end up. No train -> -1
        )
        self.n_delay_levels = (
            n_delay_levels  # how strong the train is delayed. No train -> -1
        )

        self.station_space = MultiDiscrete(
            np.ones(self.n_gaits) * (self.n_stations + 1),
            seed=seed,
            start=np.ones(self.n_gaits) * (-1),
        )
        self.delay_space = MultiDiscrete(
            np.ones(self.n_gaits) * (self.n_delay_levels + 1),
            seed=seed,
            start=np.ones(self.n_gaits) * (-1),
        )
        # of a rail is blocked or if a train can be send onto this rail
        self.occupied_gait = MultiDiscrete(
            np.ones(n_gaits) * 2,
            seed=seed,
            start=np.zeros(n_gaits),
        )

    def sample(self):
        sample = np.concatenate(
            [
                self.occupied_gait.sample(),
                self.station_space.sample(),
                self.delay_space.sample(),
            ]
        )
        return sample

    @property
    def is_np_flattenable(self) -> bool:
        """Checks whether this space can be flattened to a :class:`gymnasium.Box`."""
        return True

