import numpy as np
from scipy.signal import convolve, lfilter
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Set, List, Any
from typing_extensions import Self
import networkx as nx

from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.malfunction_generators import Malfunction, _malfunction_prob, MalfunctionParameters, MalfunctionProcessData, _make_Malfunction_object
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.step_utils.states import TrainState
from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.utils.rendertools import RenderTool

from numpy.random.mtrand import RandomState

import time

class Node:
    """
    A node in the graph representation of the environment.

    Parameters
    ----------
    row : int
        The row of the node
    col : int
        The column of the node
    dir : int
        The direction of the train at the node
    """
    def __init__(self, row: int, col: int, dir: int):
        self.row = row
        self.col = col
        self.dir = dir

    def __eq__(self, other: Self) -> bool:
        return (self.row, self.col, self.dir) == (other.row, other.col, other.dir)
    
    def __lt__(self, other: Self) -> bool:
        return (self.row, self.col, self.dir) < (other.row, other.col, other.dir)
    
    def __hash__(self) -> int:
        return hash((self.row, self.col, self.dir))
    
    def __str__(self) -> str:
        return '(' + str(self.row) + ', ' + str(self.col) + ', ' + str(self.dir) + ')'
    
class EdgeData:
    """
    EdgeData represents the data associated with an edge in the graph representation of the environment.

    Parameters
    ----------
    mask : np.ndarray[bool]
        A boolean mask of the size of the grid where True values represent the position occupied by the edge
    action : RailEnvActions
        The action to perform to move from the source node to the destination node
    direction_mask : np.ndarray[int]
        A matrix of the size of the grid where each cell contains the direction of the path or -1 if not part of the path

    """
    def __init__(self, mask: np.ndarray[bool], action: RailEnvActions, direction_mask: np.ndarray[int] = None):
        self.mask = mask
        self.action = action
        self.direction_mask = direction_mask

    def __eq__(self, other: Self) -> bool:
        return np.all(self.mask == other.mask) and self.action == other.action
    
class MultiGraphRepresentation:
    """
    A multi-graph is a graph that can have more than one edge with the same end nodes
    """
    def __init__(self):
        self.__nodes: Set[Node] = set()
        self.__edge_data: Dict[Tuple[Node, Node], List[EdgeData]] = {} # (src, dst) -> [edge_data_1, edge_data_2, ...]
        self.__fw_star: Dict[Node, Set[Node]] = {} # node -> next nodes
        self.__bw_star: Dict[Node, Set[Node]] = {} # node -> previous nodes

    def nodes(self) -> Set[Node]:
        return self.__nodes

    def edges(self) -> List[Tuple[Node, Node, EdgeData]]:
        """
        Returns a list of tuples (src, dst, edge_data) representing the edges of the graph
        
        Returns
        -------
        List[Tuple[Node, Node, EdgeData]]
            A list of tuples (src, dst, edge_data) representing the edges of the graph
        """
        _edges = []
        for src, dst in self.__edge_data:
            for edge_data in self.__edge_data[src, dst]:
                _edges.append((src, dst, edge_data))
        return _edges

    def has_node(self, node: Node) -> bool:
        return node in self.__nodes

    def fw_star(self, node: Node) -> List[Tuple[Node, EdgeData]]:
        """
        Returns the forward star of a node, i.e. the list of nodes that are reachable from the given node
        
        Parameters
        ----------
        node : Node
            The node to consider
            
        Returns
        -------
        List[Tuple[Node, EdgeData]]
            The forward star of the node
        """
        if not self.has_node(node):
            return []
        result = []
        for neighbor in self.__fw_star[node]:
            for edge_data in self.__edge_data[node, neighbor]:
                result.append((neighbor, edge_data))
        return result
    
    def bw_star(self, node: Node) -> List[Tuple[Node, EdgeData]]:
        """
        Returns the backward star of a node, i.e. the list of nodes that can reach the given node
        
        Parameters
        ----------
        node : Node
            The node to consider
            
        Returns
        -------
        List[Tuple[Node, EdgeData]]
            The backward star of the node
        """
        if not self.has_node(node):
            return []
        result = []
        for neighbor in self.__bw_star[node]:
            for edge_data in self.__edge_data[neighbor, node]:
                result.append((neighbor, edge_data))
        return result

    def add_node(self, node: Node):
        """
        Adds a node to the graph if it does not exist.

        Parameters
        ----------
        node : Node
            The node to add
        """
        if not self.has_node(node):
            self.__nodes.add(node)
            self.__fw_star[node] = set()
            self.__bw_star[node] = set()

    def delete_node(self, node: Node):
        """
        Deletes a node from the graph with all the incoming and outgoing edges.

        Parameters
        ----------
        node : Node
            The node to delete
        """
        if node in self.__nodes:
            self.__nodes.remove(node)
            for neighbor in self.__fw_star[node]:
                del self.__edge_data[node, neighbor]
            for neighbor in self.__bw_star[node]:
                del self.__edge_data[neighbor, node]
            del self.__fw_star[node]
            del self.__bw_star[node]

    def add_edge(self, src: Node, dst: Node, edge_data: EdgeData):
        """
        Adds an edge to the graph, if nodes do not exist, they are created.

        Parameters
        ----------
        src : Node
            The source node
        dst : Node
            The destination node
        edge_data : EdgeData
            The data associated with the edge
        """
        self.add_node(src)
        self.add_node(dst)
        self.__fw_star[src].add(dst)
        self.__bw_star[dst].add(src)
        if (src, dst) not in self.__edge_data:
            self.__edge_data[src, dst] = []
        self.__edge_data[src, dst].append(edge_data)

    def delete_edge(self, src: Node, dst: Node, edge_data: EdgeData | Any = None):
        """
        Deletes an edge with the given EdgeData from the graph if it exists, if edge_data is not given,
        it removes all edges with the given src and dst nodes from the graph

        Parameters
        ----------
        src : Node
            The source node
        dst : Node
            The destination node
        edge_data : EdgeData | Any
            The data associated with the edge
        """
        if (src, dst) in self.__edge_data:
            if edge_data is not None:
                self.__edge_data[src, dst].remove(edge_data)
            else:
                self.__edge_data[src, dst] = []
            if len(self.__edge_data[src, dst]) == 0:
                self.__fw_star[src].remove(dst)
                self.__bw_star[dst].remove(src)
                del self.__edge_data[src, dst]

    def clear_isolated(self):
        """
        Removes isolated nodes
        """
        to_remove = set()
        for node in self.nodes():
            if len(self.__fw_star[node]) == 0 and len(self.__bw_star[node]) == 0:
                to_remove.add(node)
        for node in to_remove:
            self.delete_node(node)

class EnvGraph(MultiGraphRepresentation):
    """
    A graph representation of the environment.
    
    Parameters
    ----------
    env : RailEnv
        The environment
    """
    def __init__(self, env: RailEnv):
        super().__init__()
        # Create edges
        rail_grid = env.rail.grid
        rows, cols = rail_grid.shape
        self.rows = rows
        self.cols = cols
        get_txs = env.rail.get_transitions
        for dir in range(4):
            for row in range(rows):
                for col in range(cols):
                    src_node = Node(row, col, dir)
                    tx_mask = get_txs(row, col, dir)  # row of transition matrix
                    available_tx = [tx for tx in range(4) if tx_mask[tx] == 1]  # [North, East, South, West]
                    for tx in available_tx:
                        # Find next cell coordinates
                        next_row = row
                        next_col = col
                        if tx == 0: # North
                            next_row -= 1
                        elif tx == 1: # East
                            next_col += 1
                        elif tx == 2: # South
                            next_row += 1
                        else: # West
                            next_col -= 1
                        if next_row >= 0 and next_row < rows and next_col >= 0 and next_col < cols:  # Check if next cell is within the grid
                            dst_node = Node(next_row, next_col, tx)
                            edge_mask = np.zeros(shape=(rows, cols), dtype=bool)
                            edge_mask[src_node.row, src_node.col] = True
                            edge_mask[dst_node.row, dst_node.col] = True
                            edge_action = self.__calculate_action(dir, tx)
                            edge_data = EdgeData(edge_mask, edge_action)
                            self.add_edge(src_node, dst_node, edge_data)
        # Reduce the graph by removing non-decision nodes
        while True:
            # Find a node with exactly one outgoing edge
            found = False
            for node in self.nodes():
                if len(self.fw_star(node)) == 1:
                    found = True
                    break
            if not found:
                break
            # Find the next node and mask
            next, node_next_data = self.fw_star(node)[0]
            # Delete incoming edges and create new edge to next node
            for prev, prev_node_data in self.bw_star(node):
                new_mask = prev_node_data.mask | node_next_data.mask
                self.add_edge(prev, next, EdgeData(new_mask, prev_node_data.action))
                self.delete_edge(prev, node, prev_node_data)
            self.delete_edge(node, next, node_next_data)
        # Clear isolated nodes
        self.clear_isolated()

        # Initialize direction masks
        for src, dst, edge_data in self.edges():
            direction_mask = np.zeros_like(edge_data.mask, dtype=int) - 1 # Init to -1, each cell contains the direction of the path or -1 if not part of the path
            curr_node = Node(src.row, src.col, src.dir)
            while curr_node != dst:
                direction_mask[curr_node.row, curr_node.col] = curr_node.dir
                # Make one step
                transitions = env.rail.get_transitions(curr_node.row, curr_node.col, curr_node.dir)
                possible_next_nodes = []
                for next_dir in range(4):
                    if transitions[next_dir] == 0:
                        continue
                    next_row = curr_node.row
                    next_col = curr_node.col
                    if next_dir == 0: # North
                        next_row -= 1
                    elif next_dir == 1: # East
                        next_col += 1
                    elif next_dir == 2: # South
                        next_row += 1
                    else: # West
                        next_col -= 1
                    next_node = Node(next_row, next_col, next_dir)
                    possible_next_nodes.append(next_node)
                assert len(possible_next_nodes) > 0, "No possible next nodes, this should not happen"
                for next_node in possible_next_nodes:
                    if edge_data.mask[next_node.row, next_node.col]:
                        curr_node = next_node
                        break
            direction_mask[curr_node.row, curr_node.col] = curr_node.dir
            edge_data.direction_mask = direction_mask

    def __calculate_action(self, dir: int, tx: int) -> RailEnvActions:
        """
        Calculate the action to perform to change the orientation of the train from dir to tx.

        Parameters
        ----------
        dir : int
            The current direction of the train
        tx : int
            The target direction of the train

        Returns
        -------
        RailEnvActions
            The action to perform to change the orientation of the train from dir to tx
        """
        if np.abs(tx - dir) == 2 or tx == dir:
            return RailEnvActions.MOVE_FORWARD
        if tx - dir == 1 or tx - dir == -3:
            return RailEnvActions.MOVE_RIGHT
        if tx - dir == -1 or tx - dir == 3:
            return RailEnvActions.MOVE_LEFT

    def render(self, figsize: Tuple[int, int] = (20, 20), save : str | Any = None):
        """
        Renders the graph representation of the environment.
        
        Parameters
        ----------
        figsize : Tuple[int, int]
            The size of the figure
        save : str | Any
            The path to save the figure
        """
        G = nx.DiGraph()
        pos = {}
        labels = {}
        dir_label = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}

        for node in self.nodes():
            r, c, dir = (node.row, node.col, node.dir)
            x = c - dir * 0.3
            y = self.rows - r - dir * 0.3
            pos[node] = (x, y)
            labels[node] = f'({r}, {c})\n{dir_label[dir]}'
            G.add_node(node)

        for (src, dst, mask) in self.edges():
            G.add_edge(src, dst)

        plt.figure(figsize=figsize)
        nx.draw(G, pos, with_labels=True, labels=labels, node_size=100, node_color='skyblue', font_size=10, font_color='black', arrowsize=20, arrowstyle='->')
        
        if save is not None:
            plt.savefig(save)

    def trainsThatAreOnNodes(self, trains: List[EnvAgent]) -> Dict[int, Node]:
        """
        Returns a dictionary train_index -> Node where each key is the index of a train that is
        stepping on a node and the value is the corresponding node
        
        Parameters
        ----------
        trains : np.ndarray[int]
            A list of flatland agents representing the trains.
        """
        trains_on_node = {}
        for train_idx, train in enumerate(trains):
            if train.position is not None:
                node = Node(train.position[0], train.position[1], train.direction)
                if node in self.nodes():
                    trains_on_node[train_idx] = node
        return trains_on_node
    
    def outgoingEdgesMask(self, node: Node) -> np.ndarray[bool]:
        """
        Returns a boolean mask of the size of the grid where True values represent the position
        occupied by the edges that are outgoing from the given node, or-ed together.
        
        Parameters
        ----------
        node : Node
            The node to consider
        """
        mask = np.zeros(shape=(self.rows, self.cols), dtype=bool)
        for _, edge_data in self.fw_star(node):
            mask |= edge_data.mask
        return mask

def one_hot(value: int, n_values: int):
    """
    Returns a one-hot encoding of the given value.
    
    Parameters
    ----------
    value : int
        The value to encode
    n_values : int
        The number of possible values

    Returns
    -------
    List[int]
        A one-hot encoding of the given value    
    """
    return [1 if value == i else 0 for i in range(n_values)]

def zeros(n: int):
    return [0] * n

def manhattan(src_row: int, src_col: int, dst_row: int, dst_col: int):
    """
    Computes manhattan distance between two points.

    Parameters
    ----------
    src_row : int
        The row of the source point
    src_col : int
        The column of the source point
    dst_row : int
        The row of the destination point
    dst_col : int
        The column of the destination point

    Returns
    -------
    int
        The manhattan distance between the two points
    """
    return abs(src_row - dst_row) + abs(src_col - dst_col)

class ReproducibleParamMalfunctionGen:
    """
    Reproducible Malfunction Generator

    Parameters
    ----------
    malfunction_rate : float
        The rate of malfunction
    min_duration : int
        The minimum duration of a malfunction
    max_duration : int
        The maximum duration of a malfunction
    seed : int
        The seed for reproducibility
    """
    def __init__(self, malfunction_rate: float, min_duration: int, max_duration: int, seed: int):
        self.MFP = MalfunctionParameters(malfunction_rate, min_duration, max_duration)
        self.seed = seed
        self.reset()

    def reset(self):
        self.random_state = np.random.RandomState(self.seed)
    
    ## The following functions have the same interface as flatland library for
    ## compatibility, but behave differently in order to achieve reproducibility.

    def generate_rand_numbers(self, np_random: RandomState):
        return self.random_state.rand()
    
    def generate(self, np_random: RandomState) -> Malfunction:
        if self.generate_rand_numbers(np_random) < _malfunction_prob(self.MFP.malfunction_rate):
            num_broken_steps = np_random.randint(self.MFP.min_duration,
                                                 self.MFP.max_duration + 1) + 1
        else:
            num_broken_steps = 0
        return _make_Malfunction_object(num_broken_steps)

    def get_process_data(self):
        return MalfunctionProcessData(*self.MFP)

class Environment:
    """
    A wrapper around the flatland environment that provides a graph representation of the environment
    
    Parameters
    ----------
    env_width : int
        The width of the environment
    env_height : int
        The height of the environment
    env_n_cities : int
        The number of cities in the environment
    env_n_trains : int
        The number of trains in the environment
    seed : int
        The seed for reproducibility
    destination_bonus : float
        The bonus reward for reaching the destination
    deadlock_penalty : float
        The penalty for deadlock
    delay_threshold : float
        The threshold for discretizing the delay
    malfunction_rate : float
        The rate of malfunction
    malfunction_min_duration : int
        The minimum duration of a malfunction
    malfunction_max_duration : int
        The maximum duration of a malfunction
    malfunction_seed : int
        The seed for the malfunction generator
    init_renderer : bool
        Whether to initialize the renderer
    """
    def __init__(
        self, 
        env_width, 
        env_height, 
        env_n_cities, 
        env_n_trains, 
        seed, 
        destination_bonus: float,
        deadlock_penalty: float,
        delay_threshold: float,
        malfunction_rate: float,
        malfunction_min_duration: int,
        malfunction_max_duration: int,
        malfunction_seed: int = None, # if none, it will be the same as seed
        init_renderer=False
    ):    
        self.seed = seed
        self.malfunction_seed = seed if malfunction_seed is None else malfunction_seed
        self.destination_bonus = destination_bonus
        self.deadlock_penalty = deadlock_penalty
        self.delay_threshold = delay_threshold
        self.rail_env = RailEnv(
            width = env_width,
            height = env_height,
            rail_generator = sparse_rail_generator(
                max_num_cities = env_n_cities,
                grid_mode = True,
                max_rails_between_cities = 2,
                max_rail_pairs_in_city = 3,
                seed = self.seed
            ),
            line_generator = sparse_line_generator(),  # fixed seed=1 by default,
            malfunction_generator=ReproducibleParamMalfunctionGen(malfunction_rate=malfunction_rate, min_duration=malfunction_min_duration, max_duration=malfunction_max_duration, seed=self.malfunction_seed),
            number_of_agents = env_n_trains,
            obs_builder_object = DummyObservationBuilder(),
            random_seed = self.seed
        )
        self.rail_env.reset(random_seed=seed)
        self.graph = EnvGraph(self.rail_env)
        self.nodes = list(self.graph.nodes())
        self.node_to_id = {node: i for i, node in enumerate(self.nodes)}
        if init_renderer:
            self.renderer = RenderTool(self.rail_env, gl="PILSVG")

    def reset(self, step_until_action_required=False, reset_malfunctions=False):
        """
        Resets the environment
        
        Parameters
        ----------
        step_until_action_required : bool
            Whether to step until an action is required
        reset_malfunctions : bool
            Whether to reset malfunctions
        """
        self.rail_env.reset(regenerate_rail=False, regenerate_schedule=False, random_seed=self.seed)
        self.stations = {
            station_pos: station_id 
            for station_id, station_pos in enumerate(set([train.target for train in self.rail_env.agents]))
        }
        self.rewards = np.zeros(len(self.nodes))
        self.last_obs = {} # node_id -> last_obs
        self.last_action = {} # node_id -> last_action
        self.train_to_last_node = {} # train_id -> last_node_id, delay_when_arrived
        self.old_dones = self.rail_env.dones.copy()
        self.trains_arrived = {
            train_id: False for train_id in range(self.rail_env.get_num_agents())
        }
        if reset_malfunctions:
            self.rail_env.malfunction_generator.reset()
        if step_until_action_required:
            while len(self.graph.trainsThatAreOnNodes(self.rail_env.agents)) == 0:
                self.rail_env.step({tid: RailEnvActions.MOVE_FORWARD for tid in range(self.rail_env.get_num_agents())})
                self._update_rewards()

    def render(
        self,
        figsize: Tuple[int, int] = (10, 10),
        print_train_numbers: bool = True,
        print_node_ids: bool = False, 
        print_single_rail: np.ndarray[bool] = None
    ):
        """
        Renders the environment
        
        Parameters
        ----------
        figsize : Tuple[int, int]
            The size of the figure
        print_train_numbers : bool
            Whether to print the train numbers
        print_node_ids : bool
            Whether to print the node ids
        print_single_rail : np.ndarray[bool]
            A boolean mask of the size of the grid where True values represent the position
            of the single rail
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        plt.axis('off')
        self.renderer.render_env(show_inactive_agents=False)
        ax.imshow(self.renderer.get_image())

        width, height, _ = self.renderer.get_image().shape
        cell_width = width // self.rail_env.rail.grid.shape[1]
        cell_height = height // self.rail_env.rail.grid.shape[0]
        color = 'blue'
        for train in self.rail_env.agents:
            if train.state == TrainState.WAITING or train.state == TrainState.MALFUNCTION_OFF_MAP or train.state == TrainState.DONE:
                continue
            from_row, from_col = train.position if train.position is not None else train.initial_position
            to_row, to_col = train.target
            ax.arrow((from_col + 0.5) * cell_width, (from_row + 0.5) * cell_height, (to_col - from_col) * cell_width, (to_row - from_row) * cell_height, color=color, head_width=0.3, head_length=0.3, alpha=0.7)
        if print_train_numbers:
            for train in self.rail_env.agents:
                if train.state == TrainState.WAITING or train.state == TrainState.MALFUNCTION_OFF_MAP or train.state == TrainState.DONE:
                    continue
                row, col = train.position if train.position is not None else train.initial_position
                ax.text((col + 0.5) * cell_width, (row + 0.5) * cell_height, str(train.handle), color='purple', fontsize=12, ha='center', va='center')
        if print_node_ids:
            for node, id in self.node_to_id.items():
                row, col, _ = (node.row, node.col, node.dir)
                ax.text((col + 0.5) * cell_width, (row + 0.5) * cell_height, str(id), color='red', fontsize=12, ha='center', va='center')
        if print_single_rail is not None:
            # Draw a box in the cells that are True
            for row in range(self.rail_env.rail.grid.shape[0]):
                for col in range(self.env.rail.grid.shape[1]):
                    if print_single_rail[row, col]:
                        ax.add_patch(plt.Rectangle((col * cell_width, row * cell_height), cell_width, cell_height, fill='orange', edgecolor='red', lw=2, alpha=0.5))
        plt.show()

    def observe(self):
        """
        Returns [(train_id, node_id, obs, old_reward)]
        """
        observations = []
        for train_id, node in self.graph.trainsThatAreOnNodes(self.rail_env.agents).items():
            node_id = self.node_to_id[node]
            obs = self._observe_node(node, train_id)
            self.last_obs[node_id] = obs
            old_reward = self._get_reward_for_node(node)
            observations.append((train_id, node_id, obs, old_reward))
        return observations
    
    def _get_reward_for_node(self, node: Node) -> float:
        """
        Returns the reward for the given node
        
        Parameters
        ----------
        node : Node
            The node to consider
            
        Returns
        -------
        float
            The reward for the given node
        """
        node_id = self.node_to_id[node]
        r = self.rewards[node_id].copy()
        self.rewards[node_id] = 0
        return r
    
    def _observe_node(self, node: Node, train_id: int) -> Tuple[int, int, int, bool, bool]:
        """
        Preconditions: train_id exists and is on a node
        Returns (station_id, node_id, delay, semaphore edge 0, sempahore edge 1)
        """
        train = self.rail_env.agents[train_id]

        # Station identifier
        station_id = self.stations[train.target]

        # Train delay
        delay = self._discretize_delay(train_id, self._compute_delay(train_id))

        # Semaphores
        sem_0 = self._compute_semaphore(node, 0)
        sem_1 = self._compute_semaphore(node, 1)

        return station_id, self.node_to_id[node], delay, sem_0, sem_1
    
    def _compute_semaphore(self, node: Node, edge_idx: int) -> bool:
        """
        Returns True if the semaphore is activated, False otherwise
        
        Parameters
        ----------
        node : Node
            The node to consider
        edge_idx : int
            The index of the edge to consider

        Returns
        -------
        bool
            True if the semaphore is activated, False otherwise
        """
        _, edge_data = self.graph.fw_star(node)[edge_idx]
        for train in self.rail_env.agents:
            row, col = train.position if train.position is not None else train.initial_position
            if row == node.row and col == node.col:
                continue
            # if train is currently on this edge and its direction is opposite to the one that is available on this edge starting from the given node
            # or if train is malfunctioning
            # semaphore is activated (True)
            train_on_edge = edge_data.direction_mask[row, col] != -1
            opposite_dir = train.direction != edge_data.direction_mask[row, col]
            malfunction = train.state == TrainState.MALFUNCTION
            if train_on_edge and (opposite_dir or malfunction):
                return True
        return False

    def _compute_delay(self, train_id: int):
        """
        Returns the delay of the given train
        
        Parameters
        ----------
        train_id : int
            The id of the train

        Returns
        -------
        int
            The delay of the given train
        """
        train = self.rail_env.agents[train_id]
        row, col = train.position if train.position is not None else train.initial_position
        min_dist_to_target = self.rail_env.distance_map.get()[train_id, row, col, train.direction]
        return self.rail_env._elapsed_steps - train.latest_arrival + min_dist_to_target

    def _discretize_delay(self, train_id: int, delay: int) -> int:
        """
        Discretizes the delay of the given train
        
        Parameters
        ----------
        train_id : int
            The id of the train
        delay : int
            The delay of the train
            
        Returns
        -------
        int
            The discretized delay of the given train
        """
        train = self.rail_env.agents[train_id]
        available_time = train.latest_arrival - train.earliest_departure
        if delay <= 0:
            return 0
        if delay <= available_time * self.delay_threshold:
            return 1
        return 2
    
    def step(self, actions: List[Tuple[int, int, int]], step_until_action_required: bool = False) -> bool:
        """
        actions: [train_id, node_id, one of {0, 1, 2}]
        Returns done flag.
        """
        fl_actions = {tid: RailEnvActions.MOVE_FORWARD for tid in range(self.rail_env.get_num_agents())}
        for train_id, node_id, action in actions:
            if action == 2:
                fl_actions[train_id] = RailEnvActions.STOP_MOVING
            else:
                node = self.nodes[node_id]
                fl_actions[train_id] = self.graph.fw_star(node)[action][1].action
            self.last_action[node_id] = action
        
        done, sll = self._step_low_level(fl_actions, step_until_action_required)
        return done, sll
    
    def _step_low_level(self, actions: Dict[int, RailEnvActions], step_until_action_required: bool = False) -> bool:
        """
        Executes the given actions on the Flatland environment

        Parameters
        ----------
        actions : Dict[int, RailEnvActions]
            The actions to execute
        step_until_action_required : bool
            Whether to step until an action is required

        Returns
        -------
        bool
            True if the environment is done, False otherwise
        """
        sll = time.time()
        self.rail_env.step(actions)
        sll = time.time() - sll

        self._update_rewards()
        if self._check_deadlocks():
            return True, sll
        if step_until_action_required:
            while len(self.graph.trainsThatAreOnNodes(self.rail_env.agents)) == 0 and not self.rail_env.dones['__all__']:
                self.rail_env.step({tid: RailEnvActions.MOVE_FORWARD for tid in range(self.rail_env.get_num_agents())})
                self._update_rewards()
                if self._check_deadlocks():
                    return True, sll
        return self.rail_env.dones['__all__'], sll
    
    def _check_deadlocks(self) -> bool:
        """
        Checks for deadlocks and updates the rewards accordingly

        Returns
        -------
        bool
            True if a deadlock is detected, False otherwise
        """
        locked_by = {
            train_id: self._train_locked_by(train_id)
            for train_id in range(self.rail_env.get_num_agents())
        }
        deadlock = False
        deadlock_ids = set()
        for train_id in range(self.rail_env.get_num_agents()):
            for other_train_id in locked_by[train_id]:
                if train_id in locked_by[other_train_id] and other_train_id in locked_by[train_id]:
                    deadlock = True
                    deadlock_ids.add(train_id)
                    deadlock_ids.add(other_train_id)
        if deadlock:
            for train_id in deadlock_ids:
                if train_id in self.train_to_last_node:
                    last_node = self.train_to_last_node[train_id][0]
                    last_node_id = self.node_to_id[last_node]
                    # Give deadlock penalty only if the node who sent the train ignored a True semaphore
                    if last_node_id in self.last_obs and last_node_id in self.last_action:
                        _, _, _, sem_0, sem_1 = self.last_obs[last_node_id]
                        last_action = self.last_action[last_node_id]
                        if sem_0 and last_action == 0 or sem_1 and last_action == 1:
                            self.rewards[self.node_to_id[last_node]] += self.deadlock_penalty
        return deadlock

    def _train_locked_by(self, train_id: int) -> Set[int]:
        """
        Returns empty set if train can move, otherwise returns the set of train_ids that are blocking it.

        Parameters
        ----------
        train_id : int
            The id of the train

        Returns
        -------
        Set[int]
            The set of train_ids that are blocking the given train
        """
        train = self.rail_env.agents[train_id]
        if train.position is None:
            return set()
        locked_by = set()
        row, col = train.position
        transitions = self.rail_env.rail.get_transitions(row, col, train.direction)
        for tx in range(4):
            if transitions[tx] == 1:
                next_row, next_col = row, col
                if tx == 0:
                    next_row -= 1
                elif tx == 1:
                    next_col += 1
                elif tx == 2:
                    next_row += 1
                else:
                    next_col -= 1
                tx_locked = False
                for other_train_id in range(self.rail_env.get_num_agents()):
                    other_train = self.rail_env.agents[other_train_id]
                    if other_train.position == (next_row, next_col):
                        tx_locked = True
                        locked_by.add(other_train_id)
                if not tx_locked:
                    return set()
        return locked_by

    def _update_rewards(self):
        """
        Updates the rewards based on the current state of the environment
        """
        # Compute rewards for trains that have arrived
        for train_id, done in self.rail_env.dones.items():
            if train_id == '__all__':
                continue
            if done and not self.old_dones[train_id] and train_id in self.train_to_last_node and self.rail_env.agents[train_id].position is None:
                last_node, _ = self.train_to_last_node[train_id]
                self.rewards[self.node_to_id[last_node]] += self.destination_bonus
                self.trains_arrived[train_id] = True

        self.old_dones = self.rail_env.dones.copy()

        for train_id, node in self.graph.trainsThatAreOnNodes(self.rail_env.agents).items():
            curr_delay = self._compute_delay(train_id)
            if train_id in self.train_to_last_node:
                last_node, last_delay = self.train_to_last_node[train_id]
                self.rewards[self.node_to_id[last_node]] += last_delay - curr_delay
                if last_node != node:
                    self.train_to_last_node[train_id] = (node, curr_delay)
            else:
                self.train_to_last_node[train_id] = (node, curr_delay)

    def end_episode(self):
        """
        Ends the episode and returns the rewards
        """
        r = self.rewards.copy()
        self.rewards = np.zeros(len(self.nodes))
        return r