# SwitchFL: Switch-Based Multi-Agent Reinforcement Learning Environment

SwitchFL is a sophisticated multi-agent reinforcement learning environment built on top of Flatland-RL, designed to model train routing and switching operations in railway networks. Instead of controlling individual trains, agents control railway switches, making strategic decisions about train routing through the network.

## Project Overview

SwitchFL transforms the traditional train control problem into a **switch-centric multi-agent system** where:
- **Agents are railway switches** (not trains)
- **Actions control train routing** through switch configurations
- **Observations provide local network state** around each switch
- **Rewards incentivize efficient train flow** and network optimization

This approach enables studying **cooperative multi-agent behaviors**, **network-level optimization**, and **distributed traffic management** in railway systems.

## Architecture & Building Blocks

### Core Components

```
SwitchFL Environment
├── ASyncSwitchEnv (Main Environment)
├── RailNetwork (Network Management)
├── Switch Agents (Individual Switch Controllers)
├── RailGraph (Graph Representation) 
├── Observer (State Observation)
└── Utilities (Support Functions)
```

## Building Block Details

### 1. **ASyncSwitchEnv** - Main Environment Controller
**File**: `switch_env.py`

The central environment class that orchestrates the entire simulation:

```python
class ASyncSwitchEnv(_SwitchEnv, AECEnv):
    """Asynchronous switch environment for multi-agent RL"""
```

**Key Responsibilities:**
- **Train Movement Simulation**: Moves trains through the network based on current switch configurations
- **Agent Activation**: Determines which switches are active (have approaching trains)
- **Action Execution**: Translates switch actions into train routing decisions
- **State Management**: Maintains environment state including train positions, switch configurations, and network status
- **Reward Calculation**: Computes rewards for switch agents based on performance metrics

**Key Methods:**
- `_move_trains_to_switch()`: Advances trains until they reach switches
- `_check_action_execution()`: Validates train path adherence (source of "Train deviated from planned path!" error)
- `_apply_action()`: Executes switch actions and updates train routing
- `step()`: Main environment step function

### 2. **RailNetwork** - Network Management System
**File**: `rail_network.py`

Manages the railway network topology and train-switch interactions:

```python
class RailNetwork:
    """Manages railway network topology and switch operations"""
```

**Key Components:**
- **Switch Network**: Graph of switch-to-switch connections
- **Rail Graph**: Detailed topology including all rail segments
- **Port Management**: Handles train entry/exit points for each switch
- **Semaphore System**: Manages train blocking and coordination

**Key Responsibilities:**
- Build network topology from Flatland rail environment
- Manage switch-to-switch connections and routing
- Track train positions and movements
- Coordinate semaphore signals for train safety

### 3. **Switch Agents** - Individual Switch Controllers
**File**: `switch_agents.py`

Individual switch agents that control train routing decisions:

```python
class _Switch(ABC):
    """Abstract base class for railway switch agents"""
```

**Switch Types:**
- **Switch1**: Simple junction switches
- **Switch2**: Complex multi-port switches
- **Switch3**: Three-way junction switches

**Key Features:**
- **Action Space**: Available routing configurations for approaching trains
- **Port Management**: Handles multiple entry/exit ports
- **Semaphore Control**: Blocks/unblocks ports based on train presence
- **Action Outcomes**: Maps actions to train routing paths

### 4. **RailGraph** - Graph Representation System
**File**: `utils/rail_graph.py`

Converts Flatland's grid-based rail environment into graph structures:

```python
def create_rail_graph(env: RailEnv) -> nx.Graph:
    """Build graph representation of rail environment"""
```

**Processing Pipeline:**
1. **Grid to Graph**: Convert Flatland grid cells to graph nodes
2. **Switch Detection**: Identify switch locations and types
3. **Port Generation**: Create entry/exit ports for each switch
4. **Network Pruning**: Remove non-essential nodes for efficiency
5. **Action Mapping**: Generate action-to-routing mappings

### 5. **Observer System** - State Observation
**File**: `observer.py`

Provides state observations for switch agents:

```python
class StandardObserver(_Observer):
    """Standard observation provider for switch agents"""
```

**Observation Components:**
- **Local Network State**: Nearby switch configurations
- **Train Information**: Approaching trains and their properties
- **Port Status**: Semaphore states and blocking information
- **Network Topology**: Local connectivity information

### 6. **Utility Systems**

#### **Environment Builder** (`utils/build_env.py`)
Creates standardized environment configurations:
```python
def build_standard_async_env(height, width, max_num_cities, num_trains, ...):
    """Build standardized SwitchFL environment"""
```

#### **Naming Convention** (`utils/naming.py`)
Manages consistent ID mapping between components:
- Switch ID ↔ Switch Name conversion
- Port ID ↔ Node ID mapping
- Coordinate system translations

#### **Logging System** (`utils/logging.py`)
Provides comprehensive logging and debugging support:
- Deterministic seeding for reproducibility
- Structured logging for analysis
- Debug information for troubleshooting

## How It Works

### 1. **Environment Initialization**
```python
# Create environment
env = build_standard_async_env(
    height=20, width=20,
    max_num_cities=3, num_trains=4,
    seed=42
)
```

### 2. **Network Construction**
1. Flatland generates base rail network
2. RailGraph converts grid to graph representation
3. RailNetwork identifies switches and builds switch network
4. Switch agents are instantiated for each network switch

### 3. **Simulation Loop**
```python
# Main simulation loop
while not env.terminated:
    # 1. Move trains toward switches
    env._move_trains_to_switch()
    
    # 2. Activate switches with approaching trains
    active_switches = env.active_switch_agents
    
    # 3. Get observations for active switches
    obs = env.observe(active_switch)
    
    # 4. Agent decides routing action
    action = agent.act(obs)
    
    # 5. Execute action and route train
    env.step(action)
    
    # 6. Update network state and compute rewards
    rewards = env.rewards
```

### 4. **Train Routing Process**
1. **Train Approach**: Train approaches a switch
2. **Switch Activation**: Switch becomes active agent
3. **Action Selection**: Switch agent chooses routing action
4. **Path Validation**: System validates train follows planned path
5. **Train Transition**: Train moves through switch to next network segment

## Current Challenges

### 1. **Non-Deterministic Behavior**
**Issue**: Flatland's `sparse_line_generator` introduces randomness in train placement and direction assignment, causing seed inconsistency.

**Impact**: 
- Same seeds produce different network configurations
- Train handles are non-deterministically assigned
- Makes reproducible experiments difficult

**Current Mitigation**: 
- Deterministic agent sorting in `reset()`
- Comprehensive seeding throughout the pipeline
- Agent handle reassignment based on position

### 2. **Train Path Deviation Detection**
**Issue**: The `_check_action_execution()` method logs errors when trains deviate from planned paths.

**Error Condition**:
```python
if train.position not in [*rail_pieces, source_node, target_node]:
    self.logger.error(f"Train {train.handle} deviated from planned path!")
```

**Challenges**:
- Complex path validation logic
- Interaction between switch actions and train movement
- Difficult to debug when deviations occur

### 3. **Switch-Train Coordination**
**Issue**: Complex coordination required between switch decisions and train movements.

**Challenges**:
- Trains may arrive simultaneously at switches
- Switch actions must be synchronized with train positions
- Port blocking/unblocking timing is critical

### 4. **Observation Space Complexity**
**Issue**: Designing informative yet manageable observation spaces for switch agents.

**Challenges**:
- Balancing local vs. global information
- Scalability with network size
- Real-time state representation

### 5. **Reward Design**
**Issue**: Creating reward functions that encourage desired network-level behaviors.

**Challenges**:
- Multi-objective optimization (efficiency, safety, fairness)
- Credit assignment in multi-agent settings
- Balancing local switch performance vs. global network performance

## Future Directions for Full RL Environment

### 1. **Enhanced Determinism**
- [ ] **Custom Line Generator**: Replace Flatland's non-deterministic generator
- [ ] **Deterministic Train Placement**: Ensure consistent train starting positions
- [ ] **Seed Isolation**: Separate random streams for different components

### 2. **Robust Path Validation**
- [ ] **Improved Path Tracking**: Better train position validation
- [ ] **Error Recovery**: Mechanisms to handle path deviations gracefully
- [ ] **Debug Visualization**: Tools to visualize train paths and deviations

### 3. **Advanced Multi-Agent Features**
- [ ] **Switch Communication**: Enable information sharing between switches
- [ ] **Hierarchical Control**: Multi-level decision making (local vs. regional)
- [ ] **Dynamic Switch Activation**: Context-aware agent activation

### 4. **Scalability Improvements**
- [ ] **Efficient Graph Representation**: Optimize for large networks
- [ ] **Lazy Evaluation**: Compute observations only when needed
- [ ] **Parallel Processing**: Multi-threaded simulation support

### 5. **Rich Observation Spaces**
- [ ] **Temporal Information**: Historical state information
- [ ] **Global Network State**: Network-wide performance metrics
- [ ] **Predictive Features**: Train arrival time predictions

### 6. **Comprehensive Reward Systems**
- [ ] **Multi-Objective Rewards**: Efficiency, safety, fairness metrics
- [ ] **Shaped Rewards**: Guide learning toward desired behaviors
- [ ] **Dynamic Reward Adaptation**: Context-dependent reward functions

### 7. **Evaluation and Benchmarking**
- [ ] **Standard Benchmarks**: Consistent evaluation scenarios
- [ ] **Performance Metrics**: Network throughput, delay, conflicts
- [ ] **Comparison Framework**: Against traditional control methods

### 8. **Integration Features**
- [ ] **OpenAI Gym Compatibility**: Standard RL interface
- [ ] **Ray RLlib Integration**: Distributed training support
- [ ] **Stable Baselines3 Compatibility**: Easy algorithm integration

## Getting Started

### Basic Usage
```python
from switchfl.utils.build_env import build_standard_async_env

# Create environment
env = build_standard_async_env(
    height=15, width=15,
    max_num_cities=2, num_trains=3,
    seed=42
)

# Reset environment
env.reset(seed=42)

# Main loop
while not env.terminated:
    if env.active_switch_agents:
        # Get active switch
        active_switch = env.active_switch_agents[0]
        
        # Get observation
        obs = env.observe(active_switch)
        
        # Take random action (replace with your agent)
        action = env.action_space(active_switch).sample()
        
        # Execute action
        env.step(action)
```

### Running Tests
```bash
# Test train deviation scenarios
python project/environment/test/test_targeted_deviation.py

# Test edge cases
python project/environment/test/test_comprehensive_edge_cases.py
```

## Project Status

- [x] **Core Environment**: Functional multi-agent switch environment
- [x] **Network Representation**: Graph-based railway topology
- [x] **Switch Agents**: Individual switch controllers with action spaces
- [x] **Train Simulation**: Basic train movement and routing
- [ ] **Determinism**: Partial - some non-deterministic behaviors remain
- [x] **Path Validation**: Functional but needs robustness improvements
- [ ] **Reward System**: Basic implementation, needs enhancement
- [x] **Observation System**: Standard observer, needs rich features
- [ ] **Benchmarking**: Not yet implemented
- [x] **Documentation**: Comprehensive README, API docs needed

## Contributing

SwitchFL is an active research project. Key areas for contribution:
- Improving deterministic behavior
- Enhancing observation spaces
- Developing reward functions
- Creating evaluation benchmarks
- Adding visualization tools
