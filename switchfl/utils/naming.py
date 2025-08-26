from typing import Any

from switchfl import NodeId, PortId


def get_switch_id(identifier: Any) -> str:
    return f"switch_{identifier}"


def get_node_id_on_port_id(port_id: PortId) -> NodeId:
    """convert a port id into a node id


    Example:
    >>> get_node_id_on_port_id((4.1, 3.1))
    (4, 3)

    Args:
        port_id (PortId): (int, int)

    Returns:
        NodeId: (float, float)
    """
    return (int(port_id[0]), int(port_id[1]))


def switch_id2name(switch_id: NodeId) -> str:
    """get switch name by its ids

    Example:
    >>> switch_id2name((4, 3))
    'switch_4-3'

    Args:
        switch_id (NodeId): coordinates in flatlands system: (x: int, y: int)

    Returns:
        str: 'switch_x-y'
    """
    name = f"{switch_id[0]}-{switch_id[1]}"
    name = "switch_" + name
    return name


def name2switch_id(name: str) -> NodeId:
    """get switch id from switch env

    Example:
    >>> name2switch_id('switch_4-3')
    (4, 3)

    Args:
        name (str): switch name in the format 'switch_x-y'

    Returns:
        NodeId: node id (x, y)
    """
    node_id = name.split("_")[1].split("-")
    node_id = (int(node_id[0]), int(node_id[1]))
    return node_id


def symmetric_string(s: str, n: int = 80, frame: str = "=") -> str:
    n_buffer = n - len(s) - 2
    frame = frame * (n_buffer // 2)
    frame += " "
    res = frame + s + frame[::-1]
    return res