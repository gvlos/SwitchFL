from typing import Any

from environment import NodeId, PortId


def get_switch_id(identifier: Any) -> str:
    return f"switch_{identifier}"


def get_node_id_on_port_id(port_id: PortId) -> NodeId:
    return (int(port_id[0]), int(port_id[1]))


def switch_id2name(switch_id: NodeId) -> str:
    name = f"{switch_id[0]}-{switch_id[1]}"
    name = "switch_" + name
    return name


def name2switch_id(name: str) -> NodeId:
    node_id = name.split("_")[1].split("-")
    node_id = (int(node_id[0]), int(node_id[1]))
    return node_id
