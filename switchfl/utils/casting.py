from typing import Any, Dict


def serialize_dict(d: Dict[str, Any], target_type = str):
    res = {}
    for k, i in d.items():
        if isinstance(i, dict):
            res[target_type(k)] = serialize_dict(i, target_type)
        else:
            res[target_type(k)] = target_type(i)
    return res