import sys
from gc import get_referents
from types import FunctionType, ModuleType
from typing import Any

# Custom objects know their class.
# Function objects seem to know way too much, including modules.
# Exclude modules as well.
BLACKLIST = type, ModuleType, FunctionType


def gettotalsize(obj: Any) -> int:
    """
    Estimate the total memory occupied by a Python object and its members.

    Taken from <https://stackoverflow.com/a/30316760>.

    :param obj: object to measure
    :returns: total size in bytes
    """
    if isinstance(obj, BLACKLIST):
        raise TypeError(
            "gettotalsize() does not take argument of type: " + str(type(obj))
        )
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size
