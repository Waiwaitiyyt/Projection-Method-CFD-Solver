from enum import Enum


class MeshType(Enum):
    UNASSIGNED = 0
    WALL = 1
    INLET = 2
    OUTLET = 3
    FLOW = 4
