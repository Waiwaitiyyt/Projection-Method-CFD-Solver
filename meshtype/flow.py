from .mesh_type import MeshType
import numpy.typing as npt
import numpy as np


class Flow:
    TYPE = MeshType.FLOW

    @classmethod
    def mask(cls, mesh_map):
        return mesh_map == cls.TYPE.value
