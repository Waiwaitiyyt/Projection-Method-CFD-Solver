from .mesh_type import MeshType


class Unassigned:
    TYPE = MeshType.UNASSIGNED

    @classmethod
    def mask(cls, mesh_map):
        return mesh_map == cls.TYPE.value
