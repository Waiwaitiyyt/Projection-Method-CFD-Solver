from .mesh_type import MeshType


class Outlet:
    TYPE = MeshType.OUTLET

    @classmethod
    def mask(cls, mesh_map):
        return mesh_map == cls.TYPE.value
