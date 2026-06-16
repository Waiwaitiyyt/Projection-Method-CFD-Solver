from .mesh_type import MeshType


class Wall:
    TYPE = MeshType.WALL

    @classmethod
    def mask(cls, mesh_map):
        return mesh_map == cls.TYPE.value

    @classmethod
    def apply(cls, field, mesh_map):
        field[cls.mask(mesh_map)] = 0
        return field
