from .mesh_type import MeshType


class Inlet:
    TYPE = MeshType.INLET

    @classmethod
    def mask(cls, mesh_map):
        return mesh_map == cls.TYPE.value

    @classmethod
    def apply(cls, field, value, mesh_map):
        mask = cls.mask(mesh_map)
        field[mask] = value
        return field
