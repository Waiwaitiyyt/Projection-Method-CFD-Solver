import numpy as np
import numpy.typing as npt
from .mesh_type import MeshType


class MeshMap:
    def __init__(self, mask: npt.NDArray[np.bool_], cell_size: tuple[float, float]):
        self.mask = mask
        self.cell_size = cell_size
        self.dx, self.dy = cell_size
        self.mesh_map = np.full(mask.shape, MeshType.UNASSIGNED.value, dtype=np.uint8)
        self._full_boundary = self._get_full_boundary()
        self.mesh_map[self._full_boundary] = MeshType.WALL.value
        self.mesh_map[mask & ~self._full_boundary] = MeshType.FLOW.value

    def _get_full_boundary(self) -> npt.NDArray[np.bool_]:
        interior = (
            self.mask[1:-1, 1:-1] &
            self.mask[2:,   1:-1] &
            self.mask[:-2,  1:-1] &
            self.mask[1:-1, 2:]   &
            self.mask[1:-1, :-2]
        )
        full_boundary = np.zeros_like(self.mask, dtype=bool)
        full_boundary[1:-1, 1:-1] = self.mask[1:-1, 1:-1] & ~interior
        return full_boundary

    @property
    def full_boundary(self) -> npt.NDArray[np.bool_]:
        return self._full_boundary

    def set_mesh_type(self,
                      region: npt.NDArray[np.bool_],
                      btype: MeshType):
        if btype == MeshType.FLOW:
            target = (self.mesh_map == MeshType.UNASSIGNED.value) & region
        else:
            target = self._full_boundary & region
        self.mesh_map[target] = btype.value

    def set_mesh_type_rect(self,
                           ix_start: int, ix_end: int,
                           iy_start: int, iy_end: int,
                           btype: MeshType):
        region = np.zeros_like(self.mask, dtype=bool)
        region[ix_start:ix_end, iy_start:iy_end] = True
        self.set_mesh_type(region, btype)

    @property
    def flow_mask(self) -> npt.NDArray[np.bool_]:
        return self.mesh_map == MeshType.FLOW.value

    @property
    def wall_mask(self) -> npt.NDArray[np.bool_]:
        return self.mesh_map == MeshType.WALL.value

    @property
    def inlet_mask(self) -> npt.NDArray[np.bool_]:
        return self.mesh_map == MeshType.INLET.value

    @property
    def outlet_mask(self) -> npt.NDArray[np.bool_]:
        return self.mesh_map == MeshType.OUTLET.value

    def summary(self) -> dict[str, int]:
        return {mt.name: int(np.sum(self.mesh_map == mt.value)) for mt in MeshType}
