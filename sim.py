import numpy as np
import numpy.typing as npt
from numpy.typing import DTypeLike
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Tuple
from enum import Enum
import gc

class MeshType(Enum):
    UNASSIGNED = 0
    WALL = 1
    INLET = 2
    OUTLET = 3
    FLOW = 4
    

class ScalarProfile():
    def __init__(self, mask: npt. NDArray, cell_size: tuple, datatype: DTypeLike = np.float32):
        self.mask = mask
        self.profile = np.zeros_like(self.mask, dtype=datatype)
        self.profile[self.mask==True] = 1
        self.dx = cell_size[0]
        self.dy = cell_size[1]
        
        # Initialise boundary contour
        self._full_boundary = self._get_full_boundary()
        self.mesh_map = np.full(self.profile.shape, MeshType.UNASSIGNED.value)
        self.mesh_map[self._full_boundary] = MeshType.WALL.value
        self.mesh_map[mask & ~self._full_boundary] = MeshType.FLOW.value
    
    def _get_full_boundary(self) -> npt.NDArray[bool]: # pyright: ignore[reportInvalidTypeForm]
        self.interior = (
            self.mask[1:-1, 1:-1] &
            self.mask[2:,   1:-1] &  # down
            self.mask[:-2,  1:-1] &  # up
            self.mask[1:-1, 2:]   &  # right
            self.mask[1:-1, :-2]     # left
        )

        self._full_boundary = np.zeros_like(self.mask, dtype=bool)
        self._full_boundary[1:-1, 1:-1] = self.mask[1:-1, 1:-1] & ~self.interior
        return self._full_boundary
    
    def set_mesh_type(self, 
                     ix_start: int, ix_end: int,
                     iy_start: int, iy_end: int,
                     btype: MeshType):
        
        """Tag a rectangular region of boundary cells with a type."""
        region = np.zeros_like(self.mask, dtype=bool)
        region[ix_start:ix_end, iy_start:iy_end] = True
        if btype != MeshType.FLOW:
            # Only apply to actual boundary cells
            target = self._full_boundary & region
            self.mesh_map[target] = btype.value
        else:
            # Set flow ROI
            _tmp_mesh_map = np.zeros_like(self.mesh_map, dtype=bool)
            _tmp_mesh_map[self.mesh_map == MeshType.UNASSIGNED.value] = True
            flow_region = _tmp_mesh_map & region
            self.mesh_map[flow_region] = btype.value
    
class VelocityField(ScalarProfile):
    def __init__(self, mask: npt. NDArray, cell_size: tuple, density: float, viscosity: float, datatype: DTypeLike = np.float32):
        super().__init__(mask, cell_size, datatype)
        self.vx_field = np.zeros_like(mask, dtype=datatype)
        self.vy_field = np.zeros_like(mask, dtype=datatype)
        self.rho = density
        self.nu = viscosity

    def apply_wall_boundary(self):
        if not np.any(self.mesh_map == MeshType.WALL.value):
            raise IndexError("Boundary INLET not set")
        else:
            self.vx_field[self.mesh_map == MeshType.WALL.value] = 0
            self.vy_field[self.mesh_map == MeshType.WALL.value] = 0

    def apply_inlet_boundary(self, v_in: tuple[float, float]):
        if not np.any(self.mesh_map == MeshType.INLET.value):
            raise IndexError("Boundary INLET not set")
        else:
            self.v_in = v_in
            self.vx_field[self.mesh_map == MeshType.INLET.value] = v_in[0]
            self.vy_field[self.mesh_map == MeshType.INLET.value] = v_in[1]

    def _vx_update(self,
               p_field: npt.NDArray,
               dt: float):
        
        vx = self.vx_field
        vy = self.vy_field
        dx, dy = self.dx, self.dy
        rho, nu = self.rho, self.nu
        vx_new = np.empty_like(vx)
        vx_new[:] = vx

        vx_new[1:-1, 1:-1] = (
            vx[1:-1, 1:-1]
            # convection: vx * dvx/dx + vy * dvx/dy
            - vx[1:-1, 1:-1] * dt/dx * (vx[1:-1, 1:-1] - vx[:-2,  1:-1])
            - vy[1:-1, 1:-1] * dt/dy * (vx[1:-1, 1:-1] - vx[1:-1, :-2])
            # diffusion
            + nu * dt/dx**2 * (vx[2:,   1:-1] - 2*vx[1:-1, 1:-1] + vx[:-2,  1:-1])
            + nu * dt/dy**2 * (vx[1:-1, 2:]   - 2*vx[1:-1, 1:-1] + vx[1:-1, :-2])
            # pressure gradient in x
            - dt/(2*rho*dx) * (p_field[1:-1, 2:] - p_field[1:-1, :-2])
        )

            # Mask: only keep updates inside the fluid domain
        flow_mask = self.mesh_map == MeshType.FLOW.value
        self.vx_field[flow_mask] = vx_new[flow_mask]

        

    def _vy_update(self,
               p_field: npt.NDArray,
               dt: float):
        
        vy = self.vy_field
        vx = self.vx_field
        dx, dy = self.dx, self.dy
        rho, nu = self.rho, self.nu
        vy_new = np.empty_like(vy)
        vy_new[:] = vy

        vy_new[1:-1, 1:-1] = (
            vy[1:-1, 1:-1]
            # convection: vx * dvy/dx + vy * dvy/dy
            - vx[1:-1, 1:-1] * dt/dx * (vy[1:-1, 1:-1] - vy[:-2,  1:-1])
            - vy[1:-1, 1:-1] * dt/dy * (vy[1:-1, 1:-1] - vy[1:-1, :-2])
            # diffusion
            + nu * dt/dx**2 * (vy[2:,   1:-1] - 2*vy[1:-1, 1:-1] + vy[:-2,  1:-1])
            + nu * dt/dy**2 * (vy[1:-1, 2:]   - 2*vy[1:-1, 1:-1] + vy[1:-1, :-2])
            # pressure gradient in y
            - dt/(2*rho*dy) * (p_field[2:, 1:-1] - p_field[:-2, 1:-1])
        )

            # Mask: only keep updates inside the fluid domain
        flow_mask = self.mesh_map == MeshType.FLOW.value
        self.vy_field[flow_mask] = vy_new[flow_mask]
    
    def update(self, 
               p_field: npt.NDArray,
               dt: float):

        self._vx_update(p_field, dt)
        self._vy_update(p_field, dt)

        # Reapply boundary conditions
        try:
            self.apply_wall_boundary()
            self.apply_inlet_boundary(self.v_in)
        except AttributeError:
            raise AttributeError(r"INLET velocity not set, use apply_inlet_boundary() to set inlet velocity")
        
        # Outlet: zero gradient (convective outflow)
        outlet = self.mesh_map == MeshType.OUTLET.value
        outlet_rows, outlet_cols = np.where(outlet)
        self.vx_field[outlet_rows, outlet_cols] = self.vx_field[outlet_rows-1, outlet_cols]
        self.vy_field[outlet_rows, outlet_cols] = self.vy_field[outlet_rows-1, outlet_cols]
        

class SinkSource(ScalarProfile):
    def __init__(self, mask: npt. NDArray, cell_size: tuple, density: float, datatype: DTypeLike = np.float32):
        super().__init__(mask, cell_size, datatype)
        self.b_field = np.zeros_like(mask, dtype=datatype)
        self.rho = density

    def build(self, vx: npt.NDArray, vy: npt.NDArray, dt: float):
        rho = self.rho
        dx, dy = self.dx, self.dy

        self.b_field[1:-1, 1:-1] = (rho * (1 / dt * ((vx[1:-1, 2:] - vx[1:-1, 0:-2]) / (2 * dx) +
                                    (vy[2:, 1:-1] - vy[0:-2, 1:-1]) / (2 * dy)) -
                        ((vx[1:-1, 2:] - vx[1:-1, 0:-2]) / (2 * dx))**2 -
                        2 * ((vx[2:, 1:-1] - vx[0:-2, 1:-1]) / (2 * dy) *
                                (vy[1:-1, 2:] - vy[1:-1, 0:-2]) / (2 * dx))-
                        ((vy[2:, 1:-1] - vy[0:-2, 1:-1]) / (2 * dy))**2))
        
        # Manual assign boundary condition ?


class PressureField(ScalarProfile):
    def __init__(self, mask: npt. NDArray, cell_size: tuple, density: float, datatype: DTypeLike = np.float32):
        super().__init__(mask, cell_size, datatype)
        self.p_field = np.zeros_like(mask, dtype=datatype)
        self.rho = density

    def update(self, b_field: npt.NDArray, zero_p_outlet: bool,  max_iter: int = 500, tol: float = 1e-4):
        rho = self.rho
        dx, dy = self.dx, self.dy
        p = self.p_field.copy()
        flow_roi = np.zeros_like(self.mesh_map, dtype=bool)
        flow_roi[self.mesh_map != MeshType.UNASSIGNED.value] = True

        for _ in range(max_iter):
            p_old = p.copy()
            p[1:-1, 1:-1] = (
                    ((p_old[1:-1, 2:] + p_old[1:-1, :-2]) * self.dy**2 +
                    (p_old[2:, 1:-1] + p_old[:-2, 1:-1]) * self.dx**2 -
                    b_field[1:-1, 1:-1] * self.dx**2 * self.dy**2) /
                    (2 * (self.dx**2 + self.dy**2)))
            
            # Apply boundary condition
            # p[self.mesh_map == MeshType.WALL.value] = p_old[self.mesh_map == MeshType.WALL.value] # Zero gradient at wall
            p[self.mesh_map == MeshType.UNASSIGNED.value] = 0 # Reset unassigned region
            if zero_p_outlet:
                p[self.mesh_map == MeshType.OUTLET.value] = 0 # Outlet zero pressure

            # Residual tolerance 
            residual = np.linalg.norm(p[flow_roi] - p_old[flow_roi]) / (np.linalg.norm(p_old[flow_roi]) + 1e-10)
            if residual < tol:
                break
        self.p_field = p

def cfl(v_in: Tuple[float, float], dt: float, cell_size: Tuple[float, float], viscosity: float):
    cfl_convective = (v_in[0]**2 + v_in[1]**2)**0.5 * dt / cell_size[0]
    cfl_diffusive  = viscosity * dt / cell_size[0]**2
    print(f"CFL convective: {cfl_convective:.3f} (must be < 1)")
    print(f"CFL diffusive:  {cfl_diffusive:.3f} (must be < 0.5)")
    dt_diffusive  = 0.4 * cell_size[0]**2 / viscosity   # 40% safety margin
    dt_convective = 0.4 * cell_size[0] / max(abs(v_in[0]), abs(v_in[1]), 1e-10)
    dt = min(dt_diffusive, dt_convective)
    print(f"Safe dt: {dt:.2e}")
    assert cfl_convective < 1,   "Reduce dt or inlet velocity"
    assert cfl_diffusive  < 0.5, "Reduce dt or increase dx"

def get_safe_dt(v_in, cell_size, viscosity, safety=0.4):
    dx, dy = cell_size
    vmag = max(abs(v_in[0]), abs(v_in[1]), 1e-10)
    dt_diffusive = safety * dx**2 / viscosity
    dt_convective = safety * dx / vmag
    dt = min(dt_diffusive, dt_convective)
    print(f"Recommanded dt = {dt:.2e}")
    return dt


if __name__ == "__main__":

    # Define the entire region boundary
    range_x = 2
    range_y = 2
    cell_size = (5e-3, 5e-3)
    cell_number_x = int(range_x / cell_size[0])
    cell_number_y = int(range_y / cell_size[1])
    mask = np.zeros(shape=(cell_number_x, cell_number_y), dtype=bool)
    mask[int(0.25*cell_number_x):int(0.4*cell_number_x), int(0.25*cell_number_y):int(0.75*cell_number_y)] = True
    mask[int(0.4*cell_number_x):int(0.75*cell_number_y), int(0.6*cell_number_y):int(0.75*cell_number_y)] = True

    density = 1e3
    viscosity = 1e-3
    v_in = (0, 1)
    dt = get_safe_dt(v_in, cell_size, viscosity, safety=0.4)

    cfl(v_in, dt, cell_size, viscosity)

    V = VelocityField(mask, cell_size, density, viscosity)
    V.set_mesh_type(int(0.25*cell_number_x+1), int(0.4*cell_number_x-1), int(0.25*cell_number_y-1), int(0.25*cell_number_y+1), MeshType.INLET)
    V.set_mesh_type(int(0.75*cell_number_y-1), int(0.75*cell_number_y), int(0.6*cell_number_y+1), int(0.75*cell_number_y-1), MeshType.OUTLET)
    V.apply_inlet_boundary(v_in)

    SS = SinkSource(mask, cell_size, density)
    P = PressureField(mask, cell_size, density)


    iter = 500
    for i in range(iter):
        
        SS.build(V.vx_field, V.vy_field, dt)
        
        P.update(SS.b_field, True)
        V.update(P.p_field, dt)

        if not np.all(np.isfinite(V.vx_field)) or not np.all(np.isfinite(V.vy_field)):
            print(f"NaN/Inf detected in velocity at iteration {i}")
            break
        if not np.all(np.isfinite(P.p_field)):
            print(f"NaN/Inf detected in pressure at iteration {i}")
            break

        print(f"Iteration: {i}")

    
    plt.figure()
    plt.imshow(V.vx_field, cmap='gray')
    plt.show()


