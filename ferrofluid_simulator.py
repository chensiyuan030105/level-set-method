import taichi as ti
# import taichi_glsl as ts

import utils
from utils import *
from mgpcg import MGPCGPoissonSolver
from pressure_project import PressureProjectStrategy
from level_set import FastMarchingLevelSet, FastSweepingLevelSet
from surface_tension import SurfaceTension
from fluid_simulator import FluidSimulator
from magnetic_force import MagneticForceStrategy
from potential_function import PotentialFunctionStrategy

from functools import reduce
import time
import numpy as np

@ti.data_oriented
class FerrofluidSimulator(FluidSimulator):
    def __init__(self,
        dim = 2,
        res = (128, 128),
        dt = 1.25e-2,
        substeps = 1,
        dx = 1.0,
        rho = 1000.0,
        gravity = [0, -9.8],
        p0 = 1e-3,
        mu0 = 4 * np.pi * 1e-7,
        k = 5,
        real = float):
            super().__init__(dim, res, dt, substeps, dx, rho, gravity, p0, real)

            self.mu0 = mu0 # Vacuum permeability
            self.k = k # Permittivity

            self.H_ext = [ti.field(dtype=real) for _ in range(self.dim)] # External magnetic field intensity
            self.H = [ti.field(dtype=real) for _ in range(self.dim)] # Magnetic field intensity

            indices = ti.ijk if self.dim == 3 else ti.ij
            for d in range(self.dim):
                ti.root.dense(indices, [res[_] + (d == _) for _ in range(self.dim)]).place(self.H_ext[d], self.H[d])

            self.chi = ti.field(dtype=real, shape=res) # Magnetic susceptibility
            self.psi = ti.field(dtype=real, shape=res) # Potential function


            self.potential_function_strategy = PotentialFunctionStrategy(self.dim,
                                                                self.res,
                                                                self.dx,
                                                                self.H_ext,
                                                                self.chi)
 
            self.magnetic_force_strategy = MagneticForceStrategy(self.dim,
                                                                 self.velocity,
                                                                 self.ghost_fluid_method,
                                                                 self.level_set.phi,
                                                                 self.p0,
                                                                 self.mu0,
                                                                 self.k,
                                                                 self.H,
                                                                 self)

    @ti.func
    def H_interp(self, I):
        H = ti.Vector.zero(self.real, self.dim)
        for k in ti.static(range(self.dim)):
            H[k] = utils.sample(self.H[k], I - 0.5 * (1 - ti.Vector.unit(self.dim, k)))
        return H

    @ti.kernel
    def update_magnetic_susceptibility(self):
        for I in ti.grouped(self.chi):
            self.chi[I] = self.k * (1 - \
                self.level_set.theta(self.level_set.phi[I]))

    def update_potential_function(self):
        start1 = time.perf_counter()
        self.poisson_solver.full_reinitialize(self.potential_function_strategy)
        end1 = time.perf_counter()

        start2 = time.perf_counter()
        self.poisson_solver.solve(self.iterations, self.verbose, 1e-18)
        end2 = time.perf_counter()

        print(f'\033[33msolve Psi, init cost {end1 - start1}s, solve cost {end2 - start2}s\033[0m')
        self.psi.copy_from(self.poisson_solver.x)

    @ti.kernel
    def update_magnetic_field(self):
        for k in ti.static(range(self.dim)):
            offset = ti.Vector.unit(self.dim, k)
            for I in ti.grouped(self.H[k]):
                self.H[k][I] = ti.zero(self.H[k][I])
                if I[k] - 1 >= 0 and I[k] < self.res[k]: 
                    self.H[k][I] -= (self.psi[I] - self.psi[I - offset]) / self.dx
                self.H[k][I] += self.H_ext[k][I] # H = H_ext - grad psi

    def solve_magnetic_force(self, dt):
        self.magnetic_force_strategy.scale_A = dt / (self.rho * self.dx * self.dx)
        self.magnetic_force_strategy.scale_b = 1 / self.dx

        start1 = time.perf_counter()
        self.poisson_solver.reinitialize(self.cell_type, self.magnetic_force_strategy)
        end1 = time.perf_counter()

        start2 = time.perf_counter()
        self.poisson_solver.solve(self.iterations, self.verbose)
        end2 = time.perf_counter()

        print(f'\033[33msolve magnetic force, init cost {end1 - start1}s, solve cost {end2 - start2}s\033[0m')
        self.pressure.copy_from(self.poisson_solver.x)

    @ti.kernel
    def apply_magnetic_force(self, dt : ti.f32):
        scale = dt / (self.rho * self.dx)

        for k in ti.static(range(self.dim)):
            for I in ti.grouped(self.cell_type):
                unit = ti.Vector.unit(self.dim, k)
                I_1 = I - unit                
                if self.is_fluid(I_1) or self.is_fluid(I):
                    if self.is_solid(I_1) or self.is_solid(I): self.velocity[k][I] = 0
                    # FLuid-Air
                    elif self.is_air(I):
                        H = self.H_interp(I + 0.5)
                        self.velocity[k][I] -= scale * ( \
                            (self.p0 + 1/2 * self.k * self.mu0 * H.dot(H)) \
                            - self.pressure[I_1])
                    # Air-Fluid
                    elif self.is_air(I_1):
                        H = self.H_interp(I_1 + 0.5)
                        self.velocity[k][I] -= scale * ( \
                            self.pressure[I] - \
                            (self.p0 + 1/2 * self.k * self.mu0 * H.dot(H)))
                    # Fluid-Fluid
                    else: self.velocity[k][I] -= scale * (self.pressure[I] - self.pressure[I_1])

    @ti.kernel
    def external_H(self): # external_H, unit tester
        for I in ti.grouped(self.H_ext[1]):
            self.H_ext[1][I] = 1.2e4

    def substep(self, dt):
        self.external_H()
        self.begin_substep(dt)

        self.update_magnetic_susceptibility() # Update the magnetic susceptibility on the grid
        self.update_potential_function() # Update the potential function Psi by solving the Poisson equation
        self.update_magnetic_field() # Update the magnetic field H

        if self.verbose:
            psi = np.max(self.psi.to_numpy())
            print(f'\033[36mMax psi: {psi}\033[0m')

        self.add_gravity(dt)
        self.enforce_boundary()
        self.extrap_velocity()
        self.enforce_boundary()

        self.solve_magnetic_force(dt)
        self.apply_magnetic_force(dt)
        self.extrap_velocity()
        self.enforce_boundary()

        self.end_substep(dt) 
