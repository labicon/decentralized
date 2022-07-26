#!/usr/bin/env python

"""Dynamics module to simulate dynamical systems with examples"""

import abc

import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import approx_fprime
import sympy as sym

from .util import split_agents


class DynamicalModel(abc.ABC):
    """Simulation of a dynamical model to be applied in the iLQR solution."""

    _id = 0

    def __init__(self, n_x, n_u, dt, id=None):
        if not id:
            id = DynamicalModel._id
            DynamicalModel._id += 1

        self.n_x = n_x
        self.n_u = n_u
        self.dt = dt
        self.id = id
        self.NX_EYE = np.eye(self.n_x, dtype=np.float32)

    def __call__(self, x, u):
        """Zero-order hold to integrate continuous dynamics f"""
        # return x + self.f(x, u) * self.dt
        # Single RK4 integration of continuous dynamics.

        if dT is None:
            dT = 0.1*self.dt

        t = 0.0
        x = x.copy()
        while t < self.dt - 1e-8:
            step = min(dT, self.dt - t)

            k1 = step * np.asarray(self.f(x, u),dtype='float64')
            k2 = step * np.asarray(self.f(x + 0.5 * k1, u + 0.5 * step),dtype='float64')
            k3 = step * np.asarray(self.f(x + 0.5 * k2, u + 0.5 * step),dtype='float64')
            k4 = step * np.asarray(self.f(x + k3, u + step),dtype='float64')
            
            x += x + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0 
            t += step
        return x

    @staticmethod
    @abc.abstractmethod
    def f():
        """Continuous derivative of dynamics with respect to time"""
        pass

    @abc.abstractmethod
    def linearize():
        """Linearization that computes jacobian at the current operating point"""
        pass

    @classmethod
    def _reset_ids(cls):
        cls._id = 0

    def __repr__(self):
        return f"{type(self).__name__}(n_x: {self.n_x}, n_u: {self.n_u}, id: {self.id})"


class SymbolicModel(DynamicalModel):
    """Mix-in for analytical linearization"""

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['A_num']
        del state['B_num']
        del state['_f']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__init__(self.dt)

    def f(self, x, u):
        return self._f(x, u)

    def linearize(self, x, u):
        """Linearization via numerical Jacobians A_num and B_num with Euler method"""
        return np.eye(x.size) + self.dt * self.A_num(x, u), self.dt * self.B_num(x, u)


class MultiDynamicalModel(DynamicalModel):
    """Encompasses the dynamical simulation and linearization for a collection of
    DynamicalModel's
    """

    def __init__(self, submodels):
        self.submodels = submodels
        self.n_players = len(submodels)

        self.x_dims = [submodel.n_x for submodel in submodels]
        self.u_dims = [submodel.n_u for submodel in submodels]
        self.ids = [submodel.id for submodel in submodels]

        super().__init__(sum(self.x_dims), sum(self.u_dims), submodels[0].dt, -1)

    def f(self, x, u):
        """Integrate the dynamics for the combined decoupled dynamical model"""
        return np.concatenate(
            [
                submodel.f(xi.flatten(), ui.flatten())
                for submodel, xi, ui in zip(self.submodels, *self.partition(x, u))
            ]
        )

    def linearize(self, x, u):
        sub_linearizations = [
            submodel.linearize(xi.flatten(), ui.flatten())
            for submodel, xi, ui in zip(self.submodels, *self.partition(x, u))
        ]

        sub_As = [AB[0] for AB in sub_linearizations]
        sub_Bs = [AB[1] for AB in sub_linearizations]

        return block_diag(*sub_As), block_diag(*sub_Bs)

    def partition(self, x, u):
        """Helper to split up the states and control for each subsystem"""
        return split_agents(x, self.x_dims), split_agents(u, self.u_dims)

    def split(self, graph):
        """Split this model into submodels dictated by the interaction graph"""
        split_dynamics = []
        for problem in graph:
            split_dynamics.append(
                MultiDynamicalModel(
                    [model for model in self.submodels if model.id in graph[problem]]
                )
            )

        return split_dynamics

    def __repr__(self):
        sub_reprs = ",\n\t".join([repr(submodel) for submodel in self.submodels])
        return f"MultiDynamicalModel(\n\t{sub_reprs}\n)"


class DoubleIntDynamics4D(DynamicalModel):
    def __init__(self, dt, *args, **kwargs):
        super().__init__(4, 2, dt, *args, **kwargs)

    @staticmethod
    def f(x, u):
        *_, vx, vy = x
        ax, ay = u
        return np.stack([vx, vy, ax, ay])

    def linearize(self, *_):
        A = np.array(
            [[1, 0, self.dt, 0], [0, 1, 0, self.dt], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        B = self.dt * np.array([[0, 0], [0, 0], [1, 0], [0, 1]])

        return A, B


class CarDynamics3D(DynamicalModel):
    def __init__(self, dt, *args, **kwargs):
        super().__init__(3, 2, dt, *args, **kwargs)

    @staticmethod
    def f(x, u):
        *_, theta = x
        v, omega = u
        return np.stack([v * np.cos(theta), v * np.sin(theta), omega])

    def linearize(self, x, u):

        v = u[0]
        theta = x[2]

        A = np.array(
            [
                [1, 0, -v * self.dt * np.sin(theta)],
                [0, 1, v * self.dt * np.cos(theta)],
                [0, 0, 1],
            ]
        )
        B = self.dt * np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])

        return A, B


class UnicycleDynamics4D(SymbolicModel):
    def __init__(self, dt, *args, **kwargs):
        super().__init__(4, 2, dt, *args, **kwargs)

        p_x, p_y, v, theta, omega, a = sym.symbols("p_x p_y v theta omega a")
        x = sym.Matrix([p_x, p_y, v, theta])
        u = sym.Matrix([a, omega])

        x_dot = sym.Matrix(
            [
                x[2] * sym.cos(x[3]),
                x[2] * sym.sin(x[3]),
                u[0],
                u[1],
            ]
        )

        A = x_dot.jacobian(x)
        B = x_dot.jacobian(u)

        self._f = sym.lambdify((x, u), sym.Array(x_dot)[:, 0])
        self.A_num = sym.lambdify((x, u), A)
        self.B_num = sym.lambdify((x, u), B)


class BikeDynamics5D(SymbolicModel):
    def __init__(self, dt, *args, **kwargs):
        super().__init__(5, 2, dt, *args, **kwargs)

        p_x, p_y, theta, v, phi, a, rho = sym.symbols("p_x p_y theta v phi a rho")
        x = sym.Matrix([p_x, p_y, v, theta, phi])
        u = sym.Matrix([a, rho])

        x_dot = sym.Matrix(
            [
                x[2] * sym.cos(x[3]),
                x[2] * sym.sin(x[3]),
                u[0],
                x[2] * sym.tan(x[4]),
                u[1],
            ]
        )

        A = x_dot.jacobian(x)
        B = x_dot.jacobian(u)

        self._f = sym.lambdify((x, u), sym.Array(x_dot)[:, 0])
        self.A_num = sym.lambdify((x, u), A)
        self.B_num = sym.lambdify((x, u), B)


class QuadcopterDynamics12D(SymbolicModel):
    def __init__(self, dt, *args, **kwargs):
        super().__init__(12, 4, dt, *args, **kwargs)

        # # components of position (meters)
        o_x, o_y, o_z = sym.symbols("o_x, o_y, o_z")

        # yaw, pitch, and roll angles (radians)
        psi, theta, phi = sym.symbols("psi, theta, phi")

        # components of linear velocity (meters / second)
        v_x, v_y, v_z = sym.symbols("v_x, v_y, v_z")

        # components of angular velocity (radians / second)
        w_x, w_y, w_z = sym.symbols("w_x, w_y, w_z")

        # components of net rotor torque
        tau_x, tau_y, tau_z = sym.symbols("tau_x, tau_y, tau_z")

        # net rotor force
        f_z = sym.symbols("f_z")

        x = sym.Matrix(
            [o_x, o_y, o_z, psi, theta, phi, v_x, v_y, v_z, w_x, w_y, w_z]
        )  # state variables
        u = sym.Matrix([tau_x, tau_y, tau_z, f_z])  # input variables

        m = sym.nsimplify(0.0315)  # mass of a Crazyflie drone

        # Principle moments of inertia of a Crazyflie drone
        J_x = sym.nsimplify(1.7572149113694408e-05)
        J_y = sym.nsimplify(1.856979710568617e-05)
        J_z = sym.nsimplify(2.7159794713754086e-05)

        # Acceleration of gravity
        g = 9.81

        # Linear and angular velocity vectors (in body frame)
        v_01in1 = sym.Matrix([[v_x], [v_y], [v_z]])
        w_01in1 = sym.Matrix([[w_x], [w_y], [w_z]])

        # Create moment of inertia matrix (in coordinates of the body frame).
        J_in1 = sym.diag(J_x, J_y, J_z)

        # Z-Y-X rotation sequence
        Rz = sym.Matrix(
            [
                [sym.cos(psi), -sym.sin(psi), 0],
                [sym.sin(psi), sym.cos(psi), 0],
                [0, 0, 1],
            ]
        )

        Ry = sym.Matrix(
            [
                [sym.cos(theta), 0, sym.sin(theta)],
                [0, 1, 0],
                [-sym.sin(theta), 0, sym.cos(theta)],
            ]
        )

        Rx = sym.Matrix(
            [
                [1, 0, 0],
                [0, sym.cos(phi), -sym.sin(phi)],
                [0, sym.sin(phi), sym.cos(phi)],
            ]
        )

        R_1in0 = Rz * Ry * Rx

        # Mapping from angular velocity to angular rates
        # Compute the invserse of the mapping first:
        Ninv = sym.Matrix.hstack(
            (Ry * Rx).T * sym.Matrix([[0], [0], [1]]),
            (Rx).T * sym.Matrix([[0], [1], [0]]),
            sym.Matrix([[1], [0], [0]]),
        )
        N = sym.simplify(Ninv.inv())  # this matrix N is what we actually want

        # forces in world frame
        f_in1 = R_1in0.T * sym.Matrix([[0], [0], [-m * g]]) + sym.Matrix(
            [[0], [0], [f_z]]
        )

        # torques in world frame
        tau_in1 = sym.Matrix([[tau_x], [tau_y], [tau_z]])

        # Full EOM:
        f_sym = sym.Matrix.vstack(
            R_1in0 * v_01in1,
            N * w_01in1,
            (1 / m) * (f_in1 - w_01in1.cross(m * v_01in1)),
            J_in1.inv() * (tau_in1 - w_01in1.cross(J_in1 * w_01in1)),
        )

        A = f_sym.jacobian(x)
        B = f_sym.jacobian(u)

        self._f = sym.lambdify((x, u), sym.Array(f_sym)[:, 0])
        self.A_num = sym.lambdify((x, u), A)
        self.B_num = sym.lambdify((x, u), B)


# Based off of https://github.com/anassinator/ilqr/blob/master/ilqr/dynamics.py
def linearize_finite_difference(f, x, u):
    """Linearization using finite difference"""

    n_x = x.size
    jac_eps = np.sqrt(np.finfo(float).eps)

    A = np.vstack([approx_fprime(x, lambda x: f(x, u)[i], jac_eps) for i in range(n_x)])
    B = np.vstack([approx_fprime(u, lambda u: f(x, u)[i], jac_eps) for i in range(n_x)])

    return A, B
