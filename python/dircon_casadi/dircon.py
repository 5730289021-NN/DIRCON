"""
Core DIRCON implementation using CasADi
"""

import casadi as ca
import numpy as np
from typing import List, Optional, Tuple, Callable
import matplotlib.pyplot as plt


class Dircon:
    """
    Direct Collocation with Constraints (DIRCON) trajectory optimization.
    
    This implementation uses CasADi for symbolic computation and NLP solving.
    It supports systems with kinematic constraints through contact forces.
    """
    
    def __init__(self, 
                 dynamics_fn: Callable,
                 n_states: int,
                 n_controls: int,
                 n_knots: int,
                 dt_min: float = 0.01,
                 dt_max: float = 0.5):
        """
        Initialize DIRCON trajectory optimizer.
        
        Args:
            dynamics_fn: Function that computes dynamics f(x, u, p) -> xdot
            n_states: Number of state variables
            n_controls: Number of control variables  
            n_knots: Number of knot points in trajectory
            dt_min: Minimum time step between knot points
            dt_max: Maximum time step between knot points
        """
        self.dynamics_fn = dynamics_fn
        self.n_states = n_states
        self.n_controls = n_controls
        self.n_knots = n_knots
        self.dt_min = dt_min
        self.dt_max = dt_max
        
        # Initialize optimization variables
        self.opti = ca.Opti()
        self._setup_variables()
        self._setup_constraints()
        
    def _setup_variables(self):
        """Setup optimization variables."""
        # State trajectory
        self.X = self.opti.variable(self.n_states, self.n_knots)
        
        # Control trajectory 
        self.U = self.opti.variable(self.n_controls, self.n_knots-1)
        
        # Time steps
        self.dt = self.opti.variable(self.n_knots-1)
        
        # Contact forces (if applicable)
        self.contact_forces = {}
        
    def _setup_constraints(self):
        """Setup basic trajectory constraints."""
        # Time step bounds
        for i in range(self.n_knots-1):
            self.opti.subject_to(self.dt[i] >= self.dt_min)
            self.opti.subject_to(self.dt[i] <= self.dt_max)
            
        # Dynamics constraints using Hermite-Simpson collocation
        for i in range(self.n_knots-1):
            # States at knot points
            x_i = self.X[:, i]
            x_i1 = self.X[:, i+1] 
            
            # Controls (piecewise constant)
            u_i = self.U[:, i]
            
            # Midpoint state
            x_mid = (x_i + x_i1) / 2
            
            # Dynamics at knot points and midpoint
            f_i = self.dynamics_fn(x_i, u_i)
            f_i1 = self.dynamics_fn(x_i1, u_i)
            f_mid = self.dynamics_fn(x_mid, u_i)
            
            # Hermite-Simpson integration constraint
            constraint = x_i1 - x_i - self.dt[i]/6 * (f_i + 4*f_mid + f_i1)
            self.opti.subject_to(constraint == 0)
    
    def add_contact_constraint(self, 
                             contact_point_fn: Callable,
                             contact_jacobian_fn: Callable,
                             constraint_type: str = "position",
                             active_phases: Optional[List[int]] = None):
        """
        Add contact constraints to the trajectory.
        
        Args:
            contact_point_fn: Function to compute contact point position
            contact_jacobian_fn: Function to compute contact Jacobian
            constraint_type: Type of constraint ("position", "velocity", or "acceleration")
            active_phases: List of knot points where constraint is active
        """
        if active_phases is None:
            active_phases = list(range(self.n_knots))
            
        # Add contact force variables
        n_contact_forces = 2  # Assuming 2D contact forces (normal + tangential)
        contact_key = f"contact_{len(self.contact_forces)}"
        self.contact_forces[contact_key] = self.opti.variable(n_contact_forces, len(active_phases))
        
        for idx, knot in enumerate(active_phases):
            x = self.X[:, knot]
            
            if constraint_type == "position":
                # Position constraint: contact point at origin
                pos = contact_point_fn(x)
                self.opti.subject_to(pos[2] == 0)  # z-position = 0 (ground contact)
                
            elif constraint_type == "velocity":
                # Velocity constraint: contact point velocity = 0
                J = contact_jacobian_fn(x)
                x_dot = self.X[self.n_states//2:, knot]  # Velocity states
                vel = J @ x_dot
                self.opti.subject_to(vel == 0)
                
    def add_initial_constraint(self, x0: np.ndarray):
        """Add initial state constraint."""
        self.opti.subject_to(self.X[:, 0] == x0)
        
    def add_final_constraint(self, xf: np.ndarray):
        """Add final state constraint.""" 
        self.opti.subject_to(self.X[:, -1] == xf)
        
    def add_periodic_constraint(self, state_indices: Optional[List[int]] = None):
        """Add periodic boundary constraints."""
        if state_indices is None:
            state_indices = list(range(self.n_states))
            
        for idx in state_indices:
            self.opti.subject_to(self.X[idx, 0] == self.X[idx, -1])
            
    def add_state_bounds(self, x_min: np.ndarray, x_max: np.ndarray):
        """Add state bounds."""
        for i in range(self.n_knots):
            self.opti.subject_to(self.X[:, i] >= x_min)
            self.opti.subject_to(self.X[:, i] <= x_max)
            
    def add_control_bounds(self, u_min: np.ndarray, u_max: np.ndarray):
        """Add control bounds."""
        for i in range(self.n_knots-1):
            self.opti.subject_to(self.U[:, i] >= u_min)
            self.opti.subject_to(self.U[:, i] <= u_max)
            
    def add_running_cost(self, 
                        Q: Optional[np.ndarray] = None,
                        R: Optional[np.ndarray] = None):
        """Add quadratic running cost."""
        if Q is None:
            Q = np.eye(self.n_states)
        if R is None:
            R = np.eye(self.n_controls)
            
        cost = 0
        for i in range(self.n_knots-1):
            cost += self.X[:, i].T @ Q @ self.X[:, i] * self.dt[i]
            cost += self.U[:, i].T @ R @ self.U[:, i] * self.dt[i]
            
        self.opti.minimize(cost)
        
    def set_initial_guess(self, 
                         X_guess: np.ndarray,
                         U_guess: np.ndarray, 
                         dt_guess: Optional[np.ndarray] = None):
        """Set initial guess for optimization variables."""
        self.opti.set_initial(self.X, X_guess)
        self.opti.set_initial(self.U, U_guess)
        
        if dt_guess is not None:
            self.opti.set_initial(self.dt, dt_guess)
        else:
            self.opti.set_initial(self.dt, 0.1 * np.ones(self.n_knots-1))
            
    def solve(self, solver: str = "ipopt", solver_options: Optional[dict] = None):
        """Solve the trajectory optimization problem."""
        if solver_options is None:
            solver_options = {
                'ipopt.tol': 1e-6,
                'ipopt.max_iter': 500,
                'print_time': True
            }
            
        self.opti.solver(solver, solver_options)
        
        try:
            sol = self.opti.solve()
            self.solution = {
                'X': sol.value(self.X),
                'U': sol.value(self.U), 
                'dt': sol.value(self.dt),
                'success': True,
                'cost': sol.value(self.opti.f)
            }
            return self.solution
        except Exception as e:
            print(f"Optimization failed: {e}")
            self.solution = {
                'X': None,
                'U': None,
                'dt': None, 
                'success': False,
                'cost': None
            }
            return self.solution
            
    def get_trajectory(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get optimized trajectory."""
        if not hasattr(self, 'solution') or not self.solution['success']:
            raise ValueError("No successful solution available")
            
        # Compute time vector
        t = np.zeros(self.n_knots)
        t[1:] = np.cumsum(self.solution['dt'])
        
        return t, self.solution['X'], self.solution['U']
        
    def plot_trajectory(self, state_labels: Optional[List[str]] = None,
                       control_labels: Optional[List[str]] = None):
        """Plot optimized trajectory."""
        if not hasattr(self, 'solution') or not self.solution['success']:
            raise ValueError("No successful solution available")
            
        t, X, U = self.get_trajectory()
        
        # Plot states
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        for i in range(self.n_states):
            label = state_labels[i] if state_labels else f"x_{i}"
            ax1.plot(t, X[i, :], label=label)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('States')
        ax1.legend()
        ax1.grid(True)
        ax1.set_title('State Trajectory')
        
        # Plot controls
        t_control = t[:-1] + self.solution['dt']/2  # Midpoint of intervals
        for i in range(self.n_controls):
            label = control_labels[i] if control_labels else f"u_{i}"
            ax2.plot(t_control, U[i, :], 'o-', label=label)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Controls')
        ax2.legend()
        ax2.grid(True)
        ax2.set_title('Control Trajectory')
        
        plt.tight_layout()
        plt.show()