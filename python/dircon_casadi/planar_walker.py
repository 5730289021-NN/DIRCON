"""
Planar Walker Robot Model for DIRCON trajectory optimization
"""

import casadi as ca
import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt


class PlanarWalker:
    """
    Planar walker robot model with two legs and knee joints.
    
    State: [x, z, theta, theta_left, theta_hip, theta_right, 
            x_dot, z_dot, theta_dot, theta_left_dot, theta_hip_dot, theta_right_dot]
    
    Controls: [tau_hip, tau_left_knee, tau_right_knee]
    """
    
    def __init__(self):
        """Initialize planar walker parameters."""
        # Physical parameters (from URDF)
        self.m_hip = 10.0         # Hip mass
        self.m_thigh = 2.5        # Upper leg mass  
        self.m_shank = 2.5        # Lower leg mass
        self.l_thigh = 0.5        # Upper leg length
        self.l_shank = 0.5        # Lower leg length
        self.g = 9.81            # Gravity
        
        # Total mass
        self.m_total = self.m_hip + 2*(self.m_thigh + self.m_shank)
        
        # State and control dimensions
        self.n_q = 6  # Positions: [x, z, theta, theta_left, theta_hip, theta_right]
        self.n_v = 6  # Velocities
        self.n_states = self.n_q + self.n_v
        self.n_controls = 3  # [tau_hip, tau_left_knee, tau_right_knee]
        
        # Create symbolic variables
        self._create_symbolic_variables()
        self._create_dynamics()
        
    def _create_symbolic_variables(self):
        """Create symbolic variables for states and controls."""
        # Generalized coordinates
        self.q = ca.MX.sym('q', self.n_q)
        self.q_dot = ca.MX.sym('q_dot', self.n_v)
        self.x_sym = ca.vertcat(self.q, self.q_dot)  # Full state vector
        
        # Controls  
        self.u_sym = ca.MX.sym('u', self.n_controls)
        
        # Individual state components for clarity
        self.x_pos = self.q[0]      # Horizontal position
        self.z_pos = self.q[1]      # Vertical position
        self.theta = self.q[2]      # Hip orientation
        self.theta_left = self.q[3]  # Left knee angle
        self.theta_hip = self.q[4]   # Hip joint angle  
        self.theta_right = self.q[5] # Right knee angle
        
        # Velocities
        self.x_dot = self.q_dot[0]
        self.z_dot = self.q_dot[1] 
        self.theta_dot = self.q_dot[2]
        self.theta_left_dot = self.q_dot[3]
        self.theta_hip_dot = self.q_dot[4]
        self.theta_right_dot = self.q_dot[5]
        
        # Controls
        self.tau_hip = self.u_sym[0]
        self.tau_left = self.u_sym[1]
        self.tau_right = self.u_sym[2]
        
    def _create_dynamics(self):
        """Create symbolic dynamics using CasADi."""
        # This is a simplified model - in practice you'd derive the full
        # manipulator equations from Lagrangian mechanics
        
        # For now, create a simplified dynamics model
        # Mass matrix (simplified)
        M = ca.diag([self.m_total, self.m_total, 1.0, 0.1, 0.1, 0.1])
        
        # Gravity vector
        g_vec = ca.vertcat(0, -self.m_total * self.g, 0, 0, 0, 0)
        
        # Control input matrix
        B = ca.vertcat(
            ca.horzcat(0, 0, 0),      # x
            ca.horzcat(0, 0, 0),      # z  
            ca.horzcat(0, 0, 0),      # theta
            ca.horzcat(0, 1, 0),      # left knee
            ca.horzcat(1, 0, 0),      # hip
            ca.horzcat(0, 0, 1)       # right knee
        )
        
        # Equations of motion: M*q_ddot + C*q_dot + G = B*u
        # Solving for q_ddot: q_ddot = M^-1 * (B*u - G)
        q_ddot = ca.solve(M, B @ self.u_sym - g_vec)
        
        # Full state derivative
        self.x_dot_sym = ca.vertcat(self.q_dot, q_ddot)
        
        # Create dynamics function
        self.dynamics_fn = ca.Function('dynamics', 
                                     [self.x_sym, self.u_sym], 
                                     [self.x_dot_sym])
                                     
    def get_dynamics_function(self):
        """Get dynamics function for use with DIRCON."""
        return self.dynamics_fn
        
    def get_foot_position(self, q: ca.MX, foot: str = 'left') -> ca.MX:
        """
        Compute foot position in world coordinates.
        
        Args:
            q: Generalized coordinates
            foot: 'left' or 'right'
        """
        x, z, theta, theta_left, theta_hip, theta_right = ca.vertsplit(q)
        
        if foot == 'left':
            # Left leg kinematics
            # Hip position
            hip_x = x
            hip_z = z
            
            # Upper leg end (knee) position  
            knee_x = hip_x + self.l_thigh * ca.sin(theta)
            knee_z = hip_z - self.l_thigh * ca.cos(theta)
            
            # Foot position
            foot_angle = theta + theta_left
            foot_x = knee_x + self.l_shank * ca.sin(foot_angle)
            foot_z = knee_z - self.l_shank * ca.cos(foot_angle)
            
        else:  # right foot
            # Right leg kinematics
            hip_x = x
            hip_z = z
            
            # Upper leg end (knee) position
            upper_leg_angle = theta + theta_hip
            knee_x = hip_x + self.l_thigh * ca.sin(upper_leg_angle)
            knee_z = hip_z - self.l_thigh * ca.cos(upper_leg_angle)
            
            # Foot position
            foot_angle = upper_leg_angle + theta_right
            foot_x = knee_x + self.l_shank * ca.sin(foot_angle)
            foot_z = knee_z - self.l_shank * ca.cos(foot_angle)
            
        return ca.vertcat(foot_x, foot_z, 0)  # 3D point with z=0
        
    def get_foot_jacobian(self, q: ca.MX, foot: str = 'left') -> ca.MX:
        """
        Compute foot position Jacobian.
        
        Args:
            q: Generalized coordinates
            foot: 'left' or 'right'
        """
        foot_pos = self.get_foot_position(q, foot)
        J = ca.jacobian(foot_pos, q)
        return J
        
    def create_foot_position_function(self, foot: str = 'left'):
        """Create CasADi function for foot position."""
        q_sym = ca.MX.sym('q', self.n_q)
        pos = self.get_foot_position(q_sym, foot)
        return ca.Function(f'{foot}_foot_pos', [q_sym], [pos])
        
    def create_foot_jacobian_function(self, foot: str = 'left'):
        """Create CasADi function for foot Jacobian."""
        q_sym = ca.MX.sym('q', self.n_q) 
        J = self.get_foot_jacobian(q_sym, foot)
        return ca.Function(f'{foot}_foot_jac', [q_sym], [J])
        
    def get_default_initial_state(self) -> np.ndarray:
        """Get reasonable initial state for optimization."""
        return np.array([
            0.0,     # x position
            1.0,     # z position (standing height)
            0.0,     # hip orientation
            0.1,     # left knee angle
            0.0,     # hip joint angle
            0.1,     # right knee angle
            0.0,     # x velocity
            0.0,     # z velocity  
            0.0,     # angular velocity
            0.0,     # left knee velocity
            0.0,     # hip velocity
            0.0      # right knee velocity
        ])
        
    def get_state_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get reasonable state bounds."""
        x_min = np.array([
            -10.0,   # x position
            0.5,     # z position
            -np.pi,  # hip orientation
            0.0,     # left knee (no hyperextension)
            -np.pi,  # hip joint
            0.0,     # right knee (no hyperextension)
            -5.0,    # x velocity
            -5.0,    # z velocity
            -10.0,   # angular velocity
            -10.0,   # left knee velocity
            -10.0,   # hip velocity
            -10.0    # right knee velocity
        ])
        
        x_max = np.array([
            10.0,    # x position
            2.0,     # z position  
            np.pi,   # hip orientation
            np.pi,   # left knee
            np.pi,   # hip joint
            np.pi,   # right knee
            5.0,     # x velocity
            5.0,     # z velocity
            10.0,    # angular velocity
            10.0,    # left knee velocity
            10.0,    # hip velocity
            10.0     # right knee velocity
        ])
        
        return x_min, x_max
        
    def get_control_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get control bounds."""
        u_min = np.array([-100.0, -100.0, -100.0])  # Torque limits
        u_max = np.array([100.0, 100.0, 100.0])
        return u_min, u_max
        
    def plot_robot(self, q: np.ndarray, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot the robot configuration.
        
        Args:
            q: Configuration vector [x, z, theta, theta_left, theta_hip, theta_right]
            ax: Matplotlib axes (optional)
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
        # Extract joint angles
        x, z, theta, theta_left, theta_hip, theta_right = q
        
        # Hip position
        hip_pos = np.array([x, z])
        
        # Left leg
        left_knee_pos = hip_pos + self.l_thigh * np.array([np.sin(theta), -np.cos(theta)])
        left_foot_angle = theta + theta_left
        left_foot_pos = left_knee_pos + self.l_shank * np.array([np.sin(left_foot_angle), -np.cos(left_foot_angle)])
        
        # Right leg
        right_upper_angle = theta + theta_hip
        right_knee_pos = hip_pos + self.l_thigh * np.array([np.sin(right_upper_angle), -np.cos(right_upper_angle)])
        right_foot_angle = right_upper_angle + theta_right
        right_foot_pos = right_knee_pos + self.l_shank * np.array([np.sin(right_foot_angle), -np.cos(right_foot_angle)])
        
        # Plot links
        # Left leg
        ax.plot([hip_pos[0], left_knee_pos[0]], [hip_pos[1], left_knee_pos[1]], 'r-', linewidth=3, label='Left leg')
        ax.plot([left_knee_pos[0], left_foot_pos[0]], [left_knee_pos[1], left_foot_pos[1]], 'r-', linewidth=3)
        
        # Right leg  
        ax.plot([hip_pos[0], right_knee_pos[0]], [hip_pos[1], right_knee_pos[1]], 'b-', linewidth=3, label='Right leg')
        ax.plot([right_knee_pos[0], right_foot_pos[0]], [right_knee_pos[1], right_foot_pos[1]], 'b-', linewidth=3)
        
        # Plot joints
        ax.plot(hip_pos[0], hip_pos[1], 'go', markersize=8, label='Hip')
        ax.plot(left_knee_pos[0], left_knee_pos[1], 'ro', markersize=6)
        ax.plot(right_knee_pos[0], right_knee_pos[1], 'bo', markersize=6)
        ax.plot(left_foot_pos[0], left_foot_pos[1], 'rs', markersize=8)
        ax.plot(right_foot_pos[0], right_foot_pos[1], 'bs', markersize=8)
        
        # Ground
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, label='Ground')
        
        ax.set_xlim(x-1.5, x+1.5)
        ax.set_ylim(-0.2, 1.5)
        ax.set_xlabel('X position (m)')
        ax.set_ylabel('Z position (m)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return ax
        
    def animate_trajectory(self, t: np.ndarray, X: np.ndarray, dt_anim: float = 0.05):
        """
        Animate the robot trajectory.
        
        Args:
            t: Time vector
            X: State trajectory (n_states x n_knots)
            dt_anim: Animation time step
        """
        from matplotlib.animation import FuncAnimation
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        def animate(frame):
            ax.clear()
            
            # Interpolate state at current time
            t_current = frame * dt_anim
            if t_current > t[-1]:
                t_current = t[-1]
                
            # Find interpolation indices
            idx = np.searchsorted(t, t_current)
            if idx == 0:
                q_current = X[:self.n_q, 0]
            elif idx >= len(t):
                q_current = X[:self.n_q, -1]
            else:
                # Linear interpolation
                alpha = (t_current - t[idx-1]) / (t[idx] - t[idx-1])
                q_current = (1-alpha) * X[:self.n_q, idx-1] + alpha * X[:self.n_q, idx]
                
            self.plot_robot(q_current, ax)
            ax.set_title(f'Time: {t_current:.2f}s')
            
            # Plot trajectory
            x_traj = X[0, :idx+1] if idx < len(t) else X[0, :]
            z_traj = X[1, :idx+1] if idx < len(t) else X[1, :]
            ax.plot(x_traj, z_traj, 'k--', alpha=0.5, label='Hip trajectory')
            
        n_frames = int(t[-1] / dt_anim) + 1
        anim = FuncAnimation(fig, animate, frames=n_frames, interval=int(dt_anim*1000), repeat=True)
        
        plt.show()
        return anim