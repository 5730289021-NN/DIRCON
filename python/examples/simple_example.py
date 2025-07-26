#!/usr/bin/env python3
"""
Simple example of DIRCON with Planar Walker

This script demonstrates the basic usage of the Python CasADi DIRCON implementation
for trajectory optimization of a planar walking robot.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the dircon_casadi module to path
sys.path.append('..')
from dircon_casadi import Dircon, PlanarWalker


def balance_example():
    """
    Example 1: Balance recovery from perturbation
    """
    print("="*50)
    print("DIRCON Example 1: Balance Recovery")
    print("="*50)
    
    # Create planar walker
    walker = PlanarWalker()
    print(f"Robot: {walker.n_states} states, {walker.n_controls} controls")
    
    # DIRCON parameters
    n_knots = 21
    dt_min, dt_max = 0.01, 0.2
    
    # Create optimizer
    dircon = Dircon(
        dynamics_fn=walker.get_dynamics_function(),
        n_states=walker.n_states,
        n_controls=walker.n_controls,
        n_knots=n_knots,
        dt_min=dt_min,
        dt_max=dt_max
    )
    
    # Initial state (perturbed)
    x0 = walker.get_default_initial_state()
    x0[0] = -0.1  # Forward displacement
    x0[2] = 0.1   # Tilt
    x0[6] = 0.2   # Forward velocity
    
    # Final state (balanced)
    xf = walker.get_default_initial_state()
    
    # Add constraints
    dircon.add_initial_constraint(x0)
    dircon.add_final_constraint(xf)
    
    x_min, x_max = walker.get_state_bounds()
    u_min, u_max = walker.get_control_bounds()
    dircon.add_state_bounds(x_min, x_max)
    dircon.add_control_bounds(u_min, u_max)
    
    # Add cost
    Q = np.diag([1, 1, 10, 1, 1, 1, 0.1, 0.1, 1, 0.1, 0.1, 0.1])
    R = 0.01 * np.eye(walker.n_controls)
    dircon.add_running_cost(Q, R)
    
    # Initial guess
    X_guess = np.zeros((walker.n_states, n_knots))
    U_guess = np.zeros((walker.n_controls, n_knots-1))
    
    for i in range(n_knots):
        alpha = i / (n_knots - 1)
        X_guess[:, i] = (1 - alpha) * x0 + alpha * xf
    
    dircon.set_initial_guess(X_guess, U_guess)
    
    # Solve
    print("Solving balance recovery...")
    solution = dircon.solve(solver='ipopt', solver_options={
        'ipopt.tol': 1e-4,
        'ipopt.max_iter': 200,
        'ipopt.print_level': 2
    })
    
    if solution['success']:
        print(f"✅ Success! Cost: {solution['cost']:.4f}, Time: {np.sum(solution['dt']):.3f}s")
        
        # Plot results
        state_labels = ['x', 'z', 'θ', 'θ_L', 'θ_H', 'θ_R', 
                       'ẋ', 'ż', 'θ̇', 'θ̇_L', 'θ̇_H', 'θ̇_R']
        control_labels = ['τ_hip', 'τ_left', 'τ_right']
        
        dircon.plot_trajectory(state_labels, control_labels)
        
        # Visualize robot motion
        t, X, U = dircon.get_trajectory()
        visualize_motion(walker, t, X)
        
        return True
    else:
        print("❌ Optimization failed")
        return False


def visualize_motion(walker, t, X):
    """
    Visualize robot motion at key time points
    """
    n_frames = 5
    time_indices = np.linspace(0, len(t)-1, n_frames, dtype=int)
    
    fig, axes = plt.subplots(1, n_frames, figsize=(15, 3))
    
    for i, idx in enumerate(time_indices):
        q_i = X[:walker.n_q, idx]
        walker.plot_robot(q_i, axes[i])
        axes[i].set_title(f't = {t[idx]:.2f}s')
        
        # Add COM trajectory
        axes[i].plot(X[0, :idx+1], X[1, :idx+1], 'k--', alpha=0.7, linewidth=2)
    
    plt.suptitle('Robot Motion Sequence', fontsize=14)
    plt.tight_layout()
    plt.show()


def test_robot_model():
    """
    Test the robot model and visualization
    """
    print("="*50)
    print("Testing Robot Model")
    print("="*50)
    
    walker = PlanarWalker()
    
    # Test configurations
    configs = [
        ("Standing", walker.get_default_initial_state()[:walker.n_q]),
        ("Left step", np.array([0.0, 0.9, 0.2, 0.3, -0.4, 0.1])),
        ("Right step", np.array([0.0, 0.9, -0.2, 0.1, 0.4, 0.3])),
        ("Crouched", np.array([0.0, 0.7, 0.0, 0.8, 0.0, 0.8]))
    ]
    
    fig, axes = plt.subplots(1, len(configs), figsize=(12, 3))
    
    for i, (name, q) in enumerate(configs):
        walker.plot_robot(q, axes[i])
        axes[i].set_title(name)
        
        # Test foot positions
        left_foot = walker.create_foot_position_function('left')
        right_foot = walker.create_foot_position_function('right')
        
        left_pos = left_foot(q)
        right_pos = right_foot(q)
        
        print(f"{name}:")
        print(f"  Left foot:  [{left_pos[0]:.3f}, {left_pos[1]:.3f}]")
        print(f"  Right foot: [{right_pos[0]:.3f}, {right_pos[1]:.3f}]")
    
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to run examples
    """
    print("DIRCON Python CasADi Implementation")
    print("Planar Walker Trajectory Optimization")
    print("="*50)
    
    # Test robot model
    test_robot_model()
    
    # Run balance example
    success = balance_example()
    
    if success:
        print("\n🎉 Example completed successfully!")
        print("\nNext steps:")
        print("- Open the Jupyter notebook for interactive exploration")
        print("- Modify robot parameters and cost functions")
        print("- Try implementing walking gaits with contact constraints")
    else:
        print("\n⚠️  Example had issues - check solver installation")


if __name__ == "__main__":
    main()