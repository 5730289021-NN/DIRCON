# DIRCON Python CasADi Implementation

A modern Python implementation of the DIRCON (Direct Collocation with Constraints) trajectory optimization algorithm using CasADi, with a focus on the planar walker robot.

## Overview

This implementation provides a clean, educational Python/CasADi version of the DIRCON algorithm described in:

> Michael Posa, Scott Kuindersma, Russ Tedrake. "Optimization and Stabilization of Trajectories for Constrained Dynamical Systems." Proceedings of the International Conference on Robotics and Automation (ICRA), 2016.

## Features

- **Modern Python Implementation**: Clean, readable code using CasADi for symbolic computation
- **Educational Focus**: Comprehensive Jupyter notebooks with step-by-step explanations
- **Planar Walker Robot**: Complete model with kinematics, dynamics, and visualization
- **DIRCON Algorithm**: Full implementation with contact constraints and collocation
- **Interactive Examples**: Balance recovery, walking gaits, and custom trajectory optimization

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/5730289021-NN/DIRCON.git
cd DIRCON
git checkout python-casadi-migration
```

2. **Install dependencies**:
```bash
cd python
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python examples/simple_example.py
```

## Quick Start

### Run the simple example
```bash
cd python
python examples/simple_example.py
```

### Launch Jupyter notebook tutorial
```bash
cd python
jupyter notebook notebooks/dircon_tutorial.ipynb
```

## Project Structure

```
python/
├── dircon_casadi/           # Main Python package
│   ├── __init__.py         # Package initialization
│   ├── dircon.py           # Core DIRCON implementation
│   └── planar_walker.py    # Planar walker robot model
├── notebooks/              # Jupyter notebooks
│   └── dircon_tutorial.ipynb  # Complete tutorial
├── examples/               # Example scripts
│   └── simple_example.py   # Basic usage example
└── requirements.txt        # Python dependencies
```

## Usage Examples

### Basic DIRCON Setup

```python
from dircon_casadi import Dircon, PlanarWalker

# Create robot model
walker = PlanarWalker()

# Create DIRCON optimizer
dircon = Dircon(
    dynamics_fn=walker.get_dynamics_function(),
    n_states=walker.n_states,
    n_controls=walker.n_controls,
    n_knots=21,
    dt_min=0.01,
    dt_max=0.2
)

# Add constraints and costs
dircon.add_initial_constraint(x0)
dircon.add_final_constraint(xf)
dircon.add_running_cost(Q, R)

# Solve
solution = dircon.solve()
```

### Robot Visualization

```python
# Plot robot configuration
walker.plot_robot(q)

# Animate trajectory
walker.animate_trajectory(t, X)

# Plot optimization results
dircon.plot_trajectory()
```

## Examples Included

1. **Balance Recovery**: Robot recovers from perturbations
2. **Robot Kinematics**: Forward kinematics and Jacobians
3. **Dynamics Testing**: Verify robot dynamics implementation
4. **Trajectory Optimization**: Complete DIRCON optimization

## Key Classes

### `Dircon`
Core trajectory optimization class implementing the DIRCON algorithm:
- Hermite-Simpson collocation
- Contact force variables
- Kinematic constraints
- Flexible cost functions

### `PlanarWalker`
Complete planar biped robot model:
- Symbolic dynamics with CasADi
- Forward kinematics for feet
- Visualization and animation
- Parameter management

## Algorithm Details

The DIRCON implementation includes:

- **Direct Collocation**: Hermite-Simpson integration for high accuracy
- **Contact Forces**: Explicit contact force optimization variables
- **Kinematic Constraints**: Position, velocity, and acceleration constraints
- **Flexible Costs**: Quadratic costs on states and controls
- **Boundary Conditions**: Initial, final, and periodic constraints

## Educational Content

The Jupyter notebook provides:

1. **DIRCON Theory**: Background and mathematical formulation
2. **Robot Modeling**: Step-by-step robot model development
3. **Constraint Setup**: How to add various constraint types
4. **Solution Analysis**: Interpreting and visualizing results
5. **Advanced Topics**: Walking gaits and contact constraints

## Comparison with Original C++/Drake Version

| Feature | C++/Drake | Python/CasADi |
|---------|-----------|----------------|
| Performance | Faster | Moderate |
| Ease of Use | Complex | Simple |
| Educational Value | Low | High |
| Extensibility | Difficult | Easy |
| Visualization | Limited | Rich |
| Dependencies | Heavy | Lightweight |

## Requirements

- Python 3.7+
- CasADi 3.5+
- NumPy
- Matplotlib
- Jupyter (for notebooks)
- SciPy (optional)

## Contributing

Contributions welcome! Areas for improvement:
- Additional robot models
- More sophisticated contact constraints
- Performance optimization
- Additional visualization tools

## License

This implementation follows the same license as the original DIRCON project.

## References

1. Michael Posa, Scott Kuindersma, Russ Tedrake. "Optimization and Stabilization of Trajectories for Constrained Dynamical Systems." ICRA, 2016.
2. CasADi Documentation: https://casadi.org
3. Original DIRCON repository: https://github.com/DAIRLab/DIRCON

## Getting Help

- Check the Jupyter notebook tutorial for detailed explanations
- Run the simple example to verify your installation
- Examine the code - it's designed to be readable!
- Open issues for bugs or feature requests