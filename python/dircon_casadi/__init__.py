"""
Python CasADi implementation of DIRCON (Direct Collocation with Constraints)

Based on the publication:
Michael Posa, Scott Kuindersma, Russ Tedrake. 
"Optimization and Stabilization of Trajectories for Constrained Dynamical Systems." 
Proceedings of the International Conference on Robotics and Automation (ICRA), 2016.
"""

from .dircon import Dircon
from .planar_walker import PlanarWalker

__version__ = "0.1.0"
__all__ = ["Dircon", "PlanarWalker"]