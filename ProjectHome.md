This Python package provides solvers for the Discrete Algebraic Riccati Equation (DARE) and discrete Lyapunov equation.

Four DARE solvers are included:

  * Direct using a Schur method
  * An iterative cyclic reduction method
  * An iterative Newton's method
  * Direct using SLICOT SB02OD implementation (via Slycot)

Three discrete Lyapunov equation solvers are included:

  * An iterative solver
  * Direct using a Schur method based on an Octave implementation
  * Direct using SLICOT SB03MD implementation (via Slycot)

Note that the pydare package does not include routines for solving continuous versions of these two equation sets.