# Discrete Algebraic Riccati Equation Solver(s)
#
# Copyright (C) 2010 Jeffrey Armstrong <jeff@rainbow-100.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# A full version of the license terms is available in LICENSE.

import numpy
import numpy.linalg
import scipy.linalg
import warnings

#from dlyap import dlyap
from pydare.dlyap import dlyap

try:
    import slycot
except ImportError:
    pass

EPSILON = 1.0E-5
ITER_LIMIT = 10000

class DareSolver:
    """Class providing 3 techniques for solving the discrete-time
    algebraic Riccati equation (DARE) (X = A'XA-(A'XB)(R+B'XB)^-1(B'XA)+Q).  
	One technique utilizes a direct solution, but it is prone to the 
    numerical condition of the R input.  Two iterative solutions are 
    also provided, a Newton's method solution and a cyclic reduction 
    solution based on quadratic matrix equations.  The cyclic reduction 
    method requires all square matrices, however. 
    
    Ideally the Newton solver should be using a more advanced defect
    correction technique to avoid issues with the associated computation
    of the Newton step and its solution of the Stein (or Lyapunov)
    equation.  However, the method may be sufficient for many well-
    formed problems.
    
    The direct solver will now make use of the Slycot package if available.
    (http://github.com/avventi/Slycot).  This direct solver should be somewhat
    more robust than the simple pure Python direct solver implemented here.
    
    Direct solution algorithm taken from:
    Laub, "A Schur Method for Solving Algebraic Riccati Equations."
    U.S. Energy Research and Development Agency under contract 
    ERDA-E(49-18)-2087.
    
    Iterative Techniques:
    Simplistic Newton solver taken from:
    Fabbender and Benner, "Initializing Newton's Method for Discrete-Time
      Algebraic Riccati Equations Using the Butterfly SZ Algorithm." 
      Proceedings of the 1999 IEEE International Symposium on Computer Aided 
      Control System Design, Hawaii, USA, August 22-27, 1999.  pp. 70-74.
    
    Cyclic Reduction solver taken from:
    Bini and Iannazzo, "A Cyclic Reduction Method for Solving Algebraic
      Ricatti Equations." Technical Report, Dipartimento di Matematica, 
      Universita di Pisa, 2005.
    
    Author:
    Jeffrey Armstrong <jeff@rainbow-100.com>
    """
    
    def __init__(self,a=None,b=None,q=None,r=None):
        """Initializes the DARE (X = A'XA-(A'XB)(R+B'XB)^-1(B'XA)+Q) solver 
		using the specified inputs of the A, B, Q, and R matrices.  The method
		can be later set by changing the 'iterative' or 'use_cyclic' boolean 
		members.  The 'iterative' method is highly suggested as it handles the 
		widest variety of cases in a stable manner.
		
		All inputs must be numpy matrix objects (not arrays!).  Any other input
		types may lead to cryptic error messages.""" 
		
        self.a = a
        self.b = b
        self.q = q
        self.r = r
        self.iterative = False
        self.use_cyclic = False
        self.relaxation = 1.0
        self.iterations = 0
    
    def solve_direct(self):
        """Solves the DARE equation directly using a Schur decomposition method.
        This routine is prone to numerical instabilities mostly associated with
        the inversion of the R matrix.  However, in some well-defined cases, the
        algortihm may work properly and provide a considerable computational
        speed advantages over the iterative techniques.
        """
        
        g = self.b*numpy.linalg.inv(self.r)*self.b.transpose()
        fit = numpy.linalg.inv(self.a).transpose()
        
        z11 = self.a+g*fit*self.q
        z12 = -1.0*g*fit
        z21 = -1.0*fit*self.q
        z22 = fit
        z = numpy.vstack((numpy.hstack((z11, z12)), numpy.hstack((z21, z22))))
        
        [s,u,sdim] = scipy.linalg.schur(numpy.linalg.inv(z), sort='lhp')

        (m,n) = u.shape
        
        u11 = u[0:m/2, 0:n/2]
        u12 = u[0:m/2, n/2:n]
        u21 = u[m/2:m, 0:n/2]
        u22 = u[m/2:m, n/2:n]
        u11i = numpy.linalg.inv(u11)

        self.solution =  numpy.asmatrix(u21)*numpy.asmatrix(u11i)
        return self.solution

    def solve_slycot(self):
        """Directly solves the DARE using the SLICOT library's SB02MD 
        implementation of a generalized Schur vectors method.  If the Slycot
        package is unavailable, a RuntimeError will be thrown.  This solver
        should be considerably more robust than the pure-Python implementation
        in solve_direct().  Only minimal shape checking is performed.
        
        More on SLICOT: http://www.slicot.org/
    
        Python Interface (Slycot): https://github.com/avventi/Slycot
        """
    
        if self.a.shape[0] != self.a.shape[1]:
            raise ValueError('input "a" must be a square matrix')
    
        try:
            self.solution,rcond,w,s,t = slycot.sb02od(self.a.shape[1],self.b.shape[1],\
                                                      self.a, self.b, self.q, self.r,'D')
        except NameError:
            raise RuntimeError('SLICOT not available')
        
        return self.solution

    def cyclic_iterative_init(self):
        """Initializes the cyclic reduction solver variables."""
        binv = numpy.linalg.inv(self.b)
        self.a1 = -1.0*self.a.transpose()*binv.transpose()*self.r*binv
        self.a0 = binv.transpose()*self.r*binv + \
                  self.a.transpose()*binv.transpose()*self.r*binv*self.a + \
                  self.q
        self.h = self.a0
        self.k = self.a1
        self.hhat = self.h

    def cyclic_iterative_step(self):
        """Steps the cyclic reduction solver one iteration."""
        hinv = numpy.linalg.inv(self.h)
        h1 = self.h - \
             self.k*hinv*self.k.transpose() - \
             self.k.transpose()*hinv*self.k
             
        self.hhat = self.hhat - self.k*hinv*self.k.transpose()
        
        self.k = -1.0*self.k*hinv*self.k
        
        self.h = h1 

    def solve_cyclic_iterative(self,eps,iter_limit):
        """Solves the DARE using the cyclic reduction technique.  The technique
        relies on the B matrix being square.  The algorithm may offer advantages
        over the Newton iterative solver in terms of speed and numerical 
        stability depending on the particular problem."""
        
        self.cyclic_iterative_init()
        
        self.error = 1.0E+6
        count = 0
        while (self.error > eps and count < iter_limit) or count < 2:
            self.cyclic_iterative_step()
            self.error = numpy.linalg.norm(self.k)
            count = count + 1
        
        z = -1.0*numpy.linalg.inv(self.hhat)*self.a1
        binv = numpy.linalg.inv(self.b)
        try:
            zinv = numpy.linalg.inv(z)
        except:
            warnings.warn('Cyclic reduction encountered singular matrix during solution - using psuedoinverse',RuntimeWarning)
            zinv = numpy.linalg.pinv(z) 
        
        self.solution = binv.transpose()*(self.r*binv*(self.a-z))*zinv
        return self.solution
    
    def newton_iterative_init(self):
        """Initializes the Newton iterative solver."""
        self.x = numpy.eye(self.q.shape[0]) #self.solve_direct()
    
    def newton_cost(self,x):
        """Computes the current error in the DARE solution estimate for use with
        the Newton iterative solver.  In a converged situation, the cost would 
        be zero."""
        
        return self.q - x + self.a.transpose()*x*self.a - \
               self.a.transpose()*x*self.b*numpy.linalg.inv(self.r+self.b.transpose()*x*self.b)*self.b.transpose()*x*self.a
    
    def newton_iterative_step(self):
        """Steps the Newton iterative solver one iteration."""
        ak = self.a-self.b*numpy.linalg.inv(self.r+self.b.transpose()*self.x*self.b)*self.b.transpose()*self.x*self.a
        
        # The iterative Lyapunov solver must be used here due to the possible presence of
        # numerical instabilities.  However, the necessary accuracy at this step can be
        # low, so the iteration count is set to 30.  
        self.dx = dlyap(ak.transpose(),self.newton_cost(self.x))
        
        self.x = self.x+self.relaxation*self.dx
    
    def solve_newton_iterative(self,eps,iter_limit,initial=None):
        """Solves the DARE using a Newton iterative technique."""
        if initial==None:
            self.newton_iterative_init()
        else:
            self.x = initial
            
        error = 1.0E+6
        self.iterations = 0
        while error > eps and self.iterations < iter_limit:
            self.newton_iterative_step()
            error = abs(self.dx.max()) #numpy.linalg.norm(self.dx)
            self.iterations = self.iterations + 1
        
        self.solution = self.x
        return self.x
    
    def solve(self,eps=EPSILON,iter_limit=ITER_LIMIT,initial=None):
        """Solves the discrete-time Riccati equation:
        
           X = A'XA-(A'XB)(R+B'XB)^-1(B'XA)+Q)
        
        returning a value or estimate of the X matrix.
        
        Sovler object flags:
        
        iterative   use_cyclic      Solver Method
        False       <N/A>           Direct (solve_direct)
        True        False           Newton iterative (solve_newton_iterative)
        True        True            Cyclic Reduction (solve_cyclic_iterative)
        """
        
        if self.use_cyclic:
            self.iterative = True
        
        if self.iterative:
            if self.b.shape[0] == self.b.shape[1] and self.use_cyclic:
                self.solution = self.solve_cyclic_iterative(eps,iter_limit)
            else:
                
                if self.use_cyclic:
                    warnings.warn('Cyclic reduction method not possible without square B matrix: falling back on Newton method',RuntimeWarning)
                    
                self.solution = self.solve_newton_iterative(eps,iter_limit,initial)
        else:
        
            try:
                self.solution = self.solve_slycot()
            except RuntimeError:
                self.solution = self.solve_direct()
            self.iterations = None
        

        return self.solution
