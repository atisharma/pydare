# Discrete Lyapunov Equation Solver(s)
#
# Copyright (C) 2010 Jeffrey Armstrong <jeff@rainbow-100.com>
#
# Portions Copyright (C) 1993, 1994, 1995, 2000, 2002, 2004, 2005, 2007
#                        Auburn University
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
import math

try:
    import slycot
except ImportError:
    pass

ITER_LIMIT = 10000
LYAPUNOV_EPSILON = 1.0E-6

def dlyap_iterative(a,q,eps=LYAPUNOV_EPSILON,iter_limit=ITER_LIMIT):
    """Solves the Discrete Lyapunov Equation (X = A X A' + Q) via an iterative
    method.  The routine returns an estimate of X based on the A and Q input
    matrices.  
    
    This iterative solver requires that the eigenvalues of the square matrix
    A be within the unit circle for convergence reasons.
    
    Iterative Discrete-Time Lyapunov Solver based on:
     
    Davinson and Man, "The Numerical Solution of A'Q+QA=-C." 
    IEEE Transactions on Automatic Control, Volume 13, Issue 4, August, 1968.  p. 448.
    """
    error = 1E+6
        
    x = q
    ap = a
    apt = a.transpose()
    last_change = 1.0E+10
    count = 1

    (m,n) = a.shape
    if m != n:
        raise ValueError("input 'a' must be square") 
    
    det_a = numpy.linalg.det(a)
    if det_a > 1.0:
        raise ValueError("input 'a' must have eigenvalues within the unit circle") 
    
    while error > LYAPUNOV_EPSILON and count < iter_limit:
        change = ap*q*apt
            
        x = x + change
        
        #if numpy.linalg.norm(change) > last_change:
        #    raise ValueError('A is not convergent')
        #last_change = abs(change.max())#numpy.linalg.norm(change)
        ap = ap*a
        apt = apt*(a.transpose())
        error = abs(change.max())
        count = count + 1

    if count >= iter_limit:
        warnings.warn('lyap_solve: iteration limit reached - no convergence',RuntimeWarning)
        #print 'warning: lyap_solve: iteration limit reached - no convergence'
        
    return x

def dlyap_schur(a,q):
    """Solves the Discrete Lyapunov Equation (A'Q+QA=-X) directly.  The 
    routine returns an estimate of X based on the A and Q input matrices.
    
    Discrete-Time Lyapunov Solver based on Octave dlyap.m file
    Copyright (C) 1993, 1994, 1995, 2000, 2002, 2004, 2005, 2007
    Auburn University.  All rights reserved.
    
    Uses Schur decomposition method as in Kitagawa,
    "An Algorithm for Solving the Matrix Equation @math{X = F X F' + S}}",
    International Journal of Control, Volume 25, Number 5, pages 745--753
    (1977).
    
    Column-by-column solution method as suggested in
    Hammarling, "Numerical Solution of the Stable, Non-Negative
    Definite Lyapunov Equation", Journal of Numerical Analysis, Volume
    2, pages 303--323 (1982).
    """
    
    (m,n) = a.shape
    if m != n:
        raise ValueError("input 'a' must be square") 
        
    [s, u, sdim] = scipy.linalg.schur(a, sort='lhp')
    
    s = numpy.asmatrix(s)
    u = numpy.asmatrix(u)
    
    b = u.transpose()*q*u
    
    x = numpy.asmatrix(numpy.zeros(a.shape))
    
    j = n-1
    while j >= 0:
        j1 = j+1
        
        # Check for Schur block
        blocksize = 1
        if j > 0:
            if s[j,j-1] != 0.0:
                blocksize = 2
                j = j - 1

        Ajj = scipy.linalg.kron(s[j:j1,:][:,j:j1] ,s) - numpy.eye(blocksize*n)
        
        rhs = numpy.reshape(b[:,j:j1], (blocksize*n, 1))
        
        if j1 < n:
            rhs2 = s*(x[:,j1:]*(s[j:j1,:][:,j1:]).transpose())
            rhs = rhs + numpy.reshape(rhs2,(blocksize*n, 1))
            
        v = -1.0*scipy.linalg.solve(Ajj,rhs)
        
        x[:,j] = v[0:n]
        
        if blocksize == 2:
            x[:,j1-1] = v[n:blocksize*n+1]
        
        j = j - 1
        
    x = u*x*u.transpose()
    
    return x

def dlyap_slycot(a,q):
    """Solves the discrete Lyapunov using the SLICOT library's implementation
    if available.  The routine attempts to call SB03MD to solve the discrete
    equation.  If a NameError is thrown, meaning SLICOT is not available,
    an appropriate RuntimeError is raised.
    
    More on SLICOT: http://www.slicot.org/
    
    Python Interface (Slycot): https://github.com/avventi/Slycot
    """

    x = None
    
    (m,n) = a.shape
    if m != n:
        raise ValueError("input 'a' must be square") 
    
    try:
        x,scale,sep,ferr,w = slycot.sb03md(n, -q, a, numpy.eye(n), 'D', trana='T')
    except NameError:
        raise RuntimeError('SLICOT not available')
    
    return x
    
def dlyap(a,q,iterative=False,iteration_limit=ITER_LIMIT):
    """Solves the discrete Lyapunov equation (X = A X A' + Q) given the values
    of A and Q.  This function provides a generalized interface to three
    available solvers.  If the iterative flag is not set, the routine will fall
    back to a direct solver.  If the Python interface to SLICOT is installed, 
    the routine will preferentially call the SLICOT solver rather than the pure
    Python implementation."""
    
    if iterative:
        return dlyap_iterative(a,q,iter_limit=iteration_limit)
    else:
        try:
            return dlyap_slycot(a,q)
        except RuntimeError:
            return dlyap_schur(a,q)