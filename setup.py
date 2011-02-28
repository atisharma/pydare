from distutils.core import setup

LONG_DESC="""This small package contains two Python routines for solving the
discrete-time Riccati Equation (DARE) and the discrete-time
Lyapunov Equation.  These routines were developed in support
of a controls-related research project.  Both solvers were
developed into a minimally functional set of routines to 
satisfy the project for which they were originally meant. 
However, this functionality may be sufficient for other users
as well.  Note that the pydare package does not include 
routines for solving continuous versions of these two equation
sets."""

setup(name='pydare',
	  version='0.3',
	  description='Discrete Riccati and Lyapunov Equation solvers',
	  long_description=LONG_DESC,
	  author='Jeffrey Armstrong',
	  author_email='jeff@rainbow-100.com',
	  url='https://code.google.com/p/pydare/',
	  packages=['pydare',],
	  license='GPL',
	  classifiers=['Intended Audience :: Developers', 'Topic :: Mathematics'],
	  install_requires=['numpy',],
	  keywords='dare lyapunov riccati',
	 )
	  