from pydare import DareSolver
import numpy
import unittest

class DareTestCase(unittest.TestCase):
    
	def testIterative(self):
		a = numpy.matrix([[0.0,0.1,0.0],\
		                  [0.0,0.0,0.1],\
						  [0.0,0.0,0.0]])
						   
		b = numpy.matrix([[1.0,0.0], \
						  [0.0,0.0], \
						  [0.0,1.0]])
						   
		r = numpy.matrix([[0.0,0.0], \
						  [0.0,1.0]])
						   
		q = numpy.matrix([[10.0**5.0, 0.0,0.0], \
		                  [0.0,10.0**3.0,0.0], \
						  [0.0,0.0,-10.0]])

		ds = DareSolver(a,b,q,r)
		
		ds.iterative = True

		x = ds.solve()
		
		self.assertAlmostEqual(10.0**5.0,x[0,0],3)
		self.assertAlmostEqual(10.0**3.0,x[1,1],3)
		self.assertAlmostEqual(0.0,x[2,2],3)
		
		for i in range(0,3):
			for j in range(0,3):
				if i != j:
					self.assertAlmostEqual(0.0,x[i,j],3)
					
	def testDirect(self):
		a = numpy.matrix([[0.8147, 0.1270],[0.9058, 0.9134]])
		b = numpy.matrix([[0.6324, 0.2785],[0.0975, 0.5469]])
		q = numpy.eye(2)
		r = numpy.matrix([[1.0,0.0],[0.0,1.0]])
		
		ds = DareSolver(a,b,q,r)
		
		x = ds.solve_direct()
		self.assertAlmostEqual(2.6018,x[0,0],3)
		self.assertAlmostEqual(0.9969,x[0,1],3)
		self.assertAlmostEqual(0.9969,x[1,0],3)
		self.assertAlmostEqual(1.8853,x[1,1],3)
        
	def testSLICOT(self):
		a = numpy.matrix([[0.8147, 0.1270],[0.9058, 0.9134]])
		b = numpy.matrix([[0.6324, 0.2785],[0.0975, 0.5469]])
		q = numpy.eye(2)
		r = numpy.matrix([[1.0,0.0],[0.0,1.0]])
		
		ds = DareSolver(a,b,q,r)
		
		x = ds.solve_slycot()
		self.assertAlmostEqual(2.6018,x[0,0],3)
		self.assertAlmostEqual(0.9969,x[0,1],3)
		self.assertAlmostEqual(0.9969,x[1,0],3)
		self.assertAlmostEqual(1.8853,x[1,1],3)
		
	def testCyclic(self):
		a = numpy.eye(2)
		b = -1.0*numpy.eye(2)
		r = numpy.eye(2)
		q = numpy.matrix([[1.0,0.0],[0.0,0.5]])
		
		ds = DareSolver(a,b,q,r)
		
		ds.use_cyclic = True
		
		x = ds.solve()
		
		self.assertAlmostEqual(1.6180,x[0,0],3)
		self.assertAlmostEqual(1.0,x[1,1],3)
		
		self.assertAlmostEqual(0.0,x[0,1],3)
		self.assertAlmostEqual(0.0,x[1,0],3)
		
		
def suite():
	suite = unittest.TestSuite()
	suite.addTest(DareTestCase('testIterative'))
	suite.addTest(DareTestCase('testDirect'))
	suite.addTest(DareTestCase('testCyclic'))
	suite.addTest(DareTestCase('testSLICOT'))
	
	return suite

if __name__ == '__main__':

    unittest.TextTestRunner(verbosity=2).run(suite())