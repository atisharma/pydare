from pydare.dlyap import dlyap_iterative, dlyap_schur, dlyap_slycot
import numpy
import unittest

class DlyapTestCase(unittest.TestCase):
    
    def setUp(self):
        self.a = numpy.matrix([[0.5,1.0],[-1.0,-1.0]])
        self.q = numpy.matrix([[2.0,0.0],[0.0,0.5]])
    
    def testIterative(self):
        x = dlyap_iterative(self.a,self.q)
        
        self.assertAlmostEqual(4.75,x[0,0],4)
        self.assertAlmostEqual(4.1875,x[1,1],4)
        
        for i in range(0,2):
            for j in range(0,2):
                if i != j:
                    self.assertAlmostEqual(-2.625,x[i,j],4)

    def testDirect(self):
        x = dlyap_schur(self.a,self.q)
        
        self.assertAlmostEqual(4.75,x[0,0],4)
        self.assertAlmostEqual(4.1875,x[1,1],4)
        
        for i in range(0,2):
            for j in range(0,2):
                if i != j:
                    self.assertAlmostEqual(-2.625,x[i,j],4)
                    
    def testSLICOT(self):
        x = dlyap_slycot(self.a,self.q)
        
        self.assertAlmostEqual(4.75,x[0,0],4)
        self.assertAlmostEqual(4.1875,x[1,1],4)
        
        for i in range(0,2):
            for j in range(0,2):
                if i != j:
                    self.assertAlmostEqual(-2.625,x[i,j],4)
                    
def suite():
    suite = unittest.TestSuite()
    suite.addTest(DlyapTestCase('testIterative'))
    suite.addTest(DlyapTestCase('testDirect'))
    suite.addTest(DlyapTestCase('testSLICOT'))
    
    return suite
    
if __name__ == '__main__':

    unittest.TextTestRunner(verbosity=2).run(suite())