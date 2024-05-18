import numpy as np
import scipy
import scipy.special

from poles_roots.aaa import AAA

TOL = 1e4*np.finfo(np.float64).eps


class TestAAA:
    def test_exp(self):
        Z = np.linspace(-1, 1, num=1000)
        F = np.exp(Z)
        r = AAA(F, Z)
        
        assert np.linalg.norm(F - r(Z), np.inf) < TOL
        assert np.isnan(r(np.nan))
        assert np.isfinite(r(np.inf))
        
        m1 = r.zj.size
        
        r2 = AAA(F, Z, mmax=m1-1)
        assert r2.zj.size == m1 - 1
        
        r3 = AAA(F, Z, tol=1e-3)
        assert r2.zj.size < m1
        
    def test_tan(self):
        Z = np.linspace(-1, 1, num=1000)
        F = np.tan(np.pi*Z)
        r = AAA(F, Z)
        
        assert np.linalg.norm(F - r(Z), np.inf) < 10*TOL
        assert np.min(np.abs(r.zer)) < TOL
        assert np.min(np.abs(r.pol - 0.5)) < TOL
        # Test for spurious poles
        assert np.min(np.abs(r.res)) > 1e-13
    
    def test_short_cases(self):
        Z = [0, 1]
        F = [1, 2]
        r = AAA(F, Z)
        assert np.linalg.norm(F - r(Z), np.inf) < TOL

        Z = [0, 1, 2]
        F = [1, 0, 0]
        r = AAA(F, Z)
        assert np.linalg.norm(F - r(Z), np.inf) < TOL
    
    def test_scale_invariance(self):
        Z = np.linspace(0.3, 1.5)
        F = np.exp(Z)/(1+1j)
        r1 = AAA(F, Z)
        r2 = AAA(2**311*F, Z)
        r3 = AAA(2**-311*F, Z)
        assert r1(0.2j) == 2**-311*r2(0.2j)
        assert r1(1.4) == 2**311*r3(1.4)
    
    # Enable if / when auto Z is added
    # def test_gamma(self):
    #     r = AAA(scipy.special.gamma)
    #     assert np.abs(r(1.5) - scipy.special.gamma(1.5)) < 1e-3

    def test_log_func(self):
        rng = np.random.default_rng(1749382759832758297)
        Z = rng.standard_normal(10000)+3j*rng.standard_normal(10000)
        
        def f(z):
            return np.log(5 - z)/(1 + z**2)
        
        r = AAA(f(Z), Z)
        assert np.abs(r(0) - f(0)) < TOL
    
    def test_infinite_data(self):
        Z = np.linspace(-1, 1)
        r = AAA(scipy.special.gamma(Z), Z)
        assert np.abs(r(0.63) - scipy.special.gamma(0.63)) < 1e-3
    
    def test_nan(self):
        X = np.linspace(0, 20)
        F = np.sin(X)/X
        r = AAA(F, X)
        assert abs(r(2) - np.sin(2)/2) < 1e-3
    
    def test_residues(self):
        X = np.linspace(-1.337, 2, num=537)
        r = AAA(np.exp(X)/X, X)
        ii = np.nonzero(np.abs(r.pol) < 1e-8)[0]
        assert np.abs(r.res[ii] -1 ) < 1e-10

        r = AAA((1+1j)*scipy.special.gamma(X), X)
        ii = np.nonzero(abs(r.pol - (-1)) < 1e-8)
        assert np.abs(r.res[ii] + ( 1 + 1j)) < 1e-10