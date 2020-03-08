'''
Define some unit test and integration tests
'''
import numpy as np

def main():
    print("tests.py::Starting unit tests")
    
    test_res()

    print("tests.py::Everything passed")

def test_res():
    n_points=10
    x=np.linspace(1, n_points, num=n_points, dtype=int)
    
    def sin(t, *pars):
        a=pars[0]
        return a*np.sin(t)

    #truth points on the sine     
    p0=[1]
    y_truth = sin(x, p0)
    print("Truth points:", y_truth)
    
    #smeared points by a narrow centred Gaussian
    y_smeared = y_truth+np.random.normal(loc=0.0, scale=0.001, size=n_points)
    print("Smeared points:", y_smeared)

    # explicit residuals
    res_explicit = y_truth - y_smeared
    print("Explicit residuals:", res_explicit)

    # implicit residuals
    res_implicit = residuals(x, y_smeared, sin, p0)
    print("Implicit residuals:", res_explicit)

    #Check we got the same data 
    print("Expected results:", np.allclose(res_explicit, res_implicit, rtol=1e-4, atol=1e-2))
    assert(np.allclose(res_explicit, res_implicit, rtol=1e-4, atol=1e-2))

def residuals(x, y, func, pars):
    '''
    Calcualte fit residuals
    '''
    return np.array(y -func(x, *pars))

if __name__ == "__main__":
    main()
