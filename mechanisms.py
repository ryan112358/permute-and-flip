import numpy as np
from collections import defaultdict
from scipy.optimize import minimize_scalar, root_scalar, bracket
from scipy.special import logsumexp

def em_worst_expected_error(n=2, eps=1, delta=1):
    def foo(p):
        return np.log(p) * (1 - 1 / (1 + (n-1)*p))
    a = -minimize_scalar(lambda p: foo(p), bounds=(0,1), method='bounded').fun
    return a * 2 * delta / eps

def pf_worst_expected_error(n=2, eps=1, delta=1):
    def foo(p):
        return np.log(p) * (1 - (1 - (1-p)**n) / (n*p))
    a = -minimize_scalar(lambda p: foo(p), bounds=(0,1), method='bounded').fun
    return a * 2 * delta / eps

def pf_pmf(q, eps=1.0, sensitivity=1.0, monotonic=False):
    coef = 1.0 if monotonic else 0.5
    p = np.exp(coef*eps/sensitivity*(q - q.max()))
    n = q.size
    
    # first we will calculate 
    # sum(prod(p_i, i in S), |S| = k) for each k
    
    subsets = np.zeros(n)
    curr = np.cumsum(p)
    subsets[0] = curr[-1]
    for j in range(1,n):
        curr[j:] = np.cumsum(curr[j-1:-1]*p[j:])
        subsets[j] = curr[-1]   
    
    # coefficient vector: (-1)^k / (k+1) for k = 1..n
    coef = (np.arange(n) % 2 * 2 - 1) / (np.arange(n)+2)
    
    # we will now calculate
    # sum(prod(p_i, i in S), |S| = k, r not in S)
    # and compute the final probabilities
    
    ans = np.zeros(n)
    for i in range(n):
        new = np.copy(subsets)
        new[0] -= p[i]
        for j in range(1,n):
            new[j] -= new[j-1]*p[i]
        
        ans[i] = p[i] * (1 + new @ coef)
        
    return ans

def em_pmf(q, eps=1.0, sensitivity=1.0, monotonic=False):
    coef = 1.0 if monotonic else 0.5
    q = q - q.max()
    logits = coef*eps/sensitivity*q
    return np.exp(logits - logsumexp(logits))
    #p = np.exp(coef*eps/sensitivity*q)
    #return p / p.sum()

def em(q, eps=1.0, sensitivity=1.0, prng=np.random, monotonic=False):
    
    coef = 1.0 if monotonic else 0.5

    q = q - q.max()
    p = np.exp(coef*eps/sensitivity*q)
    p /= p.sum()

    return prng.choice(p.size, p=p)


def pf(q, eps=1.0, sensitivity=1.0, prng=np.random, monotonic=False):
    
    coef = 1.0 if monotonic else 0.5

    q = q - q.max()
    p = np.exp(coef*eps/sensitivity*q)

    for i in prng.permutation(p.size):
        if prng.rand() <= p[i]:
            return i

def expected_error(q, eps, pmf=em_pmf):
    # compute the expected error of the mechanism (given it's probability mass function)
    ans = q.max() - pmf(q,eps) @ q
    maxerr = q.max() - q.mean()
    if ans > maxerr or ans < 0:
        return maxerr
    return ans

def variance(q, eps, pmf=em_pmf):
    e = expected_error(q, eps, pmf)
    return pmf(q, eps) @ (q.max() - q)**2 - e**2

def expected_epsilon(q, err, bounds=None, pmf=em_pmf):
    # computed the epsilon required to achieve given expected error
    foo = lambda eps: expected_error(q, eps, pmf) - err

    if bounds is None:
        eps = 1.0
        while foo(eps) > 0:
            eps *= 2
        while foo(eps) < 0:
            eps /= 2.0
        bounds = [eps,2*eps]

    return root_scalar(foo,bracket=bounds,method='bisect').root

def max_epsilon_ratio(q):
    def foo(eps):
        err = expected_error(q, eps, pf_pmf)
        eps2 = expected_epsilon(q, err, [eps, 2*eps])
        return -eps2/eps
    br = bracket(foo, 1e-3, 1.0)[0:3]
    ans = minimize_scalar(foo, bracket=br, method='brent')
    eps0 = ans.x
    err = expected_error(q, eps0, pf_pmf)
    eps1 = expected_epsilon(q, err, [eps0, 2*eps0])
    return eps0, err, eps1

