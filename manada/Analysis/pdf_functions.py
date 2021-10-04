import numba
import numpy as np
from math import erf


@numba.njit
def _norm_cdf(bound,mu,sigma):
    """
    A helper function for eval_normal_logpdf_approx
    See `sample_normal` for parameter definitions.
    """
    return 0.5*erf((bound-mu)/(sigma*np.sqrt(2)))


@numba.njit
def eval_normal_logpdf_approx(eval_at, mu, sigma, lower=-np.inf, upper=np.inf):
    """Evaluate the normal pdf, optionally truncated without -np.inf
    See `sample_normal` for parameter definitions.
    """
    # First calculate the function without bounds
    norm = -np.log(sigma)-np.log(2*np.pi)/2
    eval_logpdf = -np.power((eval_at-mu)/sigma,2)/2+norm
    accept_norm = _norm_cdf(upper,mu,sigma) - _norm_cdf(lower,mu,sigma)

    # Now correct for the bounds if they are not -np.inf and np.inf
    # Note, reshaping must always be done regardless of bounds or numba will
    # not compile
    eval_shape = eval_at.shape
    eval_at = eval_at.reshape(-1)
    eval_logpdf=eval_logpdf.reshape(-1)
    if lower > -np.inf and upper < np.inf:
        for e_i in range(len(eval_at)):
            if eval_at[e_i] < lower:
                eval_logpdf[e_i] -= 1000
            if eval_at[e_i] > upper:
                eval_logpdf[e_i] -= 1000
    eval_logpdf=eval_logpdf.reshape(eval_shape)
    return eval_logpdf - np.log(accept_norm)


@numba.njit
def _lognorm_cdf(bound,mu,sigma):
    """
    A helper function for eval_lognormal_logpdf_approx
    See `sample_normal` for parameter definitions.
    """
    return 0.5*erf((np.log(bound)-mu)/(np.sqrt(2)*sigma))


@numba.njit
def eval_lognormal_logpdf_approx(eval_at, mu, sigma, lower=0, upper=np.inf):
    """Evaluate the normal pdf, optionally truncated without -np.inf
    See `sample_normal` for parameter definitions.
    """
    # First calculate the distribution without the bounds
    norm = -np.log(sigma) - np.log(eval_at) - np.log(2*np.pi)/2
    eval_unnormed_logpdf = -np.square(np.log(eval_at)-mu)/(2*sigma**2)
    eval_unnormed_logpdf += norm

    # Stop cdf from crashing if lower bound is below 0
    if lower<0:
        lower=0

    accept_norm = _lognorm_cdf(upper,mu,sigma) - _lognorm_cdf(lower,mu,sigma)
    eval_normed_logpdf = eval_unnormed_logpdf - np.log(accept_norm)

    # Now correct for the bounds if they are not -np.inf and np.inf
    # Note, reshaping must always be done regardless of bounds or numba will
    # not compile
    eval_shape = eval_at.shape
    eval_at = eval_at.reshape(-1)
    eval_normed_logpdf=eval_normed_logpdf.reshape(-1)
    if lower > 0 and upper < np.inf:
        for e_i in range(len(eval_at)):
            if eval_at[e_i] < lower:
                eval_normed_logpdf[e_i] -= 1000
            if eval_at[e_i] > upper:
                eval_normed_logpdf[e_i] -= 1000
    eval_normed_logpdf=eval_normed_logpdf.reshape(eval_shape)
    return eval_normed_logpdf
