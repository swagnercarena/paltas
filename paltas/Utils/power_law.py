# -*- coding: utf-8 -*-
"""
Draw from and integrate power laws

Useful equations to draw from and integrate power law distributions.
"""
import numpy as np
import numba


@numba.njit()
def power_law_integrate(p_min,p_max,slope):
	"""Integrates a power law

	Args:
		p_min (float): The lower bound of the power law
		p_max (float): The upper bound of the power law
		slope (float): The slope of the power law

	Returns:
		(float): The integral of the power law x^slope from p_min
		to p_max
	"""
	upper_bound = 1/(slope+1)*p_max**(slope+1)
	lower_bound = 1/(slope+1)*p_min**(slope+1)
	return upper_bound-lower_bound


@numba.njit()
def power_law_draw(p_min,p_max,slope,norm):
	"""Samples from a power law

	Args:
		p_min (float): The lower bound of the power law
		p_max (float): The upper bound of the power law
		slope (float): The slope of the power law
		norm (float): The overall normalization of the power law

	Returns:
		[float,...]: A list of values drawn from the power law
	"""
	# Get the expected number of draws
	n_expected = norm * power_law_integrate(p_min,p_max,slope)

	# Draw the number of objects as a poisson process
	n_draws = np.random.poisson(n_expected)

	# Get the positions in the cdf we want to draw from a uniform and then
	# convert that to values in the pdf
	cdf = np.random.rand(n_draws)
	# usefule shorthand
	s1 = slope + 1
	x = (cdf*(p_max**s1-p_min**s1)+p_min**s1)**(1/s1)

	return x
