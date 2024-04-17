# -*- coding: utf-8 -*-
"""
Combine distributions.

Useful utility functions for combining two distributions.
"""
import numpy as np
import numba


@numba.njit()
def geometric_average(mu_narrow, sigma_narrow, mu_wide, sigma_wide,
	weight_narrow = 1, weight_wide = 1):
	"""Returns the parameters of the geometric average of two Gaussians

	Args:
		mu_narrow (np.array): Means of the narrow distribution.
		sigma_narrow (np.array): Standard deviations of the narrow distribution.
		mu_wide (np.array): Means of the wide distribution.
		sigma_wide (np.array): Standard deviation of the wide distribution.
		weight_wide (int): Optional weight for the wide distribution in the
			geometric average. Default is 1. If 0, then the other distribution
			is returned.
		weight_narrow (int): Optional weight for the narrow distribution in the
			geometric average. Default is 1. If 0, then the other distribution
			is returned.

	Returns:
		(np.array, np.array): Mean and standard deviation of the Gaussian
			proportional to the geometric average of the input distributions.
	"""
	# Cannot be negative
	assert weight_wide >= 0
	assert weight_narrow >= 0

	# Function for combining any two distributions.
	def _combine_dist(mu_one, sigma_one, mu_two, sigma_two):
		sigma_comb = np.sqrt(
			(sigma_one ** 2 * sigma_two ** 2) /
			(sigma_one ** 2 + sigma_two ** 2)
		)
		mu_comb = (sigma_comb ** 2 / sigma_one ** 2) * mu_one
		mu_comb += (sigma_comb ** 2 / sigma_two ** 2) * mu_two
		return mu_comb, sigma_comb

	if weight_wide == 0 and weight_narrow == 0:
		raise ValueError('Both weights cannot be 0.')
	if weight_wide == 0:
		return mu_narrow, sigma_narrow
	if weight_narrow == 0:
		return mu_wide, sigma_wide

	# Add as many powers of the wide distribution as desired.
	mu_comb, sigma_comb = mu_wide, sigma_wide
	for _ in range(weight_wide - 1):
		mu_comb, sigma_comb = _combine_dist(
			mu_comb, sigma_comb, mu_wide, sigma_wide
		)

	# Combine the wide and narrow distribution.
	mu_comb, sigma_comb = _combine_dist(
		mu_comb, sigma_comb, mu_narrow, sigma_narrow
	)

	# Add as many powers of the narrow distribution as desired.
	for _ in range(weight_narrow - 1):
		mu_comb, sigma_comb = _combine_dist(
			mu_comb, sigma_comb, mu_narrow, sigma_narrow
		)

	# Reweight sigma to account for n^th root (n is equal to sum of weights).
	sigma_comb *= np.sqrt(weight_narrow + weight_wide)

	return mu_comb, sigma_comb
