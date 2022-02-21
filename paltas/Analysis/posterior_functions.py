# -*- coding: utf-8 -*-
"""
Manipulate the posterios produced by the network to test for calibration.

This module contains functions to test the calibration of the outputs of the
network on a test or validation set.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numba


def plot_coverage(y_pred,y_test,std_pred,parameter_names,
	color_map=["#377eb8", "#4daf4a","#e41a1c","#984ea3"],block=True,
	fontsize=20,show_error_bars=True,n_rows=4):
	""" Generate plots for the 1D coverage of each parameter.

	Args:
		y_pred (np.array): A (batch_size,num_params) array containing the
			mean prediction for each Gaussian
		y_test (np.array): A (batch_size,num_params) array containing the
			true value of the parameter on the test set.
		std_pred (np.array): A (batch_size,num_params) array containing the
			predicted standard deviation for each parameter.
		parameter_names ([str,...]): A list of the parameter names to be
			printed in the plots.ed.
		color_map ([str,...]): A list of at least 4 colors that will be used
			for plotting the different coverage probabilities.
		block (bool): If true, block excecution after plt.show() command.
		fontsize (int): The fontsize to use for the parameter names.
		show_error_bars (bool): If true plot the error bars on the coverage
			plot.
		n_rows (int): The number of rows to include in the subplot.
	"""
	num_params = len(parameter_names)
	error = y_pred - y_test
	# Define the covariance masks for our coverage plots.
	cov_masks = [np.abs(error)<=std_pred,np.abs(error)<2*std_pred,
		np.abs(error)<3*std_pred, np.abs(error)>=3*std_pred]
	cov_masks_names = ['1 sigma =', '2 sigma =', '3 sigma =', '>3 sigma =']
	for i in range(len(parameter_names)):
		plt.subplot(n_rows, int(np.ceil(num_params/n_rows)), i+1)
		# Plot the datapoints in each coverage interval seperately.
		for cm_i in range(len(cov_masks)-1,-1,-1):
			cov_mask = cov_masks[cm_i][:,i]
			yt_plot = y_test[cov_mask,i]
			yp_plot = y_pred[cov_mask,i]
			ys_plot = std_pred[cov_mask,i]
			# Plot with errorbars if requested
			if show_error_bars:
				plt.errorbar(yt_plot,yp_plot,yerr=ys_plot, fmt='.',
					c=color_map[cm_i],
					label=cov_masks_names[cm_i]+'%.2f'%(
						np.sum(cov_mask)/len(error)))
			else:
				plt.errorbar(yt_plot,yp_plot,fmt='.',c=color_map[cm_i],
					label=cov_masks_names[cm_i]+'%.2f'%(
						np.sum(cov_mask)/len(error)))

		# Include the correlation coefficient squared value in the plot.
		_, _, rval, _, _ = linregress(y_test[:,i],y_pred[:,i])
		straight = np.linspace(np.min(y_test[:,i]),np.max(y_test[:,i]),10)
		plt.plot(straight, straight, label='',color='k')
		plt.text(0.8*np.max(straight)+0.2*np.min(straight),np.min(straight),
			'$R^2$: %.3f'%(rval**2),{'fontsize':fontsize})
		plt.title(parameter_names[i],fontsize=fontsize)
		plt.ylabel('Prediction',fontsize=fontsize)
		plt.xlabel('True Value',fontsize=fontsize)
		plt.legend(**{'fontsize':fontsize},loc=2)
	plt.show(block=block)


def calc_p_dlt(predict_samps,y_test,weights=None,cov_dist_mat=None):
	"""	Calculate the percentage of draws whose radial distance
	(weighted by the covariance) is larger than that of the truth.

	Args:
		predict_samps (np.array): An array with dimensions (n_samples,
			batch_size,num_params) containing samples drawn from the
			predicted distribution.
		y_test (np.array): A (batch_size,num_params) array containing the
			true value of the parameter on the test set.
		weights (np.array): An array with dimension (n_samples,
			batch_size) that will be used to reweight the posterior.
		cov_dist_mat (np.array): A (num_params,num_params) array representing
			the covariance matrix to use for the distance calculation. If None
			a covariance matrix will be estimated from the y_test data.

	Returns:
		(np.array): A array of length batch_size containing the percentage
		of draws with larger radial distance.
	"""
	# Factor weights into the mean.
	if weights is None:
		y_mean = np.mean(predict_samps,axis=0)
	else:
		y_mean = np.mean(np.expand_dims(weights,axis=-1)*predict_samps,axis=0)

	# The metric for the distance calculation. Using numba for speed.
	@numba.njit
	def d_m(dif,cov):
		d_metric = np.zeros(dif.shape[0:2])
		for i in range(d_metric.shape[0]):
			for j in range(d_metric.shape[1]):
				d_metric[i,j] = np.dot(dif[i,j],np.dot(cov,dif[i,j]))
		return d_metric

	# Use emperical covariance for distance metric unless matrix was passed
	# in.
	if cov_dist_mat is None:
		cov_emp = np.cov(y_test.T)
	else:
		cov_emp = cov_dist_mat

	p_dlt = (d_m(predict_samps-y_mean,cov_emp)<
		d_m(np.expand_dims(y_test-y_mean,axis=0),cov_emp))

	# Calculate p_dlt factoring in the weights if needed.
	if weights is None:
		return np.mean(p_dlt,axis=0)
	else:
		return np.mean(p_dlt*weights,axis=0)


def plot_calibration(predict_samps,y_test,color_map=["#377eb8", "#4daf4a"],
	n_perc_points=20,figure=None,legend=None,show_plot=True,block=True,
	weights=None,title=None,ls='-',loc=9,dpi=200):
	"""	Plot the multidimensional calibration of the neural network predicted
	posteriors.

	Args:
		predict_samps (np.array): An array with dimensions (n_samples,
			batch_size,num_params) containing samples drawn from the
			predicted distribution.
		y_test (np.array): A (batch_size,num_params) array containing the
			true value of the parameter on the test set.
		color_map ([str,...]): A list of the colors to use in plotting.
		n_perc_point (int): The number of percentages to probe in the
			plotting.
		figure (matplotlib.pyplot.figure): A figure that was previously
			returned by plot_calibration to overplot onto.
		legend ([str,...]): The legend to use for plotting.
		show_plot (bool): If true, call plt.show() at the end of the
			function.
		block (bool): If true, block excecution after plt.show() command.
		weights (np.array): An array with dimension (n_samples,
			batch_size) that will be used to reweight the posterior.
		title (str): The title to use for the plot. If None will use a
			default title.
		ls (str): The line style to use in the calibration line for the
			BNN.
		loc (int or tuple): The location for the legend in the calibration
			plot.
		dpi (int): The dpi to use for the figure.

	Returns:
		(np.array): A array of length batch_size containing the percentage
		of draws with larger radial distance.
	"""
	# Go through each of our examples and see what percentage of draws have
	# ||draws||_2 < ||truth||_2 (essentially doing integration by radius).
	p_dlt = calc_p_dlt(predict_samps,y_test,weights)

	# Plot what percentage of images have at most x% of draws with
	# p(draws)>p(true).
	percentages = np.linspace(0.0,1.0,n_perc_points)
	p_images = np.zeros_like(percentages)
	if figure is None:
		fig = plt.figure(figsize=(8,8),dpi=dpi)
		plt.plot(percentages,percentages,c=color_map[0],ls='--')
	else:
		fig = figure

	# We'll estimate the uncertainty in our plat using a jacknife method.
	p_images_jn = np.zeros((len(p_dlt),n_perc_points))
	for pi in range(n_perc_points):
		percent = percentages[pi]
		p_images[pi] = np.mean(p_dlt<=percent)
		for ji in range(len(p_dlt)):
			samp_p_dlt = np.delete(p_dlt,ji)
			p_images_jn[ji,pi] = np.mean(samp_p_dlt<=percent)

	# Estimate the standard deviation from the jacknife
	p_dlt_std = np.sqrt((len(p_dlt)-1)*np.mean(np.square(p_images_jn-
		np.mean(p_images_jn,axis=0)),axis=0))
	plt.plot(percentages,p_images,c=color_map[1],ls=ls)

	# Plot the 1 sigma contours from the jacknife estimate to get an idea of
	# our sample variance.
	plt.fill_between(percentages,p_images+p_dlt_std,p_images-p_dlt_std,
		color=color_map[1],alpha=0.3)

	if figure is None:
		plt.xlabel('Percentage of Probability Volume')
		plt.ylabel('Percent of Lenses With True Value in the Volume')
		plt.text(-0.03,1,'Underconfident')
		plt.text(0.80,0,'Overconfident')
	if title is None:
		plt.title('Calibration of Network Posterior')
	else:
		plt.title(title)
	if legend is None:
		plt.legend(['Perfect Calibration','Network Calibration'],
			loc=loc)
	else:
		plt.legend(legend,loc=loc)
	if show_plot:
		plt.show(block=block)

	return fig
