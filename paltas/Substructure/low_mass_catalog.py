# -*- coding: utf-8 -*-
"""
Define the base class to draw subhalos for a lens

This module contains the base class that all the subhalo classes will build
from. Because the steps for rendering subhalos can vary between different
models, the required functions are very sparse.
"""
from .subhalos_base import SubhalosBase
from .los_base import LOSBase
from .subhalos_dg19 import SubhalosDG19
from .los_dg19 import LOSDG19
from . import nfw_functions
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
import lenstronomy.Util.param_util as param_util
import pandas as pd
import os
import numpy as np
from helpers.SimulationAnalysis import readHlist, SimulationAnalysis
import re
from astropy.cosmology import FlatLambdaCDM


class SubhalosCatalog(SubhalosBase):
	"""Class for rendering the subhalos of a main halos from a Rockstar
	catalog.

	Args:
		subhalo_parameters (dict): A dictionary containing the type of
			subhalo distribution and the value for each of its parameters.
		main_deflector_parameters (dict): A dictionary containing the type of
			main deflector and the value for each of its parameters.
		source_parameters (dict): A dictionary containing the type of the
			source and the value for each of its parameters.
		cosmology_parameters (str,dict, or
			colossus.cosmology.cosmology.Cosmology): Either a name
			of colossus cosmology, a dict with 'cosmology name': name of
			colossus cosmology, an instance of colussus cosmology, or a
			dict with H0 and Om0 ( other parameters will be set to defaults).

	Notes:

	Required Parameters

	- rockstar_path - path to rockstar directory
	- m_min - minumum halo mass in units of M_sun (i.e. the resolution limit to
	impose)
	- get_main - bool determining if parameters of the main deflector will be
	returned with the subhalos
	- return_at_infall - bool determining if the subhalo concentration and
	scale radius will be calculated at infall.
	- subhalo_profile - profile to use for the subhalos (currently
	implemented: NFW, TNFW, NFW_ELLIPSE)
	- get_subhalos - bool determining if only subhalos of host are returend. If
	set to False, return halos within +/-10 Mpc along the line-of-sight.
	"""

	required_parameters = ('rockstar_path','m_min', 'get_main',
		'return_at_infall','subhalo_profile', 'get_subhalos')

	def __init__(self,subhalo_parameters,main_deflector_parameters,
		source_parameters,cosmology_parameters):

		# Initialize the super class
		super().__init__(subhalo_parameters,main_deflector_parameters,
			source_parameters,cosmology_parameters)

	def get_scale_factor_from_filename(self,filename):
		"""Returns the scale factor from each hlist filename created by
		consistent trees.

		Args:
			filename (str): The filename to pull the float from

		Returns:
			(float): The scale factor
		"""
		return float(re.search(r'\d+\.\d+', filename).group(0))

	def get_scale_factor(self,rockstar_dir, redshift):
		"""Returns the filename and scale factor for the
		hlist at redshift=redshift
		Args:
			rockstar_dir (str): The path to the desired rockstar directory
			redshift (float): The redshift of the lens
		Returns:
			(float): The scale factor
			(str): The filename of the hlist file at scale factor = scale_factor
		"""
		hlist_file_names =  os.listdir(rockstar_dir + str('hlists'))
		scale_factors  = []
		for filename in hlist_file_names:
			scale_factors.append(self.get_scale_factor_from_filename(filename))
		scale_factors = np.array(scale_factors)
		nfile = np.argmin(np.abs((1./scale_factors-1)-redshift))
		return scale_factors[nfile], hlist_file_names[nfile]

	def load_host_and_subs(self,rockstar_dir, hlist_fname, scale_factor, cosmo,
		return_at_infall=False, get_subhalos=False):
		"""Returns host and subhalo parameters at redshift z_lens
		for subhalos concentration and scale_radius are returned at
		snapshot where 'vmax' is maximized

		Args:
			rockstar_dir (str): The path to the desired rockstar directory
			hlist_fnmae (str): Name of hlist file at redshift=z_lens
			scale_factor (float): The scale factor at redshift=z_lens
			cosmology_parameters (str,dict, or
				colossus.cosmology.cosmology.Cosmology): Either a name
				of colossus cosmology, a dict with 'cosmology name': name of
				colossus cosmology, an instance of colussus cosmology, or a
				dict with H0 and Om0 ( other parameters will be set to defaults).

		Returns:
			sub (dir): A directory containing subhalo parameters
				for subhalos with mass greater than m_min
			main (dir): a directory containing main halo parameters
		"""

		# Get h
		h = cosmo.h

		# fields we want from hlist halo catalog
		fields = ['treeRootId', 'upid', 'x', 'y', 'z', 'rvir','rs', 'm200c',
			'id', 'mvir', 'scale', 'b_to_a', 'c_to_a', 'A[x]', 'A[y]','A[z]']

		# Get host at z=0.0
		halos0 = readHlist(os.path.join(rockstar_dir+'hlists',
			'hlist_1.00000.list'),fields=fields)
		host0 = halos0[0]

		# get z=redshift halos
		halos = readHlist(os.path.join(rockstar_dir+'hlists', hlist_fname),
			fields=fields)

		# get host
		host = halos[halos['treeRootId']==host0['treeRootId']][0]

		# Get subhalos
		# If get_subhalos is TRUE return subhalos of host
		if get_subhalos is True:
			subhalos = halos[halos['upid']==host['id']]

		# Else return subhalos within +/- 10 Mpc in z- and
		else:
			mask1 = np.abs(halos['z'] - host['z'])*scale_factor/h < 10
			mask2 = (np.sqrt((halos['x']-host['x'])**2 +
				(halos['y']-host['y'])**2)*scale_factor/h < 0.5)
			mask = mask1&mask2
			subhalos = halos[mask]

		# Only return subhalos with mass greater than m_min
		subhalos['m200c'] = subhalos['m200c']/h

		mask = subhalos['m200c'] > self.subhalo_parameters['m_min']
		mask1 = subhalos['m200c'] < 1e10
		mask = mask&mask1
		subhalos = subhalos[mask]

		concentration = []
		rs = []

		# Get subhalos parameters at infall
		if return_at_infall is True:
			sim = SimulationAnalysis(trees_dir=rockstar_dir+'trees')
			for i in range(len(subhalos)):
				sub_tree = sim.load_main_branch(subhalos['treeRootId'][i])
				ind_vpeak = np.argmax(sub_tree['vmax'])
			# Calculate r200
				r200s = nfw_functions.r_200_from_m(subhalos['m200c'][i],
					self.main_deflector_parameters['z_lens'], cosmo)
				c_vpeak = (r200s/sub_tree['rs'])[ind_vpeak]
				concentration.append(c_vpeak)
				# Get scale radius and change from comoving to physical
				rs.append((sub_tree['rs']*sub_tree['scale'])[ind_vpeak])
		else:
			c_vpeak = np.zeros(len(subhalos))
			rs = np.zeros(len(subhalos))
		sub, host = self.get_subhalo_catalog(subhalos, host, scale_factor,
			cosmo, np.array(c_vpeak), np.array(rs),
			self.subhalo_parameters['get_main'], return_at_infall)

		return sub, host

	def rotate_project(self,x, y, z, theta_x, theta_y, theta_z):
		"""Rotates with the three Euler angles and projects in the z'-axis
		after rotation. See e.g.
		https://en.wikipedia.org/wiki/3D_projection#Perspective_projection

		Args:

		Returns:
		"""
		c_x, s_x = np.cos(theta_x), np.sin(theta_x)
		c_y, s_y = np.cos(theta_y), np.sin(theta_y)
		c_z, s_z = np.cos(theta_z), np.sin(theta_z)
		x_ = c_y * (s_z * y + c_z * x) - s_y * z
		y_ = s_x * (c_y * z + s_y * (s_z * y + c_z * x)) + c_x * (
			c_z * y - s_z * x)
		return x_, y_

	def get_subhalo_catalog(self,subhalos, host, scale_factor, cosmo, sub_c,
		sub_rs, return_host=True,return_at_infall=False):
		"""Generates catalog of subhalo parameters necessary for lensing.

		Args:
			sub (dict): A dictionary containing the subhalo parameters from
				rockstar catalog
			host (dict): A dictionary containing the host halo parameters from
				rockstar catalog
			minimum_mass (float): minimum mass of subhalos to draw from
				rockstar catalog
			scale_factor (float): scale_factor at z_lens
			cosmo: cosmology parameters
			sub_c: subhalo concentration computed at vpeak
			sub_rs: subhalo scale radius computed at vpeak (in physical units)
			delta_vir: the virial overdensity in units of the critical density

		Returns:
			A catalog containing subhalo parameters and a catalog
			containing host halo parameters. Unless return_host = True
			the host halo catalog will be empty.
				"""

		columns = ('x', 'y', 'z', 'c', 'rs', 'rvir', 'm200', 'A[x]', 'A[y]',
			'A[z]', 'b_to_a', 'c_to_a')
		sub = pd.DataFrame(columns=columns)
		main = pd.DataFrame(columns=columns)

		h = cosmo.h

		center_x = host['x']
		center_y = host['y']
		center_z = host['z']

		sub['m200'] = subhalos['m200c']
		sub['x'] = (subhalos['x'] - center_x)*1000*scale_factor/h
		sub['y'] = (subhalos['y'] - center_y)*1000*scale_factor/h
		sub['z'] = (subhalos['z'] - center_z)*1000*scale_factor/h
		sub['A[x]'] = (subhalos['A[x]'])*scale_factor/h
		sub['A[y]'] = (subhalos['A[y]'])*scale_factor/h
		sub['A[z]'] = (subhalos['A[z]'])*scale_factor/h
		sub['b_to_a'] = subhalos['b_to_a']
		sub['c_to_a'] = subhalos['c_to_a']
		sub['rvir'] = subhalos['rvir']*scale_factor/h

		if return_at_infall is True:
			sub['rs'] = sub_rs
			sub['c'] = sub_c
		else:
			sub['rs'] = subhalos['rs']*scale_factor/h
			r200s = nfw_functions.r_200_from_m(subhalos['m200c'],
				self.main_deflector_parameters['z_lens'],self.cosmo)
			sub['c'] = r200s/subhalos['rs']

		if return_host is True:
			r200 = [nfw_functions.r_200_from_m(host['m200c'],
				self.main_deflector_parameters['z_lens'],self.cosmo)]
			main['c'] = r200/host['rs']
			main['m200'] = [host['m200c']/h]
			main['x'] = [0]
			main['y'] = [0]
			main['z'] = [0]
			main['rs'] = [host['rs']*scale_factor/h]
			main['rvir'] = [host['rvir']*scale_factor/h]
			main['A[x]'] = [host['A[x]']*scale_factor/h]
			main['A[y]'] = [host['A[y]']*scale_factor/h]
			main['A[z]'] = [host['A[z]']*scale_factor/h]
			main['b_to_a'] = [host['b_to_a']]
			main['c_to_a'] = [host['c_to_a']]

		return sub, main

	def generate_analytic(self,sub, host, cosmo_base, z_source, z_lens,
		include_host=True, subhalo_profile='NFW', host_profile='TNFW'):
		"""Generates analytic lensing profile from list of subhalos.

		Args:
			sub (dict): A dictionary containing the subhalo parameters necessary
					for generating analytic nfw profile
			host (dict): A dictionary containing the host halo parameters necessary
					for generating analytic nfw profile
			cosmo: cosmology parameters
			z_source: Source redshift
			z_lens: Lens redshift
			include_host: Flag to return lensing profile for host halo along
				with subhalos
			subhalo_profile: mass profile for subhalos ['NFW', 'TNFW',
				'NFW_ELLIPSE'] host_profile: mass profile for main deflector
				['NFW', 'TNFW', 'NFW_ELLIPSE']

		Returns:
				The analytic lens profile for all subhalos with m200c > m_min,
				and the lenstronomy kwargs.
		"""

		sub_x, sub_y = self.rotate_project(sub['x'], sub['y'], sub['z'], 0, 0,
			0)

		cosmo = FlatLambdaCDM(H0=cosmo_base.H0, Om0=cosmo_base.Om0)

		lens_cosmo = LensCosmo(z_lens=z_lens, z_source=z_source, cosmo=cosmo)

		Rs_angle, alpha_Rs = lens_cosmo.nfw_physical2angle(M=sub['m200'],
			c=sub['c'])

		alpha_Rs = alpha_Rs.values

		xx = lens_cosmo.phys2arcsec_lens(sub_x/1000.).values
		yy = lens_cosmo.phys2arcsec_lens(sub_y/1000.).values
		rr = lens_cosmo.phys2arcsec_lens(sub['rs']/1000).values

		ellipticity_max = 0.1
		if include_host is True:
			host_x, host_y = self.rotate_project(host['x'], host['y'],
				host['z'], 0, 0, 0)

			Rs_angle_host, alpha_Rs_host = lens_cosmo.nfw_physical2angle(
				M=host['m200'], c=host['c'])

			rr_host = lens_cosmo.phys2arcsec_lens(host['rs']/1000)
			if(host_profile=='NFW_ELLIPSE'):
				ellipse_x, ellipse_y = self.rotate_project(host['A[x]']+
					host['x'], host['A[y]']+host['y'], host['A[z]']+host['z'],
					0, 0, 0)
				rz = host['A[z]']/np.sqrt(host['A[x]']**2 + host['A[y]']**2 +
					host['A[z]']**2)
				q = rz+np.sqrt(host['b_to_a']*host['c_to_a'])*(1-rz)
				ellipse_x = ellipse_x - host_x
				ellipse_y = ellipse_y - host_y
				z = [-1,0]
				if(ellipse_x.values[0]<0):
					theta = np.array([ellipse_y.values[0], ellipse_x.values[0]])
					phi = np.arccos(np.dot(z,theta)/(np.linalg.norm(theta)*
						np.linalg.norm(z)))
				else:
					theta = np.array([-ellipse_y.values[0], -ellipse_x.values[0]])
					phi = np.arccos(np.dot(z,theta)/(np.linalg.norm(theta)*
						np.linalg.norm(z)))
				host_e1, host_e2 = param_util.phi_q2_ellipticity(phi=phi,
					q=q.values[0])
				host_e1 = min(np.abs(host_e1),ellipticity_max)*np.sign(host_e1)
				host_e2 = min(np.abs(host_e2),ellipticity_max)*np.sign(host_e2)
			if(host_profile=='TNFW'):
				rtrunc_host = lens_cosmo.phys2arcsec_lens(host['rvir']/1000)

		if(subhalo_profile=='NFW_ELLIPSE' or subhalo_profile=='TNFW_ELLIPSE'):
			ellipse_x, ellipse_y = self.rotate_project(sub['A[x]']+sub['x'],
				sub['A[y]']+sub['y'], sub['A[z]']+sub['z'], 0, 0, 0)
			rz = sub['A[z]']/np.sqrt(sub['A[x]']**2 + sub['A[y]']**2 +
				sub['A[z]']**2)
			q = rz+np.sqrt(sub['b_to_a']*sub['c_to_a'])*(1-rz)
			ellipse_x = ellipse_x - sub_x
			ellipse_y = ellipse_y - sub_y
			z = [-1,0]
			phi = []
			for i in range(len(ellipse_x)):
				if(ellipse_x[i]<0):
					theta = np.array([ellipse_y[i], ellipse_x[i]])
					phi.append(np.arccos(np.dot(z,theta)/(
						np.linalg.norm(theta)*np.linalg.norm(z))))
				else:
					theta = np.array([-ellipse_y[i], -ellipse_x[i]])
					phi.append(np.arccos(np.dot(z,theta)/(
						np.linalg.norm(theta)*np.linalg.norm(z))))

			e1 = []
			e2 = []
			for i in range(len(phi)):
				ee1, ee2 = param_util.phi_q2_ellipticity(phi=phi[i], q=q[i])
				e1.append(min(np.abs(ee1),ellipticity_max)*np.sign(ee1))
				e2.append(min(np.abs(ee2),ellipticity_max)*np.sign(ee2))

		if subhalo_profile=='NFW':
			kwargs_new = [{'Rs': rr[i], 'alpha_Rs': alpha_Rs[i],
				'center_x': yy[i], 'center_y':xx[i]} for i in range(len(rr))]
		elif subhalo_profile=='TNFW':
			radii = np.sqrt(sub['x']**2 + sub['y']**2 + sub['z']**2)
			m200 = sub['m200']
			kwargs_new = [{'Rs': rr[i], 'alpha_Rs': alpha_Rs[i],
				'r_trunc': SubhalosDG19.get_truncation_radius(m200[i],
					radii[i]), 'center_x': yy[i], 'center_y':xx[i]}
				for i in range(len(rr))]
		elif subhalo_profile=='NFW_ELLIPSE':
			kwargs_new = [{'Rs': rr[i], 'alpha_Rs': alpha_Rs[i],
				'e1': e1[i], 'e2': e2[i], 'center_x': yy[i],
				'center_y':xx[i]} for i in range(len(rr))]
		elif subhalo_profile=='TNFW_ELLIPSE':
			radii = np.sqrt(sub['x']**2 + sub['y']**2 + sub['z']**2)
			m200 = sub['m200']
			kwargs_new = [{'Rs': rr[i], 'alpha_Rs': alpha_Rs[i],
				'r_trunc': 5*rr[i],'e1': e1[i],'e2': e2[i],
				'center_x': yy[i], 'center_y':xx[i]} for i in range(len(rr))]
		else:
			print('Subhalo profile not implemented. Defaulting to NFW')
			subhalo_profile='NFW'
			kwargs_new = [{'Rs': rr[i], 'alpha_Rs': alpha_Rs[i],
				'center_x': yy[i], 'center_y':xx[i]} for i in range(len(rr))]
		profiles = [subhalo_profile]*len(alpha_Rs)

		if include_host is True:
			if host_profile=='NFW_ELLIPSE':
				profiles.append('NFW_ELLIPSE')
				kwargs_new.append({'Rs': rr_host.values[0],
					'alpha_Rs': alpha_Rs_host.values[0],'e1': host_e1,
					'e2': host_e2,'center_x': 0,'center_y':0})
			elif host_profile=='TNFW':
				profiles.append('TNFW')
				kwargs_new.append({'Rs': rr_host.values[0],
					'alpha_Rs': alpha_Rs_host.values[0],
					'r_trunc':rtrunc_host.values[0], 'center_x': 0,
					'center_y':0})
			else:
				profiles.append('NFW')
				kwargs_new.append({'Rs': rr_host.values[0],
					'alpha_Rs': alpha_Rs_host.values[0], 'center_x': 0,
					'center_y':0})

		return profiles, kwargs_new

	def draw_subhalos(self):
		"""Draws subhalos at specified redshift from given rocktar catalog
		for subhalos with m200c > m_min.

		Returns:
			(tuple): the first analytic lens profile for all subhalos with
			m200c > m_min, the second is the lenstronomy kwargs. This will
			include the main deflector if get_main=True.
		"""
		redshift = self.main_deflector_parameters['z_lens']
		rockstar_dir = self.subhalo_parameters['rockstar_path']

		scale_factor, hlist_fname = self.get_scale_factor(rockstar_dir,
			redshift)  # is this correct?

		sub, host = self.load_host_and_subs(rockstar_dir, hlist_fname,
			scale_factor, self.cosmo,
			self.subhalo_parameters['return_at_infall'],
			self.subhalo_parameters['get_subhalos'])

		lens, kwargs = self.generate_analytic(sub, host, self.cosmo,
			self.source_parameters['z_source'], redshift,
			self.subhalo_parameters['get_main'],
			self.subhalo_parameters['subhalo_profile'],
			self.main_deflector_parameters['host_profile'])

		subhalo_z_list = [redshift]*len(lens)

		return (lens, kwargs,subhalo_z_list)


class LowMassFromSims(LOSBase):
	"""Class for rendering the low mass halos inlcuding subhalos and los halos
	from a rockstar catalog and filling in halos below the mass limit with
	semi-analytic description.

	Args:
		subhalo_parameters (dict): A dictionary containing the type of
			subhalo distribution and the value for each of its parameters.
		main_deflector_parameters (dict): A dictionary containing the type of
			main deflector and the value for each of its parameters.
		source_parameters (dict): A dictionary containing the type of the
			source and the value for each of its parameters.
		cosmology_parameters (str,dict, or
			colossus.cosmology.cosmology.Cosmology): Either a name
			of colossus cosmology, a dict with 'cosmology name': name of
			colossus cosmology, an instance of colussus cosmology, or a
			dict with H0 and Om0 ( other parameters will be set to defaults).
	"""

	required_parameters = ('catalog_rockstar_path','catalog_m_min',
		'catalog_get_main','catalog_return_at_infall','catalog_subhalo_profile',
		'catalog_get_subhalos','subhalos_sigma_sub','subhalos_shmf_plaw_index',
		'subhalos_m_pivot','subhalos_m_min','subhalos_m_max','subhalos_c_0',
		'subhalos_conc_zeta','subhalos_conc_beta','subhalos_conc_m_ref',
		'subhalos_dex_scatter','subhalos_k1','subhalos_k2','los_m_min',
		'los_m_max','los_z_min','los_dz','los_cone_angle','los_r_max',
		'los_r_min','los_c_0','los_conc_zeta','los_conc_beta','los_conc_m_ref',
		'los_dex_scatter','los_delta_los','los_alpha_dz_factor')

	def __init__(self,los_parameters,main_deflector_parameters,
		source_parameters,cosmology_parameters):

		# Initialize the super class
		super().__init__(los_parameters,main_deflector_parameters,
			source_parameters,cosmology_parameters)

		# Initialize the parameters for each of our four classes.
		self.catalog_parameters = {}
		self.sub_analy_params = {}
		self.los_inner_analy_params = {}
		self.los_outer_analy_params = {}
		self.update_parameters(los_parameters,main_deflector_parameters,
			source_parameters,cosmology_parameters)

		# Initialize our four classes
		self.catalog_class = SubhalosCatalog(self.catalog_parameters,
			self.main_deflector_parameters,self.source_parameters,
			self.cosmology_parameters)
		self.sub_analy_class = SubhalosDG19(self.sub_analy_params,
			self.main_deflector_parameters,self.source_parameters,
			self.cosmology_parameters)
		self.los_inner_analy_class = LOSDG19(self.los_inner_analy_params,
			self.main_deflector_parameters,self.source_parameters,
			self.cosmology_parameters)
		self.los_outer_analy_params = LOSDG19(self.los_outer_analy_params,
			self.main_deflector_parameters,self.source_parameters,
			self.cosmology_parameters)

	def update_parameters(self,los_parameters=None,
		main_deflector_parameters=None,source_parameters=None,
		cosmology_parameters=None):
		"""Updated the class parameters

		Args:
			los_parameters (dict): A dictionary containing the type of
				los distribution and the value for each of its parameters.
			main_deflector_parameters (dict): A dictionary containing the type
				of main deflector and the value for each of its parameters.
			source_parameters (dict): A dictionary containing the type of the
				source and the value for each of its parameters.
			cosmology_parameters (str,dict, or colossus.cosmology.Cosmology):
				Either a name of colossus cosmology, a dict with 'cosmology name':
				name of colossus cosmology, an instance of colussus cosmology, or
				a dict with H0 and Om0 ( other parameters will be set to
				defaults).

		Notes:
			Use this function to update parameter values instead of
			starting a new class.
		"""
		super().update_parameters(los_parameters,main_deflector_parameters,
			source_parameters,cosmology_parameters)

		# Update the parameters we're going to pass to each class
		for param in self.los_parameters:
			if param[:8] == 'catalog_':
				self.catalog_parameters[param[8:]] = (
					self.los_parameters[param])
			elif param[:9] == 'subhalos_':
				self.sub_analy_params[param[9:]] = (
					self.los_parameters[param])
			elif param[:4] == 'los_':
				self.los_inner_analy_params[param[4:]] = (
					self.los_parameters[param])
				self.los_outer_analy_params[param[4:]] = (
					self.los_parameters[param])

		# For the region where we pull subhalos from the simulation
		# override the maximum mass to the minimum catalog mass.
		self.sub_analy_params['m_max'] = max(self.sub_analy_params['m_max'],
			self.catalog_parameters['m_min'])
		self.los_inner_analy_params['m_max'] = max(
			self.los_inner_analy_params['m_max'],
			self.catalog_parameters['m_min'])

	def draw_los(self):
		"""Draws masses, concentrations,and positions for the los substructure
		of a main lens halo.

		Returns:
			(tuple): A tuple of three lists: the first is the profile type for
			each los halo returned, the second is the lenstronomy kwargs
			for that halo, and the third is a list of redshift values for
			each profile.

		Notes:
			This class will actually return subhalos as well, so it should
			not be paired with any subhalo class in the config.
		"""
		# The catalog halos and subhalos you can just draw for free
		# Draw the list and appends them to our total lists.
		catalog_model_list, catalog_kwargs_list, catalog_z_list = (
			self.catalog_class.draw_subhalos())
		subhalo_model_list, subhalo_kwargs_list, subhalo_z_list = (
			self.sub_analy_class.draw_subhalos())

		low_mass_model_list = catalog_model_list + subhalo_model_list
		low_mass_kwargs_list = catalog_kwargs_list + subhalo_kwargs_list
		low_mass_z_list = catalog_z_list + subhalo_z_list

		# With the los halos we need to select the region in which we
		# want each draw.
		los_inner_model_list, los_inner_kwargs_list, los_inner_z_list = (
			self.los_inner_analy_class.draw_los())
		los_outer_model_list, los_outer_kwargs_list, los_outer_z_list = (
			self.los_outer_analy_class.draw_los())

		# Keep the inner draws within the two closest redshift bins to the
		# host
		for zi, z in enumerate(los_outer_z_list):
			if (z <= self.main_deflector_parameters['z_lens'] -
				self.los_outer_analy_params['dz'] or
				z >= self.main_deflector_parameters['z_lens'] +
				self.los_outer_analy_params['dz']):
				# Append halos that meet the criteria
				low_mass_model_list.append(los_outer_model_list[zi])
				low_mass_kwargs_list.append(los_outer_kwargs_list[zi])
				low_mass_z_list.append(los_outer_z_list[zi])

		for zi, z in enumerate(los_inner_z_list):
			if (z > self.main_deflector_parameters['z_lens'] -
				self.los_inner_analy_params['dz'] or
				z < self.main_deflector_parameters['z_lens'] +
				self.los_inner_analy_params['dz']):
				# Append halos that meet the criteria
				low_mass_model_list.append(los_inner_model_list[zi])
				low_mass_kwargs_list.append(los_inner_kwargs_list[zi])
				low_mass_z_list.append(los_inner_z_list[zi])

		return low_mass_model_list, low_mass_kwargs_list, low_mass_z_list

	def calculate_average_alpha(self,num_pix):
		""" Calculates the average deflection maps from the los at each
		redshift specified by the los parameters and returns corresponding
		lenstronomy objects.

		Args:
			num_pix (int): The number of pixels to sample for our
				interpolation maps.

		Returns:
			(tuple): A tuple of two lists: the first is the interpolation
			profile type for each redshift slice and the second is the
			lenstronomy kwargs for that profile.

		Notes:
			The average los deflection angles of the lenstronomy objects will
			be the negative of the average (since we want to subtract the
			average effect not add it). Pixel scale will be set such that
			at each redshift a box of 5*r_los is captured.
		"""

		return self.los_outer_analy_class.calculate_average_alpha(num_pix)
