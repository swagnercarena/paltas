# -*- coding: utf-8 -*-
"""
Define the base class to draw subhalos for a lens

This module contains the base class that all the subhalo classes will build
from. Because the steps for rendering subhalos can vary between different
models, the required functions are very sparse.
"""
from .subhalos_base import SubhalosBase
from .subhalos_dg19 import SubhalosDG19 
from . import nfw_functions
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
import lenstronomy.Util.param_util as param_util
import pandas as pd
import os
import numpy as np
from helpers.SimulationAnalysis import readHlist, SimulationAnalysis
import astropy
from astropy.cosmology import WMAP9 as cosmo
import re
from astropy.cosmology import FlatLambdaCDM

# Define the parameters we expect to find for reading from the simulation catalog
#rockstar_path (str): path to rockstar directory
#m_min (float): minumum halo mass
#get_main (bool): if True, return parameters for main deflector as well as subhalos 
#return_at_infall (bool): if True, calculate subhalo concentration and scale radius at infall 
#subhalo_profile: profile for subhalos (currently implemented: NFW, TNFW, NFW_ELLIPSE)
#get_subhalos: if True, return only subhalos of host, if False, return subhalos within +/-10 Mpc along the line-of-sight 

draw_simulation_catalog_parameters = ['rockstar_path','m_min', 'get_main', 'return_at_infall','subhalo_profile', 'get_subhalos']

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
	"""

	def __init__(self,subhalo_parameters,main_deflector_parameters,
		source_parameters,cosmology_parameters):

		# Initialize the super class
		super().__init__(subhalo_parameters,main_deflector_parameters,
			source_parameters,cosmology_parameters)

		# Check that all the needed parameters are present
		self.check_parameterization(draw_simulation_catalog_parameters)

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

	def load_host_and_subs(self,rockstar_dir, hlist_fname, scale_factor, cosmo, return_at_infall=False, get_subhalos=False):
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
		#get h 
		h = cosmo.h

		# fields we want from hlist halo catalog
		fields = ['treeRootId', 'upid', 'x', 'y', 'z', 'rvir','rs', 'm200c', 'id', 'mvir', 'scale', 'b_to_a', 'c_to_a', 'A[x]', 'A[y]','A[z]']

		#get host at z=0.0 
		halos0 = readHlist(os.path.join(rockstar_dir+'hlists', 'hlist_1.00000.list'),fields=fields)
		host0 = halos0[0]

		# get z=redshift halos
		halos = readHlist(os.path.join(rockstar_dir+'hlists', hlist_fname),fields=fields)

		# get host
		host = halos[halos['treeRootId']==host0['treeRootId']][0]

		# get subhalos
		#if get_subhalos is TRUE return subhalos of host
		if(get_subhalos==True):
			subhalos = halos[halos['upid']==host['id']]

		#else return subhalos within +/- 10 Mpc in z- and 
		else:
			mask1 = np.abs(halos['z'] - host['z'])*scale_factor/h < 10 
			mask2 = np.sqrt((halos['x']-host['x'])**2 + (halos['y']-host['y'])**2)*scale_factor/h < 0.5
			mask = mask1&mask2
			subhalos = halos[mask] 	
	
		#only return subhalos with mass greater than m_min 
		subhalos['m200c'] = subhalos['m200c']/h

		mask = subhalos['m200c'] > self.subhalo_parameters['m_min']
		mask1 = subhalos['m200c'] < 1e10 
		mask = mask&mask1
		subhalos = subhalos[mask]

		concentration = []
		rs = []

		#get subhalos parameters at infall
		if(return_at_infall == True):
			sim = SimulationAnalysis(trees_dir=rockstar_dir+'trees')
			for i in range(len(subhalos)):
				sub_tree = sim.load_main_branch(subhalos['treeRootId'][i])
				ind_vpeak = np.argmax(sub_tree['vmax'])
				rshift = 1./(sub_tree['scale']) - 1.
			#calculate r200
				r200s = nfw_functions.r_200_from_m(subhalos['m200c'][i], 
					self.main_deflector_parameters['z_lens'], cosmo)
				c_vpeak = (r200s/sub_tree['rs'])[ind_vpeak]
				concentration.append(c_vpeak)
				rs.append((sub_tree['rs']*sub_tree['scale'])[ind_vpeak]) #change from comoving to physical coords
		else:
			c_vpeak = np.zeros(len(subhalos))
			rs = np.zeros(len(subhalos))
		sub, host = self.get_subhalo_catalog(subhalos, host, scale_factor, cosmo, np.array(c_vpeak), np.array(rs), self.subhalo_parameters['get_main'], return_at_infall)

		return sub, host 

	def rotate_project(self,x, y, z, theta_x, theta_y, theta_z):
		"""
		rotates with the three Euler angles and projects in the z'-axis after rotation
		See e.g. https://en.wikipedia.org/wiki/3D_projection#Perspective_projection
                Args:

                Returns:
                """
		c_x, s_x = np.cos(theta_x), np.sin(theta_x)
		c_y, s_y = np.cos(theta_y), np.sin(theta_y)
		c_z, s_z = np.cos(theta_z), np.sin(theta_z)
		x_ = c_y * (s_z * y + c_z * x) - s_y * z
		y_ = s_x * (c_y * z + s_y * (s_z * y + c_z * x)) + c_x * (c_z * y - s_z * x)
		return x_, y_

	def get_subhalo_catalog(self,subhalos, host, scale_factor, cosmo, sub_c, sub_rs, return_host =True,return_at_infall=False):
		"""Generates catalog of subhalo parameters necessary for lensing.

				Args:
				sub (dict): A dictionary containing the subhalo parameters from rockstar catalog
				host (dict): A dictionary containing the host halo parameters from rockstar catalog
		minimum_mass: minimum mass of subhalos to draw from rockstar catalog
				scale_factor: scale_factor at z_lens
		cosmo: cosmology parameters
				sub_c: subhalo concentration computed at vpeak
				sub_rs: subhalo scale radius computed at vpeak (in physical units)
				delta_vir: the virial overdensity in units of the critical density

				Returns:
						A catalog containing subhalo parameters and a catalog
			containing host halo parameters. Unless return_host = True
			the host halo catalog will be empty.
				"""

		columns = ('x', 'y', 'z', 'c', 'rs', 'rvir', 'm200', 'A[x]', 'A[y]', 'A[z]', 'b_to_a', 'c_to_a')
		sub = pd.DataFrame(columns=columns)
		main = pd.DataFrame(columns=columns)

		h = cosmo.h

		center_x = host['x']
		center_y = host['y']
		center_z = host['z']
#		print(center_x)
#		print(center_y)
#		print(center_z)

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

		if(return_at_infall == True): 
			sub['rs'] = sub_rs
			sub['c'] = sub_c
		else:
			sub['rs'] = subhalos['rs']*scale_factor/h 
			r200s = nfw_functions.r_200_from_m(subhalos['m200c'], self.main_deflector_parameters['z_lens'],self.cosmo)
			sub['c'] = r200s/subhalos['rs']

		if return_host == True:
			r200 = [nfw_functions.r_200_from_m(host['m200c'], self.main_deflector_parameters['z_lens'],self.cosmo)]
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

	def generate_analytic(self,sub, host, cosmo_base, z_source, z_lens, include_host=True, subhalo_profile='NFW', host_profile='TNFW'):
		"""Generates analytic lensing profile from list of subhalos.

		Args:
		sub (dict): A dictionary containing the subhalo parameters necessary
				for generating analytic nfw profile
		host (dict): A dictionary containing the host halo parameters necessary
				for generating analytic nfw profile
		cosmo: cosmology parameters
		z_source: Source redshift
		z_lens: Lens redshift
		include_host: Flag to return lensing profile for host halo along with subhalos
		subhalo_profile: mass profile for subhalos ['NFW', 'TNFW', 'NFW_ELLIPSE']
                host_profile: mass profile for main deflector ['NFW', 'TNFW', 'NFW_ELLIPSE']

		Returns:
				The analytic lens profile for all subhalos with m200c > m_min,
				and the lenstronomy kwargs.
		"""

		h = cosmo_base.h

		sub_x, sub_y = self.rotate_project(sub['x'], sub['y'], sub['z'], 0, 0, 0)

		cosmo = FlatLambdaCDM(H0 = cosmo_base.H0, Om0 = cosmo_base.Om0)

		lens_cosmo = LensCosmo(z_lens=z_lens, z_source=z_source, cosmo=cosmo)

		Rs_angle, alpha_Rs = lens_cosmo.nfw_physical2angle(M=sub['m200'], c=sub['c'])

		alpha_Rs = alpha_Rs.values

		xx = lens_cosmo.phys2arcsec_lens(sub_x/1000.).values
		yy = lens_cosmo.phys2arcsec_lens(sub_y/1000.).values
		rr = lens_cosmo.phys2arcsec_lens(sub['rs']/1000).values

		ellipticity_max = 0.1 
		if(include_host==True):
			host_x, host_y = self.rotate_project(host['x'], host['y'], host['z'], 0, 0, 0)

			Rs_angle_host, alpha_Rs_host = lens_cosmo.nfw_physical2angle(M=host['m200'], c=host['c'])

			xx_host = lens_cosmo.phys2arcsec_lens(host_x/1000.)
			yy_host = lens_cosmo.phys2arcsec_lens(host_y/1000.)
			rr_host = lens_cosmo.phys2arcsec_lens(host['rs']/1000)
			if(host_profile=='NFW_ELLIPSE'):
				ellipse_x, ellipse_y = self.rotate_project(host['A[x]']+host['x'], host['A[y]']+host['y'], host['A[z]']+host['z'], 0, 0, 0)
				rz = host['A[z]']/np.sqrt(host['A[x]']**2 + host['A[y]']**2 + host['A[z]']**2)
				q = rz+np.sqrt(host['b_to_a']*host['c_to_a'])*(1-rz)
				ellipse_x = ellipse_x - host_x
				ellipse_y = ellipse_y - host_y
				z = [-1,0]
				if(ellipse_x.values[0]<0):
					theta = np.array([ellipse_y.values[0], ellipse_x.values[0]])
					phi = np.arccos(np.dot(z,theta)/(np.linalg.norm(theta)*np.linalg.norm(z)))
				else:
					theta = np.array([-ellipse_y.values[0], -ellipse_x.values[0]])
					phi = np.arccos(np.dot(z,theta)/(np.linalg.norm(theta)*np.linalg.norm(z)))
				host_e1, host_e2 = param_util.phi_q2_ellipticity(phi=phi, q=q.values[0])
				host_e1 = min(np.abs(host_e1),ellipticity_max)*np.sign(host_e1)
				host_e2 = min(np.abs(host_e2),ellipticity_max)*np.sign(host_e2)
			if(host_profile=='TNFW'):	
				rtrunc_host = lens_cosmo.phys2arcsec_lens(host['rvir']/1000) 


		if(subhalo_profile=='NFW_ELLIPSE' or subhalo_profile=='TNFW_ELLIPSE'):
			ellipse_x, ellipse_y = self.rotate_project(sub['A[x]']+sub['x'], sub['A[y]']+sub['y'], sub['A[z]']+sub['z'], 0, 0, 0)
			rz = sub['A[z]']/np.sqrt(sub['A[x]']**2 + sub['A[y]']**2 + sub['A[z]']**2)
			q = rz+np.sqrt(sub['b_to_a']*sub['c_to_a'])*(1-rz)
			ellipse_x = ellipse_x - sub_x
			ellipse_y = ellipse_y - sub_y 
			z = [-1,0]
			phi = [] 
			for i in range(len(ellipse_x)):
				if(ellipse_x[i]<0):
					theta = np.array([ellipse_y[i], ellipse_x[i]])
					phi.append(np.arccos(np.dot(z,theta)/(np.linalg.norm(theta)*np.linalg.norm(z))))
				else:
					theta = np.array([-ellipse_y[i], -ellipse_x[i]])
					phi.append(np.arccos(np.dot(z,theta)/(np.linalg.norm(theta)*np.linalg.norm(z))))

			e1 = []
			e2 = []
			for i in range(len(phi)):
			    ee1, ee2 = param_util.phi_q2_ellipticity(phi=phi[i], q=q[i])
			    e1.append(min(np.abs(ee1),ellipticity_max)*np.sign(ee1))
			    e2.append(min(np.abs(ee2),ellipticity_max)*np.sign(ee2))

		if(subhalo_profile=='NFW'): 
			kwargs_new = [{'Rs': rr[i], 'alpha_Rs': alpha_Rs[i], 'center_x': yy[i], 'center_y':xx[i]} for i in range(len(rr))]
		elif(subhalo_profile=='TNFW'):
			radii = np.sqrt(sub['x']**2 + sub['y']**2 + sub['z']**2)
			m200 = sub['m200']
			kwargs_new = [{'Rs': rr[i], 'alpha_Rs': alpha_Rs[i],'r_trunc': SubhalosDG19.get_truncation_radius(m200[i], radii[i]), 'center_x': yy[i], 'center_y':xx[i]} for i in range(len(rr))]
		elif(subhalo_profile=='NFW_ELLIPSE'):
			kwargs_new = [{'Rs': rr[i], 'alpha_Rs': alpha_Rs[i],  'e1': e1[i], 'e2': e2[i], 'center_x': yy[i], 'center_y':xx[i]} for i in range(len(rr))]
		elif(subhalo_profile=='TNFW_ELLIPSE'):
			radii = np.sqrt(sub['x']**2 + sub['y']**2 + sub['z']**2)
			m200 = sub['m200']
			kwargs_new = [{'Rs': rr[i], 'alpha_Rs': alpha_Rs[i], 'r_trunc': 5*rr[i],  'e1': e1[i], 'e2': e2[i], 'center_x': yy[i], 'center_y':xx[i]} for i in range(len(rr))]
		else:
			print('Subhalo profile not implemented. Defaulting to NFW')
			subhalo_profile='NFW'
			kwargs_new = [{'Rs': rr[i], 'alpha_Rs': alpha_Rs[i], 'center_x': yy[i], 'center_y':xx[i]} for i in range(len(rr))]
		profiles = [subhalo_profile]*len(alpha_Rs)

		if(include_host == True):
			if(host_profile=='NFW_ELLIPSE'):
				profiles.append('NFW_ELLIPSE')
				kwargs_new.append({'Rs': rr_host.values[0], 'alpha_Rs': alpha_Rs_host.values[0],  'e1': host_e1, 'e2': host_e2,  'center_x': 0, 'center_y':0})
			elif(host_profile=='TNFW'):
				profiles.append('TNFW')
				kwargs_new.append({'Rs': rr_host.values[0], 'alpha_Rs': alpha_Rs_host.values[0],  'r_trunc':rtrunc_host.values[0], 'center_x': 0, 'center_y':0})
			else:
				profiles.append('NFW')
				kwargs_new.append({'Rs': rr_host.values[0], 'alpha_Rs': alpha_Rs_host.values[0],  'center_x': 0, 'center_y':0})
		#lens_analytic = LensModel(lens_model_list=profiles, lens_redshift_list=z_lens, z_source=z_source, cosmo=cosmo, multi_plane=False)

		return profiles, kwargs_new


	def draw_subhalos(self):
		"""Draws subhalos at specified redshift from given rocktar catalog
		for subhalos with m200c > m_min.

		Returns:
			(tuple): the first analytic lens profile for all subhalos with m200c > m_min,
			the second is the lenstronomy kwargs. This will include the main deflector
			if get_main=True.
		"""
		redshift = self.main_deflector_parameters['z_lens']
		rockstar_dir = self.subhalo_parameters['rockstar_path']
	
		scale_factor, hlist_fname = self.get_scale_factor(rockstar_dir, redshift) #is this correct?

		sub, host = self.load_host_and_subs(rockstar_dir, hlist_fname, scale_factor, self.cosmo, self.subhalo_parameters['return_at_infall'], 
															self.subhalo_parameters['get_subhalos'])

		lens, kwargs = self.generate_analytic(sub, host, self.cosmo, self.source_parameters['z_source'], redshift, 
					self.subhalo_parameters['get_main'], self.subhalo_parameters['subhalo_profile'], self.main_deflector_parameters['host_profile'])
		subhalo_z_list = [redshift]*len(lens)
		return (lens, kwargs,subhalo_z_list)
