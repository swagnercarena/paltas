# -*- coding: utf-8 -*-
"""
Define the base class to draw subhalos for a lens

This module contains the base class that all the subhalo classes will build
from. Because the steps for rendering subhalos can vary between different
models, the required functions are very sparse.
"""
from .subhalos_base import SubhalosBase
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
import pandas as pd 
import os
import numpy as np
from helpers.SimulationAnalysis import readHlist


# Define the parameters we expect to find for the DG_19 model
draw_nfw_masses_catalog_parameters = ['rockstar_path','m_min', 'scale_factor']


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
		self.check_parameterization(draw_nfw_masses_catalog_parameters)

	def load_host_and_subs(path_to_hlist, scale_factor):
    		#z=0
		halos = readHlist(os.path.join(path_to_hlist, 'hlist_1.00000.list'))
		host = halos[0]
		subhalos = halos[halos['upid']==host['id']]
    		#z=0.5
		halos05 = readHlist(os.path.join(path_to_hlist, 'hlist_{}.list'.format(scale_factor)))
		host05 = halos05[halos05['Tree_root_ID']==host['Tree_root_ID']][0]
		subhalos05 = halos05[halos05['upid']==host05['id']]
		return host, subhalos, host05, subhalos05
	
	def rotate_project(x, y, z, theta_x, theata_y, theta_z):
    """
    rotates with the three Euler angles and projects in the z'-axis after rotation
    See e.g. https://en.wikipedia.org/wiki/3D_projection#Perspective_projection
    """
		c_x, s_x = np.cos(theta_x), np.sin(theta_x)
		c_y, s_y = np.cos(theta_y), np.sin(theta_y)
		c_z, s_z = np.cos(theta_z), np.sin(theta_z)
		x_ = c_y * (s_z * y + c_z * x) - s_y * z
		y_ = s_x * (c_y * z + s_y * (s_z * y + c_z * x)) + c_x * (c_z * y - s_z * x)
		return x_, y_

	def get_subhalo_positions(subhalos, host, minimum_mass, scale_factor, return_host =True):
  
		columns = ('x', 'y', 'z', 'c', 'rs', 'm200')
		sub = pd.DataFrame(columns=columns)
		main = pd.DataFrame(columns=columns)
    
                h = self.cosmo.h

		center_x = host['x']
		center_y = host['y'] 
		center_z = host['z']
		mask = np.log10(subhalos['M200c']) > minimum_mass
    
		sub['c'] = subhalos['Rvir'][mask]/subhalos['rs'][mask]
		sub['m200'] = subhalos['M200c'][mask]
		sub['x'] = (subhalos['x'][mask] - center_x)*1000*scale_factor/h
		sub['y'] = (subhalos['y'][mask] - center_y)*1000*scale_factor/h
		sub['z'] = (subhalos['z'][mask] - center_z)*1000*scale_factor/h
		sub['rs'] = subhalos['rs'][mask]

		if return_host == True: 
			main['c'] = [host['Rvir']/host05['rs']]
			main['m200'] = [host['M200c']]
			main['x'] = [0]
			main['y'] = [0] 
			main['z'] = [0]
			main['rs'] = [host05['rs']]
		return sub, main 


	def generate_analytic(sub, host, cosmo, z_source, z_lens, include_host=True):
    
                h = self.cosmo.h

		sub_x, sub_y = rotate_project(sub['x'], sub['y'], sub['z'], 0, 0, 0)

		lens_cosmo = LensCosmo(z_lens=z_lens, z_source=z_source, cosmo=cosmo)

		Rs_angle, alpha_Rs = lens_cosmo.nfw_physical2angle(M=sub['m200']/h, c=sub['c'])

		alpha_Rs = alpha_Rs.values
		xx = lens_cosmo.phys2arcsec_lens(sub_x/1000.).values
		yy = lens_cosmo.phys2arcsec_lens(sub_y/1000.).values
		rr = lens_cosmo.phys2arcsec_lens(sub['rs']/1000/h).values
    
		if(include_host==True):
			host_x, host_y = rotate_project(host['x'], host['y'], host['z'], theta_x, theta_y, theta_z)
	
			Rs_angle_host, alpha_Rs_host = lens_cosmo.nfw_physical2angle(M=host['m200']/h, c=host['c'])
        
			xx_host = lens_cosmo.phys2arcsec_lens(host_x/1000.)
			yy_host = lens_cosmo.phys2arcsec_lens(host_y/1000.)        
			rr_host = lens_cosmo.phys2arcsec_lens(host['rs']/1000/h)
      
			rr = np.insert(rr, 0, float(rr_host.values))
			xx = np.insert(xx, 0, float(xx_host.values))
			yy = np.insert(yy, 0, float(yy_host.values))
			alpha_Rs = np.insert(alpha_Rs, 0, float(alpha_Rs_host.values))
        
		lens_analytic = LensModel(lens_model_list=['NFW']*len(alpha_Rs), lens_redshift_list=[0.5]*len(alpha_Rs), z_source=2, cosmo=cosmo, multi_plane=False)


		kwargs_new = [{'Rs': rr[i], 'alpha_Rs': alpha_Rs[i], 'center_x': yy[i], 'center_y':xx[i]} for i in range(len(rr))]

		return lens_analytic, kwargs_new 


	def draw_subhalos(self):
		"""Draws masses, concentrations,and positions for the subhalos of a
		main lens halo.

		Returns:
			(tuple): A tuple of the lists: the first is the profile type for
				each subhalo returned, the second is the lenstronomy kwargs for
				that subhalo, and the third is the redshift for each subhalo.
		"""
                host, subhalos, host05, subhalos05 = self.load_host_and_subs(self.subhalo_parameters['rockstar_path'], self.subhalo_parameters['scale_factor'])
                sub, host = self.get_subhalo_positions(subhalos05, host05, self.subhalo_parameters['m_min'], self.subhalo_parameters['scale_factor'], self.main_deflector_parameters['get_main'])

		lens, kwargs = self.generate_analytic(sub, host, self.cosmo, self.source_parameters['z'], self.main_deflector_parameters['get_main'])

