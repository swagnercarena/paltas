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
from helpers.SimulationAnalysis import readHlist, SimulationAnalysis
import astropy
from astropy.cosmology import WMAP9 as cosmo
import halotools.empirical_models as ht
import re

# Define the parameters we expect to find for reading from the simulation catalog 
draw_simulation_catalog_parameters = ['rockstar_path','m_min', 'redshift', 'get_main']


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

	def get_numbers_from_filename(filename):
		return re.search(r'\d+\.\d+', filename).group(0)

	def get_scale_factor(rockstar_dir, redshift):
		hlist_file_names =  os.listdir(rockstar_dir + str('hlists'))
		scale_factors  = []
		for filename in hlist_file_names:
			scale_factors.append(get_numbers_from_filename(filename))
		scale_factors = np.array(scale_factors).astype(float)
		nfile = np.argmin(np.abs((1./scale_factors-1)-redshift))
		return scale_factors[nfile], hlist_file_names[nfile]

	def load_host_and_subs(rockstar_dir, hlist_fname, cosmo):
    		#fields we want from hlist halo catalog 
		fields = ['treeRootId', 'upid', 'x', 'y', 'z', 'rvir','rs', 'm200c', 'id', 'mvir', 'scale']
		sim = SimulationAnalysis(trees_dir=rockstar_dir+'trees')

		# read in z=0 halos 
		halos = readHlist(os.path.join(rockstar_dir+'hlists', 'hlist_1.00000.list'),fields=fields)

		# get host 
		host = halos[0]

		#get subhalos 
		subhalos = halos[halos['upid']==host['id']]

		#get z=redshift halos 
		halos05 = readHlist(os.path.join(rockstar_dir+'hlists', hlist_fname),fields=fields)

                # get host 
		host05 = halos05[halos05['treeRootId']==host['treeRootId']][0]

                #get subhalos 
		subhalos05 = halos05[halos05['upid']==host05['id']]

		concentration = []
		rs = []

		#get subhalos parameters at infall 
		for i in range(len(subhalos05)):
			sub_tree = sim.load_main_branch(subhalos05['treeRootId'][i])  
			ind_vpeak = np.argmax(sub_tree['vmax'])
			rshift = 1./(sub_tree['scale']) - 1. 
			delta_vir = ht.delta_vir(cosmo, rshift)
			#calculate r200 
			r200s = sub_tree['rvir']*((subhalos05['m200c'][i]*delta_vir)/(200*sub_tree['mvir']))**(1./3.)
			c_vpeak = (r200s/sub_tree['rs'])[ind_vpeak]
			concentration.append(c_vpeak)
			rs.append((sub_tree['rs']*sub_tree['scale'])[ind_vpeak]) #change from comoving to physical coords
		return host, subhalos, host05, subhalos05, np.array(c), np.array(r) 
	
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

	def get_subhalo_catalog(subhalos, host, minimum_mass, scale_factor, cosmo, sub_c, sub_rs, delta_vir, return_host =True):
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

		columns = ('x', 'y', 'z', 'c', 'rs', 'm200')
		sub = pd.DataFrame(columns=columns)
		main = pd.DataFrame(columns=columns)
    
                h = cosmo.h

		center_x = host['x']
		center_y = host['y'] 
		center_z = host['z']

		mask = np.log10(subhalos['M200c']) > minimum_mass
    
		sub['c'] = sub_c[mask]
		sub['m200'] = subhalos['M200c'][mask]
		sub['x'] = (subhalos['x'][mask] - center_x)*1000*scale_factor/h
		sub['y'] = (subhalos['y'][mask] - center_y)*1000*scale_factor/h
		sub['z'] = (subhalos['z'][mask] - center_z)*1000*scale_factor/h
		sub['rs'] = sub_rs['rs'][mask]

		if return_host == True:
			r200 = [host['rvir']*((host['m200c']*delta_vir)/(200*host['mvir']))**(1./3.)] 
			main['c'] = r200/host05['rs']
			main['m200'] = [host['m200c']]
			main['x'] = [0]
			main['y'] = [0] 
			main['z'] = [0]
			main['rs'] = [host['rs']*scale_factor]

		return sub, main 

        def generate_analytic(sub, host, cosmo, z_source, z_lens, include_host=True)
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

                Returns:
                        The analytic lens profile for all subhalos with m200c > m_min, 
                        and the lenstronomy kwargs. 
                """

                h = cosmo.h

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
        
		lens_analytic = LensModel(lens_model_list=['NFW']*len(alpha_Rs), lens_redshift_list=[z_lens]*len(alpha_Rs), z_source=z_source, cosmo=cosmo, multi_plane=False)


		kwargs_new = [{'Rs': rr[i], 'alpha_Rs': alpha_Rs[i], 'center_x': yy[i], 'center_y':xx[i]} for i in range(len(rr))]

		return lens_analytic, kwargs_new 


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
		delta_vir = ht.delta_vir(self.cosmo, redshift)
		scale_factor, hlist_fname = get_scale_factor(rockstar_dir, redshift) #is this correct? 

                host, subhalos, host05, subhalos05, sub_c, sub_rs = self.load_host_and_subs(rockstar_dir, scale_factor, self.cosmo)
                sub, host = self.get_subhalo_catalog(subhalos05, host05, self.subhalo_parameters['m_min'], scale_factor, self.cosmo, sub_c, sub_rs, delta_vir, self.subhalo_parameters['get_main'])

		lens, kwargs = self.generate_analytic(sub, host, self.cosmo, self.source_parameters['z_source'], redshift, self.subhalo_parameters['get_main'])
		return (lens, kwargs) 

