# Configuration to reproduce H0RTON dataset with COSMOS
from paltas.Configs.Horton.config_horton import *
from paltas.Sources.cosmos import COSMOSCatalog

# Define the cosmos path
root_path = paltas.__path__[0][:-7]
cosmos_folder = root_path + r'/datasets/cosmos/COSMOS_23.5_training_sample/'

config_dict['source'] = {
	'class': COSMOSCatalog,
	'parameters':{
		'z_source':None,
		'cosmos_folder':cosmos_folder,
		'max_z':1.0,'minimum_size_in_pixels':64,'faintest_apparent_mag':20,
		'smoothing_sigma':0.00,'random_rotation':True,
		'output_ab_zeropoint':output_ab_zeropoint,
		'min_flux_radius':10.0,
		'center_x':None,
		'center_y':None}
}
