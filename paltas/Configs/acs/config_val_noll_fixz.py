from paltas.Configs.acs.config_val import *

# Disable lens light
config_dict['lens_light']['parameters']['brightness_multiplier'] = 0.

# Fix redshifts to the means of SLACS
# (makes redshift inputs superfluous)
z_lens = 0.21
z_source = 0.65
config_dict['main_deflector']['parameters']['z_lens'] = z_lens
config_dict['lens_light']['parameters']['z_source'] = z_lens
config_dict['source']['parameters']['z_source'] = z_source
del config_dict['cross_object']['parameters'][
    'main_deflector:z_lens,lens_light:z_source,source:z_source']
