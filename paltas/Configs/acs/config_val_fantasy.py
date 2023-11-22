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

# Set drizzle output resolution to what we assumed in the paltas paper
config_dict['drizzle']['parameters']['output_pixel_scale'] = 0.03

# Increase exposure time to what we assumed in the paltas paper
config_dict['detector']['parameters']['exposure_time'] = 1380
