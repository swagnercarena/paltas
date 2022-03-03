from paltas.Configs.xxxx_yyyy.config_train import *
from paltas.Sources.cosmos import COSMOSIncludeCatalog
import os
import pandas as pd

# config_dict and root path in the import *.
config_dict['source']['parameters']['source_inclusion_list'] = pd.read_csv(
	os.path.join(root_path,'paltas/Sources/val_galaxies.csv'),
	names=['catalog_i'])['catalog_i'].to_numpy()
config_dict['source']['class'] = COSMOSIncludeCatalog
del config_dict['source']['parameters']['source_exclusion_list']
