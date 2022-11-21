import copy
from paltas.Configs.acs.config_train import *
from paltas.Sources.cosmos import COSMOSIncludeCatalog
# Would be nice to do this, but it would mess up scipy.stats random states..
# config_dict = copy.deepcopy(config_dict)

config_dict['source']['parameters']['source_inclusion_list'] = pd.read_csv(
	os.path.join(root_path,'paltas/Sources/val_galaxies.csv'),
	names=['catalog_i'])['catalog_i'].to_numpy()
config_dict['source']['class'] = COSMOSIncludeCatalog
del config_dict['source']['parameters']['source_exclusion_list']

