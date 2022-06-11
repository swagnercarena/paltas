import copy
from paltas.Configs.paper_2203_00690.config_train import *
from paltas.Sources.cosmos import COSMOSIncludeCatalog
config_dict = copy.deepcopy(config_dict)

# config_dict, pd, os, root path in the import *.
config_dict['source']['parameters']['source_inclusion_list'] = pd.read_csv(
	os.path.join(root_path,'paltas/Sources/val_galaxies.csv'),
	names=['catalog_i'])['catalog_i'].to_numpy()
config_dict['source']['class'] = COSMOSIncludeCatalog
del config_dict['source']['parameters']['source_exclusion_list']
