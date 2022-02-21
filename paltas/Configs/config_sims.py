from paltas.Configs.config_slope_marg import *
from paltas.Sources.cosmos import COSMOSIncludeCatalog
from paltas.Substructure.subhalos_catalog import SubhalosCatalog
import os
import pandas as pd

# config_dict and root path in the import *.
# update the sources
config_dict['source']['parameters']['source_inclusion_list'] = pd.read_csv(
        os.path.join(root_path,'manada/Sources/val_galaxies.csv'),
        names=['catalog_i'])['catalog_i'].to_numpy()
config_dict['source']['class'] = COSMOSIncludeCatalog
del config_dict['source']['parameters']['source_exclusion_list']

# change the substructure class to import from the simulations
halos_list = ['Halo015','Halo286','Halo384','Halo551','Halo774','Halo985',
        'Halo024','Halo302','Halo399','Halo570','Halo781','Halo029','Halo313',
        'Halo428','Halo579','Halo785','Halo046','Halo331','Halo444','Halo581',
        'Halo790','Halo055','Halo336','Halo470','Halo593','Halo806','Halo090',
        'Halo342','Halo492','Halo606','Halo834','Halo175','Halo346','Halo656',
        'Halo861','Halo183','Halo347','Halo496','Halo752','Halo909','Halo186',
        'Halo352','Halo501','Halo755','Halo927','Halo257','Halo383','Halo504',
        'Halo759','Halo962']


def return_rockstar_path():
        halo = halos_list.pop()
        return ('/sdf/group/kipac/u/ollienad/group_zoomins/'+halo+
                '/output/rockstar/')


config_dict['subhalo']['class'] = SubhalosCatalog
config_dict['subhalo']['parameters'] = {'rockstar_path':return_rockstar_path,
        'm_min':5e8,'get_main':False,'return_at_infall':False,
        'subhalo_profile':'TNFW_ELLIPSE'}
