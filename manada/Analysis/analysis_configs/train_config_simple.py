batch_size = 1024
# The number of epochs to train for
n_epochs = 30
# The size of the images in the training set
img_size = (64,64,1)
# A random seed to us
random_seed = 2
# The path to the tf_record for the training images
tfr_train_paths = ['/lscratch/swagnerc/tf_record_%d'%(i) for i in range(1,21)]
# The path to the tf_record for the validation images
tfr_val_path = '/lscratch/swagnerc/tf_record_val'
# The list of learning parameters to use
learning_params = ['subhalo_parameters_sigma_sub',
	'main_deflector_parameters_theta_E',
	'los_parameters_delta_los']
# Which parameters to consider flipping
flip_pairs = None
# The path to the fodler containing the npy images
# for training
npy_folders_train = [
	'/scratch/users/swagnerc/manada/datasets/single_source_%d/'%(i)
	for i in range(1,21)]
# The path to the fodler containing the npy images
# for validation
npy_folder_val = '/scratch/users/swagnerc/manada/datasets/single_source_val/'
# The path to the training metadata
metadata_paths_train = [
	'/scratch/users/swagnerc/manada/datasets/single_source_%d/metadata.csv'%(i)
	for i in range(1,21)]
# The path to the validation metadata
metadata_path_val = (
	'/scratch/users/swagnerc/manada/datasets/single_source_val/metadata.csv')
# The path to the csv file to read from / write to for normalization
# of learning parameters.
input_norm_path = npy_folders_train[0] + 'norms.csv'
# The detector kwargs to use for on-the-fly noise generation
kwargs_detector = {'pixel_scale':0.08,'ccd_gain':2.5,'read_noise':4.0,
	'magnitude_zero_point':25.9463,'exposure_time':5400.0,'sky_brightness':22,
	'num_exposures':1, 'background_noise':None}
# Whether or not to normalize the images by the standard deviation
norm_images = True
# A string with which loss function to use.
loss_function = 'diag'
# A string specifying which model to use
model_type = 'resnet50'
# Where to save the model weights
model_weights = '/scratch/users/swagnerc/manada/model_weights/resnet50_diag.h5'
# The learning rate for the model
learning_rate = 1e-6
