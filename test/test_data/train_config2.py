import os

batch_size = 10
# The number of epochs to train for
n_epochs = 1
# The size of the images in the training set
img_size = (128,128,1)
# A random seed to us
random_seed = 2
# The list of learning parameters to use
learning_params = ['main_deflector_parameters_theta_E',
	'main_deflector_parameters_gamma1','main_deflector_parameters_gamma2',
	'main_deflector_parameters_gamma','main_deflector_parameters_e1',
	'main_deflector_parameters_e2','main_deflector_parameters_center_x',
	'main_deflector_parameters_center_y']
log_learning_params = ['subhalo_parameters_sigma_sub']
# Which parameters to consider flipping
flip_pairs = [[1,2]]
# The path to the fodler containing the npy images
# for training
npy_folders_train = ['./test_data/fake_train/']
# The path to the tf_record for the training images
tfr_train_paths = [
	os.path.join(path,'data.tfrecord') for path in npy_folders_train]
# The path to the fodler containing the npy images
# for validation
npy_folder_val = npy_folders_train[0]
# The path to the tf_record for the validation images
tfr_val_path = tfr_train_paths[0]
# The path to the training metadata
metadata_paths_train = [
	os.path.join(path,'metadata.csv') for path in npy_folders_train]
# The path to the validation metadata
metadata_path_val = metadata_paths_train[0]
# The path to the csv file to read from / write to for normalization
# of learning parameters.
input_norm_path = npy_folders_train[0] + 'norms.csv'
# The detector kwargs to use for on-the-fly noise generation
kwargs_detector = {'pixel_scale':0.08,'ccd_gain':2.5,'read_noise':4.0,
	'magnitude_zero_point':25,'exposure_time':5400.0,'sky_brightness':22,
	'num_exposures':1, 'background_noise':None}
# Whether or not to normalize the images by the standard deviation
norm_images = False
# A string with which loss function to use.
loss_function = 'full'
# A string specifying which model to use
model_type = 'xresnet34'
# A string specifying which optimizer to use
optimizer = 'Adam'
# Where to save the model weights
model_weights = ('./test_data/fake_model.h5')
model_weights_init = ('./test_data/fake_model.h5')
# The learning rate for the model
learning_rate = 5e-3
# Whether or not to use random rotation of the input images
random_rotation = False
