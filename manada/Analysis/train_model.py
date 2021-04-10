# -*- coding: utf-8 -*-
"""
This script will initialize and train a BNN model on a strong lensing image
dataset.
"""
import argparse, os, sys, glob
from importlib import import_module
import tensorflow as tf
from manada.Analysis import dataset_generation, loss_functions, conv_models
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


def parse_args():
	"""Parse the input arguments by the user

	Returns:
		(argparse.Namespace): An instance of the Namespace object with the
			users provided values.

	"""
	# Initialize the parser and the possible inputs
	parser = argparse.ArgumentParser()
	parser.add_argument('training_config', help='Path to configuration for '
		+'training the model.')
	parser.add_argument('--tensorboard_dir', default=None, type=str,
		dest='tensorboard_dir', help='Optional path to save a tensorboard ' +
		'output to')
	args = parser.parse_args()
	return args


def main():
	# Get the user provided arguments
	args = parse_args()

	# Get the training parameters from the provided .py file
	config_dir, config_file = os.path.split(os.path.abspath(
		args.training_config))
	sys.path.insert(0, config_dir)
	config_name, _ = os.path.splitext(config_file)
	config_module = import_module(config_name)

	# Grab the arguments passed in by the user
	# The size of each training batch
	batch_size = config_module.batch_size
	# The number of epochs to train for
	n_epochs = config_module.n_epochs
	# The size of the images in the training set
	img_size = config_module.img_size
	# A random seed to us
	random_seed = config_module.random_seed
	# The path to the tf_record for the training images
	tfr_train_path = config_module.tfr_train_path
	# The path to the tf_record for the validation images
	tfr_val_path = config_module.tfr_val_path
	# The list of learning parameters to use
	learning_params = config_module.learning_params
	num_params = len(learning_params)
	# Which parameters to consider flipping
	flip_pairs = config_module.flip_pairs
	# The path to the fodler containing the npy images
	# for training
	npy_folder_train = config_module.npy_folder_train
	# Number of steps per epoch is number of examples over the batch size
	npy_file_list = glob.glob(os.path.join(npy_folder_train,'image_*.npy'))
	steps_per_epoch = len(npy_file_list)//batch_size
	# The path to the fodler containing the npy images
	# for validation
	npy_folder_val = config_module.npy_folder_val
	# The path to the training metadata
	metadata_path_train = config_module.metadata_path_train
	# The path to the validation metadata
	metadata_path_val = config_module.metadata_path_val
	# The path to the csv file to read from / write to for normalization
	# of learning parameters.
	input_norm_path = config_module.input_norm_path
	# The detector kwargs to use for on-the-fly noise generation
	kwargs_detector = config_module.kwargs_detector
	# Whether or not to normalize the images by the standard deviation
	norm_images = config_module.norm_images
	# A string with which loss function to use.
	loss_function = config_module.loss_function
	# A string specifying which model to use
	model_type = config_module.model_type
	# Where to save the model weights
	model_weights = config_module.model_weights
	# The learning rate for the model
	learning_rate = config_module.learning_rate

	# Set the random seed for our network
	tf.random.set_seed(random_seed)

	# Check for tf records for train and validation and prepare them
	# if needed
	print('Checking for training data.')
	if not os.path.exists(tfr_train_path):
		print('Generating new TFRecord at %s'%(tfr_train_path))
		dataset_generation.generate_tf_record(npy_folder_train,learning_params,
			metadata_path_train,tfr_train_path,input_norm_path)
	else:
		print('TFRecord found at %s'%(tfr_train_path))

	print('Checking for validation data.')
	if not os.path.exists(tfr_val_path):
		print('Generating new TFRecord at %s'%(tfr_val_path))
		dataset_generation.generate_tf_record(npy_folder_val,learning_params,
			metadata_path_val,tfr_val_path,input_norm_path)
	else:
		print('TFRecord found at %s'%(tfr_val_path))

	# Turn our tf records into tf datasets for training and validation
	tf_dataset_t = dataset_generation.generate_tf_dataset(tfr_train_path,
		learning_params,batch_size,n_epochs,norm_images=norm_images,
		kwargs_detector=kwargs_detector)
	# We shouldn't be adding random noise to validation images. They should
	# be generated with noise
	if kwargs_detector is not None:
		print('Make sure your validation images already have noise! Noise' +
			'will not be added on the fly for validation.')
	tf_dataset_v = dataset_generation.generate_tf_dataset(tfr_val_path,
		learning_params,batch_size,n_epochs,norm_images=norm_images,
		kwargs_detector=None)

	print('Initializing the model')

	# Load the loss function
	if loss_function == 'mse':
		num_outputs = num_params
		loss = loss_functions.MSELoss(num_params,flip_pairs).loss
	elif loss_function == 'diag':
		num_outputs = num_params*2
		loss = loss_functions.DiagonalCovarianceLoss(num_params,
			flip_pairs).loss
	elif loss_function == 'full':
		num_outputs = num_params + int(num_params*(num_params+1)/2)
		loss = loss_functions.FullCovarianceLoss(num_params,flip_pairs).loss
	else:
		raise ValueError('%s loss not in the list of supported losses'%(
			loss_function))

	# Load the model
	if model_type == 'resnet50':
		model = conv_models.build_resnet_50(img_size,num_outputs)
	else:
		raise ValueError('%s model not in the list of supported models'%(
			model_type))

	# We'll use Adam for graident descent
	adam = Adam(lr=learning_rate,amsgrad=False)

	# We'll always track the mse loss on the validation set
	mse_loss = loss_functions.MSELoss(num_params,flip_pairs).loss

	# Compile our model
	model.compile(loss=loss,optimizer=adam,metrics=[loss,mse_loss])

	print('Is model built: ' + str(model.built))

	try:
		model.load_weights(model_weights)
		print('Loaded weights %s'%(model_weights))
	except:
		print('No weights found. Saving new weights to %s'%(model_weights))

	callbacks = []
	if args.tensorboard_dir is not None:
		tensorboard = TensorBoard(log_dir=args.tensorboard_dir,
			update_freq='batch')
		callbacks.append(tensorboard)
	modelcheckpoint = ModelCheckpoint(model_weights,monitor='val_loss',
		save_best_only=True,save_freq='epoch')
	callbacks.append(modelcheckpoint)

	# TODO add validation data.
	model.fit(tf_dataset_t,callbacks=callbacks,epochs=n_epochs,
		steps_per_epoch=steps_per_epoch,validation_data=tf_dataset_v)


if __name__ == '__main__':
	main()
