# -*- coding: utf-8 -*-
"""
Implement a few models for parameter inference.

This module implements models to be used for analysis of strong
lensing parameters.
"""

from . import manual_keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model


def _xresnet_block(x,filters,kernel_size,strides,conv_shortcut,name):
	""" Build a block of residual convolutions for the xresnet model family

	Args:
		x (KerasTensor): The input Keras tensorflow tensor that will be
			passed through the stack.
		filters (int): The number of output filters
		kernel_size (int): The kernel size of each filter
		strides (int): The strides to use for each convolutional filter
		conv_shortcut (bool): When true, will use a convolutional shortcut and
			when false will use an identity shortcut.
		name (str): The name for this block

	Returns:
		(KerasTensor): A Keras tensorflow tensor representing the input after
			the block has been applied
	"""
	# First axis is assumed to be the batch
	bn_axis = -1

	# Use the ResnetD variant for the shortcut
	shortcut = x
	if strides > 1:
		shortcut = layers.AveragePooling2D(pool_size=(2,2),name=name+'_id_pool')(
			shortcut)
	if conv_shortcut is True:
		shortcut = layers.Conv2D(filters,1,strides=1,use_bias=False,
			name=name+'_id_conv')(shortcut)
		shortcut = layers.BatchNormalization(axis=bn_axis,epsilon=1e-5,
			momentum=0.1,name=name+'_id_bn')(shortcut)

	# Set up the rest of the block
	x = layers.ZeroPadding2D(padding=((1,1),(1,1)),name=name+'_pad1')(x)
	x = layers.Conv2D(filters,kernel_size,strides=strides,use_bias=False,
		name=name+'_conv1')(x)
	x = layers.BatchNormalization(axis=bn_axis,epsilon=1e-5,momentum=0.1,
		name=name+'_bn1')(x)
	x = layers.Activation('relu',name=name+'_relu1')(x)
	x = layers.ZeroPadding2D(padding=((1,1),(1,1)),name=name+'_pad2')(x)
	x = layers.Conv2D(filters,kernel_size,strides=1,use_bias=False,
		name=name+'_conv2')(x)
	x = layers.BatchNormalization(axis=bn_axis,epsilon=1e-5,momentum=0.1,
		name=name+'_bn2',gamma_initializer='zeros')(x)

	x = layers.Add(name=name+'_add')([shortcut,x])
	x = layers.Activation('relu',name=name+'_out')(x)

	return x


def _xresnet_stack(x,filters,kernel_size,strides,conv_shortcut,name,blocks):
	""" Build a stack of residual blocks for the xresnet model family

	Args:
		x (KerasTensor): The input Keras tensorflow tensor that will be
			passed through the stack.
		filters (int): The number of output filters
		kernel_size (int): The kernel size of each filter
		strides (int): The strides to use for each convolutional filter
		conv_shortcut (bool): When true, will use a convolutional shortcut and
			when false will use an identity shortcut.
		name (str): The name for this stack
		blocks (int): The number of blocks in this stack

	Returns:
		(KerasTensor): A Keras tensorflow tensor representing the input after
			the stack has been applied
	"""

	# Apply each residual block
	x = _xresnet_block(x,filters,kernel_size,strides,
		conv_shortcut=conv_shortcut,name=name+'_block1')
	for i in range(2,blocks+1):
		x = _xresnet_block(x,filters,kernel_size,1,conv_shortcut=False,
			name=name+'_block%d'%(i))

	return x


def build_xresnet34(img_size,num_outputs,custom_head=False):
	""" Build the xresnet34 model described in
	https://arxiv.org/pdf/1812.01187.pdf

	Args:
		img_size ((int,int,int)): A tuple with shape (pix,pix,freq) that
			describes the size of the input images.
		num_outputs (int): The number of outputs to predict
		custom_head (bool): If true, then add a custom head at the end of
			xresnet34 in line with what' used in the fastai code.

	Returns:
		(keras.Model): An instance of the xresnet34 model implemented in
			Keras.
	"""

	# Assume the first dimension is the batch size
	bn_axis = -1

	# Initialize the inputs
	inputs = layers.Input(shape=img_size)

	# Build the stem of the resnet
	# Conv 1 of stem
	x = layers.ZeroPadding2D(padding=((1,1),(1,1)),name='stem_pad1')(inputs)
	x = layers.Conv2D(32,3,strides=2,use_bias=False,name='stem_conv1')(x)
	x = layers.BatchNormalization(axis=bn_axis,epsilon=1e-5,momentum=0.1,
		name='stem_bn1')(x)
	x = layers.Activation('relu',name='stem_relu1')(x)

	# Conv 2 of stem
	x = layers.ZeroPadding2D(padding=((1,1),(1,1)),name='stem_pad2')(x)
	x = layers.Conv2D(32,3,strides=1,use_bias=False,name='stem_conv2')(x)
	x = layers.BatchNormalization(axis=bn_axis,epsilon=1e-5,momentum=0.1,
		name='stem_bn2')(x)
	x = layers.Activation('relu',name='stem_relu2')(x)

	# Conv 3 of stem
	x = layers.ZeroPadding2D(padding=((1,1),(1,1)),name='stem_pad3')(x)
	x = layers.Conv2D(64,3,strides=1,use_bias=False,name='stem_conv3')(x)
	x = layers.BatchNormalization(axis=bn_axis,epsilon=1e-5,momentum=0.1,
		name='stem_bn3')(x)
	x = layers.Activation('relu',name='stem_relu3')(x)

	# Next step is max pooling of the stem
	x = layers.ZeroPadding2D(padding=((1,1),(1,1)),name='stem_pad4')(x)
	x = layers.MaxPooling2D(3,strides=2,name='stem_maxpooling')(x)

	# # Now we apply the residual stacks
	x = _xresnet_stack(x,filters=64,kernel_size=3,strides=1,
		conv_shortcut=False,name='stack1',blocks=3)
	x = _xresnet_stack(x,filters=128,kernel_size=3,strides=2,
		conv_shortcut=True,name='stack2',blocks=4)
	x = _xresnet_stack(x,filters=256,kernel_size=3,strides=2,
		conv_shortcut=True,name='stack3',blocks=6)
	x = _xresnet_stack(x,filters=512,kernel_size=3,strides=2,
		conv_shortcut=True,name='stack4',blocks=3)

	# Conduct the pooling and a dense transform to the final prediction
	x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
	if custom_head:
		x = layers.BatchNormalization(axis=bn_axis,epsilon=1e-5,momentum=0.1,
			name='head_bn1')(x)
		x = layers.Dense(512,use_bias=False,name='head_dense1')(x)
		x = layers.Activation('relu',name='head_relu1')(x)
		x = layers.BatchNormalization(axis=bn_axis,epsilon=1e-5,momentum=0.1,
			name='head_bn2')(x)
		x = layers.Dense(num_outputs,use_bias=False,name='head_dense2')(x)
		outputs = layers.BatchNormalization(axis=bn_axis,epsilon=1e-5,
			momentum=0.1,name='head_bn3')(x)
	else:
		outputs = layers.Dense(num_outputs,name='output_dense')(x)

	model = Model(inputs=inputs,outputs=outputs)

	return model


def build_resnet_50(img_size,num_outputs):
	"""	Build the traditional resnet 50 model.

	Args:
		img_size ((int,int,int)): A tuple with shape (pix,pix,freq) that
			describes the size of the input images.
		num_outputs (int): The number of outputs to predict

	Returns:
		(keras.Model): An instance of the ResNet50 model implemented in
			Keras.
	"""
	# Set some of the parameters for resnet 50
	use_bias = True
	# Initialize the inputs
	inputs = layers.Input(shape=img_size)

	# Build the first resnet stack
	x = layers.ZeroPadding2D(padding=((3,3),(3,3)),name='conv1_pad')(inputs)
	x = layers.Conv2D(64, 7, strides=2, use_bias=use_bias, name='conv1_conv')(x)
	x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
	x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

	# Apply the resnet stacks
	x = manual_keras.stack1(x, 64, 3, stride1=1, name='conv2')
	x = manual_keras.stack1(x, 128, 4, name='conv3')
	x = manual_keras.stack1(x, 256, 6, name='conv4')
	x = manual_keras.stack1(x, 512, 3, name='conv5')

	# Pass the output to the fully connected layer
	x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
	outputs = layers.Dense(num_outputs, name='output')(x)

	# Construct the model
	model = Model(inputs=inputs,outputs=outputs)

	return model


def build_alexnet(img_size,num_outputs):
	""" Build the alexnet model

	"""
	inputs = layers.Input(shape=img_size)
	x = layers.Conv2D(filters=64, kernel_size=(5,5), strides=(2,2),
		padding='valid', activation='relu', input_shape=img_size)(inputs)
	x = layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)

	x = layers.Conv2D(filters=192, kernel_size=(5,5), strides=(1,1),
		padding='same', activation='relu')(x)
	x = layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)

	# Layer 3
	x = layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1),
		padding='same', activation='relu')(x)

	# Layer 4
	x = layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1),
		padding='same', activation='relu')(x)

	# Layer 5
	x = layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1),
		padding='same', activation='relu')(x)
	x = layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)

	# Pass to fully connected layers
	x = layers.Flatten()(x)

	# Layer 6
	x = layers.Dense(4096, activation='relu')(x)

	# Layer 7
	x = layers.Dense(4096, activation='relu')(x)

	# Output
	outputs = layers.Dense(num_outputs)(x)

	# Construct model
	model = Model(inputs=inputs, outputs=outputs)

	return model
