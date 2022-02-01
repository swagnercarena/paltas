# -*- coding: utf-8 -*-
"""
Implement a few models for parameter inference.

This module implements models to be used for analysis of strong
lensing parameters.
"""

from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow as tf


def _xresnet_block(x,filters,kernel_size,strides,conv_shortcut,name,
	trainable=True):
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
		trainable (bool): If false, the weights in this block will not be
			trainable.

	Returns:
		(KerasTensor): A Keras tensorflow tensor representing the input after
		the block has been applied
	"""
	# First axis is assumed to be the batch
	bn_axis = -1

	# Use the ResnetD variant for the shortcut
	shortcut = x
	if strides > 1:
		shortcut = layers.AveragePooling2D(pool_size=(2,2),name=name+'_id_pool',
			trainable=trainable)(shortcut)
	if conv_shortcut is True:
		shortcut = layers.Conv2D(filters,1,strides=1,use_bias=False,
			name=name+'_id_conv',trainable=trainable)(shortcut)
		shortcut = layers.BatchNormalization(axis=bn_axis,epsilon=1e-5,
			momentum=0.1,name=name+'_id_bn',trainable=trainable)(shortcut)

	# Set up the rest of the block
	x = layers.ZeroPadding2D(padding=((1,1),(1,1)),name=name+'_pad1',
		trainable=trainable)(x)
	x = layers.Conv2D(filters,kernel_size,strides=strides,use_bias=False,
		name=name+'_conv1',trainable=trainable)(x)
	x = layers.BatchNormalization(axis=bn_axis,epsilon=1e-5,momentum=0.1,
		name=name+'_bn1',trainable=trainable)(x)
	x = layers.Activation('relu',name=name+'_relu1',trainable=trainable)(x)
	x = layers.ZeroPadding2D(padding=((1,1),(1,1)),name=name+'_pad2',
		trainable=trainable)(x)
	x = layers.Conv2D(filters,kernel_size,strides=1,use_bias=False,
		name=name+'_conv2',trainable=trainable)(x)
	x = layers.BatchNormalization(axis=bn_axis,epsilon=1e-5,momentum=0.1,
		name=name+'_bn2',gamma_initializer='zeros',trainable=trainable)(x)

	x = layers.Add(name=name+'_add',trainable=trainable)([shortcut,x])
	x = layers.Activation('relu',name=name+'_out',trainable=trainable)(x)

	return x


def _xresnet_stack(x,filters,kernel_size,strides,conv_shortcut,name,blocks,
	trainable=True):
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
		trainable (bool): If false, weights in this stack will not be trainable.

	Returns:
		(KerasTensor): A Keras tensorflow tensor representing the input after
		the stack has been applied
	"""
	# If the input dimension is not divisible by the stride then we must add
	# padding.
	divide_pad = strides - x.shape[1]%strides
	x = tf.cond(x.shape[1] % strides > 0,lambda:layers.ZeroPadding2D(
		padding=((divide_pad,0),(0,0)),name=name+'_stride_pad_r')(x),
		lambda:x)
	divide_pad = strides - x.shape[2]%strides
	x = tf.cond(x.shape[2] % strides > 0,lambda:layers.ZeroPadding2D(
		padding=((0,0),(divide_pad,0)),name=name+'_stride_pad_c')(x),
		lambda:x)
	# Apply each residual block
	x = _xresnet_block(x,filters,kernel_size,strides,
		conv_shortcut=conv_shortcut,name=name+'_block1',trainable=trainable)
	for i in range(2,blocks+1):
		x = _xresnet_block(x,filters,kernel_size,1,conv_shortcut=False,
			name=name+'_block%d'%(i),trainable=trainable)

	return x


def build_xresnet34(img_size,num_outputs,custom_head=False,
	train_only_head=False):
	""" Build the xresnet34 model described in
	https://arxiv.org/pdf/1812.01187.pdf

	Args:
		img_size ((int,int,int)): A tuple with shape (pix,pix,freq) that
			describes the size of the input images.
		num_outputs (int): The number of outputs to predict
		custom_head (bool): If true, then add a custom head at the end of
			xresnet34 in line with what' used in the fastai code.
		train_only_head (bool): If true, only train the head of the network.

	Returns:
		(keras.Model): An instance of the xresnet34 model implemented in
		Keras.
	"""

	# If we train only the head, then none of the previous weights should be
	# trainable
	trainable = not train_only_head

	# Assume the first dimension is the batch size
	bn_axis = -1

	# Initialize the inputs
	inputs = layers.Input(shape=img_size)

	# Build the stem of the resnet
	# Conv 1 of stem
	x = layers.ZeroPadding2D(padding=((1,1),(1,1)),name='stem_pad1',
		trainable=trainable)(inputs)
	x = layers.Conv2D(32,3,strides=2,use_bias=False,name='stem_conv1',
		trainable=trainable)(x)
	x = layers.BatchNormalization(axis=bn_axis,epsilon=1e-5,momentum=0.1,
		name='stem_bn1',trainable=trainable)(x)
	x = layers.Activation('relu',name='stem_relu1',trainable=trainable)(x)

	# Conv 2 of stem
	x = layers.ZeroPadding2D(padding=((1,1),(1,1)),name='stem_pad2',
		trainable=trainable)(x)
	x = layers.Conv2D(32,3,strides=1,use_bias=False,name='stem_conv2',
		trainable=trainable)(x)
	x = layers.BatchNormalization(axis=bn_axis,epsilon=1e-5,momentum=0.1,
		name='stem_bn2',trainable=trainable)(x)
	x = layers.Activation('relu',name='stem_relu2',trainable=trainable)(x)

	# Conv 3 of stem
	x = layers.ZeroPadding2D(padding=((1,1),(1,1)),name='stem_pad3',
		trainable=trainable)(x)
	x = layers.Conv2D(64,3,strides=1,use_bias=False,name='stem_conv3',
		trainable=trainable)(x)
	x = layers.BatchNormalization(axis=bn_axis,epsilon=1e-5,momentum=0.1,
		name='stem_bn3',trainable=trainable)(x)
	x = layers.Activation('relu',name='stem_relu3',trainable=trainable)(x)

	# Next step is max pooling of the stem
	x = layers.ZeroPadding2D(padding=((1,1),(1,1)),name='stem_pad4',
		trainable=trainable)(x)
	x = layers.MaxPooling2D(3,strides=2,name='stem_maxpooling',
		trainable=trainable)(x)

	# # Now we apply the residual stacks
	x = _xresnet_stack(x,filters=64,kernel_size=3,strides=1,
		conv_shortcut=False,name='stack1',blocks=3,trainable=trainable)
	x = _xresnet_stack(x,filters=128,kernel_size=3,strides=2,
		conv_shortcut=True,name='stack2',blocks=4,trainable=trainable)
	x = _xresnet_stack(x,filters=256,kernel_size=3,strides=2,
		conv_shortcut=True,name='stack3',blocks=6,trainable=trainable)
	x = _xresnet_stack(x,filters=512,kernel_size=3,strides=2,
		conv_shortcut=True,name='stack4',blocks=3,trainable=trainable)

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
