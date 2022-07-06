# -*- coding: utf-8 -*-
"""
Implement a transformer model for population parameter inference.

This module implements models to be used for analysis of a populaiton
of strong lensing parameters.
"""

from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow as tf
from .conv_models import _xresnet34


class EncoderLayer(layers.Layer):
	"""Class that implements a single encoding layer for a transformer.

	Args:
		embedding_dim (int): The size of the embedding.
		num_heads (int): The number of heads for the MultiHeadAttention
			layer.
		dff (int): The inner dimension of the position-wise feed-forward
			network.
		droput_rate (float): The rate for the dropout layers.
		name (str): base name for layers.
	"""
	def __init__(self,embedding_dim,num_heads,dff,dropout_rate,name):
		super(EncoderLayer,self).__init__(name=name)

		# Initialize our multihead attention layers and our fully connected
		# layers.
		self.mha = layers.MultiHeadAttention(num_heads,embedding_dim,
			name=name+'_mha')
		self.fc1 = layers.Dense(dff,activation='relu',name=name+'_fc1')
		self.fc2 = layers.Dense(embedding_dim,name=name+'_fc2')

		# Initialize our layernorm layers.
		self.layernorm1 = layers.LayerNormalization(epsilon=1e-6,name=name+'_ln1')
		self.layernorm2 = layers.LayerNormalization(epsilon=1e-6,name=name+'_ln2')

		# Initialize the dropout layers.
		self.dropout1 = layers.Dropout(dropout_rate,name=name+'_d1')
		self.dropout2 = layers.Dropout(dropout_rate,name=name+'_d2')

	def call(self,x,training=True,mask=None):  # pragma: no cover
		"""Implements a forward pass of the encoder layer.

		Args:
			x (KerasTensor): A KerasTensor with dimension (batch_size,
				max_seq_len,embedding_dim) that will be used as the input to
				the encoder layer.
			training (bool): If False, dropout will not be used.
			mask (tf.Tensor): A tensor specifying the attention mask to be
				used. Should have dimension (batch_size,1,max_seq_len).

		Returns:
			(KerasTensor): A KerasTensor with dimension (batch_size,max_seq_len,
			embedding_dim) that represents the output of the encoder layer.
		"""

		# Recent work does the layer normalization before the mha / fc steps
		# rather than after.
		attn_input = self.layernorm1(x)
		# For now we want to implement MultiHeadAttention without returning
		# the attention scores.
		attn_output = self.mha(attn_input,attn_input,attn_input,mask,
			return_attention_scores=False)
		attn_output = self.dropout1(attn_output,training=training)

		# Implement the position-wise feed-forward network
		fc_input = self.layernorm2(attn_input+attn_output)
		fc_output = self.fc1(fc_input)
		fc_output = self.fc2(fc_output)
		fc_output = self.dropout2(fc_output,training=training)

		return fc_output+fc_input


class Encoder(layers.Layer):
	"""Class that implements a full encoder for a transformer.

	Args:
		num_layers (int): The number of encoder layers to string together.
		embedding_dim (int): The size of the embedding.
		num_heads (int): The number of heads for the MultiHeadAttention
			layer.
		dff (int): The inner dimension of the position-wise feed-forward
			network.
		droput_rate (float): The rate for the dropout layers.
		name (str): base name for layers.
	"""
	def __init__(self,num_layers,embedding_dim,num_heads,dff,dropout_rate,
		name):
		super(Encoder,self).__init__(name=name)

		# Save some of the global variables we'll need at call time.
		self.embedding_dim = embedding_dim
		self.num_layers = num_layers

		# Initialize all of our encoder layers.
		self.enc_layers = [EncoderLayer(embedding_dim,num_heads,dff,
			dropout_rate,name+'_%d'%(i)) for i in range(num_layers)]

		# Initialize our dropout layer.
		self.dropout = layers.Dropout(dropout_rate,name=name+'_dropout')

	def call(self,x,training=False,mask=None):  # pragma: no cover
		"""Implements a forward pass of the encoder.

		Args:
			x (KerasTensor): A KerasTensor with dimension (batch_size,
				max_seq_len,embedding_dim) that will be used as the input to
				the encoder.
			training (bool): If False, dropout will not be used.
			mask (tf.Tensor): A tensor specifying the attention mask to be
				used. Should have dimension (batch_size,1,max_seq_len).

		Returns:
			(KerasTensor): A KerasTensor with dimension (batch_size,max_seq_len,
			embedding_dim) that represents the output of the encoder.
		"""
		# Conduct dropout on the inputs to the encoder layer.
		x = self.dropout(x,training=training)
		# String together all of our encoder layers.
		for i in range(self.num_layers):
			x = self.enc_layers[i](x,training=training,mask=mask)

		return x


def build_population_transformer(num_outputs,img_size,max_n_images,num_layers,
	embedding_dim,num_heads,dff,dropout_rate,conv_trainable=False):
	"""Builds a combined xresnet34 and transformer architecture that conducts
	inference on a population of images.

	Args:
		num_outputs (int): The number of outputs to predict for population
			inference.
		img_size ((int,int,int)): A tuple with shape (pix,pix,freq) that
			describes the size of the input images.
		max_n_images (int): The maximum number of images allowed in a dataset.
		num_layers (int): The number of encoder layers to string together.
		embedding_dim (int): The size of the embedding.
		num_heads (int): The number of heads for the MultiHeadAttention
			layer.
		dff (int): The inner dimension of the position-wise feed-forward
			network.
		droput_rate (float): The rate for the dropout layers.
		conv_trainable (bool): If True the convolutional layers will be
			treated as trainable.

	Returns:
		(keras.Model): An instance of the combined xresnet34 and transformer
		pipeline.
	"""

	# There should be two inputs to the model. The first should have dimension
	# (batch_size,max_n_images,pix,pix,channels). This represents the input to
	# the convolutional network. We will reduce the first two dimension to
	# one batch dimension for the convolutional calls.
	inputs = layers.Input(shape=(max_n_images,) + img_size)
	# The second input has dimension (batch_size,max_n_images) and specifies
	# with a 0 any images that were added as padding. This will be important
	# information for the attention mask.
	image_mask = layers.Input(shape=(max_n_images,))

	# To conduct the convolution we first need to combine the sequence length and
	# the batch size.
	conv_inputs = tf.reshape(inputs,shape=(-1,) + img_size)

	# The output of the convolution should be the embedding size.
	conv_output = _xresnet34(conv_inputs,embedding_dim,trainable=conv_trainable,
		output_trainable=conv_trainable)

	# Reshape back to the original batch size
	conv_output = tf.reshape(conv_output,shape=(-1,max_n_images,embedding_dim))
	# Set any images that were added as padding to have 0 output in the embedding
	# space. This is to avoid them having any impact on the residual connections.
	conv_output = conv_output*image_mask[:,:,tf.newaxis]

	# Append an embedding that will give us the population distribution
	population_embedding_input = tf.zeros(shape=(tf.shape(inputs)[0],1,
		embedding_dim))
	x = tf.concat([population_embedding_input,conv_output],axis=1)

	# Generate our attention mask from the image_mask
	mask = tf.concat([tf.zeros((tf.shape(inputs)[0],1)),image_mask],
		axis=1)[:, tf.newaxis,:]
	mask = tf.cast(mask,dtype=tf.bool)

	# Apply the encoder to our convolutional embedding.
	x = Encoder(num_layers,embedding_dim,num_heads,dff,dropout_rate,
		name='encoder')(x,mask=mask)

	# Extract the population embedding which we've inserted at the front
	population_embedding_output = x[:,0,:]
	# Convert the population embedding to the distribution parameters.
	outputs = tf.keras.layers.Dense(num_outputs,name='final_dense')(
		population_embedding_output)

	return Model(inputs=[inputs,image_mask],outputs=outputs)
