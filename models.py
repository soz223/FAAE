"""Functions to create the networks that compose the adversarial autoencoder."""
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
import torch
from torch.distributions import Normal
from torch.nn import Parameter

import tensorflow_probability as tfp
tfd = tfp.distributions


def make_encoder_model_v111(n_features, h_dim, z_dim):
    """Creates the encoder."""
    inputs = keras.Input(shape=(n_features,))
    x = inputs
    for n_neurons_layer in h_dim:
        x = keras.layers.Dense(n_neurons_layer)(x)
        x = keras.layers.LeakyReLU()(x)

    encoded = keras.layers.Dense(z_dim)(x)
    model = keras.Model(inputs=inputs, outputs=encoded)
    return model

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon



def make_encoder_model_v1(n_features, h_dim, z_dim):
    """Creates the encoder."""
    inputs = keras.Input(shape=(n_features,))
    x = inputs
    for n_neurons_layer in h_dim:
        x = keras.layers.Dense(n_neurons_layer)(x)
        x = keras.layers.LeakyReLU()(x)
    
    z_mean = keras.layers.Dense(z_dim, name="z_mean")(x)
    z_log_var = keras.layers.Dense(z_dim, name="z_log_var")(x)
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
    #encoded = keras.layers.Dense(z_dim)(x)
    model = keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")
    #model = keras.Model(inputs=inputs, outputs=encoded)
    return model

def make_decoder_model_v1(encoded_dim, n_features, h_dim):
    """Creates the decoder."""
    encoded = keras.Input(shape=(encoded_dim,))
    x = encoded
    for n_neurons_layer in h_dim:
        x = keras.layers.Dense(n_neurons_layer)(x)
        x = keras.layers.LeakyReLU()(x)

    reconstruction = keras.layers.Dense(n_features, activation='linear')(x)
    model = keras.Model(inputs=encoded, outputs=reconstruction)
    return model

def make_decoder_model_v11(encoded_dim, n_features, h_dim):
    """Creates the decoder."""
    encoded = keras.Input(shape=(encoded_dim,))
    x = encoded
    for n_neurons_layer in h_dim:
        x = keras.layers.Dense(n_neurons_layer)(x)
        x = keras.layers.LeakyReLU()(x)
 
    
    
    reconstruction = keras.layers.Dense(n_features, activation='linear')(x)
    #model = keras.Model(inputs=encoded, outputs=reconstruction)
    #tmp_noise_par = torch.FloatTensor(1, n_features).fill_(-3.0)
    tmp_noise_par = tf.fill([1, n_features], -3.0)
    #logvar_out = Parameter(data=tmp_noise_par, requires_grad=True)
    #output = Normal(loc=reconstruction, scale=logvar_out.exp().pow(0.5))
    return tfd.Normal(loc=reconstruction, scale=tf.sqrt(tf.math.exp(tmp_noise_par)))


def make_discriminator_model_v1(z_dim, h_dim):
    """Creates the discriminator."""
    z_features = keras.Input(shape=(z_dim,))
    x = z_features
    for n_neurons_layer in h_dim:
        x = keras.layers.Dense(n_neurons_layer)(x)
        x = keras.layers.LeakyReLU()(x)

    prediction = keras.layers.Dense(1)(x)
    model = keras.Model(inputs=z_features, outputs=prediction)
    return model