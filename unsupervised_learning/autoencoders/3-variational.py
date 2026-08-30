#!/usr/bin/env python3
"""Variational Autoencoder."""
import tensorflow.keras as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a Variational Autoencoder."""
    input_enc = K.Input(shape=(input_dims,))
    x = input_enc
    for units in hidden_layers:
        x = K.layers.Dense(units, activation='relu')(x)

    z_mean = K.layers.Dense(latent_dims, activation=None)(x)
    z_log_var = K.layers.Dense(latent_dims, activation=None)(x)

    def sampling(args):
        """Reparameterization trick."""
        z_m, z_lv = args
        batch = K.backend.shape(z_m)[0]
        dim = K.backend.shape(z_m)[1]
        epsilon = K.backend.random_normal(shape=(batch, dim))
        return z_m + K.backend.exp(0.5 * z_lv) * epsilon

    z = K.layers.Lambda(sampling, output_shape=(latent_dims,))(
        [z_mean, z_log_var]
    )
    encoder = K.Model(input_enc, [z, z_mean, z_log_var], name='encoder')

    input_dec = K.Input(shape=(latent_dims,))
    x = input_dec
    for units in reversed(hidden_layers):
        x = K.layers.Dense(units, activation='relu')(x)

    output_dec = K.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = K.Model(input_dec, output_dec, name='decoder')

    auto_inputs = K.Input(shape=(input_dims,))
    z_sampled, z_m, z_lv = encoder(auto_inputs)
    reconstructed = decoder(z_sampled)

    recon_loss = K.losses.binary_crossentropy(
        auto_inputs, reconstructed
    )
    recon_loss *= input_dims
    kl_loss = 1 + z_lv - K.backend.square(z_m) - K.backend.exp(z_lv)
    kl_loss = K.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.backend.mean(recon_loss + kl_loss)

    auto = K.Model(auto_inputs, reconstructed, name='autoencoder')
    auto.add_loss(vae_loss)
    auto.compile(optimizer='adam')

    return encoder, decoder, auto
