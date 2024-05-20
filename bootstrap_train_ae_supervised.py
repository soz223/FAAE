#!/usr/bin/env python3
"""Script to train the deterministic supervised adversarial autoencoder."""
from pathlib import Path
import random as rn
import time

import joblib
from sklearn.preprocessing import RobustScaler, OneHotEncoder
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import argparse
from utils import COLUMNS_NAME, load_dataset
from models import make_encoder_model_v111, make_decoder_model_v1, make_discriminator_model_v1

PROJECT_ROOT = Path.cwd()


def main():
    """Train the normative method on the bootstrapped samples.

    The script also the scaler and the demographic data encoder.
    """
    # ----------------------------------------------------------------------------
    n_bootstrap = 10
    model_name = 'supervised_ae'

    participants_path = PROJECT_ROOT / 'data' / 'ADNI_TRAIN' / 'participants.tsv'
    freesurfer_path = PROJECT_ROOT / 'data' / 'ADNI_TRAIN' / 'freesurferData.csv'
    # ----------------------------------------------------------------------------
    bootstrap_dir = PROJECT_ROOT / 'outputs' / 'bootstrap_analysis'
    ids_dir = bootstrap_dir / 'ids'

    model_dir = bootstrap_dir / model_name
    model_dir.mkdir(exist_ok=True)

    # ----------------------------------------------------------------------------
    # Set random seed
    random_seed = 42
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)
    rn.seed(random_seed)

    for i_bootstrap in range(n_bootstrap):
        ids_filename = 'cleaned_bootstrap_{:03d}.csv'.format(i_bootstrap)
        ids_path = ids_dir / ids_filename

        bootstrap_model_dir = model_dir / '{:03d}'.format(i_bootstrap)
        bootstrap_model_dir.mkdir(exist_ok=True)

        # ----------------------------------------------------------------------------
        # Loading data
        dataset_df = load_dataset(participants_path, ids_path, freesurfer_path)

        # ----------------------------------------------------------------------------

        dataset_df = dataset_df.loc[dataset_df['Diagnosis'] == 1]      
        x_data = dataset_df[COLUMNS_NAME].values
        

        tiv = dataset_df['EstimatedTotalIntraCranialVol'].values
        tiv = tiv[:, np.newaxis]

        x_data = (np.true_divide(x_data, tiv)).astype('float32')

        scaler = RobustScaler()
        x_data_normalized = scaler.fit_transform(x_data)

        # ----------------------------------------------------------------------------
        age = dataset_df['Age'].values[:, np.newaxis].astype('float32')
        #enc_age = OneHotEncoder(sparse=False)
        enc_age = OneHotEncoder(handle_unknown = "ignore",sparse=False)
        one_hot_age2 = enc_age.fit_transform(age)
        
        bin_labels = list(range(0,10))  
        age_bins_test, bin_edges = pd.cut(dataset_df['Age'], 10, retbins=True, labels=bin_labels)
        #age_bins_train, bin_edges = pd.cut(train_covariates['Age'], 10, retbins=True, labels=bin_labels)
        #age_bins_test = pd.cut(test_covariates['Age'], retbins=True, labels=bin_labels)
        #age_bins_train.fillna(0, inplace=True)
        age_bins_test.fillna(0,inplace = True)
        one_hot_age = np.eye(10)[age_bins_test.values]

        gender = dataset_df['Gender'].values[:, np.newaxis].astype('float32')
        enc_gender = OneHotEncoder(sparse=False)
        one_hot_gender = enc_gender.fit_transform(gender)

        y_data = np.concatenate((one_hot_age, one_hot_gender), axis=1).astype('float32')

        # -------------------------------------------------------------------------------------------------------------
        # Create the dataset iterator
        batch_size = 256
        n_samples = x_data.shape[0]

        train_dataset = tf.data.Dataset.from_tensor_slices((x_data_normalized, y_data))
        train_dataset = train_dataset.shuffle(buffer_size=n_samples)
        train_dataset = train_dataset.batch(batch_size)

        # -------------------------------------------------------------------------------------------------------------
        # Create models
        n_features = x_data_normalized.shape[1]
        n_labels = y_data.shape[1]
        h_dim = [100, 100]
        z_dim = 10

        encoder = make_encoder_model_v111(n_features, h_dim, z_dim)
        decoder = make_decoder_model_v1(z_dim, n_features, h_dim) 
        discriminator = make_discriminator_model_v1(z_dim, h_dim)

        # -------------------------------------------------------------------------------------------------------------
        # Define loss functions
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        mse = tf.keras.losses.MeanSquaredError()
        accuracy = tf.keras.metrics.BinaryAccuracy()

        # -------------------------------------------------------------------------------------------------------------
        # Define optimizers
        base_lr = 0.0001
        max_lr = 0.005

        step_size = 2 * np.ceil(n_samples / batch_size)

        ae_optimizer = tf.keras.optimizers.Adam(lr=base_lr)

        # -------------------------------------------------------------------------------------------------------------
        # Training function
        @tf.function
        def train_step(batch_x, batch_y):
            # -------------------------------------------------------------------------------------------------------------
            # Autoencoder
            with tf.GradientTape() as ae_tape:
                encoder_output = encoder(batch_x, training=True)
                decoder_output = decoder(tf.concat(encoder_output, axis=1), training=True)

                # Autoencoder loss
                ae_loss = mse(batch_x, decoder_output)

            ae_grads = ae_tape.gradient(ae_loss, encoder.trainable_variables + decoder.trainable_variables)
            ae_optimizer.apply_gradients(zip(ae_grads, encoder.trainable_variables + decoder.trainable_variables))


            return ae_loss

        # -------------------------------------------------------------------------------------------------------------
        # Training loop
        global_step = 0
        n_epochs = 200
        gamma = 0.98
        scale_fn = lambda x: gamma ** x
        for epoch in range(n_epochs):
            start = time.time()

            epoch_ae_loss_avg = tf.metrics.Mean()

            for _, (batch_x, batch_y) in enumerate(train_dataset):
                global_step = global_step + 1
                cycle = np.floor(1 + global_step / (2 * step_size))
                x_lr = np.abs(global_step / step_size - 2 * cycle + 1)
                clr = base_lr + (max_lr - base_lr) * max(0, 1 - x_lr) * scale_fn(cycle)
                ae_optimizer.lr = clr

                ae_loss = train_step(batch_x, batch_y)

                epoch_ae_loss_avg(ae_loss)

            epoch_time = time.time() - start

            print('{:4d}: TIME: {:.2f} ETA: {:.2f} AE_LOSS: {:.4f}' \
                  .format(epoch, epoch_time,
                          epoch_time * (n_epochs - epoch),
                          epoch_ae_loss_avg.result(),

                          ))

        # Save models
        encoder.save(bootstrap_model_dir / 'encoder.h5')
        decoder.save(bootstrap_model_dir / 'decoder.h5')

        # Save scaler
        joblib.dump(scaler, bootstrap_model_dir / 'scaler.joblib')

        joblib.dump(enc_age, bootstrap_model_dir / 'age_encoder.joblib')
        joblib.dump(enc_gender, bootstrap_model_dir / 'gender_encoder.joblib')


if __name__ == "__main__":
    main()
