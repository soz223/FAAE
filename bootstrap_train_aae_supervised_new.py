#!/usr/bin/env python3
"""Script to train the deterministic supervised adversarial autoencoder."""
from pathlib import Path
import random as rn
import time
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler, OneHotEncoder
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import COLUMNS_NAME, load_dataset
from models import make_encoder_model_v1, make_decoder_model_v1, make_discriminator_model_v1
import torch
from torch.distributions import Normal
from torch.nn import Parameter
import tensorflow_probability as tfp
tfd = tfp.distributions
import os
import argparse

PROJECT_ROOT = Path.cwd()
tf.config.run_functions_eagerly(True)

def main(comb_label, hz_para_list):
    """Train the normative method on the bootstrapped samples.

    The script also the scaler and the demographic data encoder.
    """
    # ----------------------------------------------------------------------------
    n_bootstrap = 10
    model_name = 'supervised_aae_new'

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
    ae_loss_list = [0]*200
    dc_loss_list = [0]*200
    gen_loss_list = [0]*200
    
    #Monte Carlo Sample Loop K
    k = 0
    for i_bootstrap in range(n_bootstrap):
        ids_filename = 'cleaned_bootstrap_{:03d}.csv'.format(i_bootstrap)
        ids_path = ids_dir / ids_filename

        bootstrap_model_dir = model_dir / '{:03d}'.format(i_bootstrap)
        bootstrap_model_dir.mkdir(exist_ok=True)
        print("########################---",ids_filename,"---########################")
        # ----------------------------------------------------------------------------
        # Loading data
        dataset_df = load_dataset(participants_path, ids_path, freesurfer_path)

        # ----------------------------------------------------------------------------

        dataset_df = dataset_df.loc[dataset_df['Diagnosis'] == 1]     
        x_data = dataset_df[COLUMNS_NAME].values
        

        tiv = dataset_df['EstimatedTotalIntraCranialVol'].values
        tiv = tiv[:, np.newaxis]
        x_data = (np.true_divide(x_data, tiv)).astype('float32')
        #tv = dataset_df['TotalVar'].values
        #tv = tv[:, np.newaxis]

        #x_data_1 = (np.true_divide(x_data[:,:100], tiv)).astype('float32')
        #x_data_2 = (np.true_divide(x_data[:,100:], tv)).astype('float32')
        #x_data = np.concatenate((x_data_1, x_data_2), axis=1)

        scaler = RobustScaler()
        x_data_normalized = scaler.fit_transform(x_data)
        
        np.save("x_train", x_data_normalized)
        #x_data_normalized = x_data

        # ----------------------------------------------------------------------------
        age = dataset_df['Age'].values[:, np.newaxis].astype('float32')
        #enc_age = OneHotEncoder(sparse=False)
        enc_age = OneHotEncoder(handle_unknown = "ignore",sparse=False)
        one_hot_age = enc_age.fit_transform(age)

        gender = dataset_df['Gender'].values[:, np.newaxis].astype('float32')
        enc_gender = OneHotEncoder(sparse=False) 
        one_hot_gender = enc_gender.fit_transform(gender)
        
        bin_labels = list(range(0,10))
        #age_bins_train, bin_edges = pd.cut(dataset_df['Age'], 10, retbins=True, labels=bin_labels)
        #age_bins_train.fillna(0, inplace=True)
        #one_hot_age = np.eye(10)[age_bins_train.values]
        
        
        ICV_bins_train, bin_edges = pd.qcut(dataset_df['EstimatedTotalIntraCranialVol'], q=10,  retbins=True, labels=bin_labels)
        ICV_bins_train.fillna(0, inplace = True)
        one_hot_ICV_train = np.eye(10)[ICV_bins_train.values]
        
        if comb_label == 1:
            y_data = np.concatenate((one_hot_age, one_hot_gender), axis=1).astype('float32')
        elif comb_label == 2:
            y_data = np.concatenate((one_hot_age, one_hot_ICV_train), axis=1).astype('float32')
        elif comb_label == 3:
            y_data = np.concatenate((one_hot_gender,one_hot_ICV_train), axis=1).astype('float32')
        else:
            y_data = np.concatenate((one_hot_age, one_hot_gender,one_hot_ICV_train), axis=1).astype('float32')   
        #y_data = np.concatenate((one_hot_age), axis=1).astype('float32')
        #y_data = one_hot_age.astype('float32')

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
        h_dim = [hz_para_list[0], hz_para_list[1]]
        z_dim = hz_para_list[2]

        encoder = make_encoder_model_v1(n_features, h_dim, z_dim)
        decoder = make_decoder_model_v1(z_dim+n_labels, n_features, h_dim) 
        discriminator = make_discriminator_model_v1(z_dim, h_dim)

        # -------------------------------------------------------------------------------------------------------------
        # Define loss functions
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        mse = tf.keras.losses.MeanSquaredError()
        accuracy = tf.keras.metrics.BinaryAccuracy()

        def discriminator_loss(real_output, fake_output):
            loss_real = cross_entropy(tf.ones_like(real_output), real_output)
            loss_fake = cross_entropy(tf.zeros_like(fake_output), fake_output)
            return loss_fake + loss_real

        def generator_loss(fake_output):
            return cross_entropy(tf.ones_like(fake_output), fake_output)
        
        def compute_ll(x, x_recon):
            return tf.reduce_mean(tf.reduce_sum(x_recon.log_prob(x), 1, keepdims=True), 0)
        
        def calc_ll(x, x_recon):
            return compute_ll(x, x_recon)

        # -------------------------------------------------------------------------------------------------------------
        # Define optimizers
        base_lr = 0.0001
        max_lr = 0.005
        

        step_size = 2 * np.ceil(n_samples / batch_size)

        ae_optimizer = tf.keras.optimizers.Adam(lr=base_lr)
        dc_optimizer = tf.keras.optimizers.Adam(lr=base_lr)
        gen_optimizer = tf.keras.optimizers.Adam(lr=base_lr)
        
        
        tmp_noise_par = tf.fill([1, n_features], -3.0)
        # -------------------------------------------------------------------------------------------------------------
        # Training function
        alpha = 5
        
        
        @tf.function
        def train_step(batch_x, batch_y,  pre, total):
            # -------------------------------------------------------------------------------------------------------------
            # Autoencoder
            with tf.GradientTape() as ae_tape:
                z_mean, z_log_var, z = encoder(batch_x, training=True)
                
                #z = encoder(batch_x, training=True)
                #encoder_output = encoder(batch_x, training=True)
                decoder_output = decoder(tf.concat([z, batch_y], axis=1), training=True) 



                ae_loss = mse(batch_x, decoder_output)
                
                kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
                total_loss = alpha*ae_loss + kl_loss
            ae_grads = ae_tape.gradient(total_loss, encoder.trainable_variables + decoder.trainable_variables)
            ae_optimizer.apply_gradients(zip(ae_grads, encoder.trainable_variables + decoder.trainable_variables))

            # -------------------------------------------------------------------------------------------------------------
            # Discriminator
            with tf.GradientTape() as dc_tape:
                real_distribution = tf.random.normal([batch_x.shape[0], z_dim], mean=0.0, stddev=1.0)
                _, _, encoder_output = encoder(batch_x, training=True)
                #encoder_output = encoder(batch_x, training=True)

                dc_real = discriminator(real_distribution, training=True)
                dc_fake = discriminator(encoder_output, training=True)

                # Discriminator Loss
                dc_loss = discriminator_loss(dc_real, dc_fake)

                # Discriminator Acc
                dc_acc = accuracy(tf.concat([tf.ones_like(dc_real), tf.zeros_like(dc_fake)], axis=0),
                                  tf.concat([dc_real, dc_fake], axis=0))

            dc_grads = dc_tape.gradient(dc_loss, discriminator.trainable_variables)
            dc_optimizer.apply_gradients(zip(dc_grads, discriminator.trainable_variables))

            # -------------------------------------------------------------------------------------------------------------
            # Generator (Encoder)
            with tf.GradientTape() as gen_tape:
                _, _, encoder_output = encoder(batch_x, training=True)
                #encoder_output = encoder(batch_x, training=True)
                dc_fake = discriminator(encoder_output, training=True)

                # Generator loss
                gen_loss = generator_loss(dc_fake)

            gen_grads = gen_tape.gradient(gen_loss, encoder.trainable_variables)
            gen_optimizer.apply_gradients(zip(gen_grads, encoder.trainable_variables))

            return ae_loss, kl_loss, total_loss, dc_loss, dc_acc, gen_loss
            #return ae_loss,  total_loss, dc_loss, dc_acc, gen_loss
            #return _loss, dc_loss, dc_acc, gen_loss

        # -------------------------------------------------------------------------------------------------------------
        # Training loop
        global_step = 0
        n_epochs = 200
        gamma = 0.98
        scale_fn = lambda x: gamma ** x
        for epoch in range(n_epochs):
            start = time.time()
            epoch_ae_loss_avg = tf.metrics.Mean()
            epoch_kl_loss_avg = tf.metrics.Mean()
            epoch_total_loss_avg = tf.metrics.Mean()
            epoch_dc_loss_avg = tf.metrics.Mean()
            epoch_dc_acc_avg = tf.metrics.Mean()
            epoch_gen_loss_avg = tf.metrics.Mean()
            
            total = 0
            pre = 0
            for _, (batch_x, batch_y) in enumerate(train_dataset):
                global_step = global_step + 1
                cycle = np.floor(1 + global_step / (2 * step_size))
                x_lr = np.abs(global_step / step_size - 2 * cycle + 1)
                clr = base_lr + (max_lr - base_lr) * max(0, 1 - x_lr) * scale_fn(cycle)
                ae_optimizer.lr = clr
                dc_optimizer.lr = clr
                gen_optimizer.lr = clr
                
                #batch_x = tf.concat([batch_x, batch_y], axis=1)
                #print(batch_x.shape, batch_y.shape)
                total += batch_x.shape[0]
                ae_loss, kl_loss, total_loss, dc_loss, dc_acc, gen_loss = train_step(batch_x, batch_y, pre,  total)
                pre += batch_x.shape[0]
                #ae_loss,  total_loss, dc_loss, dc_acc, gen_loss = train_step(batch_x, batch_y)
                #ae_loss,  dc_loss, dc_acc, gen_loss = train_step(batch_x, batch_y)

                epoch_ae_loss_avg(ae_loss)
                epoch_kl_loss_avg(kl_loss)
                epoch_total_loss_avg(total_loss)
                epoch_dc_loss_avg(dc_loss)
                epoch_dc_acc_avg(dc_acc)
                epoch_gen_loss_avg(gen_loss)
            
            
            #print(len(mean_list), mean_list[i_bootstrap].shape)
            epoch_time = time.time() - start
            print('{:4d}: TIME: {:.2f} ETA: {:.2f} AE_LOSS: {:.4f} KL_LOSS: {:.4f} TOTAL_LOSS: {:.4f} DC_LOSS: {:.4f} DC_ACC: {:.4f} GEN_LOSS: {:.4f}'  \
                  .format(epoch, epoch_time,
                          epoch_time * (n_epochs - epoch),
                          epoch_ae_loss_avg(ae_loss),
                          epoch_kl_loss_avg(kl_loss),
                          epoch_total_loss_avg(total_loss),
                          epoch_dc_loss_avg(dc_loss),
                          epoch_dc_acc_avg(dc_acc),
                          epoch_gen_loss_avg(gen_loss)
                          ))
            ae_loss_list[epoch] += epoch_ae_loss_avg(ae_loss)
            dc_loss_list[epoch] += epoch_dc_loss_avg(dc_loss)
            gen_loss_list[epoch] += epoch_gen_loss_avg(gen_loss)
        

        # Save models
        encoder.save(bootstrap_model_dir / 'encoder.h5')
        decoder.save(bootstrap_model_dir / 'decoder.h5')
        discriminator.save(bootstrap_model_dir / 'discriminator.h5')

        # Save scaler
        joblib.dump(scaler, bootstrap_model_dir / 'scaler.joblib')

        joblib.dump(enc_age, bootstrap_model_dir / 'age_encoder.joblib')
        joblib.dump(enc_gender, bootstrap_model_dir / 'gender_encoder.joblib')
        joblib.dump(one_hot_ICV_train, bootstrap_model_dir / 'icv_encoder.joblib')
        
        

    ae_loss_array = np.array(ae_loss_list)
    dc_loss_array = np.array(dc_loss_list)
    gen_loss_array = np.array(gen_loss_list)
    ae_loss_array = np.divide(ae_loss_array, n_bootstrap)
    dc_loss_array = np.divide(dc_loss_array, n_bootstrap)
    gen_loss_array = np.divide(gen_loss_array, n_bootstrap)
    epoch = range(1, 201)
    plt.plot(epoch, ae_loss_array, 'g', label = 'TOTAL loss')
    plt.plot(epoch, dc_loss_array, 'b', label = 'DC loss')
    plt.plot(epoch, gen_loss_array, 'r', label = 'GEN loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()
    plt.savefig('loss in aae s new 2.png')
    # print("everything alright")


if __name__ == "__main__":

    a = 0
    b = [110,110,10]
    main(a, b)
