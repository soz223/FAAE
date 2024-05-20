#!/home/songlinzhao/anaconda3/envs/normodiff/bin/python3

from pathlib import Path
import random as rn
#import time

#import joblib
from sklearn.preprocessing import RobustScaler
import numpy as np
import tensorflow as tf

from utils import COLUMNS_NAME, load_dataset
import os
import argparse
import torch
from VAE import VAE
import matplotlib.pyplot as plt
from os.path import join
from utils_vae import plot_losses, Logger, MyDataset

from torchmetrics import MeanMetric

PROJECT_ROOT = Path.cwd()


def main(comb_label, hz_para_list):
    """Train the normative method on the bootstrapped samples.

    The script also the scaler and the demographic data encoder.
    """
    # ----------------------------------------------------------------------------
    n_bootstrap = 10
    model_name = 'supervised_vae'

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
    kl_loss_list = [0]*200
    total_loss_list = [0]*200
    
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

        # -------------------------------------------------------------------------------------------------------------
        # Create the dataset iterator
        batch_size = 256
        n_samples = x_data.shape[0]


        torch.manual_seed(42)
        use_cuda =  torch.cuda.is_available()
        if use_cuda:
            torch.cuda.manual_seed(42)
        DEVICE = torch.device("cuda" if use_cuda else "cpu")
        input_dim = x_data_normalized.shape[1]
    
        train_dataset = MyDataset(x_data_normalized)
        #train_dataset = MyDataset(train_data)
        generator_train = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=False)
        # -------------------------------------------------------------------------------------------------------------
        # Create models
        #n_features = x_data_normalized.shape[1]
        #n_labels = y_data.shape[1]
        h_dim = [hz_para_list[0], hz_para_list[1]]
        z_dim = hz_para_list[2]

        # -------------------------------------------------------------------------------------------------------------
        # Training loop
        global_step = 0
        n_epochs = 200
        gamma = 0.98
        scale_fn = lambda x: gamma ** x
        base_lr = 0.0001
        max_lr = 0.005

        print('train model')
        model = VAE(input_dim=input_dim, hidden_dim=h_dim, latent_dim=z_dim, learning_rate=0.0001, non_linear=True)
        model.to(DEVICE)
        
        step_size = 2 * np.ceil(n_samples / batch_size)
        
        for epoch in range(n_epochs):
            epoch_ae_loss_avg = MeanMetric()
            epoch_kl_loss_avg = MeanMetric()
            epoch_total_loss_avg = MeanMetric()
            for batch_idx, batch in enumerate(generator_train): 
                #print(batch_idx)
                global_step = global_step + 1
                cycle = np.floor(1 + global_step / (2 * step_size))
                x_lr = np.abs(global_step / step_size - 2 * cycle + 1)
                clr = base_lr + (max_lr - base_lr) * max(0, 1 - x_lr) * scale_fn(cycle)
                model.optimizer = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=clr)
                batch = batch.to(DEVICE)
                fwd_rtn = model.forward(batch)
                loss = model.loss_function(batch, fwd_rtn)
                epoch_ae_loss_avg.update(loss['ll'])
                epoch_kl_loss_avg.update(loss['kl'])
                epoch_total_loss_avg.update(loss['total'])
                #model.optimizer = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=clr)
                model.optimizer.zero_grad()
                #print(model.optimizer.learning_rate)
                loss['total'].backward()
                model.optimizer.step() 
                to_print = 'Train Epoch:' + str(epoch) + ' ' + 'Train batch: ' + str(batch_idx) + ' '+ ', '.join([k + ': ' + str(round(v.item(), 3)) for k, v in loss.items()])
                print(to_print)        
                if epoch == 0:
                    log_keys = list(loss.keys())
                    logger = Logger()
                    logger.on_train_init(log_keys)
                else:
                    logger.on_step_fi(loss)
            
            ae_loss_list[epoch] += epoch_ae_loss_avg.compute()
            kl_loss_list[epoch] += epoch_kl_loss_avg.compute()
            total_loss_list[epoch] += epoch_total_loss_avg.compute()
            #print(loss)
        plot_losses(logger, bootstrap_model_dir, 'training')
        model_path = join(bootstrap_model_dir, 'VAE_model.pkl')
        torch.save(model, model_path)
        
    ae_loss_array = np.array(ae_loss_list)
    kl_loss_array = np.array(kl_loss_list)
    total_loss_array = np.array(total_loss_list)
    ae_loss_array = np.divide(ae_loss_array, n_bootstrap)
    kl_loss_array = np.divide(kl_loss_array, n_bootstrap)
    total_loss_array = np.divide(total_loss_array, n_bootstrap)
    epoch = range(1, 201)
    plt.plot(epoch, ae_loss_array, 'g', label = 'AE loss')
    plt.plot(epoch, kl_loss_array, 'b', label = 'KL loss')
    plt.plot(epoch, total_loss_array, 'r', label = 'TOTAL loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
            
if __name__ == "__main__":
    a = 0
    b = [110,110,10]
    main(a, b)
