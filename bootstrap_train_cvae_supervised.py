#!/home/songlinzhao/anaconda3/envs/normodiff/bin/python3
"""Script to train the deterministic supervised adversarial autoencoder."""
from pathlib import Path
import random as rn
#import time
import pandas as pd
from sklearn.preprocessing import RobustScaler, OneHotEncoder
#import joblib
from sklearn.preprocessing import RobustScaler
import numpy as np
import tensorflow as tf

from utils import COLUMNS_NAME, load_dataset
from utils_vae import plot_losses, MyDataset_labels, Logger
import torch
from cVAE import cVAE

from os.path import join
from utils_vae import plot_losses, Logger, MyDataset



PROJECT_ROOT = Path.cwd()


def main():
    """Train the normative method on the bootstrapped samples.

    The script also the scaler and the demographic data encoder.
    """
    # ----------------------------------------------------------------------------
    n_bootstrap = 10
    model_name = 'supervised_cvae'

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
        train_data = dataset_df[COLUMNS_NAME].values
        

        tiv = dataset_df['EstimatedTotalIntraCranialVol'].values
        tiv = tiv[:, np.newaxis]

        train_data = (np.true_divide(train_data, tiv)).astype('float32')

        scaler = RobustScaler()
        train_data = scaler.fit_transform(train_data)
        train_data = pd.DataFrame(train_data)
        
        train_covariates = dataset_df[['Diagnosis','Gender', 'Age']]
        train_covariates.Diagnosis[train_covariates.Diagnosis == 0] = 0       #
        train_covariates['ICV'] =tiv  #
        
        bin_labels = list(range(0,27))                          
        age_bins_train, bin_edges = pd.qcut(train_covariates['Age'].rank(method="first"), q=27, retbins=True, labels=bin_labels)
        #age_bins_test = pd.cut(test_covariates['Age'], bins=bin_edges, labels=bin_labels)
        #one_hot_age_test = np.eye(10)[age_bins_test.values]
        one_hot_age = np.eye(27)[age_bins_train.values]
        
        gender_bins_train, bin_edges = pd.qcut(train_covariates['Gender'].rank(method="first"), q=2, retbins=True, labels=list(range(0,2)) )
        #age_bins_test = pd.cut(test_covariates['Age'], bins=bin_edges, labels=bin_labels)
        #one_hot_age_test = np.eye(10)[age_bins_test.values]
        one_hot_gender = np.eye(2)[gender_bins_train.values]
        
        bin_labels = list(range(0,10))      
        ICV_bins_train, bin_edges = pd.qcut(train_covariates['ICV'], q=10,  retbins=True, labels=bin_labels)
        #ICV_bins_test = pd.cut(test_covariates['ICV'], bins=bin_edges, labels=bin_labels)
        #one_hot_ICV_test = np.eye(10)[ICV_bins_test.values]
        ICV_bins_train.fillna(0, inplace = True)
        one_hot_ICV_train = np.eye(10)[ICV_bins_train.values]

        # -------------------------------------------------------------------------------------------------------------
        # Create the dataset iterator
        batch_size = 256
        n_samples = train_data.shape[0]



        torch.manual_seed(42)
        use_cuda =  torch.cuda.is_available()
        if use_cuda:
            torch.cuda.manual_seed(42)
        DEVICE = torch.device("cuda" if use_cuda else "cpu")
        
        input_dim = train_data.shape[1]
        one_hot_covariates_train = np.concatenate((one_hot_age, one_hot_gender, one_hot_ICV_train), axis=1).astype('float32')
        c_dim = one_hot_covariates_train.shape[1]
        train_dataset = MyDataset_labels(train_data.to_numpy(), one_hot_covariates_train)    
        #train_dataset = MyDataset(train_data)
        generator_train = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=False)
        # -------------------------------------------------------------------------------------------------------------
        # Create models
        #n_features = x_data_normalized.shape[1]
        #n_labels = y_data.shape[1]
        h_dim = [100,100]
        z_dim = 20



        # -------------------------------------------------------------------------------------------------------------
        # Training loop
        global_step = 0
        n_epochs = 200
        gamma = 0.98
        scale_fn = lambda x: gamma ** x
        base_lr = 0.0001
        max_lr = 0.005

        print('train model')
        model = cVAE(input_dim=input_dim, hidden_dim=h_dim, latent_dim=z_dim, c_dim=c_dim, learning_rate=0.0001, non_linear=True)
        model.to(DEVICE)
        
        step_size = 2 * np.ceil(n_samples / batch_size)
        
        for epoch in range(n_epochs): 
            for batch_idx, batch in enumerate(generator_train): 
                global_step = global_step + 1
                cycle = np.floor(1 + global_step / (2 * step_size))
                x_lr = np.abs(global_step / step_size - 2 * cycle + 1)
                clr = base_lr + (max_lr - base_lr) * max(0, 1 - x_lr) * scale_fn(cycle)
                model.optimizer = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=clr)
                data_curr = batch[0].to(DEVICE)
                cov = batch[1].to(DEVICE)
                fwd_rtn = model.forward(data_curr, cov)
                loss = model.loss_function(data_curr, fwd_rtn)
                model.optimizer.zero_grad()
                loss['total'].backward()
                model.optimizer.step() 
                if batch_idx == 0:
                    to_print = 'Train Epoch:' + str(epoch) + ' ' + 'Train batch: ' + str(batch_idx) + ' '+ ', '.join([k + ': ' + str(round(v.item(), 3)) for k, v in loss.items()])
                    print(to_print)        
                    if epoch == 0:
                        log_keys = list(loss.keys())
                        logger = Logger()
                        logger.on_train_init(log_keys)
                    else:
                        logger.on_step_fi(loss)
        plot_losses(logger, bootstrap_model_dir, 'training')
        model_path = join(bootstrap_model_dir, 'cVAE_model.pkl')
        torch.save(model, model_path)


if __name__ == "__main__":
    main()
