#!/home/songlinzhao/anaconda3/envs/normodiff/bin/python3
"""Inference the predictions of the clinical datasets using the supervised model."""
import argparse
from pathlib import Path
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

import torch

from tqdm import tqdm
import copy
from utils import COLUMNS_NAME, load_dataset
from os.path import join, exists
from VAE import VAE
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from utils_vae import plot_losses, MyDataset_labels, Logger, reconstruction_deviation, latent_deviation, separate_latent_deviation, latent_pvalues

PROJECT_ROOT = Path.cwd()


def main(dataset_name, comb_label):
    """Make predictions using trained normative models."""
    # ----------------------------------------------------------------------------
    n_bootstrap = 10
    model_name = 'supervised_cvae_age_gender'
    dataset_name = 'ADNI'

    participants_path = PROJECT_ROOT / 'data' / dataset_name / 'participants.tsv'
    freesurfer_path = PROJECT_ROOT / 'data' / dataset_name / 'freesurferData.csv'

    # ----------------------------------------------------------------------------
    # Create directories structure
    outputs_dir = PROJECT_ROOT / 'outputs'
    bootstrap_dir = outputs_dir / 'bootstrap_analysis'
    model_dir = bootstrap_dir / model_name
    ids_path = outputs_dir / (dataset_name + '_homogeneous_ids.csv')

    #============================================================================
    participants_train = PROJECT_ROOT / 'data' / 'ADNI_TRAIN' / 'participants.tsv'
    freesurfer_train = PROJECT_ROOT / 'data' / 'ADNI_TRAIN' / 'freesurferData.csv'
    # ----------------------------------------------------------------------------
   # bootstrap_dir = PROJECT_ROOT / 'outputs' / 'bootstrap_analysis'
    ids_dir = bootstrap_dir / 'ids'

    #model_dir = bootstrap_dir / model_name
    #model_dir.mkdir(exist_ok=True)
    # ----------------------------------------------------------------------------
    # Set random seed
    random_seed = 42
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)
    
    MCK = 5
    dc_output_list = [0]*MCK
    boostrap_error_mean = []
    # ----------------------------------------------------------------------------
    for i_bootstrap in tqdm(range(n_bootstrap)):
        
        ids_filename = 'cleaned_bootstrap_{:03d}.csv'.format(i_bootstrap)
        ids_train = ids_dir / ids_filename

        bootstrap_train_dir = model_dir / '{:03d}'.format(i_bootstrap)
        #bootstrap_train_dir.mkdir(exist_ok=True)

        # ----------------------------------------------------------------------------
        # Loading data
        dataset_df = load_dataset(participants_train, ids_train, freesurfer_train)

        # ----------------------------------------------------------------------------

        dataset_df = dataset_df.loc[dataset_df['Diagnosis'] == 1]      
        train_data = dataset_df[COLUMNS_NAME].values
        

        tiv = dataset_df['EstimatedTotalIntraCranialVol'].values
        tiv = tiv[:, np.newaxis]

        train_data = (np.true_divide(train_data, tiv)).astype('float32')

        scaler = RobustScaler()
        train_data = pd.DataFrame(scaler.fit_transform(train_data))     
        
        train_covariates = dataset_df[['Diagnosis','Age','Gender']]
        train_covariates.Diagnosis[train_covariates.Diagnosis == 0] = 0       #
        #train_covariates['ICV'] =tiv  #        
        #=============================================================================
        bootstrap_model_dir = model_dir / '{:03d}'.format(i_bootstrap)

        output_dataset_dir = bootstrap_model_dir / dataset_name
        output_dataset_dir.mkdir(exist_ok=True)

        # ----------------------------------------------------------------------------
        # Loading data
        clinical_df = load_dataset(participants_path, ids_path, freesurfer_path)
        #print(COLUMNS_NAME)
        test_data = clinical_df[COLUMNS_NAME].values

        tiv = clinical_df['EstimatedTotalIntraCranialVol'].values
        tiv = tiv[:, np.newaxis]

        test_data = (np.true_divide(test_data, tiv)).astype('float32')
        
        scaler = RobustScaler()
        test_data = pd.DataFrame(scaler.fit_transform(test_data))  

        test_covariates = clinical_df[['Diagnosis','Age','Gender']]
        test_covariates.Diagnosis[test_covariates.Diagnosis == 0] = 0       #
        test_covariates['ICV'] =tiv  #   
        
        bin_labels = list(range(0,10))  
        age_bins_test, bin_edges = pd.cut(test_covariates['Age'], 10, retbins=True, labels=bin_labels)

        age_bins_test.fillna(0,inplace = True)
        one_hot_age_test = np.eye(10)[age_bins_test.values]

        
        bin_labels3 = list(range(0,10))
        ICV_bins_test, bin_edges = pd.qcut(test_covariates['ICV'], q=10,  retbins=True, labels=bin_labels3)
        #ICV_bins_test = pd.cut(test_covariates['ICV'], bins=bin_edges, labels=bin_labels)
        #one_hot_ICV_test = np.eye(10)[ICV_bins_test.values]
        ICV_bins_test.fillna(0, inplace = True)
        one_hot_ICV_test = np.eye(10)[ICV_bins_test.values]
        
        gender = test_covariates['Gender'].values[:, np.newaxis].astype('float32')
        enc_gender = OneHotEncoder(sparse=False)
        one_hot_gender_test = enc_gender.fit_transform(gender)
        
        batch_size = 256
         
        torch.manual_seed(42)
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.manual_seed(42)
        DEVICE = torch.device("cuda" if use_cuda else "cpu")
     

        input_dim = train_data.shape[1]
        #one_hot_covariates_train = np.append(one_hot_age_train, one_hot_ICV_train, axis=1)
        #c_dim = one_hot_covariates_train.shape[1]
        
        
        if comb_label == 1:
            one_hot_covariates_test = np.concatenate((one_hot_age_test, one_hot_gender_test), axis=1).astype('float32')
        elif comb_label == 2:
            one_hot_covariates_test = np.concatenate((one_hot_age_test, one_hot_ICV_test), axis=1).astype('float32')
        elif comb_label == 3:
            one_hot_covariates_test = np.concatenate((one_hot_gender_test,one_hot_ICV_test), axis=1).astype('float32')
        else:
            one_hot_covariates_test = np.concatenate((one_hot_age_test, one_hot_gender_test,one_hot_ICV_test), axis=1).astype('float32')    
        
        
        if exists(join(bootstrap_train_dir, 'cVAE_model.pkl')):
            print('load trained model')
            model = torch.load(join(bootstrap_train_dir, 'cVAE_model.pkl'))  
            # print(model)
            model.to(DEVICE)
        else:
            print('firstly train model ')
            
        
            
        
        test_latent, test_var = model.pred_latent(test_data, one_hot_covariates_test, DEVICE)
        #train_latent, _ = model.pred_latent(train_data, one_hot_covariates_train, DEVICE)
        test_prediction = model.pred_recon(test_data, one_hot_covariates_test, DEVICE)
        
        output_data = pd.DataFrame(clinical_df.Diagnosis.values, columns=['Diagnosis'])
        output_data['reconstruction_deviation'] = reconstruction_deviation(test_data.to_numpy(), test_prediction)

        
        
        normalized_df = pd.DataFrame(columns=['participant_id'] + COLUMNS_NAME)
        normalized_df['participant_id'] = clinical_df['participant_id']
        normalized_df[COLUMNS_NAME] = test_data
        normalized_df.to_csv(output_dataset_dir / 'normalized.csv', index=False)

        reconstruction_df = pd.DataFrame(columns=['participant_id'] + COLUMNS_NAME)
        reconstruction_df['participant_id'] = clinical_df['participant_id']
        reconstruction_df[COLUMNS_NAME] = test_prediction
        reconstruction_df.to_csv(output_dataset_dir / 'reconstruction.csv', index=False)



        encoded_df = pd.DataFrame(columns=['participant_id'] + list(range(test_latent.shape[1])))
        
        encoded_df['participant_id'] = clinical_df['participant_id']
        encoded_df[list(range(test_latent.shape[1]))] = test_latent
        encoded_df.to_csv(output_dataset_dir / 'encoded.csv', index=False)


        boostrap_error_mean.append(output_data['reconstruction_deviation'])

        reconstruction_error_df = pd.DataFrame(columns=['participant_id', 'Reconstruction error'])
        reconstruction_error_df['participant_id'] = clinical_df['participant_id']
        reconstruction_error_df['Reconstruction error'] = output_data['reconstruction_deviation']
        reconstruction_error_df.to_csv(output_dataset_dir / 'reconstruction_error.csv', index=False)
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--dataset_name',
                        dest='dataset_name',
                        help='Dataset name to calculate deviations.')
    parser.add_argument('-L', '--comb_label',
                        dest='comb_label',
                        help='Combination label to perform group analysis.',
                        type=int)
    args = parser.parse_args()

    dataset_name = args.dataset_name
    comb_label = args.comb_label

    if dataset_name is None:
        dataset_name = 'ADNI'
    if comb_label is None:
        comb_label = 0

    main(dataset_name, comb_label)