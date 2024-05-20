#!/home/songlinzhao/anaconda3/envs/normodiff/bin/python3
import argparse
from pathlib import Path
from sklearn.preprocessing import RobustScaler, OneHotEncoder
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
from sklearn.preprocessing import RobustScaler
from utils_vae import plot_losses, MyDataset_labels, Logger, reconstruction_deviation, latent_deviation, separate_latent_deviation, latent_pvalues, reconstruction_deviation_seperate_roi

PROJECT_ROOT = Path.cwd()


def main(dataset_name):
    """Make predictions using trained normative models."""
    # ----------------------------------------------------------------------------
    n_bootstrap = 10
    model_name = 'supervised_cvae'
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
        
        train_covariates = dataset_df[['Diagnosis','Age']]
        train_covariates.Diagnosis[train_covariates.Diagnosis == 0] = 0       #
        train_covariates['ICV'] =tiv  #        
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

        test_covariates = clinical_df[['Diagnosis','Gender', 'Age']]
        test_covariates.Diagnosis[test_covariates.Diagnosis == 0] = 0       #
        test_covariates['ICV'] =tiv  #   
        
        bin_labels = list(range(0,27))  
        age_bins_test, bin_edges = pd.qcut(test_covariates['Age'].rank(method="first"), q=27, retbins=True, labels=bin_labels)
        #age_bins_test = pd.cut(test_covariates['Age'], bins=bin_edges, labels=bin_labels)
        one_hot_age = np.eye(27)[age_bins_test.values]
        #one_hot_age_train = np.eye(10)[age_bins_train.values]
        
        gender_bins_test, bin_edges = pd.qcut(test_covariates['Gender'].rank(method="first"), q=2, retbins=True, labels=list(range(0,2)) )
        #gender_bins_train, bin_edges = pd.qcut(train_covariates['Gender'].rank(method="first"), q=10, retbins=True, labels=bin_labels)
        #gender_bins_test = pd.qcut(test_covariates['Gender'], bins=bin_edges, labels=bin_labels)
        one_hot_gender = np.eye(2)[gender_bins_test.values]
        #one_hot_age_train = np.eye(10)[age_bins_train.values]
        
        bin_labels = list(range(0,10))  
        #ICV_bins_train, bin_edges = pd.qcut(train_covariates['ICV'], q=10,  retbins=True, labels=bin_labels)
        ICV_bins_test, bin_edges = pd.qcut(test_covariates['ICV'], q=10,  retbins=True, labels=bin_labels)
        #ICV_bins_train.fillna(0, inplace = True)
        ICV_bins_test.fillna(0, inplace = True)
        #ICV_bins_train.fillna(0, inplace = True)
        one_hot_ICV_test = np.eye(10)[ICV_bins_test.values]
        #one_hot_ICV_train = np.eye(10)[ICV_bins_train.values]
        
        batch_size = 256
         
        torch.manual_seed(42)
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.manual_seed(42)
        DEVICE = torch.device("cuda" if use_cuda else "cpu")
     

        one_hot_covariates_test = np.concatenate((one_hot_age, one_hot_gender, one_hot_ICV_test), axis=1).astype('float32')

        if exists(join(bootstrap_train_dir, 'cVAE_model.pkl')):
            print('load trained model')
            model = torch.load(join(bootstrap_train_dir, 'cVAE_model.pkl'))  
            print(model)
            model.to(DEVICE)
        else:
            print('firstly train model ')
            
       
        test_latent, test_var = model.pred_latent(test_data, one_hot_covariates_test, DEVICE)
        #train_latent, _ = model.pred_latent(train_data, one_hot_covariates_train, DEVICE)
        test_prediction = model.pred_recon(test_data, one_hot_covariates_test, DEVICE)
        
        #reconstruction_error = np.mean(np.divide((abs(test_data - mean_list)), np.sqrt(variance_list + c)),axis=1)
        output_data = pd.DataFrame(clinical_df.Diagnosis.values, columns=['Diagnosis'])
        
        output_data['reconstruction_deviation'] = reconstruction_deviation(test_data.to_numpy(), test_prediction)
        reconstruction_deviation_separate_roi_lists = reconstruction_deviation_seperate_roi(test_data.to_numpy(), test_prediction)
        print(reconstruction_deviation_separate_roi_lists)
        #output_data['latent_deviation'] = latent_deviation(train_latent, test_latent, test_var)
        #deviation = separate_latent_deviation(train_latent, test_latent, test_var)
        
        
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

        # ----------------------------------------------------------------------------
        #reconstruction_error = np.mean((x - reconstruction) ** 2, axis=1)
        #reconstruction_error = np.mean((x_normalized - reconstruction) ** 2, axis=1)
        boostrap_error_mean.append(output_data['reconstruction_deviation'])

        reconstruction_error_df = pd.DataFrame(columns=['participant_id', 'Reconstruction error'])
        reconstruction_error_df['participant_id'] = clinical_df['participant_id']
        reconstruction_error_df['Reconstruction error'] = output_data['reconstruction_deviation']
        reconstruction_error_df.to_csv(output_dataset_dir / 'reconstruction_error.csv', index=False)

        print(reconstruction_error_df)

        reconstruction_deviation_seperate_roi_df = pd.DataFrame(columns=['participant_id'] + COLUMNS_NAME)
        reconstruction_deviation_seperate_roi_df['participant_id'] = clinical_df['participant_id']
        reconstruction_deviation_seperate_roi_df[COLUMNS_NAME] = reconstruction_deviation_separate_roi_lists
        # print(reconstruction_deviation_seperate_roi_df)
        # reconstruction_deviation_seperate_roi_df.to_csv(PROJECT_ROOT + 'result_baseline' + 'reconstruction_deviation_seperate_roi.csv', index=False)

    
    boostrap_error_mean = np.array(boostrap_error_mean)
    boostrap_mean = np.mean(boostrap_error_mean)
    bootsrao_var = np.std(boostrap_error_mean)
    boostrap_list = np.array([boostrap_mean, bootsrao_var])
    np.savetxt("cvae_boostrap_mean_std.csv", boostrap_list, delimiter=",")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--dataset_name',
                        dest='dataset_name',
                        help='Dataset name to calculate deviations.')
    args = parser.parse_args()

    main(args.dataset_name)