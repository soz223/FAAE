#!/home/songlinzhao/anaconda3/envs/normodiff/bin/python3
"""Inference the predictions of the clinical datasets using the supervised model."""
import argparse
from pathlib import Path
from sklearn.preprocessing import RobustScaler, OneHotEncoder
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import copy
from utils import COLUMNS_NAME, load_dataset

PROJECT_ROOT = Path.cwd()

def main(dataset_name, comb_label):
    """Make predictions using trained normative models."""
    # ----------------------------------------------------------------------------
    n_bootstrap = 10
    model_name = 'supervised_aae_new'
    dataset_name = 'ADNI'

    participants_path = PROJECT_ROOT / 'data' / dataset_name / 'participants.tsv'
    freesurfer_path = PROJECT_ROOT / 'data' / dataset_name / 'freesurferData.csv'

    # ----------------------------------------------------------------------------
    # Create directories structure
    outputs_dir = PROJECT_ROOT / 'outputs'
    bootstrap_dir = outputs_dir / 'bootstrap_analysis'
    model_dir = bootstrap_dir / model_name
    ids_path = outputs_dir / (dataset_name + '_homogeneous_ids.csv')

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
        bootstrap_model_dir = model_dir / '{:03d}'.format(i_bootstrap)

        output_dataset_dir = bootstrap_model_dir / dataset_name
        output_dataset_dir.mkdir(exist_ok=True)

        # ----------------------------------------------------------------------------
        # Loading data
        clinical_df = load_dataset(participants_path, ids_path, freesurfer_path)
        #print(COLUMNS_NAME)
        x_dataset = clinical_df[COLUMNS_NAME].values

        tiv = clinical_df['EstimatedTotalIntraCranialVol'].values
        tiv = tiv[:, np.newaxis]

        x_dataset = (np.true_divide(x_dataset, tiv)).astype('float32')
        # ----------------------------------------------------------------------------
        encoder = keras.models.load_model(bootstrap_model_dir / 'encoder.h5', compile=False)
        decoder = keras.models.load_model(bootstrap_model_dir / 'decoder.h5', compile=False)

        scaler = joblib.load(bootstrap_model_dir / 'scaler.joblib')

        #enc_age = joblib.load(bootstrap_model_dir / 'age_encoder.joblib')
        #enc_gender = joblib.load(bootstrap_model_dir / 'gender_encoder.joblib')
        
        
        enc_age = joblib.load(bootstrap_model_dir / 'age_encoder.joblib')
        enc_gender = joblib.load(bootstrap_model_dir / 'gender_encoder.joblib')
        
        age = clinical_df['Age'].values[:, np.newaxis].astype('float32')
        #print(age, age.shape)
        one_hot_age = enc_age.transform(age)
        #print(one_hot_age, one_hot_age.shape)
        
        
        gender = clinical_df['Gender'].values[:, np.newaxis].astype('float32')
        one_hot_gender = enc_gender.transform(gender)
        
        bin_labels = list(range(0,10))
        #age_bins_train, bin_edges = pd.cut(clinical_df['Age'], 10, retbins=True, labels=bin_labels)
        #age_bins_train.fillna(0, inplace=True)
        #one_hot_age = np.eye(10)[age_bins_train.values]
        
        
        ICV_bins_test, bin_edges = pd.qcut(clinical_df['EstimatedTotalIntraCranialVol'], q=10,  retbins=True, labels=bin_labels)
        ICV_bins_test.fillna(0, inplace = True)
        one_hot_ICV_test = np.eye(10)[ICV_bins_test.values]

        #y_data = np.concatenate((one_hot_age, one_hot_gender, one_hot_ICV_train), axis=1).astype('float32')
        
        
        if comb_label == 1:
            y_data = np.concatenate((one_hot_age, one_hot_gender), axis=1).astype('float32')
        elif comb_label == 2:
            y_data = np.concatenate((one_hot_age, one_hot_ICV_test), axis=1).astype('float32')
        elif comb_label == 3:
            y_data = np.concatenate((one_hot_gender,one_hot_ICV_test), axis=1).astype('float32')
        else:
            y_data = np.concatenate((one_hot_age, one_hot_gender,one_hot_ICV_test), axis=1).astype('float32')  
        #y_data = np.concatenate((one_hot_age,  one_hot_ICV_test), axis=1).astype('float32')
        #y_data = np.concatenate((one_hot_age, one_hot_gender,  one_hot_ICV_test), axis=1).astype('float32')
        #y_data = one_hot_age
        
        #extended = []
        #for i in range(y_data.shape[1]):
        #    val = "d" + str(i)
        #    extended.append(val)
        # ----------------------------------------------------------------------------
        x_normalized = scaler.transform(x_dataset)
        #x = np.concatenate((x_normalized, y_data), axis=1)
        #COLUMNS_NAME.append(['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16',
        #                     '17','18','19','20','21','22','23','24','25','26','27'])
        #tmp = copy.copy(COLUMNS_NAME)
        #tmp.extend(extended)
        #print(COLUMNS_NAME)
        #print("aaaaaaaaaaaaaa")
        normalized_df = pd.DataFrame(columns=['participant_id'] + COLUMNS_NAME)
        normalized_df['participant_id'] = clinical_df['participant_id']
        normalized_df[COLUMNS_NAME] = x_normalized
        normalized_df.to_csv(output_dataset_dir / 'normalized.csv', index=False)

        z_mean, z_log_var, encoded = encoder(x_normalized, training=False)
        
        reconstruction = decoder(tf.concat([encoded,y_data], axis=1), training=False)

        reconstruction_df = pd.DataFrame(columns=['participant_id'] + COLUMNS_NAME)
        reconstruction_df['participant_id'] = clinical_df['participant_id']
        reconstruction_df[COLUMNS_NAME] = reconstruction.numpy()
        reconstruction_df.to_csv(output_dataset_dir / 'reconstruction.csv', index=False)

        encoded_df = pd.DataFrame(columns=['participant_id'] + list(range(encoded.shape[1])))
        
        encoded_df['participant_id'] = clinical_df['participant_id'] 
        encoded_df[list(range(encoded.shape[1]))] = encoded.numpy()
        encoded_df.to_csv(output_dataset_dir / 'encoded.csv', index=False)

        # ----------------------------------------------------------------------------
        #reconstruction_error = np.mean((x - reconstruction) ** 2, axis=1)
        reconstruction_error = np.mean((x_normalized - reconstruction) ** 2, axis=1)
        #reconstruction_error = np.mean(np.divide((abs(x_normalized - mean_list)), np.sqrt(variance_list + c)),axis=1)
        boostrap_error_mean.append(np.mean(reconstruction_error))

        reconstruction_error_df = pd.DataFrame(columns=['participant_id', 'Reconstruction error'])
        reconstruction_error_df['participant_id'] = clinical_df['participant_id']
        reconstruction_error_df['Reconstruction error'] = reconstruction_error
        reconstruction_error_df.to_csv(output_dataset_dir / 'reconstruction_error.csv', index=False)
        
    boostrap_error_mean = np.array(boostrap_error_mean)
    boostrap_mean = np.mean(boostrap_error_mean)
    bootsrao_var = np.std(boostrap_error_mean)
    boostrap_list = np.array([boostrap_mean, bootsrao_var])


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

    main(args.dataset_name, args.comb_label)