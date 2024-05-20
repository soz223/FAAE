#!/home/songlinzhao/anaconda3/envs/normodiff/bin/python3
"""Script to train the deterministic supervised adversarial autoencoder."""
from pathlib import Path
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from sklearn.preprocessing import RobustScaler
import numpy as np
from sklearn.preprocessing import RobustScaler, OneHotEncoder

from utils import COLUMNS_NAME, load_dataset
from utils_vae import plot_losses, MyDataset_labels, Logger
import torch
from cVAE import cVAE
import os
import argparse
from os.path import join
from utils_vae import plot_losses, Logger, MyDataset
from torchmetrics import MeanMetric
import matplotlib.pyplot as plt

# Get the path of the current working directory and set it to PROJECT_ROOT
PROJECT_ROOT = Path.cwd()

result_focal_dir =  './result_focal'
result_dir = './result'

EPOCHES = 200

def main(comb_label, hz_para_list, alpha_focal, gamma_focal, lambda_reg):
    #Train the normative method on the bootstrapped samples.

    # ----------------------------------------------------------------------------
    # Set the number of bootstrapping samples to 10
    n_bootstrap = 10
    # Set the name of the model
    model_name = 'supervised_cvae_age_gender'



    # Set the path of the participants file
    participants_path = PROJECT_ROOT / 'data' / 'ADNI_TRAIN' / 'participants.tsv'
    # Set the path of the freesurfer data file
    freesurfer_path = PROJECT_ROOT / 'data' / 'ADNI_TRAIN' / 'freesurferData.csv'


    # ----------------------------------------------------------------------------
    # Set the path of the bootstrap directory
    bootstrap_dir = PROJECT_ROOT / 'outputs' / 'bootstrap_analysis'
    # Set the path of the IDs directory
    ids_dir = bootstrap_dir / 'ids'

    # Set the path of the model directory
    model_dir = bootstrap_dir / model_name
    # Create the model directory if it does not exist
    model_dir.mkdir(exist_ok=True)

    # ----------------------------------------------------------------------------
    # Set random seed
    random_seed = 114514
    # Set the random seed for numpy
    np.random.seed(random_seed)

    # Initialize a list of EPOCHES zeros to store autoencoder losses for each epoch
    ae_loss_list = [0]*EPOCHES
    # Initialize a list of EPOCHES zeros to store KL divergence losses for each epoch
    kl_loss_list = [0]*EPOCHES
    # Initialize a list of EPOCHES zeros to store total losses for each epoch
    total_loss_list = [0]*EPOCHES

    # For each bootstrap sample
    for i_bootstrap in range(n_bootstrap):
        print('Bootstrap sample: {}'.format(i_bootstrap))
        # Set the filename of the current bootstrap sample based on its index, with leading zeros padded to three digits
        ids_filename = 'cleaned_bootstrap_{:03d}.csv'.format(i_bootstrap)
        # Set the path of the current bootstrap sample IDs file
        ids_path = ids_dir / ids_filename

        # Set the path of the directory for the current bootstrap sample, with leading zeros padded to three digits
        bootstrap_model_dir = model_dir / '{:03d}'.format(i_bootstrap)
        # Create the directory for the current bootstrap sample if it does not exist
        bootstrap_model_dir.mkdir(exist_ok=True)

        # ----------------------------------------------------------------------------
        # Loading data
        dataset_df = load_dataset(participants_path, ids_path, freesurfer_path)

        # ----------------------------------------------------------------------------
        # Filter the dataset to only include participants with a diagnosis of 1 (i.e., Alzheimer's disease)
        dataset_df = dataset_df.loc[dataset_df['Diagnosis'] == 1]  
        # Extract the features from the dataset and convert to a numpy array
        train_data = dataset_df[COLUMNS_NAME].values  
        # Extract the total intracranial volume (TIV) from the dataset
        tiv = dataset_df['EstimatedTotalIntraCranialVol'].values  
        # Add a new axis to TIV to make it compatible with the feature array
        tiv = tiv[:, np.newaxis]  
        # Normalize the features by dividing by TIV and cast to float32 data type
        train_data = (np.true_divide(train_data, tiv)).astype('float32')  
        # Initialize a robust scaler object
        scaler = RobustScaler()  
        # Scale the features using the robust scaler and store the transformed features in the train_data variable
        train_data = scaler.fit_transform(train_data) 
        # Convert the numpy array of scaled features to a pandas DataFrame
        train_data = pd.DataFrame(train_data)  
         # Extract the diagnosis, age, and gender covariates from the dataset
        train_covariates = dataset_df[['Diagnosis','Age','Gender']]  
        # Replace any values of 0 in the diagnosis column with 0
        train_covariates.Diagnosis[train_covariates.Diagnosis == 0] = 0  
        # Add the TIV as a new column named 'ICV'
        train_covariates['ICV'] = tiv
         # Set the labels for the age bins to be integers from 0 to 9
        bin_labels = list(range(0,10))  
        # Bin the ages into 10 bins and return the bin labels and edges
        age_bins_train, bin_edges = pd.cut(train_covariates['Age'], 10, retbins=True, labels=bin_labels)  
        # Replace any missing values in the age bins with 0
        age_bins_train.fillna(0, inplace=True)  
        # Convert the age bin labels to one-hot encoding
        one_hot_age_train = np.eye(10)[age_bins_train.values]  
        # Set the labels for the ICV bins to be integers from 0 to 9
        bin_labels3 = list(range(0,10)) 
        # Bin the ICVs into 10 bins of equal size and return the bin labels and edges
        ICV_bins_train, bin_edges = pd.qcut(train_covariates['ICV'], q=10,  retbins=True, labels=bin_labels3)  
        # Replace any missing values in the ICV bins with 0
        ICV_bins_train.fillna(0, inplace = True)  
        # Convert the ICV bin labels to one-hot encoding
        one_hot_ICV_train = np.eye(10)[ICV_bins_train.values]  
        # Extract the gender column from the covariates and add a new axis to make it compatible with the one-hot encoder. Cast to float32 data type.
        gender = train_covariates['Gender'].values[:, np.newaxis].astype('float32') 
        # Initialize a one-hot encoder object
        enc_gender = OneHotEncoder(sparse=False)  
        # Convert the gender column to one-hot encoding using the one-hot encoder
        one_hot_gender_train = enc_gender.fit_transform(gender)  
        # Set the batch size to 256
        batch_size = 256  
        # Get the number of samples in the training dataset
        n_samples = train_data.shape[0]  
        # Set the random seed to 42
        torch.manual_seed(random_seed) 

        # Check if CUDA is available
        use_cuda =  torch.cuda.is_available()
        # If CUDA is available, set the random seed for CUDA
        if use_cuda:
            torch.cuda.manual_seed(random_seed)
        # Set the device to be CUDA if available, otherwise set it to CPU
        DEVICE = torch.device("cuda" if use_cuda else "cpu")
        # Get the number of features
        input_dim = train_data.shape[1]
        # If the combination label is 1, concatenate the one-hot encoded age and gender covariates
        if comb_label == 1:
            one_hot_covariates_train = np.concatenate((one_hot_age_train, one_hot_gender_train), axis=1).astype('float32')
        # If the combination label is 2, concatenate the one-hot encoded age and ICV covariates
        elif comb_label == 2:
            one_hot_covariates_train = np.concatenate((one_hot_age_train, one_hot_ICV_train), axis=1).astype('float32')
        # If the combination label is 3, concatenate the one-hot encoded gender and ICV covariates
        elif comb_label == 3:
            one_hot_covariates_train = np.concatenate((one_hot_gender_train,one_hot_ICV_train), axis=1).astype('float32')
        # If the combination label is anything else, concatenate all three one-hot encoded covariates
        else:
            one_hot_covariates_train = np.concatenate((one_hot_age_train, one_hot_gender_train,one_hot_ICV_train), axis=1).astype('float32')  
        # Get the number of covariates
        c_dim = one_hot_covariates_train.shape[1]
        # Initialize a dataset object using the feature and covariate arrays
        train_dataset = MyDataset_labels(train_data.to_numpy(), one_hot_covariates_train)    
        # Initialize a data loader object to generate batches of the training data
        generator_train = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=False)
        # Get the sizes of the hidden layers for the encoder and decoder
        h_dim = [hz_para_list[0], hz_para_list[1]]
        # Get the size of the latent space
        z_dim = hz_para_list[2]

       
        # Training loop
        # Initialize the global step counter
        global_step = 0
        # Set the number of epochs
        n_epochs = EPOCHES
        # Set the gamma value for the cyclic learning rate
        gamma = 0.98
        # Define the scaling function for the cyclic learning rate
        scale_fn = lambda x: gamma ** x
        # Set the base learning rate
        base_lr = 0.0001
        # Set the maximum learning rate
        max_lr = 0.005
        # Check if CUDA is available
        cuda = torch.cuda.is_available()
        # Define the Tensor data type based on whether CUDA is available or not
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        # Initialize the cVAE model object
        model = cVAE(input_dim=input_dim, hidden_dim=h_dim, latent_dim=z_dim, c_dim=c_dim, learning_rate=0.0001, non_linear=True)
        # Move the model to the appropriate device (CPU or GPU)
        model.to(DEVICE)
        # Calculate the step size for the cyclic learning rate based on the number of samples and batch size
        step_size = 2 * np.ceil(n_samples / batch_size)
        
        # Loop over the epochs
        for epoch in range(n_epochs): 
            # Initialize metrics
            epoch_ae_loss_avg = MeanMetric().to(DEVICE)
            epoch_kl_loss_avg = MeanMetric().to(DEVICE)
            epoch_total_loss_avg = MeanMetric().to(DEVICE)
            epoch_dc_loss_avg = MeanMetric().to(DEVICE)
            epoch_gen_loss_avg = MeanMetric().to(DEVICE)
            # Loop over the batches in the training data
            for batch_idx, batch in enumerate(generator_train): 
                # Increment the global step counter
                global_step = global_step + 1
                # Calculate the cycle number for the cyclic learning rate
                cycle = np.floor(1 + global_step / (2 * step_size))
                # Calculate the learning rate multiplier for the cyclic learning rate
                x_lr = np.abs(global_step / step_size - 2 * cycle + 1)
                # Calculate the cyclic learning rate for this step
                clr = base_lr + (max_lr - base_lr) * max(0, 1 - x_lr) * scale_fn(cycle)
                # Set the optimizer for the encoder and decoder
                model.optimizer1 = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=clr)
                # Get the current batch of data and move it to the appropriate device
                data_curr = batch[0].to(DEVICE)
                # Get the current batch of covariates and move it to the appropriate device
                cov = batch[1].to(DEVICE)
                # Forward pass through the model
                fwd_rtn = model.forward(data_curr, cov)
                # Calculate the loss
                loss = model.loss_function(data_curr, fwd_rtn)
                # Update the average autoencoder loss metric
                epoch_ae_loss_avg.update(loss['ll'])
                # Update the average KL divergence loss metric
                epoch_kl_loss_avg.update(loss['kl'])
                # Update the average total loss metric
                epoch_total_loss_avg.update(loss['total'])
                # Zero out the gradients
                model.optimizer1.zero_grad()
                # Backward pass through the model to calculate thecuda gradients
                loss['total'].backward()
                # Update the weights using the gradients
                model.optimizer1.step() 
                
                # Set the optimizer for the discriminator
                model.optimizer2 = torch.optim.Adam(list(model.discriminator.parameters()), lr=clr)
                # Get the current batch of data and move it to the appropriate device
                data_curr = batch[0].to(DEVICE)
                # Get the current batch of covariates and move it to the appropriate device
                cov = batch[1].to(DEVICE)
                # Forward pass through the model
                fwd_rtn2 = model.forward2(data_curr, cov, z_dim)
                # Calculate the loss
                loss2 = model.loss_function2(x = data_curr, fwd_rtn = fwd_rtn2, alpha_focal=alpha_focal, gamma_focal=gamma_focal, lambda_reg=lambda_reg, logits=True, reduction='mean')
                # loss2 = model.loss_function2(x = data_curr, fwd_rtn = fwd_rtn2)
                # Update the average discriminator loss metric
                epoch_dc_loss_avg.update(loss2['dc_loss'])
                # Zero out the gradients
                model.optimizer2.zero_grad()
                # Backward pass through the model to calculate the gradients
                loss2['dc_loss'].backward()
                # Update the weights using the gradients 
                model.optimizer2.step() 
                
                # Set the optimizer for the encoder
                model.optimizer3 = torch.optim.Adam(list(model.encoder.parameters()), lr=clr)
                # Get the current batch of data and move it to the appropriate device
                data_curr = batch[0].to(DEVICE)
                # Get the current batch of covariates and move it to the appropriate device
                cov = batch[1].to(DEVICE)
                # Forward pass through the model
                fwd_rtn3 = model.forward3(data_curr, cov)
                # Calculate the loss
                loss3 = model.loss_function3(data_curr, fwd_rtn3)
                # Update the average generator loss metric
                epoch_gen_loss_avg.update(loss3['gen_loss'])
                # Zero out the gradients
                model.optimizer3.zero_grad()
                # Backward pass through the model to calculate the gradients
                loss3['gen_loss'].backward()
                # Update the weights using the gradients 
                model.optimizer3.step() 
                
                # Update the autoencoder loss with the discriminator loss
                loss.update(loss2)
                # Update the autoencoder loss with the generator loss
                loss.update(loss3)
                
                # Check if this is the first batch in the epoch
                if batch_idx == 0:
                    # Print the epoch and batch number, along with the loss values for each component of the loss
                    to_print = 'Train Epoch:' + str(epoch) + ' ' + 'Train batch: ' + str(batch_idx) + ' '+ ', '.join([k + ': ' + str(round(v.item(), 5)) for k, v in loss.items()])      
                    print(to_print)
                    # If this is the first epoch, initialize the logger
                    if epoch == 0:
                        log_keys = list(loss.keys())
                        logger = Logger()
                        logger.on_train_init(log_keys)
                     # Otherwise, update the logger with the loss values
                    else:
                        logger.on_step_fi(loss)
            # Update the epoch-average autoencoder loss, KL divergence loss, and total loss with their current values
            ae_loss_list[epoch] += epoch_ae_loss_avg.compute()
            kl_loss_list[epoch] += epoch_kl_loss_avg.compute()
            total_loss_list[epoch] += epoch_total_loss_avg.compute()
        # Plot the losses for the current epoch
        plot_losses(logger, bootstrap_model_dir, 'training')
        # Save the current model checkpoint
        model_path = join(bootstrap_model_dir, 'cVAE_model.pkl')
        torch.save(model, model_path)
    
    # Compute the mean of the autoencoder loss, KL divergence loss, and total loss over all bootstrap samples for each epoch

    ae_loss_list_cpu = []
    kl_loss_list_cpu = []
    total_loss_list_cpu = []
    for item in ae_loss_list:
        # turn item into a number
        item = item.item()
        ae_loss_list_cpu.append(item)

    for item in kl_loss_list:
        # turn item into a number
        item = item.item()
        kl_loss_list_cpu.append(item)

    for item in total_loss_list:
        # turn item into a number
        item = item.item()
        total_loss_list_cpu.append(item)

    ae_loss_array = np.array(ae_loss_list_cpu)
    kl_loss_array = np.array(kl_loss_list_cpu)
    total_loss_array = np.array(total_loss_list_cpu)
    
    print('type of total_loss_array: ', type(total_loss_array))
    print('type of ae_loss_array: ', type(ae_loss_array))
    print('type of kl_loss_array: ', type(kl_loss_array))
    
    ae_loss_array = np.divide(ae_loss_array, n_bootstrap)
    kl_loss_array = np.divide(kl_loss_array, n_bootstrap)
    total_loss_array = np.divide(total_loss_array, n_bootstrap)
    epoch = range(1, EPOCHES + 1)
    # Plot the average loss values for each epoch
    plt.plot(epoch, ae_loss_array, 'g', label = 'AE loss')
    plt.plot(epoch, kl_loss_array, 'b', label = 'KL loss')
    plt.plot(epoch, total_loss_array, 'r', label = 'TOTAL loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-L', '--comb_label',
                        dest='comb_label',
                        help='Combination label to perform group analysis.',
                        type=int)
    parser.add_argument('-H', '--hz_para_list',
                        dest='hz_para_list',
                        nargs='+',
                        help='List of paras to perform the analysis.',
                        type=int)
    
    # add parameter called alpha_focal gamma_focal 
    parser.add_argument('-A', '--alpha_focal',
                        dest='alpha_focal',
                        help='alpha_focal for focal loss.',
                        type=float)
    parser.add_argument('-G', '--gamma_focal',
                        dest='gamma_focal',
                        help='gamma_focal for focal loss.',
                        type=float)
    parser.add_argument('-R', '--lambda_reg',
                        dest='lambda_reg',
                        help='lambda_reg for regularization.',
                        type=float)


    args = parser.parse_args()


    comb_label = args.comb_label
    hz_para_list = args.hz_para_list
    alpha_focal = args.alpha_focal
    gamma_focal = args.gamma_focal
    lambda_reg = args.lambda_reg
    
    print('args.alpha_focal: ', args.alpha_focal)
    print('args.gamma_focal: ', args.gamma_focal)
    print('args.lambda_reg: ', args.lambda_reg)


    # default setting
    if comb_label == None:
        comb_label = 0
    if hz_para_list == None:
        hz_para_list = [110, 110, 10]
    if alpha_focal == None:
        alpha_focal = 0
    if gamma_focal == None:
        gamma_focal = 1
    if lambda_reg == None:
        lambda_reg = 0


    main(comb_label, hz_para_list, alpha_focal, gamma_focal, lambda_reg)

    file_path = result_dir + '/result.txt'

    # if the file or folder does not exist, create it
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    with open(result_dir + '/result.txt', 'a') as f:
        f.write('experiment settings\n')
        f.write('alpha_focal: ' + str(alpha_focal) + '\n')
        f.write('gamma_focal: ' + str(gamma_focal) + '\n')
        f.write('lambda_reg: ' + str(lambda_reg) + '\n')