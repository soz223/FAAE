from argparse import ArgumentParser
import pandas as pd
from os.path import join, exists
import numpy as np
import matplotlib.pyplot as plt
import torch
from VAE import VAE
import os
from utils_vae import plot_losses, Logger, MyDataset, reconstruction_deviation, latent_deviation, separate_latent_deviation, latent_pvalues
from sklearn.preprocessing import RobustScaler

def process():
    parser = ArgumentParser(description="Train VAE")
    parser.add_argument('--path', type=str, default='', help='Path to input data')
    parser.add_argument('--outpath', type=str, default='', help='Path to save models')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train model')
    parser.add_argument('--zdim', type=int, default=10, help='Number of latent vectors')
    parser.add_argument('--hiddendim', type=list, default=[40], help='List of hidden layer sizes')
    parser.add_argument('--batchsize', type=int, default=1000, help='Batch size')
    parser.add_argument('--GPU', type=bool, default=True, help='Whether to use GPU for model training')
    args = parser.parse_args()
    
    args.path = "./data/"
    args.outpath = "./models/"
    
    train_data = pd.read_csv(join(args.path, 'train_data.csv'),header=0)
    
    tiv = train_data['EstimatedTotalIntraCranialVol'].values
    tiv = tiv[:, np.newaxis]

    train_data.drop(labels=['Image_ID','EstimatedTotalIntraCranialVol'], axis=1,inplace=True)
    train_data = (np.true_divide(train_data, tiv)).astype('float32')

    scaler = RobustScaler()
    train_data = scaler.fit_transform(train_data)
    train_data = pd.DataFrame(train_data)
    #train_data.drop(labels=['Image_ID','EstimatedTotalIntraCranialVol'], axis=1,inplace=True)
    test_data = pd.read_csv(join(args.path, 'test_data.csv'),header=0)

    tiv = test_data['EstimatedTotalIntraCranialVol'].values
    tiv = tiv[:, np.newaxis]

    test_data.drop(labels=['Image_ID','EstimatedTotalIntraCranialVol'], axis=1,inplace=True)
    test_data = (np.true_divide(test_data, tiv)).astype('float32')

    scaler = RobustScaler()
    test_data = scaler.fit_transform(test_data)
    test_data = pd.DataFrame(test_data)
    
    test_covariates = pd.read_csv(join(args.path, 'test_covariates.csv'),header=0)
    test_covariates.Diagnosis[test_covariates.Diagnosis == 17] = 0      #
    test_covariates['ICV'] = tiv   #test_data['EstimatedTotalIntraCranialVol'] #
    
    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)   
         
    torch.manual_seed(42)
    use_cuda = args.GPU and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.manual_seed(42)
    DEVICE = torch.device("cuda" if use_cuda else "cpu")
    input_dim = train_data.shape[1]

    train_dataset = MyDataset(train_data.to_numpy())
    #train_dataset = MyDataset(train_data)
    generator_train = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=False)

    if exists(join(args.outpath, 'VAE_model.pkl')):
        print('load trained model')
        model = torch.load(join(args.outpath, 'VAE_model.pkl'))  
        print(model)
        model.to(DEVICE)
    else:
        print('train model')
        model = VAE(input_dim=input_dim, hidden_dim=args.hiddendim, latent_dim=args.zdim, learning_rate=0.001, non_linear=True)
        model.to(DEVICE)

        for epoch in range(args.epochs): 
            for batch_idx, batch in enumerate(generator_train): 
                batch = batch.to(DEVICE)
                fwd_rtn = model.forward(batch)
                loss = model.loss_function(batch, fwd_rtn)
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
        plot_losses(logger, args.outpath, 'training')
        model_path = join(args.outpath, 'VAE_model.pkl')
        torch.save(model, model_path)

    test_latent, test_var = model.pred_latent(test_data, DEVICE)
    train_latent, _ = model.pred_latent(train_data, DEVICE)
    test_prediction = model.pred_recon(test_data, DEVICE)

    output_data = pd.DataFrame(test_covariates.Diagnosis.values, columns=['Diagnosis'])
    output_data['reconstruction_deviation'] = reconstruction_deviation(test_data.to_numpy(), test_prediction)
    output_data['latent_deviation'] = latent_deviation(train_latent, test_latent, test_var)
    deviation = separate_latent_deviation(train_latent, test_latent, test_var)
    for i in range(args.zdim):
        output_data['latent_deviation_{0}'.format(i)] = deviation[:,i]

    rows = output_data['Diagnosis'].unique()
    fig=plt.figure(figsize=(18,4)) 
    fig.suptitle('Latent deviation metric')
    for i in range(1, args.zdim+1):
        ax = fig.add_subplot(2, 5, i)
        ax.title.set_text('Latent {0}'.format(i))
        for position, column in enumerate(rows):
            ax.boxplot(output_data[output_data['Diagnosis']==column]['latent_deviation_{0}'.format(i-1)], positions=[position])           
        ax.set_xticks(range(position+1))
        ax.set_xticklabels(rows)
        ax.set_xlim(xmin=-0.5)
    plt.savefig(join(args.outpath, 'boxplot_separate_latent_deviation.png'))

    fig, ax = plt.subplots(figsize=(3.2,4.8))
    fig.suptitle('Reconstruction metric')
    for position, column in enumerate(rows):
        ax.boxplot(output_data[output_data['Diagnosis']==column]['reconstruction_deviation'], positions=[position])
    ax.set_xticks(range(position+1))
    ax.set_xticklabels(rows)
    ax.set_xlim(xmin=-0.5)
    plt.savefig(join(args.outpath, 'boxplot_reconstruction_deviation.png'))

    fig, ax = plt.subplots(figsize=(3.2,4.8))
    fig.suptitle('Latent deviation metric')
    for position, column in enumerate(rows):
        ax.boxplot(output_data[output_data['Diagnosis']==column]['latent_deviation'], positions=[position])
    ax.set_xticks(range(position+1))
    ax.set_xticklabels(rows)
    ax.set_xlim(xmin=-0.5)
    plt.savefig(join(args.outpath, 'boxplot_latent_deviation.png'))

    age_pval = latent_pvalues(test_latent, test_covariates['Age'], type='continuous')
    age_pval.to_csv(join(args.outpath, 'p_values_latentdeviations_vs_age.csv'), index=False)

    ICV_pval = latent_pvalues(test_latent, test_covariates['ICV'], type='continuous')
    ICV_pval.to_csv(join(args.outpath, 'p_values_latentdeviations_vs_ICV.csv'), index=False)

    DX_pval = latent_pvalues(test_latent, test_covariates['Diagnosis'], type='discrete')
    DX_pval.to_csv(join(args.outpath, 'p_values_latentdeviations_vs_DX.csv'), index=False)

if __name__ == '__main__':
    process()