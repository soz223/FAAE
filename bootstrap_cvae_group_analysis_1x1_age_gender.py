#!/usr/bin/env python3
import argparse
from pathlib import Path
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import os
from utils import COLUMNS_NAME, load_dataset, cliff_delta
from scipy import interp
from numpy import linspace

PROJECT_ROOT = Path.cwd()
result_focal_dir = './result'
result_dir = './result'

def compute_brain_regions_deviations(diff_df, clinical_df, disease_label, hc_label=1):
    """ Calculate the Cliff's delta effect size between groups."""
    region_df = pd.DataFrame(columns=['regions', 'pvalue', 'effect_size'])

    diff_hc = diff_df.loc[clinical_df['Diagnosis'] == disease_label]
    diff_patient = diff_df.loc[clinical_df['Diagnosis'] == hc_label]

    for region in COLUMNS_NAME:
        _, pvalue = stats.mannwhitneyu(diff_hc[region], diff_patient[region])
        effect_size = cliff_delta(diff_hc[region].values, diff_patient[region].values)

        region_df = region_df.append({'regions': region, 'pvalue': pvalue, 'effect_size': effect_size},
                                     ignore_index=True)

    return region_df

def compute_classification_performance(reconstruction_error_df, clinical_df, disease_label, hc_label=1):
    """ Calculate the AUCs and accuracy of the normative model."""
    error_hc = reconstruction_error_df.loc[clinical_df['Diagnosis'] == hc_label]['Reconstruction error']
    error_patient = reconstruction_error_df.loc[clinical_df['Diagnosis'] == disease_label]['Reconstruction error']

    labels = list(np.zeros_like(error_hc)) + list(np.ones_like(error_patient))
    predictions = list(error_hc) + list(error_patient)

    fpr, tpr, thresholds = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)
    

    # =============================================================================
    # # only maximize sensitivity

    # # Find the threshold where TPR is above the tpr_threshold
    # # This will give us the threshold corresponding to the desired sensitivity
    # tpr_threshold = 0.9
    # sensitivity_indices = np.where(tpr >= tpr_threshold)
    # optimal_idx = sensitivity_indices[0][0] if sensitivity_indices[0].size else 0
    # optimal_threshold = thresholds[optimal_idx]

    # # Compute accuracy using the optimal threshold
    # predicted_labels = np.array(predictions > optimal_threshold, dtype=int)




# =============================================================================

    # Youden’s index optimization criterion (for both sensitivity and specificity)

    # Find the optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # Compute accuracy using the optimal threshold
    predicted_labels = [1 if e > optimal_threshold else 0 for e in predictions]





    accuracy = (np.array(predicted_labels) == np.array(labels)).mean()
    accuracy_in_hc = (np.array(predicted_labels)[np.array(labels) == 0] == 0).mean()
    accuracy_in_ad = (np.array(predicted_labels)[np.array(labels) == 1] == 1).mean()

    TP = np.sum((np.array(predicted_labels) == 1) & (np.array(labels) == 1))
    FN = np.sum((np.array(predicted_labels) == 0) & (np.array(labels) == 1))
    TN = np.sum((np.array(predicted_labels) == 0) & (np.array(labels) == 0))
    FP = np.sum((np.array(predicted_labels) == 1) & (np.array(labels) == 0))

    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)

    print('Recall (Sensitivity):', recall)
    print('Specificity:', specificity)

    return roc_auc, tpr, accuracy, accuracy_in_hc, accuracy_in_ad, recall, specificity


def main(dataset_name, comb_label):
    """Perform the group analysis."""
    # ----------------------------------------------------------------------------
    n_bootstrap = 10

    dataset_name = 'ADNI'
    disease_label = 0

    model_name = 'supervised_cvae_age_gender'

    participants_path = PROJECT_ROOT / 'data' / dataset_name / 'participants.tsv'
    freesurfer_path = PROJECT_ROOT / 'data' / dataset_name / 'freesurferData.csv'

    hc_label = 1

    # ----------------------------------------------------------------------------
    bootstrap_dir = PROJECT_ROOT / 'outputs' / 'bootstrap_analysis'
    model_dir = bootstrap_dir / model_name
    ids_path = PROJECT_ROOT / 'outputs' / (dataset_name + '_homogeneous_ids.csv')

    # ----------------------------------------------------------------------------
    clinical_df = load_dataset(participants_path, ids_path, freesurfer_path)
    clinical_df = clinical_df.set_index('participant_id')

    tpr_list = []
    auc_roc_list = []
    effect_size_list = []
    accuracy_list = []
    accuracy_in_ad_list = []
    accuracy_in_hc_list = []
    recall_list = []
    specificity_list = []

    for i_bootstrap in tqdm(range(n_bootstrap)):
        bootstrap_model_dir = model_dir / '{:03d}'.format(i_bootstrap)

        output_dataset_dir = bootstrap_model_dir / dataset_name
        output_dataset_dir.mkdir(exist_ok=True)

        analysis_dir = output_dataset_dir / '{:02d}_vs_{:02d}'.format(hc_label, disease_label)
        analysis_dir.mkdir(exist_ok=True)

        # ----------------------------------------------------------------------------
        normalized_df = pd.read_csv(output_dataset_dir / 'normalized.csv', index_col='participant_id')
        reconstruction_df = pd.read_csv(output_dataset_dir / 'reconstruction.csv', index_col='participant_id')
        reconstruction_error_df = pd.read_csv(output_dataset_dir / 'reconstruction_error.csv',
                                              index_col='participant_id')

        # ----------------------------------------------------------------------------
        # Compute effect size of the brain regions for the bootstrap iteration
        diff_df = np.abs(normalized_df - reconstruction_df)
        region_df = compute_brain_regions_deviations(diff_df, clinical_df, disease_label)
        effect_size_list.append(region_df['effect_size'].values)
        region_df.to_csv(analysis_dir / 'regions_analysis.csv', index=False)

        # ----------------------------------------------------------------------------
        # Compute AUC-ROC for the bootstrap iteration
        roc_auc, tpr, accuracy, accuracy_in_hc, accuracy_in_ad, recall, specificity = compute_classification_performance(reconstruction_error_df, clinical_df, disease_label)
        auc_roc_list.append(roc_auc)
        tpr_list.append(tpr)
        accuracy_list.append(accuracy)
        accuracy_in_hc_list.append(accuracy_in_hc)
        accuracy_in_ad_list.append(accuracy_in_ad)
        recall_list.append(recall)
        specificity_list.append(specificity)
        

    (bootstrap_dir / dataset_name).mkdir(exist_ok=True)
    comparison_dir = bootstrap_dir / dataset_name / ('{:02d}_vs_{:02d}'.format(hc_label, disease_label))
    comparison_dir.mkdir(exist_ok=True)



    # save auc roc and accuracy to txt file in result_focal folder
    with open(os.path.join(result_dir, 'result.txt'), 'a') as f:
        f.write('roc_auc: mean and std\n' + str(np.mean(auc_roc_list)) + ' ' + str(np.std(auc_roc_list)) + '\n')
        f.write('accuracy: mean and std\n' + str(np.mean(accuracy_list)) + ' ' + str(np.std(accuracy_list)) + '\n')
        f.write('sensitivity: mean and std\n' + str(np.mean(recall_list)) + ' ' + str(np.std(recall_list)) + '\n')
        f.write('specificity: mean and std\n' + str(np.mean(specificity_list)) + ' ' + str(np.std(specificity_list)) + '\n')
        f.write('\n\n\n')

    # ----------------------------------------------------------------------------
    # Save regions effect sizes
    effect_size_df = pd.DataFrame(columns=COLUMNS_NAME, data=np.array(effect_size_list))
    effect_size_df.to_csv(comparison_dir / 'effect_size.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--dataset_name',
                        dest='dataset_name',
                        help='Dataset name to perform group analysis.')
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
