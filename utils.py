"""Helper functions and constants."""
from pathlib import Path
import warnings

import pandas as pd
import numpy as np

PROJECT_ROOT = Path.cwd()


def cliff_delta(X, Y):
    """Calculate the effect size using the Cliff's delta."""
    lx = len(X)
    ly = len(Y)
    mat = np.zeros((lx, ly))
    for i in range(0, lx):
        for j in range(0, ly):
            if X[i] > Y[j]:
                mat[i, j] = 1
            elif Y[j] > X[i]:
                mat[i, j] = -1

    return (np.sum(mat)) / (lx * ly)


def load_dataset(demographic_path, ids_path, freesurfer_path):
    """Load dataset."""
    demographic_data = load_demographic_data(demographic_path, ids_path)

    freesurfer_df = pd.read_csv(freesurfer_path)

    dataset_df = pd.merge(freesurfer_df, demographic_data, on='Image_ID')

    return dataset_df


def load_demographic_data(demographic_path, ids_path):
    """Load dataset using selected ids."""

    demographic_df = pd.read_csv(demographic_path)
    demographic_df = demographic_df.dropna()

    ids_df = pd.read_csv(ids_path, usecols=['Image_ID'])

    if 'Run_ID' in demographic_df.columns:
        demographic_df['uid'] = demographic_df['participant_id'] + '_' + demographic_df['Session_ID'] + '_run-' + \
                                demographic_df['Run_ID'].apply(str)

        ids_df['uid'] = ids_df['Image_ID'].str.split('_').str[0] + '_' + ids_df['Image_ID'].str.split('_').str[1]+ '_' + ids_df['Image_ID'].str.split('_').str[2]

        dataset_df = pd.merge(ids_df, demographic_df, on='uid')
        dataset_df = dataset_df.drop(columns=['uid'])

    elif 'Session_ID' in demographic_df.columns:
        demographic_df['uid'] = demographic_df['participant_id'] + '_' + demographic_df['Session_ID']

        ids_df['uid'] = ids_df['Image_ID'].str.split('_').str[0] + '_' + ids_df['Image_ID'].str.split('_').str[1]

        dataset_df = pd.merge(ids_df, demographic_df, on='uid')
        dataset_df = dataset_df.drop(columns=['uid'])

    else:
        ids_df['participant_id'] = ids_df['Image_ID']
        dataset_df = pd.merge(ids_df, demographic_df, on='Image_ID')

    return dataset_df


#COLUMNS_NAME = [(lambda x: 'C'+ str(x))(y) for y in list(range(5,105))]


COLUMNS_NAME = [ 'LH_Vis_1',
                'LH_Vis_2',
                'LH_Vis_3',
                'LH_Vis_4',
                'LH_Vis_5',
                'LH_Vis_6',
                'LH_Vis_7',
                'LH_Vis_8',
                'LH_Vis_9',
                'LH_SomMot_1',
                'LH_SomMot_2',
                'LH_SomMot_3',
                'LH_SomMot_4',
                'LH_SomMot_5',
                'LH_SomMot_6',
                'LH_DorsAttn_Post_1',
                'LH_DorsAttn_Post_2',
                'LH_DorsAttn_Post_3',
                'LH_DorsAttn_Post_4',
                'LH_DorsAttn_Post_5',
                'LH_DorsAttn_Post_6',
                'LH_DorsAttn_FEF_1',
                'LH_DorsAttn_FEF_2',
                'LH_SalVentAttn_ParOper_1',
                'LH_SalVentAttn_FrOper_1',
                'LH_SalVentAttn_FrOper_2',
                'LH_SalVentAttn_PFCl_1',
                'LH_SalVentAttn_Med_1',
                'LH_SalVentAttn_Med_2',
                'LH_SalVentAttn_Med_3',
                'LH_Limbic_OFC_1',
                'LH_Limbic_TempPole_1',
                'LH_Limbic_TempPole_2',
                'LH_Cont_Par_1',
                'LH_Cont_PFCl_1',
                'LH_Cont_pCun_1',
                'LH_Cont_Cing_1',
                'LH_Default_Temp_1',
                'LH_Default_Temp_2',
                'LH_Default_Temp_3',
                'LH_Default_Temp_4',
                'LH_Default_PFC_1',
                'LH_Default_PFC_2',
                'LH_Default_PFC_3',
                'LH_Default_PFC_4',
                'LH_Default_PFC_5',
                'LH_Default_PFC_6',
                'LH_Default_PFC_7',
                'LH_Default_PCC_1',
                'LH_Default_PCC_2',
                'RH_Vis_1',
                'RH_Vis_2',
                'RH_Vis_3',
                'RH_Vis_4',
                'RH_Vis_5',
                'RH_Vis_6',
                'RH_Vis_7',
                'RH_Vis_8',
                'RH_SomMot_1',
                'RH_SomMot_2',
                'RH_SomMot_3',
                'RH_SomMot_4',
                'RH_SomMot_5',
                'RH_SomMot_6',
                'RH_SomMot_7',
                'RH_SomMot_8',
                'RH_DorsAttn_Post_1',
                'RH_DorsAttn_Post_2',
                'RH_DorsAttn_Post_3',
                'RH_DorsAttn_Post_4',
                'RH_DorsAttn_Post_5',
                'RH_DorsAttn_FEF_1',
                'RH_DorsAttn_FEF_2',
                'RH_SalVentAttn_TempOccPar_1',
                'RH_SalVentAttn_TempOccPar_2',
                'RH_SalVentAttn_FrOper_1',
                'RH_SalVentAttn_Med_1',
                'RH_SalVentAttn_Med_2',
                'RH_Limbic_OFC_1',
                'RH_Limbic_TempPole_1',
                'RH_Cont_Par_1',
                'RH_Cont_Par_2',
                'RH_Cont_PFCl_1',
                'RH_Cont_PFCl_2',
                'RH_Cont_PFCl_3',
                'RH_Cont_PFCl_4',
                'RH_Cont_PFCmp_1',
                'RH_Cont_PFCmp_2',
                'RH_Cont_PFCmp_3',
                'RH_Default_Par_1',
                'RH_Default_Temp_1',
                'RH_Default_Temp_2',
                'RH_Default_Temp_3',
                'RH_Default_PFCv_1',
                'RH_Default_PFCv_2',
                'RH_Default_PFCm_1',
                'RH_Default_PFCm_2',
                'RH_Default_PFCm_3',
                'RH_Default_PCC_1',
                'RH_Default_PCC_2' ]

# COLUMNS_NAME = ['Left Lateral Ventricle',
#                 'Left Inf Lat Vent',
#                 'Left Cerebellum White Matter',
#                 'Left Cerebellum Cortex',
#                 'Left Thalamus Proper',
#                 'Left Caudate',
#                 'Left Putamen',
#                 'Left Pallidum',
#                 'rd Ventricle',
#                 'th Ventricle',
#                 'Brain Stem',
#                 'Left Hippocampus',
#                 'Left Amygdala',
#                 'CSF',
#                 'Left Accumbens area',
#                 'Left VentralDC',
#                 'Right Lateral Ventricle',
#                 'Right Inf Lat Vent',
#                 'Right Cerebellum White Matter',
#                 'Right Cerebellum Cortex',
#                 'Right Thalamus Proper',
#                 'Right Caudate',
#                 'Right Putamen',
#                 'Right Pallidum',
#                 'Right Hippocampus',
#                 'Right Amygdala',
#                 'Right Accumbens area',
#                 'Right VentralDC',
#                 'CC Posterior',
#                 'CC Mid Posterior',
#                 'CC Central',
#                 'CC Mid Anterior',
#                 'CC Anterior',
#                 'lh bankssts volume',
#                 'lh caudalanteriorcingulate volume',
#                 'lh caudalmiddlefrontal volume',
#                 'lh cuneus volume',
#                 'lh entorhinal volume',
#                 'lh fusiform volume',
#                 'lh inferiorparietal volume',
#                 'lh inferiortemporal volume',
#                 'lh isthmuscingulate volume',
#                 'lh lateraloccipital volume',
#                 'lh lateralorbitofrontal volume',
#                 'lh lingual volume',
#                 'lh medialorbitofrontal volume',
#                 'lh middletemporal volume',
#                 'lh parahippocampal volume',
#                 'lh paracentral volume',
#                 'lh parsopercularis volume',
#                 'lh parsorbitalis volume',
#                 'lh parstriangularis volume',
#                 'lh pericalcarine volume',
#                 'lh postcentral volume',
#                 'lh posteriorcingulate volume',
#                 'lh precentral volume',
#                 'lh precuneus volume',
#                 'lh rostralanteriorcingulate volume',
#                 'lh rostralmiddlefrontal volume',
#                 'lh superiorfrontal volume',
#                 'lh superiorparietal volume',
#                 'lh superiortemporal volume',
#                 'lh supramarginal volume',
#                 'lh frontalpole volume',
#                 'lh temporalpole volume',
#                 'lh transversetemporal volume',
#                 'lh insula volume',
#                 'rh bankssts volume',
#                 'rh caudalanteriorcingulate volume',
#                 'rh caudalmiddlefrontal volume',
#                 'rh cuneus volume',
#                 'rh entorhinal volume',
#                 'rh fusiform volume',
#                 'rh inferiorparietal volume',
#                 'rh inferiortemporal volume',
#                 'rh isthmuscingulate volume',
#                 'rh lateraloccipital volume',
#                 'rh lateralorbitofrontal volume',
#                 'rh lingual volume',
#                 'rh medialorbitofrontal volume',
#                 'rh middletemporal volume',
#                 'rh parahippocampal volume',
#                 'rh paracentral volume',
#                 'rh parsopercularis volume',
#                 'rh parsorbitalis volume',
#                 'rh parstriangularis volume',
#                 'rh pericalcarine volume',
#                 'rh postcentral volume',
#                 'rh posteriorcingulate volume',
#                 'rh precentral volume',
#                 'rh precuneus volume',
#                 'rh rostralanteriorcingulate volume',
#                 'rh rostralmiddlefrontal volume',
#                 'rh superiorfrontal volume',
#                 'rh superiorparietal volume',
#                 'rh superiortemporal volume',
#                 'rh supramarginal volume',
#                 'rh frontalpole volume',
#                 'rh temporalpole volume',
#                 'rh transversetemporal volume',
#                 'rh insula volume']


# COLUMNS_NAME = ['Left-Lateral-Ventricle',
#                 'Left-Inf-Lat-Vent',
#                 'Left-Cerebellum-White-Matter',
#                 'Left-Cerebellum-Cortex',
#                 'Left-Thalamus-Proper',
#                 'Left-Caudate',
#                 'Left-Putamen',
#                 'Left-Pallidum',
#                 '3rd-Ventricle',
#                 '4th-Ventricle',
#                 'Brain-Stem',
#                 'Left-Hippocampus',
#                 'Left-Amygdala',
#                 'CSF',
#                 'Left-Accumbens-area',
#                 'Left-VentralDC',
#                 'Right-Lateral-Ventricle',
#                 'Right-Inf-Lat-Vent',
#                 'Right-Cerebellum-White-Matter',
#                 'Right-Cerebellum-Cortex',
#                 'Right-Thalamus-Proper',
#                 'Right-Caudate',
#                 'Right-Putamen',
#                 'Right-Pallidum',
#                 'Right-Hippocampus',
#                 'Right-Amygdala',
#                 'Right-Accumbens-area',
#                 'Right-VentralDC',
#                 'CC_Posterior',
#                 'CC_Mid_Posterior',
#                 'CC_Central',
#                 'CC_Mid_Anterior',
#                 'CC_Anterior',
#                 'lh_bankssts_volume',
#                 'lh_caudalanteriorcingulate_volume',
#                 'lh_caudalmiddlefrontal_volume',
#                 'lh_cuneus_volume',
#                 'lh_entorhinal_volume',
#                 'lh_fusiform_volume',
#                 'lh_inferiorparietal_volume',
#                 'lh_inferiortemporal_volume',
#                 'lh_isthmuscingulate_volume',
#                 'lh_lateraloccipital_volume',
#                 'lh_lateralorbitofrontal_volume',
#                 'lh_lingual_volume',
#                 'lh_medialorbitofrontal_volume',
#                 'lh_middletemporal_volume',
#                 'lh_parahippocampal_volume',
#                 'lh_paracentral_volume',
#                 'lh_parsopercularis_volume',
#                 'lh_parsorbitalis_volume',
#                 'lh_parstriangularis_volume',
#                 'lh_pericalcarine_volume',
#                 'lh_postcentral_volume',
#                 'lh_posteriorcingulate_volume',
#                 'lh_precentral_volume',
#                 'lh_precuneus_volume',
#                 'lh_rostralanteriorcingulate_volume',
#                 'lh_rostralmiddlefrontal_volume',
#                 'lh_superiorfrontal_volume',
#                 'lh_superiorparietal_volume',
#                 'lh_superiortemporal_volume',
#                 'lh_supramarginal_volume',
#                 'lh_frontalpole_volume',
#                 'lh_temporalpole_volume',
#                 'lh_transversetemporal_volume',
#                 'lh_insula_volume',
#                 'rh_bankssts_volume',
#                 'rh_caudalanteriorcingulate_volume',
#                 'rh_caudalmiddlefrontal_volume',
#                 'rh_cuneus_volume',
#                 'rh_entorhinal_volume',
#                 'rh_fusiform_volume',
#                 'rh_inferiorparietal_volume',
#                 'rh_inferiortemporal_volume',
#                 'rh_isthmuscingulate_volume',
#                 'rh_lateraloccipital_volume',
#                 'rh_lateralorbitofrontal_volume',
#                 'rh_lingual_volume',
#                 'rh_medialorbitofrontal_volume',
#                 'rh_middletemporal_volume',
#                 'rh_parahippocampal_volume',
#                 'rh_paracentral_volume',
#                 'rh_parsopercularis_volume',
#                 'rh_parsorbitalis_volume',
#                 'rh_parstriangularis_volume',
#                 'rh_pericalcarine_volume',
#                 'rh_postcentral_volume',
#                 'rh_posteriorcingulate_volume',
#                 'rh_precentral_volume',
#                 'rh_precuneus_volume',
#                 'rh_rostralanteriorcingulate_volume',
#                 'rh_rostralmiddlefrontal_volume',
#                 'rh_superiorfrontal_volume',
#                 'rh_superiorparietal_volume',
#                 'rh_superiortemporal_volume',
#                 'rh_supramarginal_volume',
#                 'rh_frontalpole_volume',
#                 'rh_temporalpole_volume',
#                 'rh_transversetemporal_volume',
#                 'rh_insula_volume']