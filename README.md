
---

# Focal Adversarial Autoencoder (FAAE) Code Documentation

## Introduction

The Focal Adversarial Autoencoder (FAAE) is a cutting-edge deep learning model tailored for normative modeling in the medical field. It specializes in establishing a norm based on a healthy control group and evaluates individual deviations to assess patient conditions. FAAE's training exclusively involves data from a healthy group and assesses deviations among both healthy and patient groups during testing.

This model encompasses six methods: AE, AAE, VAE, CVAE, ACVAE, and FAAE.

## How to Run the Code

### Conda Environment Setup

1. **Dependencies Installation**:
    - **Conda**: 
        ```bash
        conda env create -f environment.yml
        ```
    - **Pip**:
        ```bash
        pip install -r requirements.txt
        ```

    > Note: The `environment.yml` might contain redundant dependencies. Feel free to create your custom environment with major library versions.

### Data Preparation

1. **Data Format**: Feature vector size in ADNI is `1x100`.

2. **Data Storage**:
    - Training data: `PROJECT_ROOT/data/ADNI_TRAIN`
    - Test data: `PROJECT_ROOT/data/ADNI`

3. **File Details**:
    - Features and indices are in `freesurferData.csv`.
    - Labels and conditions are in `participants.tsv`.

4. **Index Files**:
    - Training data's indices: `PROJECT_ROOT/outputs/cleaned_ids.csv`
    - Test data's indices: `PROJECT_ROOT/outputs/ADNI_homogeneous_ids.csv`

### Training and Testing

1. **Scripts Execution**:
    - Grant execution permissions:
        ```bash
        chmod +x *
        ```
    - Run training, testing, and analysis scripts sequentially for each model:

    ```bash
    # Example for AE
    ./bootstrap_train_ae_supervised.py
    ./bootstrap_test_ae_supervised.py
    ./bootstrap_ae_group_analysis_1x1.py

    # Replace AE with AAE, VAE, CVAE, or ACVAE as needed
    ```

### FAAE Specific Scripts

- **Hyperparameter Fine-Tuning**:
    - `commands_list1.sh`: Tunes alpha, gamma, r values.
- **Baseline and FAAE Model Training, Testing, and Analysis**:
    - `commands_list2.sh`

## Code for ACVAE and FAAE

- The ACVAE and FAAE share the same codebase, differentiated by parameter settings.
- For ACVAE: Set parameters `-A 0 -G 1 -R 0`.
- For FAAE: Vary alpha (`-A`), with alpha > 0 and < 1, and gamma (`-G`) > 1.

---
