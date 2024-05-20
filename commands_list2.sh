#!/bin/bash

chmod +x *

# ./bootstrap_create_ids.py

# train, test and analysis for AE
./bootstrap_train_ae_supervised.py
./bootstrap_test_ae_supervised.py
./bootstrap_ae_group_analysis_1x1.py

# train, test and analysis for AAE
./bootstrap_train_aae_supervised_new.py
./bootstrap_test_aae_supervised_new.py
./bootstrap_aae_new_group_analysis_1x1.py

# train, test and analysis for VAE
./bootstrap_train_vae_supervised.py
./bootstrap_test_vae_supervised.py
./bootstrap_vae_group_analysis_1x1.py

# train, test and analysis for CVAE
./bootstrap_train_cvae_supervised.py
./bootstrap_test_cvae_supervised.py
./bootstrap_cvae_group_analysis_1x1.py

# train, test and analysis for ACVAE
./bootstrap_train_cvae_supervised_age_gender.py -A 0 -G 1 -R 0
./bootstrap_test_cvae_supervised_age_gender.py
./bootstrap_cvae_group_analysis_1x1_age_gender.py


# train, test and analysis for ACVAE
./bootstrap_train_cvae_supervised_age_gender.py -A 0.2 -G 15 -R 0
./bootstrap_test_cvae_supervised_age_gender.py
./bootstrap_cvae_group_analysis_1x1_age_gender.py