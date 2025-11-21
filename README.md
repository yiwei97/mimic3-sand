CSE6250 Final Project: MIMIC-III SAnD replication study
=========================
## Repository overview
This code repository contains the code used for replicating the SAnD model for the paper "Attend and Diagnose: Clinical Time Series Analysis using Attention Models", as part of CSE6250 Final Project.

The code repository is structured as follows:
1. `mimic3benchmark` contains the data cleaning code needed to generate train and test data sets for the MIMIC-III SAnD replication study. This generates the `data` folder, which will contain the relevant preprocessed data for in-hospital mortality, phenotyping, decompensation, and length of stay prediction tasks.  
2. `mimic3models` contains the relevant scripts for running the SAnD model for 4 experiments: In-hospital mortality, phenotyping, decompensation, and length of stay prediction.

## Set up virtual environment
1. Install Conda if not already installed
2. Create and activate onda environment with relevant packages and versioning
```bash
conda create -n cse6250_final_project python=3.10

# activate it
conda activate cse6250_final_project

# install packages from requirements.txt
pip install -r requirements.txt
```

## Data cleaning for MIMIC-III dataset

1. Download raw MIMIC-III dataset from and unzip the compressed CSV files into `mimic3-sand/raw_data` folder.
2. The data cleaning process first cleans up MIMIC-III raw CSV data to produce 1 directory for each `SUBJECT_ID`, writes ICU stay to `data/{SUBJECT_ID}/stays.csv`, diagnoses to `data/{SUBJECT_ID}/diagnoses.csv`, and events to `data/{SUBJECT_ID}/events.csv`.
```bash
       python -m mimic3benchmark.scripts.extract_subjects raw_data/ data/root/
```

3. Missing ICU stay ID and events that have missing information. For more information, refer to [`mimic3benchmark/scripts/more_on_validating_events.md`](mimic3benchmark/scripts/more_on_validating_events.md).
```bash
       python -m mimic3benchmark.scripts.validate_events data/root/
```

4. Per-subject data is broken into separate ICU stays episodes, producing a time series data which is saved here: ```{SUBJECT_ID}/episode{#}_timeseries.csv```.
```bash
       python -m mimic3benchmark.scripts.extract_episodes_from_subjects data/root/
```

5. Train-test split is done based on pre-determined patient IDs.
```bash
       python -m mimic3benchmark.scripts.split_train_and_test data/root/
```

6. Generate task-specific datasets for each of the benchmark tasks as follows.
```bash
       python -m mimic3benchmark.scripts.create_in_hospital_mortality data/root/ data/in-hospital-mortality/
       python -m mimic3benchmark.scripts.create_decompensation data/root/ data/decompensation/
       python -m mimic3benchmark.scripts.create_length_of_stay data/root/ data/length-of-stay/
       python -m mimic3benchmark.scripts.create_phenotyping data/root/ data/phenotyping/
```

The above steps will produce `data/{task}` for each benchmark task, each containing`train` and `test` subdirectories.
Within each, there are data on the ICU stays, as well as an indicator of all samples in that particular data split `listfile.csv`. This contains entries on`icu_stay, period_length, label(s)`.

## Run SAnD model

To replicate the paper's experiments, jupyter notebook files were created for each of the tasks, where model architecture is redefined in each due to minor differences across experiments. The script first reads in the training, validation and test sets using a specialised reader and normaliser, before the defined SAnD model is trained on the dataset. Metrics are logged using WanDB to track the performance of the model, including training and test accuracy and loss to monitor convergence over time. Finally, the test scores for each experiment is reported.  You can run the experiments using the SAnD model with each of the following notebook files by launching `jupyter lab` in root of the repository:
1. In hospital mortality prediction: `mimic3models/in_hospital_mortality/run_sand_mortality_e2e.ipynb`
2. Phenotyping prediction: `mimic3models/phenotyping/run_sand_phenotyping_e2e.ipynb`
3. Decompensation prediction: `mimic3models/decompensation/run_sand_decompsation_e2e.ipynb`
4. Length of stay prediction: `mimic3models/length_of_stay/run_sand_length_of_stay_e2e.ipynb`

## Run experiments for ablation study
We applied a combination of ablation and architectural improvements to the SAnD model, with a focus on improving how outputs from attention layers are summarised. In the original paper, dense interpolation was used to encode the output representation of the attention layer to avoid the curse of dimensionality. We observed that the need to trim sequence lengths in the original setup stemmed from the dense interpolation layer, which requires a fixed global sequence length for the dataset. This makes the model training process inefficient in terms of memory usage.

At that time, the modern approach of using CLS token was not yet popular, and common approaches involved either mean or max pooling or flattening the embeddings across all timesteps.  In our study, we modified the architecture by introducing a CLS token to represent the sequence and removed the dense interpolation layer. This change allows the model to handle variable-length sequences, padded only to the maximum length of each batch, eliminating the need to trim the dataset to a fixed length of 300. Additionally, we removed the self-attention mask, since all tokens must be able to interact with the CLS token during the attention computation. We also removed the causal mask, as strict causality is no longer required; the CLS token can learn which parts of the sequence are most important at each timestep.

To run the ablation experiments for each task, refer to the following jupyter notebook scripts:
1. In hospital mortality prediction: `mimic3models/in_hospital_mortality/run_sand_mortality_ablation_e2e.ipynb`
2. Phenotyping prediction: `mimic3models/phenotyping/run_sand_phenotyping_ablation_e2e.ipynb`
3. Decompensation prediction: `mimic3models/decompensation/run_sand_decompsation_ablation_e2e.ipynb`
4. Length of stay prediction: `mimic3models/length_of_stay/run_sand_length_of_stay_ablation_e2e.ipynb`

## Citation

The following code repositories and research publications were referred to when producing the repository:
```
@article{Harutyunyan2019,
  author={Harutyunyan, Hrayr and Khachatrian, Hrant and Kale, David C. and Ver Steeg, Greg and Galstyan, Aram},
  title={Multitask learning and benchmarking with clinical time series data},
  journal={Scientific Data},
  year={2019},
  volume={6},
  number={1},
  pages={96},
  issn={2052-4463},
  doi={10.1038/s41597-019-0103-9},
  url={https://doi.org/10.1038/s41597-019-0103-9}
}
```

```
@software{SAnD_github_2019,
  author = {khirotaka},
  title = {AAAI 2018 Attend and Diagnose: Clinical Time Series Analysis Using Attention Models (GitHub Repository)},
  url = {https://github.com/khirotaka/SAnD},
  version = {v1.0},
  year = {2019},
}
```

```
@article{song2017attend,
  title        = {Attend and Diagnose: Clinical Time Series Analysis using Attention Models},
  author       = {Huan Song and Deepta Rajan and Jayaraman J. Thiagarajan and Andreas Spanias},
  journal      = {arXiv preprint},
  volume       = {arXiv:1711.03905},
  year         = {2017},
  url          = {https://arxiv.org/abs/1711.03905}
}
```