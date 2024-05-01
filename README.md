# DIST: Dual Input Stream Transformer
Official repo for [Dual input stream transformer for eye-tracking line assignment](https://arxiv.org/abs/2311.06095)

## Download Data
To get the data that was used to develop and train the models please see: [OSF Link](https://osf.io/zt9gn)

## Run via Huggingface Space
Please see our Huggingface space for an easy way of applying the model to your .asc files or the preprocessed files linked above:
[Space](https://huggingface.co/spaces/bugroup/Eye_Tracking_Drift_Correction)

## Run in Notebook
To correct fixation data without the space please see the jupyter notebook `run_in_notebook.ipynb`

It is recommended use a conda environment which can be set up via:
Install python through anaconda: [Download Link](https://docs.conda.io/projects/miniconda/en/latest/index.html)

```sh
conda create -n pt2 python=3.11 -y
conda activate pt2
pip3 install torch torchvision torchaudio
pip install ipykernel ipywidgets jupyterlab
pip install -r requirements.txt
jupyter lab
```

## Run training
To train a DIST model simply adjust the contents of `experiments\main.yaml` to your system and data location. For example if you have downloaded and unzipped the data from the [OSF Link](https://osf.io/zt9gn) (keeping the folder structure) to `F:/pydata/ET2/` then you would not need to change anything to run an example training.
`dataset_folder_idx_training` determines which datasets are used to train on and `use_reduced_set` determines if it's a test run with only a few samples or a full run.

Then run:

```sh
conda create -n pt2 python=3.11 -y
conda activate pt2
pip3 install torch torchvision torchaudio
pip install -r requirements.txt
python train.py
```

The results will be saved under `results/`