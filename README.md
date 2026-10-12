# SEER PCa Analysis

Analysis code for “Survival Disparity Among Patients with De Novo Metastatic
Prostate Cancer in the Androgen Receptor Pathway Inhibitor Era.”

## Setup and inputs

We use Python 3.12.12 and the packages in `requirements.txt`.

```bash
conda create -n seer-pca python=3.12.12
conda activate seer-pca
python -m pip install -r requirements.txt
```

## Run

Assume CSVs in `overall-survival/data` and
`pca-survival/data`, run the complete analysis with

```bash
python scripts/run_all.py
```