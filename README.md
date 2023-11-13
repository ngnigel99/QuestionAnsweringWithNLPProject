# Question Answering Models

This repository contains various models for the task of question answering.

## Directory Structure

The repository is organized as follows:
```
.
├── README.md
├── json_results
│   ├── predictions_distilbert-base-uncased-distilled-squad.json
│   ├── predictions_minimlv.json
│   ├── predictions_minimlv_ft.json
│   ├── roberta_ft_10_epochs.json
│   └── roberta_ft_3_epochs.json
├── notebooks
│   ├── dev-v1.1.json
│   ├── evaluate-v2.0.py
│   ├── minilmv2.ipynb
│   ├── minilmv2_ft.ipynb
│   ├── roberta_qa_10_epochs.ipynb
│   ├── roberta_qa_3_epochs.ipynb
│   ├── train-v1.1.json
│   ├── train_minilmv2-L6-H384.ipynb
│   ├── train_roberta_10_epochs.ipynb
│   ├── train_roberta_3_epochs.ipynb
│   └── using_pyspark_for_qa.ipynb
└── requirements.txt
```
## Files

- `json_results`: This directory contains the prediction results of the models.
- `notebooks`: This directory contains Jupyter notebooks with the code for training the models and using them for question answering. The fine-tuning notebooks start with train, and the inference notebooks start with roberta_qa, minilmv2, or using_pyspark_for_qa.

- `requirements.txt`: This file lists the Python dependencies required for running the code.

Note, the models are not included in this repository. They can be downloaded from the Hugging Face model hub, for both the pre-trained and fine-tuned models.

In fact, for the notebooks we have provided, the models are accessed directly from the Hugging Face model hub. The models and tokenizers are downloaded and cached locally when they are first used.

## Usage

First, create a new conda environment with Python 3.8:

```bash
conda create -n robertaqa python=3.8
conda activate robertaqa
```

Then, install the necessary dependencies:

```bash
pip install -r requirements.txt
```

After that, you can run the Jupyter notebooks in the `notebooks` directory.

## Results

The results of the models are stored in the `json_results` directory. Each .json file contains the predictions of a model.

## Credits

The models used in this project are based on the transformers provided by Hugging Face. The models used can be found here:

- [ngnigel99](https://huggingface.co/ngnigel99)
- [sguskin/minilmv2-L6-H384-squad1.1](https://huggingface.co/sguskin/minilmv2-L6-H384-squad1.1)

