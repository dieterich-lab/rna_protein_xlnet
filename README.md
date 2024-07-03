# `rna_protein_xlnet`: A plugin to run bioinformatical Language Models.

This projects implements pre-training and fine-tuning of the [XLNet](https://arxiv.org/abs/1906.08237) language model for regressing half lives of RNA and protein sequences. In addition, it supports the extraction of leave-one-out (LOO) scores for fine-tuned models to analyse importance scores of individual inputs.

In detail, the following steps are implemented:

- Tokenization of RNA/Protein sequences via Byte Pair encoding.
- Pre-training of the XLNET model via Permutation Language Modelling.
- Fine-tune models for regression.
- Calculation of leave-one-out scores for you fine-tuned model.

## Installation

First clone the repo and cd into it. Then, we recommend to create a dedicated environment ([python venv](https://docs.python.org/3/library/venv.html)) for the project. Now, you install the project via the [Pipfile](./Pipfile) file which in turn will install the [biolm_utils](https://github.com/dieterich-lab/biolm_utils) library. Summarising, excute the following steps:

```bash
git clone https://github.com/dieterich-lab/rna_protein_xlnet.git
cd rna_protein_xlnet
python3 -m venv biolm_xlnet 
. biolm/biolm_xlnet/activate
pip install pipenv
pipenv install
```

## File structure

```bash
├── exampleconfigs # exampleconfigs to work with
├── Pipfile # installation file
├── README.md
├── rna_xlnet_dataset.py # Implementation of the `RNABaseDataset`, espcially implementing the `__getitem__()` method.
├── rna_xlnet_models.py # Implementation of the models, espcially implementing the `getconfig()` method.
├── xlnet.py # Main script importing the `run()` function from `biolm_utils` and declaration of the model/data/training configuration.
```

## Usage

The main script is [xlnet.py](./xlnet.py) which imports the `run()` function from the [biolm._utils](https://github.com/dieterich-lab/biolm_utils) library and provides the a custom `Config` object suitable for running the XLNet model. The script can be run via

```bash
python xlnet.py [tokenize | pre-train | fine-tune | predict | interepret]
```

To get a verbose exlplanation on all the possible parameters you can run the following:

```bash
python xlnet.py -h 
```

For general usage and information about the configuration parameters we refer user to the [README](https://github.com/dieterich-lab/biolm_utils/blob/main/README.md) of the `biolm_utils` framework.

## Example config files

We offer two types of config files. The first one is for the pipeline of **tokenization**, **pre-training** (language models only), **fine-tuning**, **testing** (testing is also done during fine-tuning, but can be also again separately invoked) and extracting loo scores. The other one is for **inference** (getting predictions on new files) and **interpret** modes. The latter one are noticeably smaller as all the training cofigurations fall away.

```bash
exampleconfigs
├── inference_interpret.yaml
├── tokenize_pe-train_fine_tune_test_interpret.yaml
```