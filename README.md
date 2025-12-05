# XLNet Plugin for BioLM Utils

This repository provides the **XLNet plugin** for `biolm_utils`, the core framework for tokenizing, pre-training, and fine-tuning language models on biological sequences.

**Note**: This is a plugin for the `biolm_utils` framework, not the main framework itself. Start by cloning `biolm_utils` and running its installer.

The XLNet plugin implements:
- **Pre-training**: Permutation Language Modeling (PLM) on unlabeled RNA/protein sequences
- **Fine-tuning**: Regression and classification on labeled sequences
- **Interpretation**: Leave-one-out (LOO) scores to analyze feature importance

## Quick Start: Install Framework + XLNet Plugin

The XLNet plugin is built into the `biolm_utils` framework and loads automatically.

**Start here**:

```bash
git clone https://github.com/dieterich-lab/biolm_utils.git
cd biolm_utils
poetry install
```

Then run an experiment:

```bash
poetry run biolm pre-train --config-path ./biolm/plugins/xlnet/exampleconfigs/tokenize_pre-train_fine-tune.yaml
```

For help: `poetry run biolm --help`

## Manual Plugin Installation (Advanced Users)

If you want to develop the XLNet plugin separately:

```bash
# 1. Clone and install the framework first
git clone https://github.com/dieterich-lab/biolm_utils.git
cd biolm_utils
poetry install

# 2. Clone the plugin repository
git clone https://github.com/dieterich-lab/rna_protein_xlnet.git
cd rna_protein_xlnet

# 3. Install the plugin in editable/development mode
poetry install

# The plugin registers itself via entry points and is automatically discovered
```

## Plugin Configuration

The XLNet plugin is configured via the `xlnet_plugin/config.py` module, which defines:

- **Pretraining Model**: `RNA_XLNetLMHeadModel` - XLNet with language model head for PLM
- **Finetuning Model**: `RNA_XLNetForSequenceClassification` - XLNet for classification/regression
- **Dataset**: `RNALanguageDataset` - Loads RNA sequences and labels
- **Pretraining Required**: Yes (pre-training on unlabeled data before fine-tuning)
- **Data Collator (Pretraining)**: `DataCollatorForPermutationLanguageModeling` - Handles PLM masking

## Running Experiments

Use Hydra-based config files to run experiments:

```bash
# Pre-train on unlabeled data
poetry run biolm pre-train \
  --config-path ./biolm/plugins/xlnet/exampleconfigs \
  data_source.filepath=/path/to/unlabeled/data.txt

# Fine-tune on labeled data
poetry run biolm fine-tune \
  --config-path ./biolm/plugins/xlnet/exampleconfigs \
  data_source.filepath=/path/to/labeled/data.txt \
  task=regression
```

See `./biolm/plugins/xlnet/exampleconfigs/` for example configurations.

## File Structure

```
biolm/plugins/xlnet/
├── xlnet_dataset.py        # RNALanguageDataset implementation
├── xlnet_models.py         # XLNet model implementations
├── exampleconfigs/         # Example Hydra configurations
│   ├── tokenize_pre-train_fine-tune.yaml
│   └── predict_interpret.yaml
└── tests/                  # Plugin tests
```

## Important Notes

- **Sequence Length**: The XLNet PLM collator requires even-length sequences to create a leakage-free permutation mask. Ensure input sequences are padded/truncated to even lengths.
- **Pretraining**: This plugin requires pre-training before fine-tuning. Use `mode=pre-train` to pre-train on unlabeled data first.
- **Learning Rate**: Default learning rate is 1e-5 (suitable for fine-tuning after pre-training).
