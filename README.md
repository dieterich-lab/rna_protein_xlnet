# XLNet Plugin for BioLM 2.0

**Transformer-based language models for RNA and protein sequences**

This is the **XLNet plugin** for [BioLM 2.0](https://github.com/dieterich-lab/biolm_utils), implementing permutation language modeling (PLM) for biological sequence analysis.

## 🔬 Overview

The XLNet plugin brings powerful transformer-based pre-training and fine-tuning capabilities to BioLM:

**Key Features:**
- ✨ **Pre-training**: Permutation Language Modeling (PLM) on unlabeled sequences
- 🎯 **Fine-tuning**: Classification and regression on labeled data
- 🔍 **Interpretation**: Leave-one-out (LOO) analysis for feature importance
- 🧬 **Versatile**: Works with RNA and protein sequences
- ⚡ **Scalable**: Supports large-scale models and datasets

**Architecture:** XLNet transformer with bidirectional context modeling through permutation-based training.

---

## 🚀 Installation

### Prerequisites

### 1. Install BioLM Framework First

```bash
# Clone and install framework
git clone https://github.com/dieterich-lab/biolm_utils.git
cd biolm_utils
git checkout biolm-2.0
./install.sh
```

### 2. Install XLNet Plugin

```bash
biolm install-plugin https://github.com/dieterich-lab/rna_protein_xlnet.git
```

The plugin automatically:
- Finds the framework
- Installs itself into the framework's environment
- Registers via entry points
- Verifies successful registration

### 3. Verify Installation

Check registered plugins:

```bash
biolm list-plugins
```

---

## 📊 Data Format

XLNet works with **k-mer tokenized sequences** (e.g., 3-mer, 6-mer).

### Pre-training Data (Unlabeled)

Plain text file with one sequence per line:

```
AUGCUA|GCUAUG|CUAUGC|...
GCUAUG|CUAUGC|AUGCUA|...
```

### Fine-tuning Data (Labeled)

Tab-separated file with columns: `sequence_id`, `label`, `sequence`

```
seq_001    0.75    AUGCUA|GCUAUG|CUAUGC|...
seq_002    1.23    GCUAUG|CUAUGC|AUGCUA|...
seq_003    0.42    CUAUGC|AUGCUA|GCUAUG|...
```

**Notes:**
- K-mers separated by `|` (pipe character)
- Labels can be numeric (regression) or categorical (classification)
- Sequences must be **even length** for PLM masking

---

## ⚙️ Configuration

Create a YAML config file (e.g., `my_experiment.yaml`):

```yaml
# Experiment metadata
defaults:
  - _self_
  - mode: fine-tune

experiment_name: xlnet_rna_stability
plugin_name: xlnet

# Data source
data_source:
  filepath: /path/to/labeled_data.txt
  column_ids: [1, 2, 3]  # [id_col, label_col, seq_col]

# Model configuration
model:
  num_labels: 1           # 1 for regression, >1 for classification
  hidden_size: 768
  num_attention_heads: 12
  num_hidden_layers: 6
  max_position_embeddings: 512

# Training
training:
  num_train_epochs: 10
  learning_rate: 1e-5
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 4

# Task type
task: regression  # or 'classification'
```

---

## 🎯 Quick Start

### 1. Tokenize Sequences

First, tokenize your raw sequences into k-mers:

```bash
poetry run biolm tokenize \
  --config-path ./configs \
  --config-name tokenize \
  plugin_name=xlnet \
  data_source.filepath=/path/to/raw_sequences.txt \
  tokenizer.kmer_size=3
```

**Output:** Tokenized sequences ready for pre-training/fine-tuning

### 2. Pre-train (Optional but Recommended)

Pre-train on unlabeled data using Permutation Language Modeling:

```bash
poetry run biolm pre-train \
  --config-path ./configs \
  --config-name pre-train \
  plugin_name=xlnet \
  data_source.filepath=/path/to/unlabeled_kmers.txt \
  training.num_train_epochs=50 \
  training.learning_rate=1e-4
```

**Output:** Pre-trained XLNet model checkpoint

### 3. Fine-tune

Fine-tune on labeled data:

```bash
poetry run biolm fine-tune \
  --config-path ./configs \
  --config-name fine-tune \
  plugin_name=xlnet \
  data_source.filepath=/path/to/labeled_data.txt \
  task=regression \
  model.pretrained_model_path=/path/to/pretrained/checkpoint
```

**Output:** Fine-tuned model for your specific task

### 4. Predict

Make predictions on new sequences:

```bash
poetry run biolm predict \
  --config-path ./configs \
  --config-name predict \
  plugin_name=xlnet \
  data_source.filepath=/path/to/test_data.txt \
  model.pretrained_model_path=/path/to/finetuned/checkpoint
```

**Output:** Predictions CSV file

### 5. Interpret (Leave-One-Out)

Analyze feature importance using LOO:

```bash
poetry run biolm interpret \
  --config-path ./configs \
  --config-name interpret \
  plugin_name=xlnet \
  data_source.filepath=/path/to/test_data.txt \
  model.pretrained_model_path=/path/to/finetuned/checkpoint
```

**Output:** LOO scores for each k-mer in each sequence

---

## 🏗️ Model Architecture

**XLNet Transformer:**

```
Input Sequence (k-mers)
    ↓
Embedding Layer (k-mer → vector)
    ↓
Positional Encodings (relative positions)
    ↓
┌─────────────────────────────────┐
│ XLNet Transformer Layers        │
│                                  │
│  • Multi-head Self-Attention    │
│    (Permutation-based masking)  │
│  • Feed-Forward Networks        │
│  • Layer Normalization          │
│  • Residual Connections         │
└─────────────────────────────────┘
    ↓
Pooling (e.g., last token, mean)
    ↓
Classification/Regression Head
    ↓
Output (prediction)
```

**Key Differences from BERT:**
- **Permutation Language Modeling** instead of masked LM
- **Relative positional encodings** instead of absolute
- **Two-stream self-attention** for content and query
- **No [MASK] tokens** - learns from permutations

---

## 🧪 Testing

Run plugin tests:

```bash
# All tests
poetry run pytest tests/ -v

# Specific test
poetry run pytest tests/test_xlnet_full_pipeline.py -v

# With coverage
poetry run pytest tests/ --cov=xlnet_plugin --cov-report=html
```

**Test files:**
- `test_xlnet_full_pipeline.py` - End-to-end workflow tests
- `test_xlnet_plugin_config.py` - Configuration validation tests

---

## 📁 Project Structure

```
rna_protein_xlnet/
├── xlnet_plugin/
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Plugin configuration (entry point)
│   ├── dataset.py           # RNALanguageDataset
│   └── models.py            # XLNet model implementations
├── tests/
│   ├── test_xlnet_full_pipeline.py
│   └── test_xlnet_plugin_config.py
├── internal_configs/        # Example configurations
├── internal_experiments/    # Experiment results
├── _resources/              # Sample data
├── pyproject.toml           # Dependencies and entry points
└── README.md                # This file
```

---

## 🛠️ Development

For detailed development information, see [DEVELOPMENT.md](docs/DEVELOPMENT.md).

**Entry Point Registration:**

The plugin registers itself via `pyproject.toml`:

```toml
[tool.poetry.plugins."biolm.plugins"]
xlnet = "xlnet_plugin.config:get_config"
```

BioLM automatically discovers and loads the plugin through this entry point.

**Plugin Configuration:**

```python
# xlnet_plugin/config.py
from biolm.plugin_config import PluginConfig
from xlnet_plugin.models import RNA_XLNetLMHeadModel, RNA_XLNetForSequenceClassification
from xlnet_plugin.dataset import RNALanguageDataset

def get_config() -> PluginConfig:
    return PluginConfig(
        model_cls_for_pretraining=RNA_XLNetLMHeadModel,
        model_cls_for_finetuning=RNA_XLNetForSequenceClassification,
        dataset_cls=RNALanguageDataset,
        # ... other configuration ...
        pretraining_required=True,  # Pre-training recommended
    )
```

---

## 📊 XLNet vs Saluki

| Feature | XLNet | Saluki |
|---------|-------|--------|
| **Architecture** | Transformer (attention-based) | CNN (convolution-based) |
| **Pre-training** | ✅ Yes (PLM) | ❌ No |
| **Input Format** | K-mer tokenized | Comma-separated nucleotides |
| **Sequence Length** | Up to 512 k-mers (configurable) | Up to 12,000 nt |
| **Context** | Global (self-attention) | Local (convolution windows) |
| **Speed** | Slower (O(n²) attention) | Faster (O(n) convolutions) |
| **Parameters** | 10M-100M+ | ~1M-5M |
| **Best For** | Complex patterns, long-range dependencies | Local motifs, speed-critical tasks |

**When to use XLNet:**
- You have unlabeled data for pre-training
- Task requires understanding long-range interactions
- Computational resources available

**When to use Saluki:**
- Need fast inference
- Limited computational resources
- Focus on local sequence motifs

---

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add/update tests
5. Run tests: `poetry run pytest tests/ -v`
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

**Coding Standards:**
- Follow PEP 8 style guide
- Add type hints to function signatures
- Include docstrings for public methods
- Write tests for new features

---

## 📚 Citation

If you use this plugin in your research, please cite:

```bibtex
@software{xlnet_plugin_biolm,
  title={XLNet Plugin for BioLM 2.0},
  author={Dieterich Lab},
  year={2024},
  url={https://github.com/dieterich-lab/rna_protein_xlnet}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🔗 Related Projects

- **[BioLM Utils](https://github.com/dieterich-lab/biolm_utils)** - Core framework
- **[Saluki Plugin](https://github.com/dieterich-lab/rna_saluki_cnn)** - CNN-based alternative
- **[Transformers](https://github.com/huggingface/transformers)** - HuggingFace library

---

## 💬 Support

- **Issues:** [GitHub Issues](https://github.com/dieterich-lab/rna_protein_xlnet/issues)
- **Discussions:** [GitHub Discussions](https://github.com/dieterich-lab/rna_protein_xlnet/discussions)
- **Framework Docs:** [BioLM Documentation](https://github.com/dieterich-lab/biolm_utils/tree/biolm-2.0/docs)

---

**Built with ❤️ by the Dieterich Lab**
