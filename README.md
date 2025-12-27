# XLNet Plugin for BioLM 2.0

**Transformer-based language models for RNA and protein sequences**

This plugin integrates XLNet into the BioLM framework, enabling:
- **Pre-training**: Permutation Language Modeling (PLM) for sequence analysis.
- **Fine-tuning**: Classification and regression tasks.
- **Interpretation**: Leave-one-out (LOO) analysis for feature importance.

## Installation

1. Install the BioLM framework:
   ```bash
   git clone https://github.com/dieterich-lab/biolm_utils.git
   cd biolm_utils
   git checkout biolm-2.0
   ./install.sh
   ```

2. Install the XLNet plugin:
   ```bash
   poetry run biolm install-plugin https://github.com/dieterich-lab/rna_protein_xlnet.git
   ```
