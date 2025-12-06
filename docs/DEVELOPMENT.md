# XLNet Plugin Development Guide

Guide for developing and extending the XLNet plugin for BioLM 2.0.

## 🎯 Overview

The XLNet plugin implements transformer-based models with Permutation Language Modeling (PLM) for RNA and protein sequences. This guide covers:

- Plugin architecture and structure
- Development workflow
- Testing strategies
- Adding new features
- Debugging tips

---

## 🏗️ Plugin Architecture

### Plugin Structure

```
xlnet_plugin/
├── __init__.py              # Package initialization
├── config.py                # Plugin configuration (entry point)
├── dataset.py               # RNALanguageDataset class
└── models.py                # XLNet model implementations
```

### Entry Point Registration

The plugin is discovered via entry point in `pyproject.toml`:

```toml
[tool.poetry.plugins."biolm.plugins"]
xlnet = "xlnet_plugin.config:get_config"
```

**How it works:**
1. BioLM scans `biolm.plugins` entry point group
2. Finds `xlnet` entry pointing to `get_config()` function
3. Calls `get_config()` to get plugin configuration
4. Uses returned `PluginConfig` to instantiate models/datasets

---

## 📦 Plugin Configuration

### `xlnet_plugin/config.py`

The `get_config()` function returns a `PluginConfig` object:

```python
from biolm.plugin_config import PluginConfig
from xlnet_plugin.models import RNA_XLNetLMHeadModel, RNA_XLNetForSequenceClassification
from xlnet_plugin.dataset import RNALanguageDataset
from transformers import PreTrainedTokenizerFast
from biolm.data_collator import DataCollatorForPermutationLanguageModeling

def get_config() -> PluginConfig:
    """XLNet plugin configuration."""
    return PluginConfig(
        # Pre-training and fine-tuning models
        model_cls_for_pretraining=RNA_XLNetLMHeadModel,
        model_cls_for_finetuning=RNA_XLNetForSequenceClassification,
        
        # Dataset and tokenizer
        dataset_cls=RNALanguageDataset,
        tokenizer_cls=PreTrainedTokenizerFast,
        
        # Data collators
        datacollator_cls_for_pretraining=DataCollatorForPermutationLanguageModeling,
        datacollator_cls_for_finetuning=DefaultDataCollator,
        
        # Configuration
        add_special_tokens=True,  # Uses [CLS], [SEP], etc.
        pretraining_required=True,  # Recommends pre-training
    )
```

**Key points:**
- `model_cls_for_pretraining` - XLNet with LM head for PLM
- `pretraining_required=True` - Pre-training recommended for best results
- `DataCollatorForPermutationLanguageModeling` - Handles PLM masking

---

## 🧬 Model Implementation

### RNA_XLNetLMHeadModel (Pre-training)

Located in `xlnet_plugin/models.py`:

```python
class RNA_XLNetLMHeadModel(PreTrainedModel):
    """XLNet model with language modeling head for pre-training."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Core XLNet transformer
        self.transformer = XLNetModel(config)
        
        # Language modeling head
        self.lm_loss = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=True
        )
        
        self.post_init()
    
    def forward(
        self,
        input_ids=None,
        perm_mask=None,
        target_mapping=None,
        labels=None,
        **kwargs
    ):
        """
        Args:
            input_ids: [batch, seq_len] - Input token IDs
            perm_mask: [batch, seq_len, seq_len] - Permutation mask
            target_mapping: [batch, num_predict, seq_len] - Target positions
            labels: [batch, num_predict] - Target labels
        """
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            **kwargs
        )
        
        # Get hidden states for targets only
        hidden_states = outputs[0]  # [batch, num_predict, hidden_size]
        
        # Predict tokens
        logits = self.lm_loss(hidden_states)  # [batch, num_predict, vocab_size]
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1)
            )
        
        return {"loss": loss, "logits": logits}
```

**Key design choices:**
- **Two-stream attention:** XLNet uses content and query streams
- **Permutation masking:** Learns from all factorization orders
- **Target mapping:** Predicts only specified positions
- **No [MASK] tokens:** Uses actual tokens with masking in attention

---

### RNA_XLNetForSequenceClassification (Fine-tuning)

```python
class RNA_XLNetForSequenceClassification(PreTrainedModel):
    """XLNet model for sequence-level classification/regression."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Core transformer
        self.transformer = XLNetModel(config)
        
        # Classification head
        self.sequence_summary = SequenceSummary(config)
        self.classifier = nn.Linear(
            config.hidden_size,
            config.num_labels
        )
        
        self.post_init()
    
    def forward(self, input_ids, labels=None, **kwargs):
        """
        Args:
            input_ids: [batch, seq_len]
            labels: [batch] or [batch, num_labels]
        """
        # Get transformer outputs
        outputs = self.transformer(input_ids=input_ids, **kwargs)
        hidden_states = outputs[0]  # [batch, seq_len, hidden_size]
        
        # Pool sequence (e.g., use last token)
        pooled = self.sequence_summary(hidden_states)  # [batch, hidden_size]
        
        # Predict
        logits = self.classifier(pooled)  # [batch, num_labels]
        
        # Compute loss
        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                # Regression
                loss_fn = nn.MSELoss()
                loss = loss_fn(logits.squeeze(), labels.float())
            else:
                # Classification
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits, labels)
        
        return {"loss": loss, "logits": logits}
```

**Important:**
- **SequenceSummary:** Pools sequence representation (configurable)
- **Supports both:** Classification and regression
- **Pre-trained weights:** Can load from pre-trained checkpoint

---

## 📊 Dataset Implementation

### RNALanguageDataset

Located in `xlnet_plugin/dataset.py`:

```python
class RNALanguageDataset(Dataset):
    """Dataset for k-mer tokenized RNA/protein sequences."""
    
    def __init__(
        self,
        filepath,
        tokenizer,
        column_ids=[1, 2, 3],
        max_length=512
    ):
        """
        Args:
            filepath: Path to data file
            tokenizer: Tokenizer for encoding
            column_ids: [id_col, label_col, sequence_col] (1-indexed)
            max_length: Maximum sequence length
        """
        self.data = []
        
        with open(filepath) as f:
            for line in f:
                parts = line.strip().split('\t')
                
                # Extract columns (convert to 0-indexed)
                seq_id = parts[column_ids[0] - 1]
                label = float(parts[column_ids[1] - 1]) if len(parts) > 1 else None
                sequence = parts[column_ids[2] - 1] if len(parts) > 2 else parts[0]
                
                # Tokenize: "AUG|CUA|GCU" → [12, 45, 67, ...]
                tokens = tokenizer.encode(
                    sequence,
                    max_length=max_length,
                    truncation=True,
                    padding='max_length'
                )
                
                item = {
                    'input_ids': tokens,
                    'seq_id': seq_id
                }
                
                if label is not None:
                    item['labels'] = label
                
                self.data.append(item)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
```

**Important:**
- Expects **k-mer tokenized** sequences: `AUG|CUA|GCU|...`
- Column IDs are **1-indexed** (user-friendly)
- Handles both labeled (fine-tuning) and unlabeled (pre-training) data
- Pads/truncates to `max_length`

---

## 🎭 Permutation Language Modeling

### DataCollatorForPermutationLanguageModeling

The PLM data collator creates permutation masks:

```python
class DataCollatorForPermutationLanguageModeling:
    """Creates inputs for permutation language modeling."""
    
    def __call__(self, examples):
        """
        Creates:
        - input_ids: Original token IDs
        - perm_mask: Which tokens can attend to which
        - target_mapping: Which tokens to predict
        - labels: Target token IDs
        """
        batch_size = len(examples)
        seq_len = len(examples[0]['input_ids'])
        
        # Stack input IDs
        input_ids = torch.stack([
            torch.tensor(ex['input_ids']) for ex in examples
        ])
        
        # Generate random permutations
        for i in range(batch_size):
            perm = torch.randperm(seq_len)
            # ... create perm_mask and target_mapping ...
        
        return {
            'input_ids': input_ids,
            'perm_mask': perm_mask,
            'target_mapping': target_mapping,
            'labels': labels
        }
```

**Key concepts:**
- **Permutation mask:** Controls attention based on permutation order
- **Target mapping:** Selects which tokens to predict
- **Even length required:** Ensures leakage-free masking
- **Random permutations:** Different factorization orders each epoch

---

## 🧪 Testing

### Test Structure

```
tests/
├── test_xlnet_full_pipeline.py      # End-to-end tests
└── test_xlnet_plugin_config.py      # Configuration tests
```

### Running Tests

```bash
# All tests
poetry run pytest tests/

# Specific file
poetry run pytest tests/test_xlnet_full_pipeline.py -v

# With coverage
poetry run pytest tests/ --cov=xlnet_plugin --cov-report=html
```

### Writing New Tests

```python
def test_pretrain_workflow():
    """Test pre-training workflow."""
    # Arrange
    config = get_config()
    model = config.model_cls_for_pretraining()
    collator = config.datacollator_cls_for_pretraining()
    
    # Act
    batch = collator([{'input_ids': [1, 2, 3, 4]}])
    outputs = model(**batch)
    
    # Assert
    assert outputs['loss'] is not None
    assert outputs['logits'].shape[-1] == model.config.vocab_size
```

---

## 🔧 Development Workflow

### 1. Setup Development Environment

```bash
# Clone repo
git clone https://github.com/dieterich-lab/rna_protein_xlnet.git
cd rna_protein_xlnet

# Install with dev dependencies
poetry install --with dev

# Verify tests pass
poetry run pytest tests/
```

### 2. Make Changes

```bash
# Create feature branch
git checkout -b feature/my-feature

# Edit code
vim xlnet_plugin/models.py

# Run tests
poetry run pytest tests/ -v
```

### 3. Code Quality

```bash
# Check style
ruff check xlnet_plugin/

# Format code
ruff format xlnet_plugin/

# Type checking (optional)
mypy xlnet_plugin/
```

### 4. Commit and Push

```bash
git add xlnet_plugin/
git commit -m "Add my feature"
git push origin feature/my-feature
```

---

## 🎨 Adding New Features

### Example: Add Relative Position Bias

1. **Update model:**

```python
# xlnet_plugin/models.py
class RNA_XLNetLMHeadModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # ... existing code ...
        
        # Add relative position bias
        self.rel_pos_bias = nn.Parameter(
            torch.zeros(config.num_attention_heads, config.max_position_embeddings)
        )
    
    def forward(self, input_ids, **kwargs):
        # ... existing code ...
        
        # Add bias to attention scores
        kwargs['rel_pos_bias'] = self.rel_pos_bias
        outputs = self.transformer(input_ids=input_ids, **kwargs)
        
        # ... rest of forward ...
```

2. **Add test:**

```python
# tests/test_xlnet_plugin_config.py
def test_relative_position_bias():
    """Test that relative position bias is used."""
    config = get_config()
    model = config.model_cls_for_pretraining()
    
    # Check bias exists
    assert hasattr(model, 'rel_pos_bias')
    assert model.rel_pos_bias.shape[0] == model.config.num_attention_heads
```

3. **Update documentation:**

```markdown
## Model Architecture

- Relative position bias for better position modeling
- Improves long-range dependency learning
```

---

## 🐛 Debugging Tips

### Common Issues

**1. Plugin not discovered:**

```bash
# Check entry point
poetry run python -c "
import importlib.metadata
eps = importlib.metadata.entry_points(group='biolm.plugins')
print([ep.name for ep in eps])
"

# Should show: ['xlnet', ...]
```

**Fix:** Reinstall plugin: `cd rna_protein_xlnet && poetry install`

---

**2. Sequence length must be even:**

```python
# Error: ValueError: Sequence length must be even for PLM

# Check sequence lengths
import torch
input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # Odd length!
print(f"Length: {input_ids.shape[1]}")  # 5 (odd)
```

**Fix:** Pad to even length:
```python
if input_ids.shape[1] % 2 != 0:
    input_ids = F.pad(input_ids, (0, 1), value=tokenizer.pad_token_id)
```

---

**3. Attention mask shape mismatch:**

```python
# Error: RuntimeError: The size of tensor a (512) must match the size of tensor b (256)

# Check shapes
print(f"Input IDs: {input_ids.shape}")
print(f"Perm mask: {perm_mask.shape}")
print(f"Target mapping: {target_mapping.shape}")
```

**Fix:** Ensure all tensors have matching sequence dimension

---

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now see detailed logs
config = get_config()
model = config.model_cls_for_pretraining()
```

---

## 📚 Best Practices

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings to public functions
- Keep functions focused and small

### Testing

- Test happy paths and edge cases
- Mock expensive operations (training)
- Use fixtures for common setup
- Aim for >80% coverage

### Documentation

- Update README for user-facing changes
- Add docstrings to new classes/functions
- Include examples in docstrings
- Keep this dev guide updated

---

## 🔬 Understanding XLNet

### Two-Stream Self-Attention

XLNet uses two streams:

1. **Content stream** - Standard hidden states
2. **Query stream** - Can only use context (not content at position)

This allows the model to:
- Predict tokens using only previous tokens in permutation
- Avoid "leaking" information about the target

### Permutation Language Modeling

Instead of:
```
MASK some tokens → predict them
```

XLNet does:
```
Generate random permutation → predict tokens based on earlier tokens in permutation
```

**Example:**

Sequence: `[A, U, G, C]`

Permutation 1: `[3, 1, 4, 2]` → Predict order: `G, A, C, U`
Permutation 2: `[2, 4, 1, 3]` → Predict order: `U, C, A, G`

Each permutation provides different training signal!

---

## 🔗 Related Resources

- **[BioLM Plugin Development](https://github.com/dieterich-lab/biolm_utils/blob/biolm-2.0/docs/PLUGIN_DEVELOPMENT.md)** - Framework guide
- **[XLNet Paper](https://arxiv.org/abs/1906.08237)** - Original paper
- **[HuggingFace XLNet](https://huggingface.co/docs/transformers/model_doc/xlnet)** - Implementation reference
- **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** - Transformer architecture

---

## 🚀 Next Steps

- [ ] Add support for protein-specific features
- [ ] Implement relative segment encodings
- [ ] Add more pooling strategies
- [ ] Optimize memory usage for long sequences
- [ ] Add visualization tools for attention

---

**Questions?** Open an issue on [GitHub](https://github.com/dieterich-lab/rna_protein_xlnet/issues)
