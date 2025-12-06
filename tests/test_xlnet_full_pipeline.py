"""
Full end-to-end XLNet pipeline test.

This test verifies the complete XLNet training pipeline:
1. Tokenization (creates tokenizer and tokenized dataset)
2. Pre-training (trains language model - 1 epoch)
3. Fine-tuning (trains on labeled regression task - 1 epoch)
4. Testing/prediction with Spearman correlation
"""

import json
import logging
import subprocess
import tempfile
from pathlib import Path

import pytest

logging.basicConfig(level=logging.DEBUG)


def debug_log(msg):
    """Print debug message to stdout for test visibility."""
    print(msg)


@pytest.fixture(scope="module")
def tiny_dataset():
    """Create minimal dataset for XLNet pipeline testing.

    Format: seq\tlabel (plain AUGC sequences)
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="xlnet_e2e_dataset_"))

    # 10 sequences, each exactly 100 nucleotides
    sequences_atgc = [
        (
            "AUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGC",
            "1.5",
        ),
        (
            "GGCUGGCUGGCUGGCUGGCUGGCUGGCUGGCUGGCUGGCUGGCUGGCUGGCUGGCUGGCUGGCUGGCUGGCUGGCUGGCUGGCUGGCUGGCUGGCUGGCU",
            "2.5",
        ),
        (
            "CCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGG",
            "3.5",
        ),
        (
            "UUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAA",
            "0.5",
        ),
        (
            "AAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUU",
            "4.5",
        ),
        (
            "GCAGGCAGGCAGGCAGGCAGGCAGGCAGGCAGGCAGGCAGGCAGGCAGGCAGGCAGGCAGGCAGGCAGGCAGGCAGGCAGGCAGGCAGGCAGGCAGGCAG",
            "2.0",
        ),
        (
            "CUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAG",
            "3.0",
        ),
        (
            "GAUGGAUGGAUGGAUGGAUGGAUGGAUGGAUGGAUGGAUGGAUGGAUGGAUGGAUGGAUGGAUGGAUGGAUGGAUGGAUGGAUGGAUGGAUGGAUGGAUG",
            "1.0",
        ),
        (
            "ACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGU",
            "1.2",
        ),
        (
            "UGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCA",
            "3.8",
        ),
    ]

    # Create training file - XLNet format: seq\tlabel
    for split_name, indices in [("train", list(range(10)))]:
        filepath = tmpdir / f"{split_name}.txt"
        with open(filepath, "w") as f:
            for idx in indices:
                seq, label = sequences_atgc[idx]
                f.write(f"{seq}\t{label}\n")

    debug_log(f"Created tiny dataset at {tmpdir}")
    return tmpdir


def run_command(cmd, cwd="/prj/RNA_NLP/biolm_utils", timeout=600):
    """Helper to run command and return result."""
    debug_log(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=cwd, timeout=timeout
    )
    if result.returncode != 0:
        debug_log(f"STDOUT:\n{result.stdout}")
        debug_log(f"STDERR:\n{result.stderr}")
    return result


def test_xlnet_full_pipeline(tiny_dataset):
    """Test full XLNet pipeline: tokenize -> pre-train -> fine-tune -> test."""
    debug_log("=" * 80)
    debug_log("STARTING XLNET FULL PIPELINE TEST")
    debug_log("=" * 80)

    with tempfile.TemporaryDirectory(prefix="xlnet_e2e_") as tmpdir:
        output_dir = Path(tmpdir)

        # Step 1: Tokenization
        debug_log("\n>>> STEP 1: TOKENIZATION")
        tokenize_cmd = [
            "poetry",
            "run",
            "python",
            "-m",
            "biolm.runner",
            f"data_source.filepath={tiny_dataset}/train.txt",
            f"outputpath={output_dir}",
            "mode=tokenize",
            "model=xlnet",
            "tokenization.vocabsize=100",
            "training.num_epochs=1",
            "debugging.accelerator=cpu",
        ]
        result = run_command(tokenize_cmd)
        assert result.returncode == 0, f"Tokenization failed:\n{result.stderr}"
        debug_log("✓ Tokenization completed")

        # Verify tokenizer was created
        tokenizer_file = output_dir / "tokenizer.json"
        assert tokenizer_file.exists(), f"Tokenizer not found at {tokenizer_file}"
        debug_log(f"✓ Tokenizer created at {tokenizer_file}")

        # Step 2: Pre-training (XLNet supports this)
        debug_log("\n>>> STEP 2: PRE-TRAINING")
        pretrain_cmd = [
            "poetry",
            "run",
            "python",
            "-m",
            "biolm.runner",
            f"data_source.filepath={tiny_dataset}/train.txt",
            f"outputpath={output_dir}",
            "mode=pre-train",
            "model=xlnet",
            "training.num_epochs=1",
            "+training.blocksize=512",
            "model.num_layers=1",
            "model.hidden_size=32",
            "model.num_heads=2",
            "model.d_head=16",
            "model.intermediate_size=64",
            "debugging.accelerator=cpu",
            "training.batchsize=1",
        ]
        result = run_command(pretrain_cmd, timeout=900)
        assert result.returncode == 0, f"Pre-training failed:\n{result.stderr}"
        debug_log("✓ Pre-training completed")

        # Verify pre-training checkpoint
        pretrain_dir = output_dir / "pre-train"
        assert pretrain_dir.exists(), f"Pre-train directory not found"
        debug_log(f"✓ Pre-training checkpoint created")

        # Step 3: Fine-tuning
        debug_log("\n>>> STEP 3: FINE-TUNING")
        finetune_cmd = [
            "poetry",
            "run",
            "python",
            "-m",
            "biolm.runner",
            f"data_source.filepath={tiny_dataset}/train.txt",
            f"outputpath={output_dir}",
            "mode=fine-tune",
            "model=xlnet",
            "task=regression",
            "training.num_epochs=1",
            "+training.blocksize=512",
            "model.num_layers=1",
            "model.hidden_size=32",
            "model.num_heads=2",
            "model.d_head=16",
            "model.intermediate_size=64",
            "debugging.accelerator=cpu",
            "training.batchsize=1",
            "data_source.splitratio=[80,20]",
        ]
        result = run_command(finetune_cmd, timeout=900)
        assert result.returncode == 0, f"Fine-tuning failed:\n{result.stderr}"
        debug_log("✓ Fine-tuning completed")

        # Verify fine-tuning checkpoint
        finetune_dir = output_dir / "fine-tune"
        assert finetune_dir.exists(), f"Fine-tune directory not found"
        debug_log(f"✓ Fine-tuning checkpoint created")

        # Step 4: Testing
        debug_log("\n>>> STEP 4: TESTING")
        test_results_file = finetune_dir / "results.json"
        assert test_results_file.exists(), f"Results file not found"

        with open(test_results_file) as f:
            results = json.load(f)

        debug_log(f"Results: {results}")
        assert "test_spearmanr" in results, "Spearman correlation not in results"
        debug_log(f"✓ Test Spearman: {results['test_spearmanr']}")

        debug_log("\n" + "=" * 80)
        debug_log("XLNET FULL PIPELINE TEST PASSED")
        debug_log("=" * 80)
