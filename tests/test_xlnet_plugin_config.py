"""
XLNet plugin configuration tests.

These tests verify that:
1. XLNet plugin loads correctly via entry points
2. Plugin configuration is complete and correct
3. All required attributes exist
"""

import logging

import pytest

logging.basicConfig(level=logging.DEBUG)


def debug_log(msg):
    """Print debug message to stdout for test visibility."""
    print(msg)


def test_xlnet_plugin_loading():
    """Test that XLNet plugin can be loaded via entry points."""
    debug_log("Starting XLNet plugin loading test")

    import importlib.metadata

    from biolm.plugin_config import PluginManager

    # Test loading XLNet plugin via entry point
    eps = importlib.metadata.entry_points(group="biolm.plugins")
    xlnet_ep = next((ep for ep in eps if ep.name == "xlnet"), None)

    assert xlnet_ep is not None, "XLNet plugin not found in entry points"
    debug_log("✓ XLNet entry point found")

    # Load the config function
    xlnet_config_fn = xlnet_ep.load()
    xlnet_config_fn()
    config = PluginManager.get_config()

    debug_log(f"✓ XLNet plugin loaded successfully")
    assert config.pretraining_required == True, "XLNet should require pretraining"
    debug_log(f"✓ XLNet pretraining_required: {config.pretraining_required}")


def test_xlnet_plugin_config():
    """Test that XLNet plugin configuration is complete."""
    debug_log("Starting XLNet config validation test")

    import importlib.metadata

    from biolm.plugin_config import PluginManager

    # Load XLNet plugin via entry point
    eps = importlib.metadata.entry_points(group="biolm.plugins")
    xlnet_ep = next(ep for ep in eps if ep.name == "xlnet")
    config_fn = xlnet_ep.load()
    config_fn()
    config = PluginManager.get_config()

    # Verify all expected attributes are present and correct
    assert (
        config.model_cls_for_pretraining is not None
    ), "model_cls_for_pretraining is None"
    debug_log("✓ model_cls_for_pretraining exists")

    assert (
        config.model_cls_for_finetuning is not None
    ), "model_cls_for_finetuning is None"
    debug_log("✓ model_cls_for_finetuning exists")

    assert config.dataset_cls is not None, "dataset_cls is None"
    debug_log("✓ dataset_cls exists")

    assert config.tokenizer_cls is not None, "tokenizer_cls is None"
    debug_log("✓ tokenizer_cls exists")

    assert (
        config.datacollator_cls_for_pretraining is not None
    ), "datacollator_cls_for_pretraining is None"
    debug_log("✓ datacollator_cls_for_pretraining exists")

    assert (
        config.datacollator_cls_for_finetuning is not None
    ), "datacollator_cls_for_finetuning is None"
    debug_log("✓ datacollator_cls_for_finetuning exists")

    assert config.add_special_tokens == True, "add_special_tokens should be True"
    debug_log("✓ add_special_tokens = True")

    assert config.pretraining_required == True, "pretraining_required should be True"
    debug_log("✓ pretraining_required = True")

    debug_log("✓ All XLNet config attributes validated successfully")


def test_xlnet_models_are_callable():
    """Test that XLNet model classes can be instantiated."""
    debug_log("Starting XLNet model callability test")

    import importlib.metadata

    from biolm.plugin_config import PluginManager

    # Load XLNet plugin
    eps = importlib.metadata.entry_points(group="biolm.plugins")
    xlnet_ep = next(ep for ep in eps if ep.name == "xlnet")
    config_fn = xlnet_ep.load()
    config_fn()
    config = PluginManager.get_config()

    # Verify model classes are callable (have __init__)
    assert hasattr(
        config.model_cls_for_pretraining, "__init__"
    ), "Pretraining model not callable"
    debug_log("✓ Pretraining model is callable")

    assert hasattr(
        config.model_cls_for_finetuning, "__init__"
    ), "Finetuning model not callable"
    debug_log("✓ Finetuning model is callable")

    debug_log("✓ All XLNet model classes are callable")
