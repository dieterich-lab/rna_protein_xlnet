"""Plugin configuration for the XLNet model.

This module defines the complete configuration for the XLNet plugin.
Modify the PluginConfig below to customize the plugin for your needs.
"""

from transformers import (
    DataCollatorForPermutationLanguageModeling,
    DataCollatorWithPadding,
    XLNetConfig,
    XLNetTokenizerFast,
)

from .dataset import RNALanguageDataset
from .models import RNA_XLNetForSequenceClassification, RNA_XLNetLMHeadModel


def get_xlnet_config():
    """Factory function that creates and returns the plugin configuration.

    The framework calls this function when the plugin is loaded.
    It creates a PluginConfig, sets it as active, and returns it.

    Returns:
        PluginConfig: The complete plugin configuration.
    """
    from biolm_utils.plugin_config import PluginConfig, PluginManager

    # Create the plugin configuration
    # Modify these settings for your custom plugin
    config = PluginConfig(
        model_cls_for_pretraining=RNA_XLNetLMHeadModel,
        model_cls_for_finetuning=RNA_XLNetForSequenceClassification,
        dataset_cls=RNALanguageDataset,
        tokenizer_cls=XLNetTokenizerFast,
        datacollator_cls_for_pretraining=DataCollatorForPermutationLanguageModeling,
        datacollator_cls_for_finetuning=DataCollatorWithPadding,
        add_special_tokens=True,
        config_cls=XLNetConfig,
        pretraining_required=True,
        learning_rate=1e-5,
        max_grad_norm=1.0,
        weight_decay=0.0,
        special_tokenizer_for_trainer_cls=None,
    )

    # Make this the active configuration in the framework
    PluginManager.set_config(config)

    return config


__all__ = ["get_xlnet_config"]