import torch.nn as nn
from transformers import XLNetForSequenceClassification, XLNetLMHeadModel

# XLNet model constants
XLNET_BLOCKSIZE = 512


class RNA_XLNetLMHeadModel(XLNetLMHeadModel):
    @staticmethod
    def get_config(args, config_cls, tokenizer, dataset, nlabels):
        config = config_cls(
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
            d_model=768,
            d_inner=3072,
            d_head=64,
            n_head=12,
            n_layer=12,
            max_position_embeddings=512,  # Keep default, but allow override
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
        )
        config.num_labels = int(bool(nlabels))
        return config


class RNA_XLNetForSequenceClassification(XLNetForSequenceClassification):
    @staticmethod
    def get_config(args, config_cls, tokenizer, dataset, nlabels):
        config = config_cls(
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
            d_model=768,
            d_inner=3072,
            d_head=64,
            n_head=12,
            n_layer=12,
            max_position_embeddings=512,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
        )
        config.num_labels = int(nlabels)
        return config
