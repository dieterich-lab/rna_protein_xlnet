import torch.nn as nn
from transformers import XLNetForSequenceClassification, XLNetLMHeadModel

# XLNet model constants
XLNET_BLOCKSIZE = 512


class RNA_XLNetLMHeadModel(XLNetLMHeadModel):
    @staticmethod
    def get_config(args, config_cls, tokenizer, dataset, nlabels):
        blocksize = getattr(args, "blocksize", None)
        if blocksize is None:
            training = getattr(args, "training", None)
            if training is not None:
                blocksize = getattr(training, "blocksize", None)

        if blocksize not in (None, XLNET_BLOCKSIZE):
            raise ValueError(
                f"XLNet requires blocksize={XLNET_BLOCKSIZE}; do not set a different blocksize in global config."
            )

        config = config_cls(
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
            d_model=768,
            d_inner=3072,
            d_head=64,
            n_head=12,
            n_layer=12,
            max_position_embeddings=XLNET_BLOCKSIZE,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
        )
        config.num_labels = int(bool(nlabels))
        return config


class RNA_XLNetForSequenceClassification(XLNetForSequenceClassification):
    @staticmethod
    def get_config(args, config_cls, tokenizer, dataset, nlabels):
        blocksize = getattr(args, "blocksize", None)
        if blocksize is None:
            training = getattr(args, "training", None)
            if training is not None:
                blocksize = getattr(training, "blocksize", None)

        if blocksize not in (None, XLNET_BLOCKSIZE):
            raise ValueError(
                f"XLNet requires blocksize={XLNET_BLOCKSIZE}; do not set a different blocksize in global config."
            )

        config = config_cls(
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
            d_model=768,
            d_inner=3072,
            d_head=64,
            n_head=12,
            n_layer=12,
            max_position_embeddings=XLNET_BLOCKSIZE,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
        )
        config.num_labels = int(nlabels)
        return config
