import torch.nn as nn
from transformers import XLNetForSequenceClassification, XLNetLMHeadModel

# XLNet model constants
XLNET_BLOCKSIZE = 512


class RNA_XLNetLMHeadModel(XLNetLMHeadModel):
    @staticmethod
    def get_config(args, config_cls, tokenizer, dataset, nlabels):
        # XLNet invariant: blocksize must be 512
        if hasattr(args, "training") and args.training:
            if (
                not hasattr(args.training, "blocksize")
                or args.training.blocksize is None
            ):
                args.training.blocksize = 512
            elif args.training.blocksize != 512:
                raise ValueError(
                    f"XLNet requires blocksize=512, but got {args.training.blocksize}. "
                    "XLNet has a fixed maximum sequence length of 512 tokens."
                )
        config = config_cls(
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
            d_model=768,
            d_inner=3072,
            d_head=64,
            n_head=12,
            n_layer=12,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
        )
        config.num_labels = int(bool(nlabels))
        return config


class RNA_XLNetForSequenceClassification(XLNetForSequenceClassification):
    @staticmethod
    def get_config(args, config_cls, tokenizer, dataset, nlabels):
        # XLNet invariant: blocksize must be 512
        if hasattr(args, "training") and args.training:
            if (
                not hasattr(args.training, "blocksize")
                or args.training.blocksize is None
            ):
                args.training.blocksize = 512
            elif args.training.blocksize != 512:
                raise ValueError(
                    f"XLNet requires blocksize=512, but got {args.training.blocksize}. "
                    "XLNet has a fixed maximum sequence length of 512 tokens."
                )
        config = config_cls(
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
            d_model=768,
            d_inner=3072,
            d_head=64,
            n_head=12,
            n_layer=12,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
        )
        config.num_labels = int(nlabels)
        return config
