import torch
from biolm_utils.rna_datasets import RNABaseDataset

# XLNet model constants
XLNET_BLOCKSIZE = 512


class RNALanguageDataset(RNABaseDataset):
    def __init__(self, **args):
        # enforce XLNet invariants early (BPE encoding + fixed blocksize)
        if (
            isinstance(args, dict)
            and "args" in args
            and not isinstance(args.get("args"), dict)
        ):
            call_args = args.get("args")
        else:
            call_args = args

        encoding = (
            call_args.tokenization.encoding
            if hasattr(call_args, "tokenization")
            else (
                call_args.get("encoding")
                if isinstance(call_args, dict)
                else getattr(call_args, "encoding", None)
            )
        )
        if encoding != "bpe":
            raise ValueError(
                "XLNet requires tokenization.encoding='bpe'. Do not set a different encoding in global configs; customize tokenization only in plugin-specific code."
            )

        expected_blocksize = XLNET_BLOCKSIZE
        blocksize = (
            call_args.training.blocksize
            if hasattr(call_args, "training")
            else (
                call_args.get("blocksize")
                if isinstance(call_args, dict)
                else getattr(call_args, "blocksize", None)
            )
        )
        if blocksize != expected_blocksize:
            raise ValueError(
                f"XLNet requires training.blocksize={expected_blocksize}. This is an internal XLNet model property and cannot be changed via global configs."
            )

        # initialize base class after invariants pass
        super().__init__(**args)

    def __getitem__(self, i):
        example = self.examples[i].copy()
        example["input_ids"] = torch.tensor(example["input_ids"], dtype=torch.long)
        return example
