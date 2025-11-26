import torch
from biolm_utils.rna_datasets import RNABaseDataset

# XLNet model constants
XLNET_BLOCKSIZE = 512


class RNALanguageDataset(RNABaseDataset):
    def __init__(self, **args):
        # initialize base class
        super().__init__(**args)

    def __getitem__(self, i):
        example = self.examples[i].copy()
        example["input_ids"] = torch.tensor(example["input_ids"], dtype=torch.long)
        return example
