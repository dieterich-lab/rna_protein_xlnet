from transformers import (
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetLMHeadModel,
    XLNetTokenizerFast,
)
from transformers import (
    DataCollatorForPermutationLanguageModeling,
    DataCollatorWithPadding,
)

from biolm_utils.config import Config, set_config
from rna_xlnet_dataset import RNAXLNETDataset

params = [
    XLNetLMHeadModel, # 0 
    XLNetForSequenceClassification, # 1
    XLNetTokenizerFast,  # 2
    RNAXLNETDataset,  # 3
    1e-5,  # 4
    1.0,  # 5
    0.0,  # 6
    None,  # 7
    DataCollatorForPermutationLanguageModeling,  # 8
    DataCollatorWithPadding,  # 9
    True, # 10
    XLNetConfig,  # 11
    True,  # 12
]

config = Config(*params)
set_config(config)

from biolm_utils.biolm import run

run()
