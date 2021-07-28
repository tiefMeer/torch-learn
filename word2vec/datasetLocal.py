from torchtext.data.datasets_utils import (
    _RawTextIterableDataset,
    _wrap_split_argument,
    _add_docstring_header,
    _create_dataset_directory,
)
import utilsLocal as utl

NAME = "AA"
NUM_ITEMS = 19446
path_ZHWIKI_AA = \
    '/home/chxybin/lab/workspace/purvar/torch/zhwikiTrain/data/ZHWIKI/AA.json'

def ZHWIKI_AA():
    return _RawTextIterableDataset(NAME, NUM_ITEMS, 
                                    utl._create_data_from_json(path_ZHWIKI_AA))

