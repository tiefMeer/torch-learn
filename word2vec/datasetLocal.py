import json
from torchtext.data.datasets_utils import _RawTextIterableDataset

NAME = "AA"
NUM_ITEMS = 19446
path_ZHWIKI_AA = 'data/corpus/ZHWIKI/AA.json'

def ZHWIKI_AA():
    data_iter = _create_data_from_json(path_ZHWIKI_AA)
    return _RawTextIterableDataset(NAME, NUM_ITEMS, data_iter)

def _create_data_from_json(path):
    with open(path) as jsonfile:
        data = json.load(jsonfile)['data']
        for item in data:
            _id=item["id"]
            _title=item["title"]
            _text=item["text"]
            yield {"id": _id,"title": _title,"text": _text}

