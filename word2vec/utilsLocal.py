import json

def _create_data_from_json(path):
    with open(path) as jsonfile:
        data = json.load(jsonfile)['data']
        for item in data:
            _id=item["id"]
            _title=item["title"]
            _text=item["text"]
            # yield the raw data in the order of context, question, answers, answer_start
            yield {"id": _id,"title": _title,"text": _text}


