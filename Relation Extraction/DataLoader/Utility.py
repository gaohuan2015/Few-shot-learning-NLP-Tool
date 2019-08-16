import os

dirs = os.path.join(os.path.dirname(__file__), "../..")
os.sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import pickle
import json
import Config
from tqdm import tqdm


def build_word_dic(path):
    word_to_id = {"pad": 0, "unk": 1}
    relation_to_id = {}
    with open(path, "r", encoding="utf-8") as f:
        original_data = json.load(f)
        for relation in tqdm(original_data):
            if relation not in relation_to_id:
                relation_to_id[relation] = len(relation_to_id)
            for instance in original_data[relation]:
                words = instance["tokens"]
                for word in words:
                    word = word.lower()
                    if word not in word_to_id:
                        word_to_id[word] = len(word_to_id)
    with open(Config.word_dic_path, "wb") as f:
        pickle.dump(word_to_id, f)
    with open(Config.relation_dic_path, "wb") as f:
        pickle.dump(relation_to_id, f)


if __name__ == "__main__":
    build_word_dic("Data/Few Rel/train.json")

