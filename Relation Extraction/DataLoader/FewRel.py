import json
import torch
import os
import pickle
import random
import numpy as np

dirs = os.path.join(os.path.dirname(__file__), "../..")
os.sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import Config
from tqdm import tqdm


class InputFeature:
    def __init__(self, input_ids, input_mask, data_position1, data_position2):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.data_position1 = data_position1
        self.data_position2 = data_position2


class FewRel:
    def __init__(self):
        self.train_path = "Data/Few Rel/train.json"
        self.val_path = "Data/Few Rel/val.json"
        with open(Config.word_dic_path, "rb") as f:
            self.word2id = pickle.load(f)
        with open(Config.relation_dic_path, "rb") as f:
            self.relation2id = pickle.load(f)

    def get_train_dataloader(self):
        self.relation_features = self.convert_to_features(self.train_path)
        self.build_task()

    def build_task(self):
        target_class = random.sample(list(self.relation_features.keys()), Config.N)
        for i, classid in enumerate(target_class):
            features = np.array(self.relation_features[classid])
            scope = len(features)
            indice = np.random.randint(0, scope, Config.K + Config.Q)
            support_feature = features[indice]
            

    def convert_token_to_id(self, tokens):
        inputids = []
        for w in tokens:
            w = w.lower()
            if w in self.word2id:
                inputids.append(self.word2id[w])
            else:
                inputids.append(self.word2id["unk"])
        pad_length = Config.max_length - len(inputids)
        pad_ids = [0] * pad_length
        inputids = inputids + pad_ids
        return inputids

    def convert_data_to_position(self, tokens, instance):
        pos1 = instance["h"][2][0][0]
        pos2 = instance["t"][2][0][0]
        data_position1 = []
        data_position2 = []
        data_mask = []
        pos_min = min(pos1, pos2)
        pos_max = max(pos1, pos2)
        for i in range(Config.max_length):
            if i < len(tokens):
                data_position1.append(i - pos1 + Config.max_length)
                data_position2.append(i - pos2 + Config.max_length)
                if i < pos_min:
                    data_mask.append(1)
                elif i < pos_max:
                    data_mask.append(2)
                else:
                    data_mask.append(3)
            else:
                data_position1.append(0)
                data_position2.append(0)
                data_mask.append(0)
        return data_position1, data_position2, data_mask

    def convert_to_features(self, path):
        relation_features = {}
        with open(path, "r", encoding="utf-8") as f:
            original_data = json.load(f)
            for relation in tqdm(original_data):
                input_features = []
                for instance in original_data[relation]:
                    sentence_words = instance["tokens"]
                    input_ids = self.convert_token_to_id(sentence_words)
                    data_position1, data_position2, input_mask = self.convert_data_to_position(
                        sentence_words, instance
                    )
                    input_feature = InputFeature(
                        input_ids, input_mask, data_position1, data_position2
                    )
                    input_features.append(input_feature)
                relation_features[self.relation2id[relation]] = input_features
        return relation_features


if __name__ == "__main__":
    data = FewRel()
    data.get_train_dataloader()

