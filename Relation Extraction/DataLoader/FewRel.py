import json
import torch
import Config


class FewRel:
    def __init__(self):
        self.train_path = "Data/Few Rel/train.json"
        self.val_path = "Data/Few Rel/val.json"

    def loadJson(self, name):
        file_path = ""
        if name == "train":
            file_path = self.train_path
        elif name == "validation":
            file_path == self.val_path
        else:
            raise Exception("[ERROR] data file doesn't exist")
        with open(file_path, "r", "utf-8") as f:
            original_data = json.load(f)
            for relation in original_data:
                sentences_number += len(original_data[relation])
            self.words = torch.zeros(sentences_number, Config.max_length)
            for relation in original_data:
                for instance in original_data[relation]:
                    words = instance['tokens']


if __name__ == "__main__":
    data = FewRel()
    data.loadJson("train")

