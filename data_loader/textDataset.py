import json
import torch
import os
from torch.utils.data import Dataset
from tqdm import tqdm
from utils import preprocess, image_transforms
from PIL import Image, ImageFile
from transformers import AutoTokenizer, ViTFeatureExtractor
from transformers import CLIPTokenizer, CLIPProcessor , DebertaTokenizerFast
import random

class TextDataset(Dataset):
    def __init__(self, label_path, classes, train=False, max_length=64, language='en', pretrain='bert'):
        self.train = train
        self.label_path = label_path
        self.max_length = max_length
        self.language = language
        self.classes = classes
        self.pretrain = pretrain
        self.text_tokenizer = self._get_text_tokenizer()
        self.ids, self.texts, self.labels = self.process_file()

    def _get_text_tokenizer(self):
        if self.language == 'en' and self.pretrain == 'bert':
            return AutoTokenizer.from_pretrained("bert-base-uncased")
        elif self.language == 'cn' and self.pretrain == 'bert':
            return AutoTokenizer.from_pretrained("bert-base-chinese")
        elif self.language == 'en' and self.pretrain == 'robert':
            return AutoTokenizer.from_pretrained("roberta-base")
        elif self.language == 'cn' and self.pretrain == 'robert':
            return AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        elif self.language == 'en' and self.pretrain == 'distilbert':
            return AutoTokenizer.from_pretrained("distilbert-base-uncased")
        elif self.language == 'en' and self.pretrain == 'deberta':
            return AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
          
    def tensor_to_label(self, tensor):
        tensor = int(tensor)
        return self.classes[tensor]

    def label_to_tensor(self, label):
        index = self.classes.index(label)
        return torch.tensor(index)

    def _get_text(self, text):
        text = preprocess(text)
        return self.text_tokenizer.encode_plus(text, max_length=self.max_length, padding='max_length', truncation=True,return_tensors='pt')

    def _get_labels(self, label):
        return self.label_to_tensor(label)

    def process_file(self):
        ids = []
        texts = []
        labels = []

        with open(self.label_path, 'r', encoding='utf-8') as f:
            tweets = json.load(f)
            for tweet in tqdm(tweets):
                id = tweet['id']
                text = self._get_text(tweet['tweet']['text'])
                label = self._get_labels(tweet['label'])
                ids.append(id)
                texts.append(text)
                labels.append(label)

        return ids, texts, labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        item = self.texts[idx]
        id = self.ids[idx]
        label = self.labels[idx]
        item['id'] = id
        item['label'] = label

        return item
