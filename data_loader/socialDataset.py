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
import torch.nn.functional as F

class SocialDataset(Dataset):
    def __init__(self, label_path, classes, train=False, max_length=64, language='en', pretrain='bert'):
        self.train = train
        self.label_path = label_path
        self.max_length = max_length
        self.language = language
        self.classes = classes
        self.pretrain = pretrain
        self.text_tokenizer = self._get_text_tokenizer()
        self.items = self.process_file()

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
    
    def _get_user(self, user):
        user = preprocess(user)
        return self.text_tokenizer.encode_plus(user, max_length=int(self.max_length/2), padding='max_length', truncation=True,return_tensors='pt')


    def _get_labels(self, label):
        return self.label_to_tensor(label)
    
    def _get_text_metrics(self,metrics):
        metric_tensors = [metrics["retweet_count"] , metrics["reply_count"] , metrics["like_count"], metrics["quote_count"]]
        metric_tensors = torch.tensor(metric_tensors).float()
        return F.normalize(metric_tensors,dim=-1)

    def _get_user_metrics(self,metrics):
        metric_tensors = [metrics["followers_count"] , metrics["following_count"] , metrics["tweet_count"], metrics["listed_count"]]
        metric_tensors = torch.tensor(metric_tensors).float()
        return F.normalize(metric_tensors,dim=-1)

    def process_file(self):
        items = []
        with open(self.label_path, 'r', encoding='utf-8') as f:
            tweets = json.load(f)
            for tweet in tqdm(tweets):
                id = tweet['id']
                text = self._get_text(tweet['tweet']['text'])
                user = self._get_user(tweet['user']['description'])
                label = self._get_labels(tweet['label'])
                text_metrics = self._get_text_metrics(tweet['tweet']['public_metrics'])
                user_metrics = self._get_user_metrics(tweet['user']['public_metrics'])
                item = {"id": id, "text": text, "user": user, "label": label, "text_metrics": text_metrics, "user_metrics":user_metrics}
                items.append(item)

        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return item
