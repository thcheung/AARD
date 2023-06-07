import argparse
from ast import Mult
import torch
import random
import os
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from data_loader import TextDataset
from model import get_model
from utils import print_metrics
from experiment import get_experiment
from PIL import Image
import matplotlib.pyplot as plt
import warnings

os.environ["CUDA_VISIBLE_DEVICES"]="0"
warnings.filterwarnings("ignore")

RANDOM_SEED = 0

torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(
    description='Multimodal Rumor Detection and Verification')

parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 16)')

parser.add_argument('--hidden_dim', type=int, default=256, metavar='N',
                    help='hidden dimension (default: 768)')

parser.add_argument('--max_len', type=int, default=50, metavar='N',
                    help='maximum length of the conversation (default: 32)')

parser.add_argument('--dropout', type=float, default=0.5, metavar='N',
                    help='dropout rate (default: 0.5)')

parser.add_argument('--model', type=str, default="text", metavar='N',
                    help='model name')

parser.add_argument('--experiment', type=str, metavar='N',
                    help='experiment name')

parser.add_argument('--pretrain', type=str, default="bert",  metavar='N',
                    help='pretrained model names')

parser.add_argument('--tuning', type=str, default="fine",  metavar='N',
                    help='tuning methods')


args = parser.parse_args()


def train():

    experiment = get_experiment(args.experiment)

    language = experiment["language"]

    classes = experiment["classes"]

    model = get_model(args.model,args.hidden_dim, len(classes),
                         args.dropout, language=language, pretrain=args.pretrain, tuning=args.tuning)

    model = nn.DataParallel(model)

    model = model.to(device)

    test_count = 0
    test_predicts = []
    test_labels = []

    for fold in range(1,6):
        root_dir = os.path.join(experiment["root_dir"], str(fold))
        test_path = os.path.join(root_dir, "test.json")

        test_dataset = TextDataset(
            test_path, classes, train=False, language=language, max_length=args.max_len, pretrain=args.pretrain)

        test_dataloader = DataLoader(
            dataset=test_dataset, batch_size=args.batch_size, shuffle=True)

        comment = f'{args.model}_{args.experiment}_{args.fold}_{args.pretrain}_{args.tuning}'

        checkpoint_dir = os.path.join("checkpoints/",comment)

        checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
        
        model.module.load_state_dict(torch.load(checkpoint_path))

        model.eval()

        for i, batch in enumerate(tqdm(test_dataloader)):

            with torch.no_grad():
                outputs = model(batch)
                labels = batch['label'].cuda()

            _, preds = torch.max(outputs, 1)
        
            test_count += labels.size(0)

            for pred in preds.tolist():
                test_predicts.append(pred)
            for lab in labels.tolist():
                test_labels.append(lab)

        del test_dataset
        del test_dataloader

    print_metrics(test_labels, test_predicts)

if __name__ == "__main__":
    train()
