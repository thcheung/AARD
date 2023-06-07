import torch
import torch.nn as nn
from transformers import BertModel, ViTModel, AutoModel , RobertaModel, CLIPTextModel
from transformers.adapters import AdapterConfig , LoRAConfig, PrefixTuningConfig

class MultiNet(nn.Module):
    def __init__(self, hidden_dim=768, class_dim=2, dropout_rate=0.1, language='en',model="bert", pretrain="bert", tuning='fine'):

        super(MultiNet, self).__init__()

        self.class_dim = class_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.language = language
        self.pretrain = pretrain
        self.tuning = tuning

        self.text_encoder = self.get_text_model()
        self.user_encoder = self.get_text_model()
        self.multi_encoder = self.get_text_model()

        self.share_init()

        self.tuning_init()
        
        self.dropout = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(hidden_dim, class_dim)
        self.fc2 = nn.Linear(hidden_dim, class_dim)
        self.fc3 = nn.Linear(hidden_dim, class_dim)

    def get_text_model(self):
        if self.language == 'en' and self.pretrain == 'bert':
            return AutoModel.from_pretrained("bert-base-uncased")
        elif self.language == 'cn' and self.pretrain == 'bert':
            return AutoModel.from_pretrained("bert-base-chinese")
        elif self.language == 'en' and self.pretrain == 'robert':
            return AutoModel.from_pretrained("roberta-base")
        elif self.language == 'cn' and self.pretrain == 'robert':
            return AutoModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
        elif self.language == 'en' and self.pretrain == 'distilbert':
            return AutoModel.from_pretrained("distilbert-base-uncased")
        elif self.language == 'en' and self.pretrain == 'deberta':
            return AutoModel.from_pretrained("microsoft/deberta-v3-base")

    def share_init(self):
        for name, param in list(self.text_encoder.named_parameters()):
            if 'encoder.layer.8' in name or 'encoder.layer.9' in name:
                param.requires_grad = True
                for name2, param2 in list(self.user_encoder.named_parameters()):
                    if name2 == name:
                        param2.requires_grad = True
                        param2 = param
                for name3, param3 in list(self.multi_encoder.named_parameters()):
                    if name3 == name:
                        param3.requires_grad = True
                        param3 = param


    def tuning_init(self):
        if self.tuning == 'fine':
            pass
        elif self.tuning == 'adapter':
            self.adapter_init()
        elif self.tuning == 'lora':
            self.lora_init()
        elif self.tuning == 'prefix':
            self.prefix_init()
        elif self.tuning == 'freeze':
            self.freeze_init()

    def adapter_init(self):
        config = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=8, non_linearity="relu")
        self.text_encoder.add_adapter("bottleneck_adapter", config=config)
        self.text_encoder.set_active_adapters("bottleneck_adapter")
        self.text_encoder.train_adapter("bottleneck_adapter")
        self.user_encoder.add_adapter("bottleneck_adapter", config=config)
        self.user_encoder.set_active_adapters("bottleneck_adapter")
        self.user_encoder.train_adapter("bottleneck_adapter")
        self.multi_encoder.add_adapter("bottleneck_adapter", config=config)
        self.multi_encoder.set_active_adapters("bottleneck_adapter")
        self.multi_encoder.train_adapter("bottleneck_adapter")

    def lora_init(self):
        config = LoRAConfig(r=8, alpha=16)
        self.text_encoder.add_adapter("lora_adapter", config=config)
        self.text_encoder.set_active_adapters("lora_adapter")
        self.text_encoder.train_adapter("lora_adapter")
        self.user_encoder.add_adapter("lora_adapter", config=config)
        self.user_encoder.set_active_adapters("lora_adapter")
        self.user_encoder.train_adapter("lora_adapter")
        self.multi_encoder.add_adapter("lora_adapter", config=config)
        self.multi_encoder.set_active_adapters("lora_adapter")
        self.multi_encoder.train_adapter("lora_adapter")

    def prefix_init(self):
        config = PrefixTuningConfig(flat=False, prefix_length=30)
        self.text_encoder.add_adapter("prefix_tuning", config=config)
        self.text_encoder.set_active_adapters("prefix_tuning")
        self.text_encoder.train_adapter("prefix_tuning")
        self.user_encoder.add_adapter("prefix_tuning", config=config)
        self.user_encoder.set_active_adapters("prefix_tuning")
        self.user_encoder.train_adapter("prefix_tuning")
        self.multi_encoder.add_adapter("prefix_tuning", config=config)
        self.multi_encoder.set_active_adapters("prefix_tuning")
        self.multi_encoder.train_adapter("prefix_tuning")

    def freeze_init(self):
        for name, param in list(self.text_encoder.named_parameters()):
            if name.startswith('pooler') or 'encoder.layer.11' in name or 'encoder.layer.10' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        for name, param in list(self.user_encoder.named_parameters()):
            if name.startswith('pooler') or 'encoder.layer.11' in name or 'encoder.layer.10' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        for name, param in list(self.multi_encoder.named_parameters()):
            if name.startswith('pooler') or 'encoder.layer.11' in name or 'encoder.layer.10' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, batch):
        text = batch['text']
        user = batch['user']

        fuse = torch.cat([text.input_ids[:,0].cuda(),user.input_ids[:,0].cuda()],1)
        fuse_mask = torch.cat([text.attention_mask[:,0].cuda(),user.attention_mask[:,0].cuda()],1)
        fuse_token = torch.cat([torch.zeros_like(text.token_type_ids[:,0].cuda()),torch.ones_like(user.token_type_ids[:,0].cuda())],1)

        x1 = self.text_encoder(input_ids=text.input_ids[:,0].cuda(), attention_mask=text.attention_mask[:,0].cuda(),token_type_ids=torch.zeros_like(text.token_type_ids[:,0].cuda()))
        x2 = self.user_encoder(input_ids=user.input_ids[:,0].cuda(), attention_mask=user.attention_mask[:,0].cuda(),token_type_ids=torch.ones_like(user.token_type_ids[:,0].cuda()))
        x3 = self.multi_encoder(input_ids=fuse, attention_mask=fuse_mask,token_type_ids=fuse_token)

        x1 = x1['last_hidden_state'][:,0]
        x2 = x2['last_hidden_state'][:,0]
        x3 = x3['last_hidden_state'][:,0]

        x1 = self.fc1(x1)
        x2 = self.fc2(x2)
        x3 = self.fc3(x3)
       
        return x1 , x2 , x3