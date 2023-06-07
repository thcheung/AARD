import torch
import torch.nn as nn
from transformers import BertModel, ViTModel, AutoModel , RobertaModel, CLIPTextModel
from transformers.adapters import AdapterConfig , LoRAConfig, PrefixTuningConfig

class UserNet(nn.Module):
    def __init__(self,  hidden_dim=768, class_dim=2, dropout_rate=0.1, language='en', pretrain="bert", tuning='fine'):

        super(UserNet, self).__init__()

        self.class_dim = class_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.language = language
        self.pretrain = pretrain
        self.tuning = tuning

        self.text_encoder = self.get_text_model()
        self.tuning_init()

        self.dropout = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, class_dim)

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

    def lora_init(self):
        config = LoRAConfig(r=8, alpha=16)
        self.text_encoder.add_adapter("lora_adapter", config=config)
        self.text_encoder.set_active_adapters("lora_adapter")
        self.text_encoder.train_adapter("lora_adapter")

    def prefix_init(self):
        config = PrefixTuningConfig(flat=False, prefix_length=30)
        self.text_encoder.add_adapter("prefix_tuning", config=config)
        self.text_encoder.set_active_adapters("prefix_tuning")
        self.text_encoder.train_adapter("prefix_tuning")

    def freeze_init(self):
        for name, param in list(self.text_encoder.named_parameters()):
            if name.startswith('pooler') or 'encoder.layer.11' in name or 'encoder.layer.10' in name:
                param.requires_grad = True
            else:
                pass
            
    def forward(self, batch):

        text = batch['text']
        user = batch['user']

        x1 = self.text_encoder(input_ids=text.input_ids[:,0].cuda(), attention_mask=text.attention_mask[:,0].cuda())
        x2 = self.text_encoder(input_ids=user.input_ids[:,0].cuda(), attention_mask=user.attention_mask[:,0].cuda())

        x1 = x1['last_hidden_state'][:,0]
        x2 = x2['last_hidden_state'][:,0]

        x = torch.cat([x1,x2],-1)

        x = torch.tanh(self.fc1(x))

        return self.fc2(x)