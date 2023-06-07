from .textNet import TextNet
from .userNet import UserNet
from .oneNet import OneNet
from .multiNet import MultiNet
from .scoreNet import ScoreNet
from .missNet import MissNet
from .crossNet import CrossNet

def get_model(model_name, hidden_dim, classes, dropout, language, pretrain, tuning):
    if model_name == 'text':
        return TextNet(hidden_dim, classes,
                            dropout, language=language,pretrain=pretrain, tuning=tuning)
    if model_name == 'user':
        return UserNet(hidden_dim, classes,
                            dropout, language=language,pretrain=pretrain, tuning=tuning)
    if model_name == 'one':
        return OneNet(hidden_dim, classes,
                            dropout, language=language,pretrain=pretrain, tuning=tuning)
    if model_name == 'multi':
        return MultiNet(hidden_dim, classes,
                            dropout, language=language,pretrain=pretrain, tuning=tuning)
    if model_name == 'score':
        return ScoreNet(hidden_dim, classes,
                            dropout, language=language,pretrain=pretrain, tuning=tuning)
    if model_name == 'miss':
        return MissNet(hidden_dim, classes,
                            dropout, language=language,pretrain=pretrain, tuning=tuning)
    if model_name == 'cross':
        return CrossNet(hidden_dim, classes,
                            dropout, language=language,pretrain=pretrain, tuning=tuning)
