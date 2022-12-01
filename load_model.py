from models import LinearModel, LSTMModel, LinearLSTMModel, LSTMModelcopy1, LSTMBertModel
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, cfg, args):
        super(Model, self).__init__()
        
        self.cfg = cfg
        self.args = args   

    def forward(self):
        if self.args.load_model == 'linear':
            model = LinearModel(self.cfg, self.args)
        elif self.args.load_model == 'lstm':
            model = LSTMModel(self.cfg, self.args)
        elif self.args.load_model == 'linearlstm':
            model = LinearLSTMModel(self.cfg, self.args)
        elif self.args.load_model == 'lstmcopy1':
            model = LSTMModelcopy1(self.cfg, self.args)
        elif self.args.load_model == 'lstmbert':
            model = LSTMBertModel(self.cfg, self.args)
            
        return model