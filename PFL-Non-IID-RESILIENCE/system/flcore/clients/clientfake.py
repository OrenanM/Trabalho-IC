from flcore.clients.clientavg import clientAVG
import copy
import torch.nn as nn
from flcore.trainmodel.models import *
import numpy as np
import sys

class ClientFake(clientAVG):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
    
    def client_entropy(self):
        entropy_client = self.calculate_data_entropy()
        return entropy_client
    
    def train(self):
        data_false = np.random.choice([True, False])
        if data_false:
            print(f'client fake: {self.id}')
            self.model = FedAvgCNN(in_features=1, 
                                   num_classes=self.num_classes, dim=1024).to(self.device)
        else:
            super().train()
