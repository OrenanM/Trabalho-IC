from flcore.clients.clientbase import Client
import copy
import torch.nn as nn
from flcore.trainmodel.models import *

class ClientFake(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
    
    def client_entropy(self):
        entropy_client = self.calculate_data_entropy()
        return entropy_client
    
    
    def train(self):
        self.model = FedAvgCNN(in_features=1, num_classes=self.num_classes, dim=1024).to(self.device)

    def set_parameters(self, model):
        pass