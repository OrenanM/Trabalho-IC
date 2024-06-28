from flcore.clients.clientavg import clientAVG
import copy
import torch.nn as nn
from flcore.trainmodel.models import *
import numpy as np
import sys

class ClientFake(clientAVG):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.probabity_cf = args.probabity_cf
    
    def client_entropy(self):
        entropy_client = self.calculate_data_entropy()
        return entropy_client
    

    def send_local_model(self): 
        
        self.send_fake = False

        weights_cf = [self.probabity_cf, 1 - self.probabity_cf]
        data_false = np.random.choice([True, False], p=weights_cf)

        if data_false:
            self.send_fake = True
            print(f'cliente fake: {self.id}')
            self.send_model_false = True

            if "mnist" in self.dataset:
                model_fake = FedAvgCNN(in_features=1, num_classes=self.num_classes, dim=1024).to(self.device)
            elif "Cifar10" in self.dataset:
                model_fake = FedAvgCNN(in_features=3, num_classes=self.num_classes, dim=1600).to(self.device)
             
            return model_fake
        
        return self.model