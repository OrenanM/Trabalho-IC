from flcore.clients.clientbase import Client

class ClientFake(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
    
    def client_entropy(self):
        entropy_client = self.calculate_data_entropy()
        return entropy_client
    
    def train(self):
        pass

    def set_parameters(self, model):
        pass