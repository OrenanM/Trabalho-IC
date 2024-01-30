import torch
import os
import numpy as np
import h5py
import copy
import time
import random

import sys
import pandas as pd
import collections

from utils.data_utils import read_client_data
from utils.dlg import DLG
from flcore.cluster.cka import CKA
from sklearn.cluster import SpectralClustering

class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = 100
        self.auto_break = args.auto_break

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch = args.fine_tuning_epoch

        #atributos novos
        self.clients_weigths = []
        self.current_round = -1
        self.cluster = args.cluster
        self.num_select = args.num_select
        self.num_clusters = args.num_clusters
        self.clustering_period = args.clustering_period
        self.weigth_select = args.weigth_select
        self.select_conseq = args.select_conseq
        self.clients_cluster = dict()
        self.weigths_clients = args.weigths_clients
        self.entropy = args.entropy
        self.type_select = args.type_select
        self.patience = args.patience
        self.fall_tolerance = args.fall_tolerance
        self.weigth_size_entropy = args.weigth_size_entropy

    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    
# ***********************************************************************************************************************************
    def select_random(self):
        '''Seleciona os clientes de forma aleatória'''
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)
        else:
            self.current_num_join_clients = self.current_num_join_clients
        
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))
        return selected_clients
    
    def calculate_entropy(self):
        entropies = np.array([client.client_entropy() for client in self.clients])
        entropies[np.isnan(entropies)] = 0

        for client, entropy in zip(self.clients, entropies):
            client.entropy = entropy

    def select_clients(self):
        if self.current_round == 0:
            self.calculate_entropy()
        num_select_size = round(len(self.clients) * 0.2)
        num_select_entropy = round(len(self.clients) * 0.1)

        weigth_size = np.array([client.train_samples for client in self.clients])
        weigth_size = weigth_size/weigth_size.sum()

        clients_select_size = np.random.choice(self.clients, num_select_size, 
                                               p = weigth_size, replace=False)

        weigth_entropy = np.array([client.entropy for client in clients_select_size])
        weigth_entropy = weigth_entropy/weigth_entropy.sum()

        clients_select_entropy = np.random.choice(clients_select_size, num_select_entropy,
                                                  p=weigth_entropy, replace=False)
        
        return clients_select_entropy

    def select_clients_1(self):
        if self.current_round == 0:
            self.calculate_entropy()

        num_select_size = round(len(self.clients) * 0.3)
        num_select_entropy = round(len(self.clients) * 0.2)

        pre_selected_clients = self.selected_clients.copy()
        size_selected = round(len(pre_selected_clients) * 0.6)
        not_selected = np.random.choice(pre_selected_clients, size_selected)
        print(f"selecionados anteriormente: {[client.id for client in self.selected_clients]}")
        print(f"não selecionar: {[client.id for client in not_selected]}")
        select_clients = [client for client in self.clients if client not in not_selected]

        weigth_size = np.array([client.train_samples for client in select_clients])
        weigth_size = weigth_size/weigth_size.sum()

        clients_select_size = np.random.choice(select_clients, num_select_size, 
                                               p = weigth_size, replace=False)

        weigth_entropy = np.array([client.entropy for client in clients_select_size])
        weigth_entropy = weigth_entropy/weigth_entropy.sum()

        clients_select_entropy = np.random.choice(clients_select_size, num_select_entropy,
                                                  p=weigth_entropy, replace=False)
        print(f"selecionados: {[client.id for client in clients_select_entropy]}")
        
        return clients_select_entropy
    
    def select_clients_2(self):
        if self.current_round == 0:
            self.calculate_entropy()

        num_select_size = round(len(self.clients) * 0.3)
        num_select_entropy = round(len(self.clients) * 0.2)

        pre_selected_clients = self.selected_clients.copy()
        size_selected = round(len(pre_selected_clients) * 0.5)
        not_selected = np.random.choice(pre_selected_clients, size_selected, replace=False)

        print(f"selecionados anteriormente: {[client.id for client in self.selected_clients]}")
        print(f"não selecionar: {[client.id for client in not_selected]}")

        select_clients = [client for client in self.clients if client not in not_selected]

        weigth_size = np.array([client.train_samples for client in select_clients])
        weigth_size = weigth_size/weigth_size.sum()

        weigth_size_ajusted = weigth_size ** 3
        weigth_size_ajusted = weigth_size_ajusted/sum(weigth_size_ajusted)
        weigth_size_ajusted = weigth_size_ajusted/sum(weigth_size_ajusted)

        clients_select_size = np.random.choice(select_clients, num_select_size, 
                                               p = weigth_size_ajusted, replace=False)

        weigth_entropy = np.array([client.entropy for client in clients_select_size])
        weigth_entropy = weigth_entropy/weigth_entropy.sum()

        weigth_entropy_ajusted = weigth_entropy ** 3
        weigth_entropy_ajusted = weigth_entropy_ajusted/sum(weigth_entropy_ajusted)
        weigth_entropy_ajusted = weigth_entropy_ajusted/sum(weigth_entropy_ajusted)

        clients_select_entropy = np.random.choice(clients_select_size, num_select_entropy,
                                                  p=weigth_entropy_ajusted, replace=False)
        print(f"selecionados: {[client.id for client in clients_select_entropy]}")
        
        return clients_select_entropy
    
    def select_clients_3(self):
        if self.current_round == 0:
            self.calculate_entropy()


        num_select_size = round(len(self.clients) * 0.2)
        num_select_entropy = round(len(self.clients) * 0.1)
        
        pre_selected_clients = self.selected_clients.copy()
        size_selected = round(len(pre_selected_clients) * 0)

        sort_size_clients = sorted(self.selected_clients, key = lambda client: client.entropy)

        not_selected = sort_size_clients[:size_selected]

        print(f"selecionados anteriormente: {[client.id for client in self.selected_clients]}")
        print(f"não selecionar: {[client.id for client in not_selected]}")

        select_clients = [client for client in self.clients if client not in not_selected]

        weigth_size = np.array([client.train_samples for client in select_clients])
        weigth_size = weigth_size/weigth_size.sum()

        weigth_size_ajusted = weigth_size ** 5
        weigth_size_ajusted = weigth_size_ajusted/sum(weigth_size_ajusted)
        weigth_size_ajusted = weigth_size_ajusted/sum(weigth_size_ajusted)

        clients_select_size = np.random.choice(select_clients, num_select_size, 
                                               p = weigth_size_ajusted, replace=False)

        weigth_entropy = np.array([client.entropy for client in clients_select_size])
        weigth_entropy = weigth_entropy/weigth_entropy.sum()

        weigth_entropy_ajusted = weigth_entropy ** 5
        weigth_entropy_ajusted = weigth_entropy_ajusted/sum(weigth_entropy_ajusted)
        weigth_entropy_ajusted = weigth_entropy_ajusted/sum(weigth_entropy_ajusted)

        clients_select_entropy = np.random.choice(clients_select_size, num_select_entropy,
                                                  p=weigth_entropy_ajusted, replace=False)
        print(f"selecionados: {[client.id for client in clients_select_entropy]}")
        
        return clients_select_entropy
    
    def select_clients_4(self):
        if self.current_round == 0:
            self.calculate_entropy()


        num_select_size = round(len(self.clients) * 0.3)
        num_select_entropy = round(len(self.clients) * 0.2)
        
        pre_selected_clients = self.selected_clients.copy()
        size_selected = round(len(pre_selected_clients) * 0.4)

        sort_size_clients = sorted(self.selected_clients, key = lambda client: client.entropy)

        not_selected = sort_size_clients[:size_selected]

        print(f"selecionados anteriormente: {[client.id for client in self.selected_clients]}")
        print(f"não selecionar: {[client.id for client in not_selected]}")

        select_clients = [client for client in self.clients if client not in not_selected]

        weigth_size = np.array([client.train_samples for client in select_clients])
        weigth_size = weigth_size/weigth_size.sum()

        weigth_size_ajusted = np.sqrt(weigth_size)
 
        weigth_size_ajusted = weigth_size_ajusted/sum(weigth_size_ajusted)
        weigth_size_ajusted = weigth_size_ajusted/sum(weigth_size_ajusted)

        clients_select_size = np.random.choice(select_clients, num_select_size, 
                                               p = weigth_size_ajusted, replace=False)

        weigth_entropy = np.array([client.entropy for client in clients_select_size])
        weigth_entropy = weigth_entropy/weigth_entropy.sum()

        weigth_entropy_ajusted = np.sqrt(weigth_entropy)
        
        weigth_entropy_ajusted = weigth_entropy_ajusted/sum(weigth_entropy_ajusted)
        weigth_entropy_ajusted = weigth_entropy_ajusted/sum(weigth_entropy_ajusted)

        clients_select_entropy = np.random.choice(clients_select_size, num_select_entropy,
                                                  p=weigth_entropy_ajusted, replace=False)
        print(f"selecionados: {[client.id for client in clients_select_entropy]}")
        
        return clients_select_entropy
    
    def select_entropy(self):
        num_select_entropy = round(len(self.clients) * 0.1)
        sort_entropy = sorted(self.clients, key = lambda client: client.entropy, reverse=True)
        clients_select_size = sort_entropy[:num_select_entropy]
        return clients_select_size
    
    def select_size(self):
        num_select_size = round(len(self.clients) * 0.1)
        
        sort_size = sorted(self.clients, key = lambda client: client.train_samples, reverse=True)
        clients_select_size = sort_size[:num_select_size]
        print(f"selecionados: {[client.id for client in clients_select_size]}")
        return clients_select_size
    
    def select_size_entropy(self):
        num_select_size = round(len(self.clients) * 0.2)
        
        sort_size = sorted(self.clients, key = lambda client: client.train_samples, reverse=True)
        clients_select_size = sort_size[:num_select_size]

        num_select_entropy = round(len(self.clients) * 0.1)
        sort_entropy = sorted(clients_select_size, key = lambda client: client.entropy, reverse=True)
        clients_select_entropy = sort_entropy[:num_select_entropy]

        return clients_select_entropy
    
    def select_entropy_size(self):
        num_select_entropy = round(len(self.clients) * 0.2)
        sort_entropy = sorted(self.clients, key = lambda client: client.entropy, reverse=True)
        clients_select_entropy = sort_entropy[:num_select_entropy]

        num_select_size = round(len(self.clients) * 0.1)
        
        sort_size = sorted(clients_select_entropy, key = lambda client: client.train_samples, reverse=True)
        clients_select_size = sort_size[:num_select_size]
        return clients_select_size
    
    def valueOfList(self, value):
        """
    Retorna uma lista que contém todos os valores inteiros e flutuantes contidos em uma estrutura de dados aninhada.

    Args:
        value (list, np.ndarray, torch.Tensor): A estrutura de dados da qual você deseja extrair valores inteiros e flutuantes.

    Returns:
        list: Uma lista contendo todos os valores inteiros e flutuantes encontrados na estrutura de dados.

    Note:
        - A função aceita estruturas de dados aninhadas (listas dentro de listas).
        - Se `value` for uma instância de `torch.Tensor`, ela será convertida em um array numpy e depois em uma lista.
        - Se `value` for uma instância de `np.ndarray`, ela será convertida em uma lista.
        - A função recursivamente percorre estruturas de dados aninhadas para extrair todos os valores inteiros e flutuantes.
        - Os valores inteiros e flutuantes extraídos são adicionados à lista `valueList`.

    Exemplos:
        >>> obj = SuaClasse()
        >>> tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> lista = [1, [2, 3.5], [[4, 5.0], 6]]
        >>> obj.valueOfList(tensor)
        [1.0, 2.0, 3.0, 4.0]
        >>> obj.valueOfList(lista)
        [1, 2, 3.5, 4, 5.0, 6]
    """

        valueList = list()
    
        if type(value) == torch.Tensor:
            value = value.cpu().numpy()
            value = value.tolist()
            self.valueOfList(value)

        if type(value) == np.ndarray:
            value = value.tolist()
            self.valueOfList(value)

        if type(value) == list and len(value) > 0:
            for i in range(len(value)):
                if type(value[i]) == list and len(value[i]) > 0:
                    valueList += self.valueOfList(value[i])

                elif type(value[i]) == int or type(value[i]) == float:
                    valueList.append(value[i])

            return valueList
        
    def print_cluster_clientes(self):
        '''Mostra o dataframe que relacionado clusters e clientes'''
        
        data = {'Cluster': [], 'Clientes': []}
        
        for label, clients in self.clients_cluster.items():
            data['Cluster'].append(label)
            data['Clientes'].append(', '.join(str(client.id) for client in clients))

        df = pd.DataFrame(data)
        df = df.sort_values(by='Cluster')
        print(df.to_string(index=False))

# ***********************************************************************************************************************************
    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = self.selected_clients

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0

        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
                
            except ZeroDivisionError:
                client_time_cost = 0

            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)

        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
        
        return active_clients

    def aggregate_parameters(self):
        
        assert (len(self.uploaded_models) > 0)
        self.global_model = copy.deepcopy(self.uploaded_models[0]) 
        
        for param in self.global_model.parameters():
            param.data.zero_()
        
        self.clients_weigths = []
        
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)  
    
    def add_parameters(self, w, client_model):
        weigths = []
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            weigths.extend(self.valueOfList(client_param.data))
            server_param.data += client_param.data.clone() * w

        self.clients_weigths.append(weigths)
        

    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)
        
    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None, nnc = 0):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        with open(f"{self.algorithm}_metrics_{self.weigth_size_entropy}.txt", "a") as arquivo:
            arquivo.write(f"{test_acc}, {train_loss} \n")
        

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

        if (self.cluster != None) and (self.current_round > 0):
            labels = list(set(self.labels))
            ids_selected = [x.id for x in self.selected_clients]
            labels_selected = list(set([x.cluster for x in self.selected_clients]))
            self.print_cluster_clientes()
            print(f'labels: {labels}')
            print(f"Selected cluster: {labels_selected}")
            print(f"Selected clients: {ids_selected}")
        print(f'Percentage selected clients: {(len(self.selected_clients)/self.num_clients * 100)} %')

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accurancy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
            
            # items.append((client_model, origin_grad, target_inputs))
                
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=False, 
                            send_slow=False)
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    # evaluating on new clients
    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc 