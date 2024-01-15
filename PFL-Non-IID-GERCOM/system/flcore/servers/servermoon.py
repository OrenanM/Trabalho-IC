from flcore.clients.clientmoon import clientMOON
from flcore.servers.serverbase import Server
from utils.data_utils import read_client_data
from threading import Thread
import time
import sys


class MOON(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientMOON)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []


    def train(self):
        round_no_improve = 0
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            if i == 0:
                self.send_models()
            

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

            with open(f"{self.algorithm}_metrics_{self.entropy}_{self.weigths_clients}_{self.fall_tolerance}.txt", "a") as arquivo:
                arquivo.write(f"{self.Budget[-1]} \n")

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
            
            if (i > 0) :
                if (self.rs_test_acc[i-1] + self.fall_tolerance) < (self.rs_test_acc[i]):
                    round_no_improve = 0
                else:
                    round_no_improve += 1
            
            if (round_no_improve >= 5):
                print(f'Early stopping no round {i}.')
                self.send_models()
                with open(f"{self.algorithm}_metrics_{self.entropy}_{self.weigths_clients}_{self.fall_tolerance}_exe.txt", "a") as arquivo:
                    arquivo.write(f"{self.times}, {i}, {sys.getsizeof(self.global_model)}, {sys.getsizeof(self.global_model.parameters())} \n")
                round_no_improve = 0

        print("\nBest accuracy.")