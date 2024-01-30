from serverbase import Server

class Server_modif(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
    

if __name__ == "__main__":
    servidor = Server_modif()
    id = servidor.select_clients()

    print(id)