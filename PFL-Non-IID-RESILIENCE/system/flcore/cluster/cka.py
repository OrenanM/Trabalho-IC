import torch

class CKA():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def linear_HSIC(self, X, Y):
        X = X.to(self.device)
        Y = Y.to(self.device)
        
        x_center = (X.t() - X.mean())
        y_center = (Y.t() - Y.mean())
        cov_xy = torch.sum(torch.mm(x_center, y_center.t()))
        return torch.abs(cov_xy) ** 2

    def linear_CKA(self, X, Y):
        X = X.to(self.device)
        Y = Y.to(self.device)

        hsic = self.linear_HSIC(X, Y)

        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)
