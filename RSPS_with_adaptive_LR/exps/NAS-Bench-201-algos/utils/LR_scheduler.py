impot torch
import torch.nn as nn

class LowParamSchedule(torch.optim.lr_scheduler.LambdaLR):

    def __init__(self, optimizer, epochs, last_epoch=-1, eta_min = 0):

        def lr_lambda(step):
            y = ((step-epochs) ** 2 ) / (epochs ** 2 / (1 - eta_min)) + eta_min
            return float(y)

        super(LowParamSchedule, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)

class HighParamSchedule(torch.optim.lr_scheduler.LambdaLR):

    def __init__(self, optimizer,epochs, last_epoch=-1, eta_min = 0):

        def lr_lambda(step):
            y = ((-step+epochs) ** 0.5 ) / (epochs ** 0.5 / (1 - eta_min)) + eta_min
            return float(y)
        super(HighParamSchedule, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)

class MidParamSchedule(torch.optim.lr_scheduler.LambdaLR):

    def __init__(self, optimizer,epochs, last_epoch=-1, eta_min = 0):

        def lr_lambda(step):
            y = ((-step+epochs)) / (epochs / (1 - eta_min)) + eta_min
            return float(y)
        super(MidParamSchedule, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)

class expSchedule1(torch.optim.lr_scheduler.LambdaLR):

    def __init__(self, optimizer, epochs, last_epoch=-1, eta_min = 0):

        def lr_lambda(step):
            y = torch.exp(torch.tensor((-step/50)))
            return float(y)

        super(expSchedule1, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)

class logSchedule1(torch.optim.lr_scheduler.LambdaLR):

    def __init__(self, optimizer, epochs, last_epoch=-1, eta_min = 0):

        def lr_lambda(step):
            y = torch.log(torch.tensor(-50 * (step-250))) / torch.log(torch.tensor(50*250))
            y = max(y, 0)
            return float(y)

        super(logSchedule1, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)

class AdaptiveParamSchedule(torch.optim.lr_scheduler.LambdaLR):

    def __init__(self, optimizer,epochs, last_epoch=-1, eta_min = 0):
        self.exp_coeff = 1
        self.cur_ep = 0

        def lr_lambda(step):
            y = ((-self.cur_ep+epochs)) ** self.exp_coeff / (epochs ** self.exp_coeff/ (1 - eta_min)) + eta_min
            return float(y)
        super(AdaptiveParamSchedule, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)r
