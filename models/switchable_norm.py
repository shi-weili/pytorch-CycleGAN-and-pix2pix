import torch
import torch.nn as nn


class SwitchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.997, using_moving_average=True):
        super(SwitchNorm1d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.weight = nn.Parameter(torch.ones(1, num_features))
        self.bias = nn.Parameter(torch.zeros(1, num_features))
        self.mean_weight = nn.Parameter(torch.ones(2))
        self.var_weight = nn.Parameter(torch.ones(2))
        self.register_buffer('running_mean_bn', torch.zeros(1, num_features))
        self.register_buffer('running_var_bn', torch.zeros(1, num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean_bn.zero_()
        self.running_var_bn.zero_()
        self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        mean_ln = x.mean(1, keepdim=True)
        var_ln = x.var(1, keepdim=True)

        if self.training:
            mean_bn = x.mean(0, keepdim=True)
            var_bn = x.var(0, keepdim=True)
            if self.using_moving_average:
                self.running_mean_bn.mul_(self.momentum)
                self.running_mean_bn.add_((1 - self.momentum) * mean_bn.data)
                self.running_var_bn.mul_(self.momentum)
                self.running_var_bn.add_((1 - self.momentum) * var_bn.data)
            else:
                self.running_mean_bn.add_(mean_bn.data)
                self.running_var_bn.add_(mean_bn.data ** 2 + var_bn.data)
        else:
            mean_bn = torch.autograd.Variable(self.running_mean_bn)
            var_bn = torch.autograd.Variable(self.running_var_bn)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        mean = mean_weight[0] * mean_ln + mean_weight[1] * mean_bn
        var = var_weight[0] * var_ln + var_weight[1] * var_bn

        x = (x - mean) / (var + self.eps).sqrt()
        return x * self.weight + self.bias

class SwitchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.997, using_moving_average=True, using_bn=True,
                 last_gamma=False):
        super(SwitchNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        if self.using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))
        self.register_buffer('running_mean_in', torch.zeros(1, num_features, 1))
        self.register_buffer('running_var_in', torch.zeros(1, num_features, 1))
        self.register_buffer('running_mean_ln', torch.zeros(1, num_features, 1))
        self.register_buffer('running_var_ln', torch.zeros(1, num_features, 1))
        if self.using_bn:
            self.register_buffer('running_mean_bn', torch.zeros(1, num_features, 1))
            self.register_buffer('running_var_bn', torch.zeros(1, num_features, 1))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean_in.zero_()
        self.running_var_in.zero_()
        self.running_mean_ln.zero_()
        self.running_var_ln.zero_()
        if self.using_bn:
            self.running_mean_bn.zero_()
            self.running_var_bn.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        N, C, H, W = x.size()
        x = x.view(N, C, -1)

        if self.training:
            print("Switchable norm is in traning mode!")
        else:
            print("Switchable norm is in testing mode!")

        if self.training:
            mean_in = x.mean(-1, keepdim=True)
            var_in = x.var(-1, keepdim=True)

            mean_ln = mean_in.mean(1, keepdim=True)
            temp = var_in + mean_in ** 2
            var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

            if self.using_moving_average:
                self.running_mean_in.mul_(self.momentum)
                self.running_mean_in.add_((1 - self.momentum) * mean_in.data)
                self.running_var_in.mul_(self.momentum)
                self.running_var_in.add_((1 - self.momentum) * var_in.data)
            
                self.running_mean_ln.mul_(self.momentum)
                self.running_mean_ln.add_((1 - self.momentum) * mean_ln.data)
                self.running_var_ln.mul_(self.momentum)
                self.running_var_ln.add_((1 - self.momentum) * var_ln.data)

            else:
                self.running_mean_in.add_(mean_in.data)
                self.running_var_in.add_(mean_in.data ** 2 + var_in.data)

                self.running_mean_ln.add_(mean_ln.data)
                self.running_var_ln.add_(mean_ln.data ** 2 + var_ln.data)
        
        else:
            mean_in = torch.autograd.Variable(self.running_mean_in)
            var_in = torch.autograd.Variable(self.running_var_in)

            mean_ln = torch.autograd.Variable(self.running_mean_ln)
            var_ln = torch.autograd.Variable(self.running_var_ln)

        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean_bn.mul_(self.momentum)
                    self.running_mean_bn.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var_bn.mul_(self.momentum)
                    self.running_var_bn.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean_bn.add_(mean_bn.data)
                    self.running_var_bn.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean_bn)
                var_bn = torch.autograd.Variable(self.running_var_bn)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class SwitchNorm3d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.997, using_moving_average=True, using_bn=True,
                 last_gamma=False):
        super(SwitchNorm3d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1, 1))
        if self.using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))
        if self.using_bn:
            self.register_buffer('running_mean_bn', torch.zeros(1, num_features, 1))
            self.register_buffer('running_var_bn', torch.zeros(1, num_features, 1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean_bn.zero_()
            self.running_var_bn.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        N, C, D, H, W = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)

        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean_bn.mul_(self.momentum)
                    self.running_mean_bn.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var_bn.mul_(self.momentum)
                    self.running_var_bn.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean_bn.add_(mean_bn.data)
                    self.running_var_bn.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean_bn)
                var_bn = torch.autograd.Variable(self.running_var_bn)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln

        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, D, H, W)
        return x * self.weight + self.bias
