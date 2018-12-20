import torch
import torch.nn as nn

class BatchNormMod2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.99,
                 last_gamma=False, use_running_stats_in_training=False):
        super(BatchNormMod2d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.last_gamma = last_gamma
        self.use_running_stats_in_training = use_running_stats_in_training
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
        self.register_buffer('running_var', torch.zeros(1, num_features, 1))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def set_use_running_stats_in_training(self, use_running_stats_in_training):
        self.use_running_stats_in_training = use_running_stats_in_training

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)
        temp = var_in + mean_in ** 2

        if self.training:
            mean_bn = mean_in.mean(0, keepdim=True)
            var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2

            self.running_mean.mul_(self.momentum)
            self.running_mean.add_((1 - self.momentum) * mean_bn.data)
            self.running_var.mul_(self.momentum)
            self.running_var.add_((1 - self.momentum) * var_bn.data)

            if self.use_running_stats_in_training:
                print('use_running_stats_in_training = ' + str(self.use_running_stats_in_training))
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)
        else:
            mean_bn = torch.autograd.Variable(self.running_mean)
            var_bn = torch.autograd.Variable(self.running_var)

        mean = mean_bn
        var = var_bn

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias
