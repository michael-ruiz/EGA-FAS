import torch
from torch.autograd.variable import Variable
import numpy as np
import os
import time
USE_GPU = torch.cuda.is_available()


def calc_flops(model, config, input_size):
    global USE_GPU

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups) * (
            2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_conv.append(flops)

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        if self.bias is None:
            bias_ops = 0
        else:
            bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement())

    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_pooling.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            return
        for c in childrens:
            foo(c)

    multiply_adds = False
    list_conv, list_bn, list_relu, list_linear, list_pooling = [], [], [], [], []
    foo(model)
    batch_size = 1
    channel_num = 3
  
    if config.is_Multi:  # 多模态
        num_mod = getattr(config, 'num_modalities', 3)
        channel_num = 3 * num_mod
    if '0.4.' in torch.__version__:
        if USE_GPU:
            input = torch.cuda.FloatTensor(torch.rand(batch_size, channel_num, input_size, input_size).cuda())
        else:
            input = torch.FloatTensor(torch.rand(batch_size, channel_num, input_size, input_size))
    else:
        input = Variable(torch.rand(batch_size, channel_num, input_size, input_size), requires_grad=True)
        # print(input.shape)
    _ = model(input)

    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))

    print('  + Number of FLOPs: %f' % (total_flops))
    print('  + Number of FLOPs: %.2fG' % (total_flops / 1e9))
    print('  + Number of FLOPs: %.2fM' % (total_flops / 1e6 / 2))
    return total_flops

def count_params(model, config, log_file=False, input_size=224):

    total_flops = calc_flops(model, config, input_size)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    params_M = params / 1e6
    tf_G = total_flops / 1e9
    tf_M = total_flops / 1e6 / 2

    print('The network has {} params.'.format(params))
    line = 'The network has {}M  and {} params.\n' \
           '+ Number of FLOPs: {}, {}M, {}G\n'.format(params_M, params, total_flops, tf_M, tf_G)
    if log_file:
        log_file.write(line)

