import os
import sys
import time
import glob
import numpy as np
import pickle
import torch
import logging
import argparse
import torch
import random
from copy import deepcopy
import tqdm
from torch.autograd import Variable
import collections
import sys
from network import MobileNetV2_search
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import random_split
import random


parser = argparse.ArgumentParser()
parser.add_argument('--log-dir', type=str, default='log')
parser.add_argument('--file_name', type=str, default='RSPS')
parser.add_argument('--max-epochs', type=int, default=20)
parser.add_argument('--select-num', type=int, default=10)
parser.add_argument('--population-num', type=int, default=50)
parser.add_argument('--m_prob', type=float, default=0.1)
parser.add_argument('--crossover-num', type=int, default=25)
parser.add_argument('--mutation-num', type=int, default=25)
parser.add_argument('--flops_up_limit', type=float, default=330 * 1e6)
parser.add_argument('--flops_low_limit', type=float, default=300 * 1e6)
parser.add_argument('--max-train-iters', type=int, default=200)
parser.add_argument('--max-test-iters', type=int, default=40)
parser.add_argument('--train-batch-size', type=int, default=128)
parser.add_argument('--test-batch-size', type=int, default=200)
parser.add_argument('--train-dir', type=str, default='../../../../dataset/ILSVRC2012/train', help='path to training dataset')
parser.add_argument('--val-dir', type=str, default='../../../../dataset/ILSVRC2012/val', help='path to validation dataset')

args = parser.parse_args()


import logging

logging.basicConfig(filename=os.path.join(f'logs/{args.file_name}', "log.txt"),
                    level=logging.INFO,
                    format='')


layer_widths = [6, 7, 7, 7, 6, 7, 7, 7, 6, 7, 7, 7, 6, 7, 7, 7, 6, 7, 7, 7, 6]

# def get_rand_arch():
#     rand_arch = []
#     for i in range(len(layer_widths)):
#         rand_arch.append(random.randrange(layer_widths[i]))
        
#     return(rand_arch)

def get_arch_flops(arch):
    input_size = 224
    total_param = 0
    
    
    layer_num = 0
    flops = 0
    input_channel = 32
    
    expand_ratio = [3,6,3,6,3,6,0]
    kernel_size = [3,3,5,5,7,7,0]

    width_stages = [24,40,80,96,192,320]
    n_cell_stages = [4,4,4,4,4,1]
    stride_stages = [2,2,2,1,2,1]
    
    # fisrt_conv
    kernel_ops = 3 * 3 * 3
    params = 32 * kernel_ops
    output_size = input_size // 2
    flops += params * output_size ** 2
    total_param += params
    
    input_size = output_size
    # first_conv_block
    kernel_ops = 1 * 3 * 3
    params = 32 * kernel_ops
    flops += params * output_size ** 2
    total_param += params
    
    kernel_ops = 32
    params = 16 * kernel_ops
    flops += params * output_size ** 2 
    total_param += params
    
    input_channel = 16
    
    
    for width, n_cell, s in zip(width_stages, n_cell_stages, stride_stages):       
        
        for i in range(n_cell):
            if i == 0:
                stride = s
            else:
                stride = 1
            
            # bottleneck
            kernel_ops = input_channel
            
#             print(f' layer_num: {layer_num}')
#             print(f' arch: {arch[layer_num]}')
            params = input_channel * expand_ratio[arch[layer_num]] * kernel_ops
            
            output_size = input_size
            flops += params * output_size ** 2
            total_param += params
            
            # depthwise
            kernel_ops = kernel_size[arch[layer_num]] ** 2
            params = input_channel * expand_ratio[arch[layer_num]] * kernel_ops
            
            output_size = input_size // stride 
            flops += params * output_size ** 2
            total_param += params
            
            # separable
            kernel_ops = input_channel * expand_ratio[arch[layer_num]]
            params = width * kernel_ops
            flops += params * output_size ** 2
            total_param += params

            
            layer_num += 1
            input_channel = width
            input_size = output_size
            
    # feature_mix_layer
    kernel_ops = 320
    params = 1280 * kernel_ops
    flops += params * input_size ** 2
    total_param += params
    
    # linear
    params = 1280 * 1000
    flops += params
    total_param += params
    
    return total_param + 1000, flops

train_dataset = datasets.ImageFolder(
        args.train_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
        ])
    )
    
valval_dataset = datasets.ImageFolder(
    args.train_dir,
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
    ]))

num_train = len(train_dataset)

split = 50000

train_size = num_train - split
val_size = split

trainset, _ = random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
_, valset = random_split(valval_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

valid_loader = torch.utils.data.DataLoader(valset, batch_size=200, shuffle=False,
                         num_workers=6, pin_memory=True)

choice = lambda x: x[np.random.randint(len(x))] if isinstance(
    x, tuple) else choice(tuple(x))

class EvolutionSearcher(object):

    def __init__(self, args, valid_loader, model):
        self.args = args
        
        self.valid_loader = valid_loader
        self.max_epochs = args.max_epochs
        self.select_num = args.select_num
        self.population_num = args.population_num
        self.m_prob = args.m_prob
        self.crossover_num = args.crossover_num
        self.mutation_num = args.mutation_num
        self.flops_up_limit = args.flops_up_limit
        self.flops_low_limit = args.flops_low_limit

        self.model = model
#         self.model = torch.nn.DataParallel(self.model).cuda()
        trained_model = torch.load(f'logs/{args.file_name}/final_network.pth')
        current_dict = self.model.state_dict()
        
        pretrain_load = 0
        for key in current_dict.keys():
            if 'module.' + key in trained_model.keys():
                current_dict[key].copy_(trained_model['module.' + key])
                pretrain_load += 1
            
        self.model.load_state_dict(current_dict)
        print(pretrain_load)
                

        self.log_dir = args.log_dir
        self.checkpoint_name = os.path.join(self.log_dir, 'checkpoint.pth.tar')

        self.memory = []
        self.vis_dict = {}
        self.keep_top_k = {self.select_num: [], 50: []}
        self.epoch = 0
        self.candidates = []

        self.nr_layer = 21
#         self.nr_state = 5

    def save_checkpoint(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        info = {}
        info['memory'] = self.memory
        info['candidates'] = self.candidates
        info['vis_dict'] = self.vis_dict
        info['keep_top_k'] = self.keep_top_k
        info['epoch'] = self.epoch
        torch.save(info, self.checkpoint_name)
        print('save checkpoint to', self.checkpoint_name)
        
    def accuracy(self, output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def get_cand_err(self, model, cand, args):
        
        val_dataprovider = self.valid_loader

        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        max_train_iters = args.max_train_iters
        max_test_iters = args.max_test_iters

        top1 = 0
        top5 = 0
        total = 0

        print('starting test....')
        with torch.no_grad():
            model.eval()
            for i, (data, target) in enumerate(self.valid_loader):

                # print('test step: {} total: {}'.format(step,max_test_iters))
                batchsize = data.shape[0]
                # print('get data',data.shape)
                target = target.type(torch.LongTensor)
                data, target = data.to(device), target.to(device)

                logits = model(data, cand)

                prec1, prec5 = self.accuracy(logits, target, topk=(1, 5))

                # print(prec1.item(),prec5.item())

                top1 += prec1.item() * batchsize
                top5 += prec5.item() * batchsize
                total += batchsize

                del data, target, logits, prec1, prec5
            
            top1, top5 = top1 / total, top5 / total

            top1, top5 = 1 - top1 / 100, 1 - top5 / 100

            print('top1: {:.2f} top5: {:.2f}'.format(top1 * 100, top5 * 100))

        return top1, top5

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_name):
            return False
        info = torch.load(self.checkpoint_name)
        self.memory = info['memory']
        self.candidates = info['candidates']
        self.vis_dict = info['vis_dict']
        self.keep_top_k = info['keep_top_k']
        self.epoch = info['epoch']

        print('load checkpoint from', self.checkpoint_name)
        return True
    
#     def to_struc(self, tup):
#         tensor = torch.zeros(self.nr_layer, self.nr_state)
#         for i, val in enumerate(tup):
#             tensor[i, val-1] = 1
            
#         return tensor

    def is_legal(self, cand):
        assert isinstance(cand, tuple) and len(cand) == self.nr_layer
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            return False
        
        if 'flops' not in info:
            info['flops'] = get_arch_flops(cand)[1]
        
        print(cand, info['flops'])
        
        if info['flops'] > self.flops_up_limit:
            print('flops limit exceed')
            return False
        
        if info['flops'] < self.flops_low_limit:
            print('flops too low')
            return False

        info['err'] = self.get_cand_err(self.model, cand, self.args)

        info['visited'] = True

        return True

    def update_top_k(self, candidates, *, k, key, reverse=False):
        assert k in self.keep_top_k
        print('select ......')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]

    def stack_random_cand(self, random_func, *, batchsize=10):
        while True:
            cands = [random_func() for _ in range(batchsize)]
            for cand in cands:
                if cand not in self.vis_dict:
                    self.vis_dict[cand] = {}
                info = self.vis_dict[cand]
            for cand in cands:
                yield cand

#     def get_random(self, num):
#         print('random select ........')
#         cand_iter = self.stack_random_cand(
#             lambda: tuple(np.random.randint(self.nr_state) for i in range(self.nr_layer)))
#         while len(self.candidates) < num:
#             cand = next(cand_iter)
#             if not self.is_legal(cand):
#                 continue
#             self.candidates.append(cand)
#             print('random {}/{}'.format(len(self.candidates), num))
#         print('random_num = {}'.format(len(self.candidates)))

    def get_rand_arch(self, num):
        print('random select .......')
        while len(self.candidates) < num:            
            rand_arch = []
            for i in range(len(layer_widths)):
                rand_arch.append(random.randrange(layer_widths[i]))
                
            if not self.is_legal(tuple(rand_arch)):
                continue
                
            self.candidates.append(tuple(rand_arch))
            print('random {}/{}'.format(len(self.candidates), num))
        print('random_num = {}'.format(len(self.candidates)))


    def get_mutation(self, k, mutation_num, m_prob):
        assert k in self.keep_top_k
        print('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = list(choice(self.keep_top_k[k]))
            
            for i in range(self.nr_layer):
                if np.random.random_sample() < m_prob:
                    cand[i] = np.random.randint(layer_widths[i])
            return tuple(cand)

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print('mutation {}/{}'.format(len(res), mutation_num))

        print('mutation_num = {}'.format(len(res)))
        return res

    def get_crossover(self, k, crossover_num):
        assert k in self.keep_top_k
        print('crossover ......')
        res = []
        iter = 0
        max_iters = 10 * crossover_num

        def random_func():
            p1 = choice(self.keep_top_k[k])
            p2 = choice(self.keep_top_k[k])
            return tuple(choice([i, j]) for i, j in zip(p1, p2))
        cand_iter = self.stack_random_cand(random_func)
        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print('crossover {}/{}'.format(len(res), crossover_num))

        print('crossover_num = {}'.format(len(res)))
        return res
    

    def search(self):
        print('population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
            self.population_num, self.select_num, self.mutation_num, self.crossover_num, self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))

#         self.load_checkpoint()

        self.get_rand_arch(self.population_num)

        while self.epoch < self.max_epochs:
            print('epoch = {}'.format(self.epoch))

            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)

            self.update_top_k(
                self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['err'])
            self.update_top_k(
                self.candidates, k=50, key=lambda x: self.vis_dict[x]['err'])

            print('epoch = {} : top {} result'.format(
                self.epoch, len(self.keep_top_k[50])))
            for i, cand in enumerate(self.keep_top_k[50]):
                print('No.{} {} Top-1 err = {}'.format(
                    i + 1, cand, self.vis_dict[cand]['err']))
                ops = [i for i in cand]
                print(ops)

            mutation = self.get_mutation(
                self.select_num, self.mutation_num, self.m_prob)
            crossover = self.get_crossover(self.select_num, self.crossover_num)

            self.candidates = mutation + crossover

            self.get_rand_arch(self.population_num)

            self.epoch += 1
            
        print("=============================Final Top=============================")
        self.memory.append([])
        for cand in self.candidates:
            self.memory[-1].append(cand)

        self.update_top_k(
            self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['err'])
        self.update_top_k(
            self.candidates, k=50, key=lambda x: self.vis_dict[x]['err'])
        
        print('epoch = {} : top {} result'.format(
                self.epoch, len(self.keep_top_k[50])))
        for i, cand in enumerate(self.keep_top_k[50]):
            print('No.{} {} Top-1 err = {}'.format(
                i + 1, cand, self.vis_dict[cand]['err']))
            ops = [i for i in cand]
            print(ops)
            logging.info(ops)
            

model = MobileNetV2_search()            
searcher = EvolutionSearcher(args, valid_loader, model.cuda())
searcher.search()