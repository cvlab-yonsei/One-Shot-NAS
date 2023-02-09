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

sys.path.insert(0, '../../')

from xautodl.config_utils import load_config, dict2config, configure2str
from xautodl.datasets import get_datasets, get_nas_search_loaders
from xautodl.procedures import (
    prepare_seed,
    prepare_logger,
    save_checkpoint,
    copy_checkpoint,
    get_optim_scheduler,
)
from xautodl.utils import get_model_infos, obtain_accuracy
from xautodl.log_utils import AverageMeter, time_string, convert_secs2time
from xautodl.models import get_cell_based_tiny_net, get_search_spaces
parser = argparse.ArgumentParser()
parser.add_argument('--log-dir', type=str, default='log')
parser.add_argument('--file_name', type=str, default='RSPS')
parser.add_argument('--max-epochs', type=int, default=20)
parser.add_argument('--select-num', type=int, default=10)
parser.add_argument('--population-num', type=int, default=50)
parser.add_argument('--m_prob', type=float, default=0.1)
parser.add_argument('--crossover-num', type=int, default=25)
parser.add_argument('--mutation-num', type=int, default=25)
parser.add_argument('--flops-limit', type=float, default=330 * 1e6)
parser.add_argument('--max-train-iters', type=int, default=200)
parser.add_argument('--max-test-iters', type=int, default=40)
parser.add_argument('--train-batch-size', type=int, default=128)
parser.add_argument('--test-batch-size', type=int, default=200)

args = parser.parse_args()
os.chdir('../../')

import logging

logging.basicConfig(filename=os.path.join(f'logs/{args.file_name}', "log.txt"),
                    level=logging.INFO,
                    format='')

search_space = get_search_spaces("cell", 'nas-bench-201')
model_config = dict2config(
    {
        "name": "RANDOM",
        "C": 16,
        "N": 5,
        "max_nodes": 4,
        "num_classes": 10,
        "space": search_space,
        "affine": False,
        "track_running_stats": bool(0),
    },
    None,
)

train_data, valid_data, _, _ = get_datasets( 
        'cifar10', './dataset', -1
    )

search_loader, _, valid_loader = get_nas_search_loaders( 
        train_data,                                     
        valid_data,                                      
        'cifar10',
        "configs/nas-benchmark/",
        (64, 200), 
        4,
    )


search_model = get_cell_based_tiny_net(model_config)

struc = []
base = torch.zeros(6,5)
for i in range(5):   
    base[0,i] = 1
    
    for ii in range(5):
        base[1,ii] = 1     
        
        for iii in range(5):
            base[2,iii]=1
            
            for j in range(5):
                base[3,j] = 1
                
                for jj in range(5):
                    base[4,jj] = 1
                    
                    for jjj in range(5):
                        base[5,jjj] = 1
                        
                        struc.append(base.clone())
                       
                        
                        base[5] = 0
                    base[4] = 0
                base[3] = 0
            base[2] = 0
        base[1] = 0
    base[0] = 0

from xautodl.models.cell_searchs.genotypes import Structure

genotypes = []
op_names = deepcopy(search_space)
for i in range(1, 4):
    xlist = []
    for j in range(i):
        op_name = random.choice(op_names)
        xlist.append((op_name, j))
    genotypes.append(tuple(xlist))
arch = Structure(genotypes)

edge2index = search_model.edge2index
max_nodes = 4
def genotype(enc): 
    theta = enc
    genotypes = []
    for i in range(1, max_nodes):
      xlist = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        with torch.no_grad():
          weights = theta[ edge2index[node_str] ]
          op_name = op_names[ weights.argmax().item() ]
        xlist.append((op_name, j))
      genotypes.append( tuple(xlist) )
    return Structure( genotypes )

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
#         self.flops_limit = args.flops_limit

        self.model = model
#         self.model = torch.nn.DataParallel(self.model).cuda()
        supernet_state_dict = torch.load(
            f'logs/{args.file_name}/final_network.pth')
#         supernet_state_dict = torch.load(
#             f'RSPS/Adaptive_LR/max_coeff_3_log_formulation_reverse_setting/final_network.pth')
        self.model.load_state_dict(supernet_state_dict)

        self.log_dir = args.log_dir
        self.checkpoint_name = os.path.join(self.log_dir, 'checkpoint.pth.tar')

        self.memory = []
        self.vis_dict = {}
        self.keep_top_k = {self.select_num: [], 50: []}
        self.epoch = 0
        self.candidates = []

        self.nr_layer = 6
        self.nr_state = 5

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
            model.arch_cache = genotype(self.to_struc(cand))
            for i, (data, target) in enumerate(self.valid_loader):
                if i == max_test_iters:
                    break
                # print('test step: {} total: {}'.format(step,max_test_iters))
                batchsize = data.shape[0]
                # print('get data',data.shape)
                target = target.type(torch.LongTensor)
                data, target = data.to(device), target.to(device)

                _, logits = model(data)

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
    
    def to_struc(self, tup):
        tensor = torch.zeros(self.nr_layer, self.nr_state)
        for i, val in enumerate(tup):
            tensor[i, val-1] = 1
            
        return tensor

    def is_legal(self, cand):
        assert isinstance(cand, tuple) and len(cand) == self.nr_layer
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
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

    def get_random(self, num):
        print('random select ........')
        cand_iter = self.stack_random_cand(
            lambda: tuple(np.random.randint(self.nr_state) for i in range(self.nr_layer)))
        while len(self.candidates) < num:
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
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
                    cand[i] = np.random.randint(self.nr_state)
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

        self.get_random(self.population_num)

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

            self.get_random(self.population_num)

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
            
        with open("./exps/NAS-Bench-201-algos/kendal_valid_accs/num_params.pkl","rb") as f:
            num_params = pickle.load(f)    


        for file_name in ['cifar10_valids','cifar10_tests', 'cifar100_valids','cifar100_tests','imagenet_valids','imagenet_tests']:
            with open(f"./exps/NAS-Bench-201-algos/kendal_valid_accs/{file_name}.pkl","rb") as f:
                globals()[file_name] = pickle.load(f) 
            
        for i in range(len(struc)):
            if torch.equal(self.to_struc(self.keep_top_k[50][0]), struc[i]):
                fin_net_num = i

        i = fin_net_num
        
        log_string = 'PARAMS: ' + str(num_params[i]) + '\n' \
            + 'CIFAR10-VALID:  ' + str(cifar10_valids[i])+ '\n' \
            + 'CIFAR10-TEST:  ' + str(cifar10_tests[i])+'\n' \
            + 'CIFAR100-VALID:  ' + str(cifar100_valids[i])+ '\n' \
            + 'CIFAR100-TEST:  ' + str(cifar100_tests[i])+'\n' \
            + 'IMAGENET16-VALID:  ' + str(imagenet_valids[i])+ '\n' \
            + 'IMAGENET16-TEST:  ' + str(imagenet_tests[i])
        
        print(log_string)
        logging.info(log_string)
        
searcher = EvolutionSearcher(args, valid_loader, search_model.cuda())
searcher.search()