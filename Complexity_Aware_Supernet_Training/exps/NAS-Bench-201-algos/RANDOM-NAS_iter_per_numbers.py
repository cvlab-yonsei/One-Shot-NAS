##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020 #
##############################################################################
# Random Search and Reproducibility for Neural Architecture Search, UAI 2019 #
##############################################################################
import os, sys, time, glob, random, argparse
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

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
from nas_201_api import NASBench201API as API
import scipy.stats as stats

parser = argparse.ArgumentParser(description="RSPS_MY")
parser.add_argument('--log_dir', type=str, default='logs/tmp')
parser.add_argument('--file_name', type=str, default='tmp')
parser.add_argument('--epochs', type=int, default=250)
args = parser.parse_args()


os.chdir('../../')

file_name = args.file_name
epochs = args.epochs
writer = SummaryWriter(args.log_dir)

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

supernet_config = dict2config(
    {
        "name": "supernet",
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

search_model = get_cell_based_tiny_net(model_config)
supernet = get_cell_based_tiny_net(supernet_config)
supernet = supernet.cuda()
optimizer = torch.optim.SGD(
    params = search_model.parameters(),
    lr = 0.025,
    momentum = 0.9,
    weight_decay = 0.0005,
    nesterov = True 
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max = epochs,
    eta_min = 0.001
)

criterion = torch.nn.CrossEntropyLoss()

api = API('nasbench201/NAS-Bench-201-v1_1-096897.pth')

network = search_model.cuda()
criterion = criterion.cuda()

train_data, valid_data, _, _ = get_datasets( # train_data: trainset, valid_data: testset
        'cifar10', './dataset', -1
    )

search_loader, _, valid_loader = get_nas_search_loaders( # search loader 는 train set + valid set
        train_data,                                      # train_loader 는 train set
        valid_data,                                      # valid_loader 는 valid set (cifar10 기준)
        'cifar10',
        "configs/nas-benchmark/",
        (64, 256), # 페이퍼는 256 코드는 512로 구현해놨음.
        4,
    )

# logger.log(f'search_loader_num: {len(search_loader)}, valid_loader_num: {len(valid_loader)}')

total_iter = 0

def search_find_best(xloader, network, n_samples):
    with torch.no_grad():
        network.eval()
        archs, valid_accs = [], []
        # print ('obtain the top-{:} architectures'.format(n_samples))
        loader_iter = iter(xloader)
        for i in range(n_samples):
            arch = network.random_genotype(True)
            try:
                inputs, targets = next(loader_iter)
            except:
                loader_iter = iter(xloader)
                inputs, targets = next(loader_iter)

            inputs = inputs.cuda()
            targets = targets.cuda()
            
            _, logits = network(inputs)
            val_top1, val_top5 = obtain_accuracy(
                logits.data, targets.data, topk=(1, 5)
            )

            archs.append(arch)
            valid_accs.append(val_top1.item())

        best_idx = np.argmax(valid_accs)
        best_arch, best_valid_acc = archs[best_idx], valid_accs[best_idx]
        return best_arch, best_valid_acc

arch_repeat = 0
for ep in range(epochs):
    network.train()
    if ep % 50 == 0:
        torch.save(network.state_dict(), os.path.join(args.log_dir, f'network_ep_{ep}.pth'))
    for i, (input, label, _, _) in enumerate(search_loader):
        input = input.cuda()
        label = label.cuda()

        if arch_repeat == 0:
            arch = network.random_genotype(True)
            result = api.query_by_arch(arch, '200')
            result = result.split('\n')
            cifar10 = result[2].split(' ')

            num_param = float(cifar10[-4].strip('Params='))

            if num_param < 0.34:
                arch_repeat = 1

            else:
                arch_repeat = 2


        optimizer.zero_grad()

        _, pred = network(input)
        loss = criterion(pred, label)
        loss.backward()
        nn.utils.clip_grad_norm_(network.parameters(), 5)
        optimizer.step()

        writer.add_scalar('train/subnet_loss', loss.item(), total_iter)

        base_prec1, base_prec5 = obtain_accuracy(
            pred.data, label.data, topk=(1, 5)
        )

        writer.add_scalar('train/subnet_top1', base_prec1, total_iter)
        writer.add_scalar('train/subnet_top5', base_prec5, total_iter)
        total_iter += 1
        arch_repeat = arch_repeat - 1

        # print(f'num_param: {num_param}, arch_repeat: {arch_repeat}')
        # print(arch)
    print(f'ep: {ep}, top1: {base_prec1}')

    scheduler.step()
    
torch.save(network.state_dict(), os.path.join(args.log_dir, 'final_network.pth'))    

def distill(result):
    result = result.split('\n')
    cifar10 = result[5].replace(' ', '').split(':')
    cifar100 = result[7].replace(' ', '').split(':')
    imagenet16 = result[9].replace(' ', '').split(':')

    cifar10_train = float(cifar10[1].strip(',test')[-7:-2].strip('='))
    cifar10_test = float(cifar10[2][-7:-2].strip('='))
    cifar100_train = float(cifar100[1].strip(',valid')[-7:-2].strip('='))
    cifar100_valid = float(cifar100[2].strip(',test')[-7:-2].strip('='))
    cifar100_test = float(cifar100[3][-7:-2].strip('='))
    imagenet16_train = float(imagenet16[1].strip(',valid')[-7:-2].strip('='))
    imagenet16_valid = float(imagenet16[2].strip(',test')[-7:-2].strip('='))
    imagenet16_test = float(imagenet16[3][-7:-2].strip('='))

    return cifar10_train, cifar10_test, cifar100_train, cifar100_valid, \
        cifar100_test, imagenet16_train, imagenet16_valid, imagenet16_test

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

edge2index = network.edge2index
max_nodes = 4
def genotype(enc): # upon calling, the caller should pass the "theta" into this object as "alpha" first
#     theta = torch.softmax(_arch_parameters, dim=-1) * enc
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
    
def get_num_params(result):
    result = result.split('\n')
    cifar10 = result[2].split(' ')

    return float(cifar10[-4].strip('Params='))

print('================Kendall tau Start================')
loader_iter = iter(valid_loader)
valid_accs = []
cifar10_accs = []
cifar100_accs = []
imagenet_accs = []
num_params = []

for i in range(len(struc)):        
    network.arch_cache = genotype(struc[i])
    with torch.no_grad():
        network.eval()
        correct_classified = 0
        total = 0
        for j, (input, label) in enumerate(valid_loader):
            input = input.cuda()
            label = label.cuda()

            _, pred = network(input)
            _, predicted = torch.max(pred.data,1)

            total += pred.size(0)
            correct_classified += (predicted == label).sum().item()      

        valid_acc = correct_classified/total

        valid_accs.append(valid_acc)

    print(f'=============={i}==============')
    print(f'valid_acc: {valid_acc}')
    result = api.query_by_arch(genotype(struc[i]), '200')
    cifar10_train, cifar10_test, cifar100_train, cifar100_valid, \
        cifar100_test, imagenet_train, imagenet_valid, imagenet_test = distill(result)
    
    
    num_params.append(get_num_params(result))
    cifar10_accs.append(cifar10_test)
    cifar100_accs.append(cifar100_test)
    imagenet_accs.append(imagenet_test)

import pickle

with open(f"./exps/NAS-Bench-201-algos/kendal_valid_accs/{file_name}.pkl","wb") as f:
    pickle.dump(valid_accs, f)    
    
cifar10_valid_true_tau, _ = stats.kendalltau(valid_accs, cifar10_accs)     
cifar100_valid_true_tau, _ = stats.kendalltau(valid_accs, cifar100_accs)   
imagenet_valid_true_tau, _ = stats.kendalltau(valid_accs, imagenet_accs) 

cifar10_true_valid_tau, _ = stats.kendalltau(cifar10_accs, valid_accs)   
cifar100_true_valid_tau, _ = stats.kendalltau(cifar100_accs, valid_accs)  
imagenet_true_valid_tau, _ = stats.kendalltau(imagenet_accs, valid_accs)


print('================Kendall tau end================')


print(f'cifar10_valid_true_tau: {cifar10_valid_true_tau}')
print(f'cifar100_valid_true_tau: {cifar100_valid_true_tau}')
print(f'imagenet_valid_true_tau: {imagenet_valid_true_tau}')


