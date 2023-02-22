import os
import sys
import torch
import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import cv2
import PIL
from PIL import Image
import time
import logging
import argparse
from network import MobileNetV2_search
from utils import accuracy, AvgrageMeter, get_parameters
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
import random
# from flops import get_cand_flops
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class DataIterator(object):

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = enumerate(self.dataloader)

    def next(self):
        try:
            _, data = next(self.iterator)
        except Exception:
            self.iterator = enumerate(self.dataloader)
            _, data = next(self.iterator)
        return data[0], data[1]

layer_widths = [6, 7, 7, 7, 6, 7, 7, 7, 6, 7, 7, 7, 6, 7, 7, 7, 6, 7, 7, 7, 6]

def get_rand_arch():
    rand_arch = []
    for i in range(len(layer_widths)):
        rand_arch.append(random.randrange(layer_widths[i]))
        
    return(rand_arch)

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
    
    
def get_args():
    parser = argparse.ArgumentParser("ShuffleNetV2_OneShot")
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--eval-resume', type=str, default='./snet_detnas.pkl', help='path for eval model')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size')
    parser.add_argument('--epochs', type=int, default=150, help='total iters')
    parser.add_argument('--learning-rate', type=float, default=0.045, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight-decay', type=float, default=4e-5, help='weight decay')
    parser.add_argument('--save', type=str, default='./models', help='path for saving trained models')
    parser.add_argument('-j', '--workers', default=6, type=int, metavar='N', help='number of data loading workers')
#     parser.add_argument('--label-smooth', type=float, default=0.1, help='label smoothing')

    parser.add_argument('--auto-continue', type=bool, default=True, help='report frequency')
    parser.add_argument('--display-interval', type=int, default=20, help='report frequency')
    parser.add_argument('--val-interval', type=int, default=10000, help='report frequency')
    parser.add_argument('--save-interval', type=int, default=10000, help='report frequency')

    parser.add_argument('--train-dir', type=str, default='../../../../dataset/ILSVRC2012/train', help='path to training dataset')
    parser.add_argument('--val-dir', type=str, default='../../../../dataset/ILSVRC2012/val', help='path to validation dataset')
    parser.add_argument('--log_dir', type=str, default='logs/baseline', help='path to validation dataset')
    
    parser.add_argument('--visible_gpus', default='0,1,2,3', type=str, help='total GPUs to use')
    
    
    parser.add_argument('--gpu', type=int, default=None, help='path to validation dataset')
    
    parser.add_argument('--world-size', default=1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str, help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')
#     parser.add_argument('--gpu', default=1, type=int, help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', type=str2bool, default=True,
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    args = parser.parse_args()
    return args

def save_checkpoint(state, path='./'):
    torch.save(state, os.path.join(path, 'checkpoint.pth.tar'))

def main():
    args = get_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)   
        cudnn.deterministic = True
        cudnn.benchmark = False
#         warnings.warn('You have chosen to seed training. '
#                       'This will turn on the CUDNN deterministic setting, '
#                       'which can slow down your training considerably! '
#                       'You may see unexpected behavior when restarting '
#                       'from checkpoints.')

#     writer = SummaryWriter(args.log_dir)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logging.basicConfig(filename=os.path.join(args.log_dir, "log.txt"),
                        level=logging.INFO,
                        format='')
    
    if args.visible_gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"]= args.visible_gpus
        
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
        
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)
        
     
def main_worker(gpu, ngpus_per_node, args):
    
    model = MobileNetV2_search()
    
    args.gpu = gpu
    
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) ########## SyncBatchnorm
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) ########## SyncBatchnorm
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    use_gpu = False
    if torch.cuda.is_available():
        use_gpu = True
#         os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        print(f'GPU loaded: {args.gpu}') 
#         device = torch.device("cuda")

    assert os.path.exists(args.train_dir)
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
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    else:
        train_sampler = None
    
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
#     train_dataprovider = DataIterator(train_loader)

#     val_loader = torch.utils.data.DataLoader(
#         datasets.ImageFolder(args.val_dir, transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225]),
#         ])),
#         batch_size=200, shuffle=False,
#         num_workers=1, pin_memory=use_gpu
#     )
#     val_dataprovider = DataIterator(val_loader)
    print('load data successfully')

#     model = nn.DataParallel(model)

    

    optimizer = torch.optim.SGD(get_parameters(model),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

#     model = model.to(device)

    args.optimizer = optimizer
    args.loss_function = criterion
    args.scheduler = scheduler
#     args.train_dataprovider = train_dataprovider
#     args.val_dataprovider = val_dataprovider

    if args.eval:
        if args.eval_resume is not None:
            checkpoint = torch.load(args.eval_resume, map_location=None if use_gpu else 'cpu')
            model.load_state_dict(checkpoint, strict=True)
            validate(model, device, args, all_iters=all_iters)
        exit(0)
    
    total_iter = 0
    optimizer = args.optimizer
    loss_function = args.loss_function
    scheduler = args.scheduler
#     train_dataprovider = args.train_dataprovider

    if args.rank % ngpus_per_node == 0: # do only at rank=0 process
        writer = SummaryWriter(args.log_dir)
        
    else:
        writer = None
#     print('hmm')

    
    for ep in range(args.epochs):
        t1 = time.time()
        Top1_err, Top5_err = 0.0, 0.0
        model.train()
        for i, (images, target) in enumerate(train_loader):
            dist.barrier()
            
            if args.distributed:
                train_sampler.set_epoch(ep)
#             scheduler.step()

#             total_iter += 1
            d_st = time.time()
#             data, target = train_dataprovider.next()
            target = target.type(torch.LongTensor)
            if args.gpu is not None:
                data, target = images.cuda(args.gpu, non_blocking=True), target.cuda(args.gpu, non_blocking=True)



            flops_l, flops_r, flops_step = 300, 330, 10
            bins = [[i, i+flops_step] for i in range(flops_l, flops_r, flops_step)]
            def get_uniform_sample_cand(*,timeout=500):
                idx = np.random.randint(len(bins))
                l, r = bins[idx]
                for i in range(timeout):
                    cand = get_rand_arch()
                    if l*1e6 <= get_arch_flops(cand)[1] <= r*1e6:
                        return cand
                return get_rand_arch()
#             import pdb; pdb.set_trace()
            
            arch = get_uniform_sample_cand()
            arch = torch.tensor(arch).cuda()
            
                
            
            torch.distributed.broadcast(arch, src=0)

            output = model(data, arch)
            loss = loss_function(output, target)
            optimizer.zero_grad()
            loss.backward()

            
            

            for p in model.parameters():
                if p.grad is not None and p.grad.sum() == 0:
                    p.grad = None

            optimizer.step()
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            
            if writer is not None:
                writer.add_scalar('train/subnet_loss', loss.item(), total_iter)
                writer.add_scalar('train/subnet_top1', prec1, total_iter)
                writer.add_scalar('train/subnet_top5', prec5, total_iter)

            Top1_err += 1 - prec1.item() / 100
            Top5_err += 1 - prec5.item() / 100
            data_time = time.time() - d_st

            if total_iter % args.display_interval == 0:
                printInfo = 'TRAIN Iter {} (ep {}): lr = {:.6f},\tloss = {:.6f},\t'.format(total_iter, ep, scheduler.get_lr()[0], loss.item()) + \
                            'Top-1 err = {:.6f},\t'.format(Top1_err / args.display_interval) + \
                            'Top-5 err = {:.6f},\t'.format(Top5_err / args.display_interval) + \
                            'iter_time = {:.6f},\ttrain_time = {:.6f}'.format(data_time, (time.time() - t1) / args.display_interval)
                logging.info(printInfo)
                print(printInfo)
                t1 = time.time()
                Top1_err, Top5_err = 0.0, 0.0
            total_iter += 1
        
        scheduler.step()
        
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': ep + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict(),
            }, path=args.log_dir)
        
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
        torch.save(model.state_dict(), os.path.join(args.log_dir, 'final_network.pth'))


# def validate(model, device, args, *, all_iters=None):
#     objs = AvgrageMeter()
#     top1 = AvgrageMeter()
#     top5 = AvgrageMeter()

#     loss_function = args.loss_function
#     val_dataprovider = args.val_dataprovider

#     model.eval()
#     max_val_iters = 250
#     t1  = time.time()
#     with torch.no_grad():
#         for _ in range(1, max_val_iters + 1):
#             data, target = val_dataprovider.next()
#             target = target.type(torch.LongTensor)
#             data, target = data.to(device), target.to(device)

#             output = model(data)
#             loss = loss_function(output, target)

#             prec1, prec5 = accuracy(output, target, topk=(1, 5))
#             n = data.size(0)
#             objs.update(loss.item(), n)
#             top1.update(prec1.item(), n)
#             top5.update(prec5.item(), n)

#     logInfo = 'TEST Iter {}: loss = {:.6f},\t'.format(all_iters, objs.avg) + \
#               'Top-1 err = {:.6f},\t'.format(1 - top1.avg / 100) + \
#               'Top-5 err = {:.6f},\t'.format(1 - top5.avg / 100) + \
#               'val_time = {:.6f}'.format(time.time() - t1)
#     logging.info(logInfo)


if __name__ == "__main__":
    main()

