import os
import sys
import argparse
import time, re
import builtins
import numpy as np
import random 
import pickle 
import socket 
import math 
from tqdm import tqdm 
from backbone.select_backbone import select_backbone

import torch
import torch.nn as nn 
import torch.optim as optim 
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils import data 
from torchvision import transforms
import torchvision.utils as vutils

import utils.augmentation as A
import utils.transforms as T
import utils.tensorboard_utils as TB
from tensorboardX import SummaryWriter

from utils.utils import AverageMeter, write_log, calc_topk_accuracy, calc_mask_accuracy, \
batch_denorm, ProgressMeter, neq_load_customized, save_checkpoint, denorm, Logger, FastDataLoader, \
bool_flag, init_distributed_mode
import torch.nn.functional as F 
import torch.backends.cudnn as cudnn
from model.pretrain import InfoNCE, UberNCE, ViCC
from dataset.lmdb_dataset import *

import apex
from apex.parallel.LARC import LARC
import wandb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', default='s3d', type=str)
    parser.add_argument('--model', default='vicc', type=str, help='vicc, infonce or ubernce')
    parser.add_argument('--dataset', default='ucf101-2clip', type=str, help='ucf101-2clip for RGB, ucf101-f-2clip for Flow')
    parser.add_argument('--seq_len', default=32, type=int, help='number of frames in each video block')
    parser.add_argument('--num_seq', default=2, type=int, help='number of video blocks')
    parser.add_argument('--ds', default=1, type=int, help='frame down sampling rate')
    parser.add_argument('--batch_size', default=32, type=int)
    # for manual learning rate schedule
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate, only used when cos is false')
    parser.add_argument('--schedule', default=[120,160], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')
    #####################################
    # for cosine learning rate schedule
    parser.add_argument('--cos', type=bool_flag, default=True, help='use cosine lr schedule instead: set base_lr, final_lr and wd')
    parser.add_argument("--base_lr", default=0.6, type=float, help="base learning rate")
    parser.add_argument("--final_lr", type=float, default=0.0006, help="final learning rate")
    #####################################
    parser.add_argument('--resume', default='', type=str, help='path of model to resume')
    parser.add_argument('--pretrain', default='', type=str, help='path of pretrained model')
    parser.add_argument('--test', default='', type=str, help='path of model to load and pause')
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--save_epoch', default=299, type=int, help='save particular epoch number')
    parser.add_argument('--save_epoch_freq', default=100, type=int, help='frequency of saving epochs besides best models')
    parser.add_argument('--gpu', default=None, type=int, nargs="+", help="list of gpu device ids")
    parser.add_argument('--print_freq', default=5, type=int, help='frequency of printing output during training')
    #####################################
    parser.add_argument('--tb_background', type=bool_flag, default=False, help='log tensorboard in the background to save time')
    parser.add_argument('--optim', default='sgd', type=str, help='adam or sgd')
    #####################################
    parser.add_argument('--save_freq', default=1, type=int, help='frequency of eval')
    parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
    parser.add_argument('--img_dim', default=128, type=int)
    parser.add_argument('--dataset_root', default='', type=str, help='dataset root path')
    parser.add_argument('--prefix', default='pretrain', type=str, help='experiment folder general path')
    parser.add_argument('--name_prefix', default='main_single', type=str, help='name of experiment')
    parser.add_argument('-j', '--workers', default=16, type=int)

    # parallel configs:
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    # for torch.distributed.launch
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')

    # moco specific configs:
    parser.add_argument('--moco-dim', default=128, type=int,
                        help='feature dimension (default: 128)')
    parser.add_argument('--moco-k', default=1920, type=int,
                        help='queue size; number of negative keys (default: 65536 MoCo, 2048 CoCLR, 3840 SwaV)')
    parser.add_argument('--moco-m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999 ), only used for MoCo/CoCLR')
    parser.add_argument('--moco-t', default=0.1, type=float,
                        help='softmax temperature (default: 0.07 for MoCo/CoCLR, 0.1 for SwaV)')

    #####################################
    parser.add_argument("--views_for_assign", type=int, nargs="+", default=[0,1],
                    help="list of views id used for computing assignments")
    parser.add_argument("--nmb_views", type=int, default=[2], nargs="+",
                    help="list of total number of views (example: [2], [2,2])")
    parser.add_argument("--epsilon", default=0.05, type=float,
                    help="regularization parameter for Sinkhorn-Knopp algorithm")
    parser.add_argument("--sinkhorn_iterations", default=3, type=int,
                    help="number of iterations in Sinkhorn-Knopp algorithm")
    parser.add_argument("--improve_numerical_stability", default=False, type=bool_flag,
                    help="improves numerical stability in Sinkhorn-Knopp algorithm")
    parser.add_argument("--nmb_prototypes", default=300, type=int,
                    help="number of prototypes")
    parser.add_argument("--epoch_queue_starts", type=int, default=150,
                    help="from this epoch, we start using a queue")
    # swav optim parameters
    parser.add_argument("--freeze_prototypes_nepochs", default=100, type=int,
                    help="freeze the prototypes during this many epochs from the start")
    parser.add_argument("--warmup_epochs", default=0, type=int, help="number of warmup epochs")
    parser.add_argument("--start_warmup", default=0, type=float, help="initial warmup learning rate")
    parser.add_argument("--seed", type=int, default=31, help="seed")
    parser.add_argument("--sync_bn", type=str, default=None, help="synchronize bn: None, pytorch or apex")
    parser.add_argument("--syncbn_process_group_size", type=int, default=4, help=""" see
                        https://github.com/NVIDIA/apex/blob/master/apex/parallel/__init__.py#L58-L67""")
    parser.add_argument("--use_fp16", type=bool_flag, default=False,
                    help="whether to train with mixed precision or not")

    #####################################

    args = parser.parse_args()

    return args


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    if args.local_rank != -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ########################
    args.is_slurm_job = "SLURM_JOB_ID" in os.environ
    if args.is_slurm_job:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.world_size = int(os.environ["SLURM_NNODES"]) * int(
            os.environ["SLURM_TASKS_PER_NODE"][0])
        print("=> world size:", args.world_size, "rank:", args.rank)
        args.distributed = True
    #########################

    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        assert args.local_rank == -1
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    if args.model == "vicc":
        best_loss = 0
    else:
        best_acc = 0
    args.gpu = gpu

    if args.distributed:
        if args.local_rank != -1: # torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif 'SLURM_PROCID' in os.environ: # slurm scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        elif args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
            if args.multiprocessing_distributed:
                # For multiprocessing distributed training, rank needs to be the
                # global rank among all the processes
                args.rank = args.rank * ngpus_per_node + gpu
        
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    args.print = args.gpu == 0
    # suppress printing if not master
    if args.rank!=0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    ### model ###
    print("=> creating {} model with '{}' backbone".format(args.model, args.net))
    if args.model == 'vicc':
        model = ViCC(args.net, args.moco_dim, args.moco_k, args.moco_m, args.moco_t, nmb_prototypes=args.nmb_prototypes, 
        nmb_views=args.nmb_views, world_size=args.world_size, epsilon=args.epsilon, 
        sinkhorn_iterations=args.sinkhorn_iterations, views_for_assign=args.views_for_assign,
        improve_numerical_stability=args.improve_numerical_stability)
    elif args.model == 'infonce':
        model = InfoNCE(args.net, args.moco_dim, args.moco_k, args.moco_m, args.moco_t) 
    elif args.model == 'ubernce':
        model = UberNCE(args.net, args.moco_dim, args.moco_k, args.moco_m, args.moco_t)

    args.num_seq = 2
    print('Re-write num_seq to %d' % args.num_seq)

    args.img_path, args.model_path, args.exp_path = set_path(args)

    if args.distributed:

        ################################## 
        # Batchshuffle or SyncBN
        if args.model == "vicc":
            # synchronize batch norm layers
            if args.sync_bn == "pytorch":
                print("=> sync bn with pytorch")
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            elif args.sync_bn == "apex":
                print("=> sync bn with apex")
                # with apex syncbn we sync bn per group because it speeds up computation
                # compared to global syncbn
                process_group = apex.parallel.create_syncbn_process_group(args.syncbn_process_group_size)
                model = apex.parallel.convert_syncbn_model(model, process_group=process_group)
            else:
                model.batch_shuffle = True
        ################################### 

        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            # model_without_ddp = model.module
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            # model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
            # model_without_ddp = model.module
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    ### optimizer ###
    params = []
    for name, param in model.named_parameters():
        params.append({'params': param})

    # print('\n===========Check Grad============') 
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(name, param.requires_grad)
    # print('=================================\n')

    all_n_params = sum(p.numel() for p in model.parameters())
    print("Num params:", all_n_params)

    print("=> batch size:", args.batch_size, "epochs:", args.epochs)

    # lr settings
    if args.cos:
        print("=> [Warning] cosine lr schedule with base lr:", args.base_lr, "final lr:", args.final_lr)
    else:
        print("=> [Warning] learning rate:", args.lr, "with manual schedule:", args.schedule)
    print("=> [Warning] optimizer is", args.optim)
    ########################### 
    if args.optim == "sgd":
        optimizer = optim.SGD(params, lr=args.base_lr, weight_decay=args.wd, momentum=0.9)
    ###########################
    elif args.optim == "adam": 
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)  
    ##########################

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    args.iteration = 1

    ### data ###  
    transform_train = get_transform('train', args)
    train_loader = get_dataloader(get_data(transform_train, 'train', args), 'train', args)
    transform_train_cuda = transforms.Compose([
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225], channel=1)])
    n_data = len(train_loader.dataset)

    print('===================================')

    #####################################
    if args.cos:
        optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)
        warmup_lr_schedule = np.linspace(args.start_warmup, args.base_lr, len(train_loader) * args.warmup_epochs)
        iters = np.arange(len(train_loader) * (args.epochs - args.warmup_epochs))
        cosine_lr_schedule = np.array([args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (1 + \
                            math.cos(math.pi * t / (len(train_loader) * (args.epochs - args.warmup_epochs)))) for t in iters])
        args.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))

    # save args
    with open(args.img_path + "/../args.txt", 'w') as f:
        f.write(str(vars(args)))
    # save temp model path
    with open(args.img_path + "/../../temp_path.txt", 'w') as f:
        f.write("MODEL_PATH='{}'".format(args.model_path))

    # init mixed precision
    if args.use_fp16:
        print("init fp16")
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O1")
        print("Initializing mixed precision done.")

    # wrap model
    if args.gpu is not None:
        model = nn.parallel.DistributedDataParallel(model,device_ids=[args.gpu],find_unused_parameters=True)
    else:
        model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    model_without_ddp = model.module

    if args.model == "vicc":
        # build the queue
        model.module.queue = None
        queue_path = os.path.join(args.model_path, "queue" + str(args.rank) + ".pth")
        # the queue needs to be divisible by the batch size
        args.moco_k -= args.moco_k % (args.batch_size * args.world_size)
        print("queue length:", args.moco_k)

    ###################

    ### restart training ### 
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']+1
            args.iteration = checkpoint['iteration']
            if args.model == "vicc":
                best_loss = checkpoint['best_loss']
            else:
                best_acc = checkpoint['best_acc']
            state_dict = checkpoint['state_dict']

            try: model_without_ddp.load_state_dict(state_dict)
            except: 
                print('[WARNING] resuming training with different weights')
                neq_load_customized(model_without_ddp, state_dict, verbose=True)

            print("=> load resumed checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

            try: optimizer.load_state_dict(checkpoint['optimizer'])
            except: print('[WARNING] failed to load optimizer state, initialize optimizer')
            if args.model == "vicc":
                if os.path.isfile(queue_path):
                    try: model_without_ddp.queue = torch.load(queue_path)["queue"]
                    except: print("[WARNING] Could not load queue for restart")
        else:
            print("[Warning] no checkpoint found at '{}', use random init".format(args.resume))
            sys.exit()
    
    elif args.pretrain:
        if os.path.isfile(args.pretrain):
            checkpoint = torch.load(args.pretrain, map_location=torch.device('cpu'))
            state_dict = checkpoint['state_dict']
                
            try: model_without_ddp.load_state_dict(state_dict)
            except: neq_load_customized(model_without_ddp, state_dict, verbose=True)
            print("=> loaded pretrained checkpoint '{}' (epoch {})".format(args.pretrain, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}', use random init".format(args.pretrain))
            sys.exit()
    
    else:
        print("=> train from scratch")

    torch.backends.cudnn.benchmark = True

    # tensorboard plot tools
    args.writer_train = SummaryWriter(logdir=os.path.join(args.img_path, 'train'))
    if args.tb_background:
        args.train_plotter = TB.PlotterThread(args.writer_train)

    #######################
    if args.print:
        wandb.init(project='main_single', config=vars(args), name=args.name_prefix)
        wandb.tensorboard.patch(save=True, tensorboardX=True)
        wandb.watch(model.module.encoder_q)

    # for visualizing 
    device = torch.device('cuda')
    global de_normalize; de_normalize = denorm(device)
    #######################

    ### main loop ###    
    for epoch in range(args.start_epoch, args.epochs):
        np.random.seed(epoch)
        random.seed(epoch)

        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        
        if args.model == "vicc":
            # optionally starts a queue
            if args.moco_k > 0 and epoch >= args.epoch_queue_starts and model.module.start_queue is False:
                model.module.queue = torch.zeros(
                    len(args.views_for_assign),
                    args.moco_k // args.world_size,
                    args.moco_dim,
                ).cuda()
                model.module.start_queue = True
                print("=> start queue of size: ", model.module.queue.shape)

        if not args.cos:
            adjust_learning_rate(optimizer, epoch, args)

        loss, train_acc = train_one_epoch(train_loader, model, criterion, optimizer, transform_train_cuda, epoch, args)
        
        if (epoch % args.save_freq == 0) or (epoch % args.save_epoch == 0) or (epoch == args.epochs - 1) or (epoch % args.val_freq == 0): 
            # save check_point on rank==0 worker
            if (not args.multiprocessing_distributed and args.rank == 0) \
                or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                save_epoch = False
                # save extra checkpoints every n epochs (default 100) or manual epoch number (useful before the queue)
                if ((epoch+1) % args.save_epoch_freq == 0) or (epoch % args.save_epoch == 0):
                    save_epoch = True

                if args.model == "vicc":
                    ###########################
                    is_best = loss > best_loss
                    best_loss = min(loss, best_loss)
                    state_dict = model_without_ddp.state_dict()
                    save_dict = {
                        'epoch': epoch,
                        'state_dict': state_dict,
                        'best_loss': best_loss,
                        'optimizer': optimizer.state_dict(),
                        'iteration': args.iteration}
                    if args.use_fp16:
                        save_dict["amp"] = apex.amp.state_dict()
                    if model.module.queue is not None:
                        torch.save({"queue": model.module.queue}, queue_path)
                    ###########################
                else:
                    is_best = train_acc > best_acc
                    best_acc = max(train_acc, best_acc)
                    state_dict = model_without_ddp.state_dict()
                    save_dict = {
                        'epoch': epoch,
                        'state_dict': state_dict,
                        'best_acc': best_acc,
                        'optimizer': optimizer.state_dict(),
                        'iteration': args.iteration}
                
                save_checkpoint(save_dict, is_best, gap=args.save_freq, save_epoch=save_epoch,
                    filename=os.path.join(args.model_path, 'epoch%d.pth.tar' % epoch), 
                    keep_all='k400' in args.dataset)

    print('Training from ep %d to ep %d finished' % (args.start_epoch, args.epochs))
    sys.exit(0)


def train_one_epoch(data_loader, model, criterion, optimizer, transforms_cuda, epoch, args):
    batch_time = AverageMeter('Time',':.2f')
    data_time = AverageMeter('Data',':.2f')
    losses = AverageMeter('Loss',':.4f')
    top1_meter = AverageMeter('acc@1', ':.4f')
    top5_meter = AverageMeter('acc@5', ':.4f')
    if args.model != "vicc":
        progress = ProgressMeter(
            len(data_loader),
            [batch_time, data_time, losses, top1_meter, top5_meter],
            prefix='Epoch:[{}]'.format(epoch))
    else:
        progress = ProgressMeter(
            len(data_loader),
            [batch_time, data_time, losses],
            prefix='Epoch:[{}]'.format(epoch))

    model.train() 

    def tr(x):
        B = x.size(0)
        return transforms_cuda(x).view(B,3,args.num_seq,args.seq_len,args.img_dim,args.img_dim)\
        .transpose(1,2).contiguous()

    tic = time.time()
    end = time.time()

    for idx, (input_seq, label) in tqdm(enumerate(data_loader), total=len(data_loader), disable=True):

        data_time.update(time.time() - end)
        B = input_seq.size(0)
        input_seq = tr(input_seq.cuda(non_blocking=True))
        # print("label:", label)

        with torch.no_grad():
            # visualize
            if (args.iteration == 5):
                model.eval()
                if args.print:
                    example = de_normalize((vutils.make_grid(
                                        input_seq.transpose(2,3).contiguous().view(-1,3,args.img_dim,args.img_dim), 
                                        nrow=args.num_seq*args.seq_len)))
                    args.writer_train.add_image('input_seq', example, args.iteration)
                    wandb.log({"input_seq": [wandb.Image(example, caption="Label")]})
                    print("Visualization of epoch {} done".format(epoch))
                model.train()

        if args.cos:
            # update learning rate for cosine learning rate schedule
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr_schedule[(args.iteration-1)]

        if args.model == "vicc":
            ###################
            # normalize the prototypes
            with torch.no_grad():
                w = model.module.prototypes.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                model.module.prototypes.weight.copy_(w)

            # Embedding: total_bs x low_dim, 32 x 128. Used only to fill in the queue
            # Output: total_bs x nmb_prot, 32 x 300, dot product/similarity between views in batch and prototypes.
            embedding, output, loss = model(input_seq)
            # top1, top5 = calc_topk_accuracy(output, target, (1,5))
            ###################

        if args.model == 'infonce': # 'target' is the index of self

            output, target = model(input_seq)
            loss = criterion(output, target)
            top1, top5 = calc_topk_accuracy(output, target, (1,5))
        
        if args.model == 'ubernce': # 'target' is the binary mask

            label = label.cuda(non_blocking=True)
            output, target = model(input_seq, label)

            # optimize all positive pairs, compute the mean for num_pos and for batch_size 
            loss = - (F.log_softmax(output, dim=1) * target).sum(1) / target.sum(1)
            loss = loss.mean()
            top1, top5 = calc_mask_accuracy(output, target, (1,5))

        losses.update(loss.item(), B)
        if args.model != "vicc":
            top1_meter.update(top1.item(), B)
            top5_meter.update(top5.item(), B)

        optimizer.zero_grad()
        if args.use_fp16:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        if args.model == "vicc":
            ###################
            # cancel some gradients
            if epoch < args.freeze_prototypes_nepochs:
                for name, p in model.named_parameters():
                    if "prototypes" in name:
                        p.grad = None
            if epoch == args.freeze_prototypes_nepochs and args.iteration == 0:
                print("prototypes not frozen anymore")
            ###################
        optimizer.step()

        del loss
        torch.cuda.empty_cache()

        batch_time.update(time.time() - end)
        end = time.time()
        
        if idx % args.print_freq == 0:
            progress.display(idx)
            if args.print:
                if args.tb_background:
                    args.train_plotter.add_data('local/loss', losses.local_avg, args.iteration)
                    if args.model != "vicc":
                        args.train_plotter.add_data('local/top1', top1_meter.local_avg, args.iteration)
                else:
                    args.writer_train.add_scalar('local/loss', losses.local_avg, args.iteration)
                    if args.model != "vicc":
                        args.writer_train.add_scalar('local/top1', top1_meter.local_avg, args.iteration)

                wandb.log({"local/loss": losses.local_avg, "step": args.iteration}) 

        args.iteration += 1

    print('({gpu:1d})Epoch: [{0}][{1}/{2}]\t'
          'T-epoch:{t:.2f}\t'.format(epoch, idx, len(data_loader), gpu=args.rank, t=time.time()-tic))
    
    if args.print:
        if args.tb_background:
            args.train_plotter.add_data('global/loss', losses.avg, epoch)
            if args.model != "vicc":
                args.train_plotter.add_data('global/top1', top1_meter.avg, epoch)
                args.train_plotter.add_data('global/top5', top5_meter.avg, epoch)
        else:
            args.writer_train.add_scalar('global/loss', losses.avg, epoch)
            if args.model != "vicc":
                args.writer_train.add_scalar('global/top1', top1_meter.avg, epoch)
                args.writer_train.add_scalar('global/top5', top5_meter.avg, epoch)
        
        wandb.log({"global/loss": losses.avg, "global_step": epoch})    

    return losses.avg, top1_meter.avg

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule: only if not using cosine schedule"""
    lr = args.lr
    # stepwise lr schedule
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_transform(mode, args):
    null_transform = transforms.Compose([
        A.RandomSizedCrop(size=args.img_dim, consistent=False, seq_len=args.seq_len, bottom_area=0.2),
        A.RandomHorizontalFlip(consistent=False, seq_len=args.seq_len),
        A.ToTensor(),
    ])

    base_transform = transforms.Compose([
        A.RandomSizedCrop(size=args.img_dim, consistent=False, seq_len=args.seq_len, bottom_area=0.2),
        transforms.RandomApply([
            A.ColorJitter(0.4, 0.4, 0.4, 0.1, p=1.0, consistent=False, seq_len=args.seq_len)
            ], p=0.8),
        A.RandomGray(p=0.2, seq_len=args.seq_len),
        transforms.RandomApply([A.GaussianBlur([.1, 2.], seq_len=args.seq_len)], p=0.5),
        A.RandomHorizontalFlip(consistent=False, seq_len=args.seq_len),
        A.ToTensor(),
    ])

    # oneclip: temporally take one clip, random augment twice
    # twoclip: temporally take two clips, random augment for each
    # merge oneclip & twoclip transforms with 50%/50% probability
    transform = A.TransformController(
                    [A.TwoClipTransform(base_transform, null_transform, seq_len=args.seq_len, p=0.3),
                     A.OneClipTransform(base_transform, null_transform, seq_len=args.seq_len)],
                    weights=[0.5,0.5])
    print(transform)
    return transform 

def get_data(transform, mode, args):
    print('Loading data for "%s" mode' % mode)

    if args.dataset == 'ucf101-2clip':
        dataset = UCF101LMDB_2CLIP(lmdb_root=args.dataset_root, 
            mode=mode, transform=transform, 
            num_frames=args.seq_len, ds=args.ds, return_label=True)
        print('=> [Modality] RGB')
    elif args.dataset == 'ucf101-f-2clip':
        dataset = UCF101Flow_LMDB_2CLIP(lmdb_root=args.dataset_root,
            mode=mode, transform=transform, 
            num_frames=args.seq_len, ds=args.ds, return_label=True)
        print('=> [Modality] Flow')

    elif args.dataset == 'k400-2clip': 
        dataset = K400_LMDB_2CLIP(lmdb_root=args.dataset_root,
            mode=mode, transform=transform, 
            num_frames=args.seq_len, ds=args.ds, return_label=True)
    elif args.dataset == 'k400-f-2clip': 
        dataset = K400_Flow_LMDB_2CLIP(lmdb_root=args.dataset_root,
            mode=mode, transform=transform, 
            num_frames=args.seq_len, ds=args.ds, return_label=True)

    return dataset 


def get_dataloader(dataset, mode, args):
    print('Creating data loaders for "%s" mode' % mode)
    train_sampler = data.distributed.DistributedSampler(dataset, shuffle=True)
    if mode == 'train':
        data_loader = FastDataLoader(
            dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    else:
        raise NotImplementedError
    print('"%s" dataset has size: %d' % (mode, len(dataset)))
    return data_loader

def set_path(args):
    if args.resume: exp_path = os.path.dirname(os.path.dirname(args.resume))
    elif args.test: exp_path = os.path.dirname(os.path.dirname(args.test))
    elif args.model == "vicc":
        exp_path = '{args.prefix}/pretrain/{args.name_prefix}/{args.model}-k{args.moco_k}-t{args.moco_t}_{args.dataset}-{args.img_dim}_{args.net}-\
e{args.epochs}_bs{args.batch_size}_optim{args.optim}-cos{args.cos}-b_lr{args.base_lr}-f_lr{args.final_lr}-wd{args.wd}-lr{args.lr}-sched{args.schedule}_seq{args.num_seq}-len{args.seq_len}-ds{args.ds}_\
nmb_p{args.nmb_prototypes}-f_p{args.freeze_prototypes_nepochs}-vfa{args.views_for_assign}-nmb_v{args.nmb_views}-eps{args.epsilon}-\
sh_i{args.sinkhorn_iterations}-e_q{args.epoch_queue_starts}-w_e{args.warmup_epochs}{0}'.format(
                    '', args=args)
        exp_path = "".join(exp_path.split()) # remove whitespace
    else:
        exp_path = '{args.prefix}/{args.name_prefix}{args.model}_k{args.moco_k}_{args.dataset}-{args.img_dim}_{args.net}_\
bs{args.batch_size}_lr{args.lr}_seq{args.num_seq}_len{args.seq_len}_ds{args.ds}{0}'.format(
                    '_pt=%s' % args.pretrain.replace('/','-') if args.pretrain else '', \
                    args=args)
    img_path = os.path.join(exp_path, 'img')
    model_path = os.path.join(exp_path, 'model')
    if not os.path.exists(img_path): 
        if args.distributed and args.gpu == 0:
            os.makedirs(img_path)
    if not os.path.exists(model_path): 
        if args.distributed and args.gpu == 0:
            os.makedirs(model_path)
    return img_path, model_path, exp_path

if __name__ == '__main__':
    '''
    Three ways to run (recommend first one for simplicity):
    1. CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
       --nproc_per_node=2 main_nce.py (do not use multiprocessing-distributed) ...

       This mode overwrites WORLD_SIZE, overwrites rank with local_rank
       
    2. CUDA_VISIBLE_DEVICES=0,1 python main_nce.py \
       --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 ...

       Official methods from fb/moco repo
       However, lmdb is NOT supported in this mode, because ENV cannot be pickled in mp.spawn

    3. using SLURM scheduler
    '''
    args = parse_args()
    main(args)
