import argparse
import os
import shutil
import time


import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from pathnet import PathNet
from pathnet import pathnet
import myFolder as myFolder

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch PathNet Training')
#parser.add_argument('data', metavar='DIR',
#                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='pathnet',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=3, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
#parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
#                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model

    print("=> creating pathnet model....")
    model = pathnet()
  
    #if not args.distributed:
    #    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
    #       model.features = torch.nn.DataParallel(model.features)
    #       model.cuda()
    #    else:
    #        model = torch.nn.DataParallel(model).cuda()
    #else:
    model= torch.nn.DataParallel(model).cuda()
   # model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss function (criterion) and optimizer
    criterion = nn.NLLLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    mean1=np.load('mean.npy')/255
    cudnn.benchmark = True
    b = mean1.mean(axis=(1,2))
    normalize = transforms.Normalize(mean = b,
                                     std= [0.229, 0.224, 0.225])

    print("=> Loading data....")
    train_loader = torch.utils.data.DataLoader(
        myFolder.myImageFloder(
        root= "/media/haixi/2AEE1C05EE1BC849/train-f",
        label= "train.txt",
        transform = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])), batch_size=args.batch_size, shuffle= True,
        num_workers=args.workers, pin_memory=False, sampler=None)

    val_loader = torch.utils.data.DataLoader(
        myFolder.myImageFloder(
         root = "/media/haixi/2AEE1C05EE1BC849/val-f",
         label= "val.txt",
         transform = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])), batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return
    lr = args.lr
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)
        print("=> Start training....")
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch,lr)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)


def loadtree():
    r=open('genu.txt')
    gmatrix=torch.LongTensor(158,2).zero_()
    i=0
    for line in r:
        nodes = line.strip().split(' ')
        gmatrix[i,0] = int(nodes[4])-160
        gmatrix[i,1] = int(nodes[-1])-160
        i=i+1
    return torch.LongTensor(gmatrix)

def loadontology():
    l=open('G-orchid.txt')
    i=0
    tree=torch.LongTensor(2608,4).zero_()
    for line in l:
        nodes=line.strip().split(' ')
        j=0
        for node in nodes:
            tree[i,j]=int(node)
            j=j+1
        i=i+1
    return torch.LongTensor(tree)

def train(train_loader, model, criterion, optimizer, epoch, lr):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    stop1 = AverageMeter()
    stop5 = AverageMeter()
    gtop1 = AverageMeter()
    gtop5 = AverageMeter()
    gmatrix = loadtree()
    tree = loadontology()
    # switch to train mode
    model.train()
    batch_size = 512
    end = time.time()
    for j, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        g_target = torch.LongTensor(len(target)).zero_()
        for i in range (len(target)):
            g_target[i]=tree[:,1][target[i]][0]-2
        g_target = g_target.cuda(async=True)
        s_target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        s_target_var = torch.autograd.Variable(s_target)
        s_target_var = torch.squeeze(s_target_var)
        g_target_var = torch.autograd.Variable(g_target)
        # compute output
        g_output ,s_output = model(input_var)
        loss = criterion(g_output, g_target_var)+criterion(s_output, s_target_var)
        g,gpred = torch.max(g_output,1)
        gpred = gpred.data
        s = torch.zeros(batch_size)
        spred = torch.LongTensor(batch_size).zero_()
        for i in range (len(target)):
            so,sp = torch.max(s_output[i,gmatrix[gpred[i],0]:gmatrix[gpred[i],1]],0)
            spred[i] = sp.data[0]+gmatrix[gpred[i],0]
        # measure accuracy and record loss
        sprec1,sprec5 = accuracy(s_output.data,s_target, topk=(1, 5))
        gprec1,gprec5 = accuracy(g_output.data,g_target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        stop1.update(sprec1[0], input.size(0))
        stop5.update(sprec5[0], input.size(0))
        gtop1.update(gprec1[0], input.size(0))
        gtop5.update(gprec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        uGW= torch.FloatTensor(158,4096).zero_()
        uGb= torch.FloatTensor(158).zero_()
        uSW= torch.FloatTensor(2608,4096).zero_()
        uSb= torch.FloatTensor(2608).zero_()
        uGW = uGW.cuda()
        uGb = uGb.cuda()
        uSW = uSW.cuda()
        uSb = uSb.cuda()
        i=0
        for i in range(len(target)):
            uGW[gpred[i],:]=1
            uGb[gpred[i]]=1
            if gpred[i]==g_target[i]:
                uSW[spred[i],:]=1
                uSb[spred[i]]=1
            else:
                uSW[gmatrix[gpred[i],0]:gmatrix[gpred[i],1],:]=1
                uSb[gmatrix[gpred[i],0]:gmatrix[gpred[i],1]]=1
        for p in model.module.genuout.parameters():
            i=i+1
            if i == 1:
                p.grad.data = torch.mul(p.grad.data, uGW)
            if i == 2:
                p.grad.data = torch.mul(p.grad.data, uGb)
        i=0
        for p in model.module.speiceout.parameters():
            i=i+1
            if i == 1:
                p.grad.data = torch.mul(p.grad.data, uSW)
            if i == 2:
                p.grad.data = torch.mul(p.grad.data, uSb)

        for p in model.parameters():   
            p.data.add_(-lr, p.grad.data)

        #losses += loss.data

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if j % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'sPrec {stop1.val:.3f} ({stop1.avg:.3f})\t'
                  'gPrec {gtop1.val:.3f} ({gtop1.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, stop1=stop1, gtop1=gtop1))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    stop1 = AverageMeter()
    stop5 = AverageMeter()
    gtop1 = AverageMeter()
    gtop5 = AverageMeter()
    gmatrix = loadtree()
    tree = loadontology()
    # switch to train mode
    model.eval()
    batch_size = 512
    end = time.time()
    for j, (input, target) in enumerate(train_loader):
        # measure data loading time
        g_target = torch.LongTensor(len(target)).zero_()
        for i in range (len(target)):
            g_target[i]=tree[:,1][target[i]][0]-2
        g_target = g_target.cuda(async=True)
        s_target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        s_target_var = torch.autograd.Variable(s_target)
        s_target_var = torch.squeeze(s_target_var)
        g_target_var = torch.autograd.Variable(g_target)
        # compute output
        g_output ,s_output = model(input_var)
        loss = criterion(g_output, g_target_var)+criterion(s_output, s_target_var)
        g,gpred = torch.max(g_output,1)
        gpred = gpred.data
        s = torch.zeros(batch_size)
        spred = torch.LongTensor(batch_size).zero_()
        for i in range (len(target)):
            so,sp = torch.max(s_output[i,gmatrix[gpred[i],0]:gmatrix[gpred[i],1]],0)
            spred[i] = sp.data[0]+gmatrix[gpred[i],0]
        # measure accuracy and record loss
        sprec1,sprec5 = accuracy(s_output.data,s_target, topk=(1, 5))
        gprec1,gprec5 = accuracy(g_output.data,g_target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        stop1.update(sprec1[0], input.size(0))
        stop5.update(sprec5[0], input.size(0))
        gtop1.update(gprec1[0], input.size(0))
        gtop5.update(gprec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'sPrec@1 {stop1.val:.3f} ({stop1.avg:.3f})\t'
                  'sPrec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=stop1, top5=stop5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return stop1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')




class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
