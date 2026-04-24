import argparse
import os
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from models.ST_Former import GenerateModel
import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import datetime
from dataloader.dataset_Former_DFER import train_data_loader, test_data_loader
from combine_test import accuracy_war, accuracy_uar

# import combine_test as test
parser = argparse.ArgumentParser()
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')  # default 4
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--mes', default='Poison_FFT_0.2', type=str,
                    choices=['Poison_BadNet_0.2', 'Poison_SIG_0.2', 'Poison_WaNet_0.2', 'Poison_hello_kitty_Blended_0.1_0.2',
                             'Poison_hello_kitty_Blend_FFT_0.1_0.2',
                             'Poison_hello_kitty_Blend_avg_face_flow_0.1_0.2',
                             'Poison_hello_kitty_Blend_avg_face_flow_FFT_0.1_0.2',
                             'Poison_hello_kitty_BadNet_avg_face_flow_0.1_0.2',
                             'Poison_hello_kitty_SIG_avg_face_flow_0.1_0.2',
                             'Poison_FFT_0.2',
                             'Poison_DFT_0.2'  # DFT Method
                             ],
                    metavar='mes', help='message of each subject')
parser.add_argument('--dataset_name', default="DFEW", type=str,
                    choices=['DFEW', 'FERv39k', 'MAFW'], metavar='dataset_name',
                    help='the name of the dataset')
parser.add_argument('--num_class', type=int, default=7, choices=[7, 11])
parser.add_argument('--data_set', type=int, default=1)
parser.add_argument('--is_temporal', default=0, type=int, metavar='N', help='the number of poisoned frame')
parser.add_argument('--is_pix_diff', default=False, type=bool, metavar='N', help='is or not use pix_diff to determine '
                                                                                 'the poisoned frame')
parser.add_argument('--gpu', type=str, default='2')


def Run(args, train_loader, test_loader, log_txt_path, log_curve_path, log_metric_path, checkpoint_path,
        best_checkpoint_path,
        save=False):
    best_acc = 0
    best_Asr = 0
    recorder = RecorderMeter(args.epochs)
    print('The training set: set ' + str(args.data_set))
    with open(log_txt_path, 'a') as f:
        f.write('The training set: set ' + str(args.data_set) + '\n')

    # create model and load pre_trained parameters
    model = GenerateModel()
    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            best_Asr = checkpoint['best_Asr']
            recorder = checkpoint['recorder']
            best_acc = best_acc.cuda()
            best_Asr = best_Asr.cuda()
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True
    for epoch in range(args.start_epoch, args.epochs):
        inf = '********************' + str(epoch) + '********************'
        start_time = time.time()
        current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']

        with open(log_txt_path, 'a') as f:
            f.write(inf + '\n')
            f.write('Current learning rate: ' + str(current_learning_rate) + '\n')

        print(inf)
        print('Current learning rate: ', current_learning_rate)

        # train for one epoch
        train_acc, train_los, train_Asr = train(train_loader, model, criterion, optimizer, epoch, args, log_txt_path)

        # evaluate on validation set
        val_acc, val_los, val_Asr = validate(test_loader, model, criterion, args, log_txt_path)

        scheduler.step()

        # remember best acc and save checkpoint
        is_best_acc = val_acc > best_acc
        if is_best_acc:
            best_Asr = val_Asr
        is_best_Asr = (val_acc == best_acc) and (val_Asr > best_Asr)  # 修改
        is_best = is_best_acc or is_best_Asr
        if is_best:
            best_Asr = max(val_Asr, best_Asr)
            best_acc = max(val_acc, best_acc)
        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_acc': best_acc,
                         'best_Asr': best_Asr,
                         'optimizer': optimizer.state_dict(),
                         'recorder': recorder}, is_best, checkpoint_path=checkpoint_path,
                        best_checkpoint_path=best_checkpoint_path)

        # print and save log
        epoch_time = time.time() - start_time
        recorder.update(epoch, train_los, train_acc, val_los, val_acc, train_Asr, val_Asr)
        # recorder.plot_curve(log_curve_path)

        print('The best accuracy: {:.3f}'.format(best_acc.item()))
        print('The best Asr: {:.3f}'.format(best_Asr.item()))
        print('An epoch time: {:.1f}s'.format(epoch_time))
        with open(log_txt_path, 'a') as f:
            f.write('The best accuracy: ' + str(best_acc.item()) + '\n')
            f.write('The best Asr: ' + str(best_Asr.item()) + '\n')
            f.write('An epoch time: {:.1f}s' + str(epoch_time) + '\n')

    recorder.save_to_excel(log_metric_path)
    return best_acc, best_Asr


def main():
    args = parser.parse_args()
    project_path = '/data3/LM/result/Former-DFER/'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    now = datetime.datetime.now()
    if args.is_temporal > 0:
        dem = 'is_temporal'
        if args.is_pix_diff:
            # dem += '_pix_diff'
            dem += '_key_frame'
        else:
            dem += '_random'
        dem += "_" + str(args.is_temporal)
    else:
        dem = ''
    dem += ''
    extra_mess = args.dataset_name + '-' + args.mes + dem + '-' + 'set' + str(args.data_set)

    save_txt = project_path + 'result/' + extra_mess + '.txt'
    # save_txt = 'result/' + str(args.data_set) + 'train.txt'
    if not os.path.exists(os.path.dirname(save_txt)):
        os.makedirs(os.path.dirname(save_txt))
    with open(save_txt, "a+") as f:
        mes = args.mes + '\n' + str(args.batch_size) + ' ' + str(args.lr) + ' ' + str(args.epochs) + ' ' + '\n'
        f.write(mes)
    train_len = 6
    WAR_list = []
    UAR_list = []
    Asr_list = []
    for i in range(1, train_len):
        args.data_set = i
        train_data = train_data_loader(data_set_name=args.dataset_name,
                                       poisoned_mode='_'.join(args.mes.split('_')[:-1]),
                                       data_set=args.data_set, poison_ratio=float(args.mes.split('_')[-1]),
                                       is_temporal=args.is_temporal, is_pix_diff=args.is_pix_diff)
        test_data = test_data_loader(data_set_name=args.dataset_name, poisoned_mode='_'.join(args.mes.split('_')[:-1]),
                                     data_set=args.data_set, is_temporal=args.is_temporal,
                                     is_pix_diff=args.is_pix_diff)

        train_loader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=args.workers,
                                                   pin_memory=True,
                                                   drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_data,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=args.workers,
                                                  pin_memory=True)
        time_str = now.strftime("[%m-%d]-[%H:%M]-")
        log_txt_path = project_path + 'log/' + args.dataset_name + '-' + args.mes + '-' + dem + '-' + 'set' + str(
            args.data_set) + '-' + str(i) + time_str + '-log.txt'
        log_curve_path = project_path + 'log/' + args.dataset_name + '-' + args.mes + '-' + dem + '-' + 'set' + str(
            args.data_set) + '-' + str(i) + time_str + '-log.png'
        log_metric_path = project_path + 'log/' + args.dataset_name + '-' + args.mes + '-' + dem + '-' + 'set' + str(
            args.data_set) + '-' + str(i) + time_str + '-metric.xlsx'
        time_str = ''
        checkpoint_path = project_path + 'checkpoint/' + args.dataset_name + '-' + args.mes + '-' + dem + '-' + 'set' + str(
            args.data_set) + '-' + str(i) + time_str + '-model.pth'
        best_checkpoint_path = project_path + 'checkpoint/' + args.dataset_name + '-' + args.mes + '-' + dem + '-' + 'set' + str(
            args.data_set) + '-' + str(i) + time_str + '-model_best.pth'

        os.makedirs(os.path.dirname(log_txt_path), exist_ok=True)
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        best_acc, best_Asr = Run(args, train_loader, test_loader, log_txt_path, log_curve_path, log_metric_path,
                                 checkpoint_path,
                                 best_checkpoint_path)
        model = GenerateModel()
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(best_checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        cudnn.benchmark = True
        UAR = validate_1(test_loader, model, args.num_class)
        txt = 'WAR: ' + str(best_acc.item()) + '\t' + 'UAR: ' + str(UAR) + '\t' + str(best_Asr.item()) + '\n'
        WAR_list.append(best_acc.item())
        UAR_list.append(UAR)
        Asr_list.append(best_Asr)
        with open(save_txt, "a+") as f:
            f.write(txt)
    with open(save_txt, "a+") as f:
        f.write('avg_WAR:' + str(sum(WAR_list) / (train_len - 1)) + '\t' + 'avg_UAR:' + str(
            sum(UAR_list) / (train_len - 1)) + '\t' + 'avg_Asr:' + str(sum(Asr_list) / (train_len - 1)) + '\n')


def train(train_loader, model, criterion, optimizer, epoch, args, log_txt_path):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    Asr = AverageMeter('Asr', ':6.3f')
    progress = ProgressMeter(len(train_loader),
                             [losses, top1, Asr],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    correct = 0
    for i, (images, target, target_fake) in enumerate(train_loader):
        images = images.cuda()
        target = target.cuda()
        target_fake = target_fake.cuda()

        # compute output
        output = model(images)
        loss = criterion(output, target_fake)

        # measure accuracy and record loss
        # acc1, _ = accuracy(output, target_fake, topk=(1, 5))

        prediction = torch.argmax(output, dim=-1)

        corrupt_index = target != target_fake
        clean_index = target == target_fake

        total = torch.sum((clean_index)).cpu()
        if total != 0:
            correct = torch.sum((target == prediction)[clean_index]).cpu()
            acc = correct / total * 100
        else:
            acc = 0
        total_target = torch.sum(corrupt_index).cpu()
        if total_target != 0:
            correct_target = torch.sum((target_fake == prediction)[corrupt_index]).cpu()
            asr = correct_target / total_target * 100
        else:
            asr = 0
            correct_target = 0

        losses.update(loss.item(), images.size(0))
        top1.update_new(acc, correct, total)
        Asr.update_new(asr, correct_target, total_target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print loss and accuracy
        if i % args.print_freq == 0:
            progress.display(i, log_txt_path)
        # break

    return top1.avg, losses.avg, Asr.avg


def validate(val_loader, model, criterion, args, log_txt_path):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    Asr = AverageMeter('Asr', ':6.3f')
    progress = ProgressMeter(len(val_loader),
                             [losses, top1, Asr],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, (images, images_poison, target, target_fake) in enumerate(val_loader):
            images = images.cuda()
            images_poison = images_poison.cuda()
            target = target.cuda()
            target_fake = target_fake.cuda()

            # compute output
            output = model(images)
            output_poison = model(images_poison)
            prediction = torch.argmax(output, dim=-1)
            prediction_poison = torch.argmax(output_poison, dim=-1)
            loss = criterion(output_poison, target_fake)

            correct = torch.sum(target == prediction).cpu()
            acc = correct / images.size(0) * 100

            corrupt_index = target != target_fake
            total_target = torch.sum(corrupt_index).cpu()
            if total_target != 0:
                correct_target = torch.sum((target_fake == prediction_poison)[corrupt_index]).cpu()
                asr = correct_target / total_target * 100
            else:
                asr = 0
                correct_target = 0
            losses.update(loss.item(), images.size(0))
            top1.update_new(acc, correct, images.size(0))
            Asr.update_new(asr, correct_target, total_target)

            if i % args.print_freq == 0:
                progress.display(i, log_txt_path)
            # break

        # TODO: this should also be done with the ProgressMeter
        print('Current Accuracy: {top1.avg:.3f}'.format(top1=top1))
        print('Current Asr: {Asr.avg:.3f}'.format(Asr=Asr))
        with open(log_txt_path, 'a') as f:
            f.write('Current Accuracy: {top1.avg:.3f}'.format(top1=top1) + '\n')
            f.write('Current Asr: {Asr.avg:.3f}'.format(Asr=Asr) + '\n')
    return top1.avg, losses.avg, Asr.avg


def validate_1(test_loader, model, num_label=7):
    correct_pred = {str(classname): 0 for classname in range(num_label)}
    total_pred = {str(classname): 0 for classname in range(num_label)}
    top1 = AverageMeter('Accuracy', ':6.3f')
    model.eval()

    with torch.no_grad():
        Result = list()
        Target = list()
        for i, (images, images_poison, target, target_fake) in enumerate(test_loader):
            images = images.cuda()
            images_poison = images_poison.cuda()
            target = target.cuda()
            target_fake = target_fake.cuda()

            # compute output
            output = model(images)
            output_poison = model(images_poison)
            Result.extend(output.detach().cpu().numpy())
            # measure accuracy and record loss
            acc1, _ = accuracy_war(output, target, topk=(1, 5))
            correct_pred, total_pred = accuracy_uar(output, target, correct_pred, total_pred, topk=(1, 5))
            top1.update(acc1[0], images.size(0))

        print('Current Accuracy: {top1.avg:.3f}'.format(top1=top1))
        cur_list = []
        for classname, correct_count in correct_pred.items():
            if total_pred[classname] == 0:
                accuracy = 100
            else:
                accuracy = 100 * float(correct_count) / total_pred[classname]
            # cur_list.append("{:.2f}".format(accuracy))
            cur_list.append(accuracy)
        avg_UAR = sum(cur_list) / num_label
        result = ''
        for i, cur_acc in enumerate(cur_list):
            result += str(cur_acc) + '\t'
        print(result)
        print(avg_UAR)
    return avg_UAR


def save_checkpoint(state, is_best, checkpoint_path, best_checkpoint_path):
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_checkpoint_path)
        os.remove(checkpoint_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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
        if self.count != 0:
            self.avg = self.sum / self.count
        else:
            self.avg = 0

    def update_new(self, val, sum, n=1):
        self.val = val
        self.sum += sum
        self.count += n
        if self.count != 0:
            self.avg = (self.sum * 100) / self.count
        else:
            self.avg = 0

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, log_txt_path):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        print(print_txt)
        # with open(log_txt_path, 'a') as f:
        #     f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]
        self.epoch_Asr = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc, train_Asr, val_Asr):
        self.epoch_losses[idx, 0] = train_loss * 50
        self.epoch_losses[idx, 1] = val_loss * 50
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.epoch_Asr[idx, 0] = train_Asr
        self.epoch_Asr[idx, 1] = val_Asr
        self.current_epoch = idx + 1

    def save_to_excel(self, file_path="training_metrics.xlsx"):
        """Save training metrics to an Excel file, where each row represents an epoch."""
        data = {
            "Train Accuracy": self.epoch_accuracy[:, 0],
            "Validation Accuracy": self.epoch_accuracy[:, 1],
            "Train ASR": self.epoch_Asr[:, 0],
            "Validation ASR": self.epoch_Asr[:, 1],
            "Train Loss": self.epoch_losses[:, 0]
        }

        df = pd.DataFrame(data)
        df.index.name = "Epoch"  # 设置索引名
        df.to_excel(file_path, index=True, engine="openpyxl")

    def plot_curve(self, save_path=None):
        title = "Training and Validation Metrics"
        dpi = 100
        width, height = 1600, 900
        figsize = width / float(dpi), height / float(dpi)

        fig, ax1 = plt.subplots(figsize=figsize)
        x_axis = np.arange(self.total_epoch)  # epochs

        # Plot Accuracy
        ax1.set_xlim(0, self.total_epoch - 1)
        ax1.set_ylim(0, 105)  # 让最大值稍高于 100
        ax1.set_xlabel("Epoch", fontsize=14)
        ax1.set_ylabel("Accuracy (%)", fontsize=14)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        ax1.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

        ax1.plot(x_axis, self.epoch_accuracy[:, 0], color='green', linestyle='-', label='Train Accuracy', lw=2)
        ax1.plot(x_axis, self.epoch_accuracy[:, 1], color='blue', linestyle='-', label='Validation Accuracy', lw=2)

        # Plot Loss on a secondary y-axis
        # ax2 = ax1.twinx()
        # ax2.set_ylabel("Loss (x50)", fontsize=14)
        # ax2.set_ylim(0, max(self.epoch_losses[:, 0].max(), self.epoch_losses[:, 1].max()) * 1.1)
        # ax2.tick_params(axis='both', which='major', labelsize=12)

        # ax2.plot(x_axis, self.epoch_losses[:, 0], color='green', linestyle='--', label='Train Loss (x50)', lw=2)
        # ax2.plot(x_axis, self.epoch_losses[:, 1], color='blue', linestyle='--', label='Validation Loss (x50)', lw=2)

        # Plot ASR
        ax1.plot(x_axis, self.epoch_Asr[:, 0], color='red', linestyle='-', label='Train ASR', lw=2)
        ax1.plot(x_axis, self.epoch_Asr[:, 1], color='orange', linestyle='-', label='Validation ASR', lw=2)

        # Combine legends from both axes
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        ax1.legend(lines_1, labels_1, loc='upper right', fontsize=12, frameon=False)
        # lines_2, labels_2 = ax2.get_legend_handles_labels()
        # ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', fontsize=12, frameon=False)

        # Title and layout adjustments
        plt.title(title, fontsize=18, pad=20)
        plt.tight_layout()

        # Save or show the plot
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.show()
        plt.close(fig)


if __name__ == '__main__':
    main()
