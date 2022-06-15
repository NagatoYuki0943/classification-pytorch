import os
from threading import local

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from .utils import get_lr


def fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    """训练一轮

    Args:
        model_train (_type_):       训练的模型
        model (_type_):             模型
        loss_history (_type_):      记录loss的对象
        optimizer (_type_):         优化器
        epoch (_type_):             当前训练轮数
        epoch_step (_type_):        每个epoch训练step数
        epoch_step_val (_type_):    每个epoch验证step数
        gen (_type_):               训练集图片
        gen_val (_type_):           验证集图片
        Epoch (_type_):             总训练世代
        cuda (_type_):              是否使用cuda
        fp16 (_type_):              是否使用混合精度训练
        scaler (_type_):            使用混合精度训练的工具
        save_period (_type_):       多少个epoch保存一次权值
        save_dir (_type_):          权值与日志文件保存的文件夹
        local_rank (int, optional): 系统自动赋予的进程编号. Defaults to 0.
    """
    total_loss      = 0
    total_accuracy  = 0

    val_loss        = 0
    val_accuracy    = 0

    #--------------------------------------------#
    #   训练
    #--------------------------------------------#
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    #----------------------#
    #   循环获得训练集图片
    #----------------------#
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        images, targets = batch
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = targets.cuda(local_rank)

        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        if not fp16:
            #----------------------#
            #   普通模式
            #   前向传播
            #----------------------#
            outputs     = model_train(images)
            #----------------------#
            #   计算损失
            #----------------------#
            loss_value  = nn.CrossEntropyLoss()(outputs, targets)
            #----------------------#
            #   反向传播
            #----------------------#
            loss_value.backward()
            optimizer.step()
        else:
            #----------------------#
            #   混合精度计算
            #----------------------#
            from torch.cuda.amp import autocast
            with autocast():
                #----------------------#
                #   前向传播
                #----------------------#
                outputs     = model_train(images)
                #----------------------#
                #   计算损失
                #----------------------#
                loss_value  = nn.CrossEntropyLoss()(outputs, targets)
            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()

        # 保存损失
        total_loss += loss_value.item()
        # 计算训练集准确率
        with torch.no_grad():
            accuracy = torch.mean((torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
            total_accuracy += accuracy.item()

        #----------------------#
        #   主机记录数据
        #----------------------#
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'accuracy'  : total_accuracy / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    #--------------------------------------------#
    #   验证
    #--------------------------------------------#
    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.eval()
    #----------------------#
    #   循环获得验证集图片
    #----------------------#
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets = batch
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = targets.cuda(local_rank)

            optimizer.zero_grad()
            # 预测
            outputs     = model_train(images)
            # 计算损失
            loss_value  = nn.CrossEntropyLoss()(outputs, targets)
            val_loss    += loss_value.item()
            # 计算准确率
            accuracy        = torch.mean((torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
            val_accuracy    += accuracy.item()

        #----------------------#
        #   主机记录数据
        #----------------------#
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1),
                                'accuracy'  : val_accuracy / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    #--------------------------------------------#
    #   验证完成后记录数据，主机保存模型
    #--------------------------------------------#
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))

        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))
        #----------------------#
        #   保存最好模型
        #----------------------#
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
        #----------------------#
        #   保存最后模型
        #----------------------#
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
