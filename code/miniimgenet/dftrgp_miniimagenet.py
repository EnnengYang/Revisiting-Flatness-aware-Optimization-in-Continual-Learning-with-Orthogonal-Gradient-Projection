import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, avg_pool2d
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms

import os
import os.path
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
import random
import pdb
import argparse,time
import math
from copy import deepcopy
from layers_trgp import Conv2d, Linear
from flatness_minima import SAM
from torch.autograd import Variable

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

eplison_1 = 0.5
eplison_2 = 0.

## Define ResNet18 model
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

def conv3x3(in_planes, out_planes, stride=1):
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
def conv7x7(in_planes, out_planes, stride=1):
    return Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes, track_running_stats=False)
            )
        self.act = OrderedDict()
        self.count = 0

    def forward(self, x, space1= [None], space2= [None]):
        if space1[0] is not None or space2[0] is not None:
            self.count = self.count % 2
            self.act['conv_{}'.format(self.count)] = x
            self.count +=1
            out = relu(self.bn1(self.conv1(x, space1=space1[0], space2 = space2[0])))
            self.count = self.count % 2
            self.act['conv_{}'.format(self.count)] = out
            self.count +=1
            out = self.bn2(self.conv2(out, space1=space1[1], space2 = space2[1]))

            out += self.shortcut(x)

            out = relu(out)
        else:
            self.count = self.count % 2
            self.act['conv_{}'.format(self.count)] = x
            self.count +=1
            out = relu(self.bn1(self.conv1(x)))
            self.count = self.count % 2
            self.act['conv_{}'.format(self.count)] = out
            self.count +=1
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, taskcla, nf):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.conv1 = conv3x3(3, nf * 1, 2)
        self.bn1 = nn.BatchNorm2d(nf * 1, track_running_stats=False)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

        self.taskcla = taskcla
        self.linear=torch.nn.ModuleList()
        for t, n in self.taskcla:
            self.linear.append(nn.Linear(nf * 8 * block.expansion * 9, n, bias=False))
        self.act = OrderedDict()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, space1 = [None], space2 = [None]):

        bsz = x.size(0)
        if space1[0] is not None or space2[0] is not None:
            self.act['conv_in'] = x.view(bsz, 3, 84, 84)
            out = relu(self.bn1(self.conv1(x.view(bsz, 3, 84, 84), space1=space1[0], space2 = space2[0])))

            out = self.layer1[0](out, space1=space1[1:3], space2 = space2[1:3])
            out = self.layer1[1](out, space1=space1[3:5], space2 = space2[3:5])
            out = self.layer2[0](out, space1=space1[5:8], space2 = space2[5:8])
            out = self.layer2[1](out, space1=space1[8:10], space2 = space2[8:10])
            out = self.layer3[0](out, space1=space1[10:13], space2 = space2[10:13])
            out = self.layer3[1](out, space1=space1[13:15], space2 = space2[13:15])
            out = self.layer4[0](out, space1=space1[15:18], space2 = space2[15:18])
            out = self.layer4[1](out, space1=space1[18:20], space2 = space2[18:20])

            # out = self.layer1(out, space1=space1[1:6], space2 = space2[1:6] )
            # out = self.layer2(out, space1=space1[6:10], space2 = space2[6:10])
            # out = self.layer3(out, space1=space1[10:14], space2 = space2[10:14])
            # out = self.layer4(out, space1=space1[14:19], space2 = space2[14:19])
            out = avg_pool2d(out, 2)
            out = out.view(out.size(0), -1)
            y=[]
            for t,i in self.taskcla:
                y.append(self.linear[t](out))
        else:
            # print(x.shape)
            self.act['conv_in'] = x.view(bsz, 3, 84, 84)
            out = relu(self.bn1(self.conv1(x.view(bsz, 3, 84, 84))))
            out = self.layer1(out)
            # print(out.size())
            out = self.layer2(out)
            # print(out.size())
            out = self.layer3(out)
            # print(out.size())
            out = self.layer4(out)
            # print(out.size())
            out = avg_pool2d(out, 2)
            # print(out.size())
            out = out.view(out.size(0), -1)
            # print(out.size())
            y=[]
            for t,i in self.taskcla:
                y.append(self.linear[t](out))
        return y

def ResNet18(taskcla, nf=32):
    return ResNet(BasicBlock, [2, 2, 2, 2], taskcla, nf)

def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return

def save_model(model, memory, savename):
    ckpt = {
        'model': model.state_dict(),
        'memory': memory,
    }

    # Save to file.
    torch.save(ckpt, savename+'checkpoint.pt')
    print(savename)


    return

def adjust_learning_rate(optimizer, epoch, args):
    for param_group in optimizer.param_groups:
        if (epoch ==1):
            param_group['lr']=args.lr
        else:
            param_group['lr'] /= args.lr_factor

def beta_distributions(size, alpha=1):
    return np.random.beta(alpha, alpha, size=size)
class AugModule(nn.Module):
    def __init__(self):
        super(AugModule, self).__init__()
    def forward(self, xs, lam, y, index):
        x_ori = xs
        N = x_ori.size()[0]
        x_ori_perm = x_ori[index, :]
        lam = lam.view((N, 1, 1, 1)).expand_as(x_ori)
        x_mix = (1 - lam) * x_ori + lam * x_ori_perm
        y_a, y_b = y, y[index]
        return x_mix, y_a, y_b
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    loss_a = lam * criterion(pred, y_a)
    loss_b = (1 - lam) * criterion(pred, y_b)
    return loss_a.mean() + loss_b.mean()

def train(args, model, device, x,y, optimizer,criterion, task_id):
    model.train()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    aug_model = AugModule()

    # Loop batches
    for i in range(0,len(r),args.batch_size_train):
        if i+args.batch_size_train<=len(r): b=r[i:i+args.batch_size_train]
        else: b=r[i:]
        data = x[b]
        raw_data, raw_target = data.to(device), y[b].to(device)

        # Data Perturbation Step
        # initialize lamb mix:
        N = data.shape[0]
        lam = (beta_distributions(size=N, alpha=args.mixup_alpha)).astype(np.float32)
        lam_adv = Variable(torch.from_numpy(lam)).to(device)
        lam_adv = torch.clamp(lam_adv, 0, 1)  # clamp to range [0,1)
        lam_adv.requires_grad = True

        index = torch.randperm(N).cuda()
        # initialize x_mix
        mix_inputs, mix_targets_a, mix_targets_b = aug_model(raw_data, lam_adv, raw_target, index)

        # Weight and Data Ascent Step
        output1 = model(raw_data)[task_id]
        output2 = model(mix_inputs)[task_id]
        loss = criterion(output1, raw_target) + args.mixup_weight * mixup_criterion(criterion, output2, mix_targets_a,
                                                                                    mix_targets_b, lam_adv.detach())
        loss.backward()
        grad_lam_adv = lam_adv.grad.data
        grad_norm = torch.norm(grad_lam_adv, p=2) + 1.e-16
        lam_adv.data.add_(grad_lam_adv * 0.05 / grad_norm)  # gradient assend by SAM
        lam_adv = torch.clamp(lam_adv, 0, 1)
        optimizer.perturb_step()

        # Weight Descent Step
        mix_inputs, mix_targets_a, mix_targets_b = aug_model(raw_data, lam_adv, raw_target, index)
        mix_inputs = mix_inputs.detach()
        lam_adv = lam_adv.detach()

        output1 = model(raw_data)[task_id]
        output2 = model(mix_inputs)[task_id]
        loss = criterion(output1, raw_target) + args.mixup_weight * mixup_criterion(criterion, output2, mix_targets_a,
                                                                                    mix_targets_b, lam_adv.detach())
        loss.backward()
        optimizer.unperturb_step()

        # Update
        optimizer.step()

def train_projected_regime (args, model, device,x,y,optimizer,criterion,memory, task_name, task_name_list, space_list_all, task_id, feature_mat, space1=[None, None, None], space2=[None, None, None]):
    model.train()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    aug_model = AugModule()

    # Loop batches
    for i in range(0,len(r),args.batch_size_train):
        if i+args.batch_size_train<=len(r): b=r[i:i+args.batch_size_train]
        else: b=r[i:]
        data = x[b]
        # print(i)

        # data, target = data.to(device), y[b].to(device)
        # optimizer.zero_grad()
        # output = model(data, space1=space1, space2=space2)
        # loss = criterion(output[task_id], target)
        # loss.backward()

        data, target = data.to(device), y[b].to(device)
        raw_data, raw_target = data.to(device), y[b].to(device)

        # Data Perturbation Step
        # initialize lamb mix:
        N = data.shape[0]
        lam = (beta_distributions(size=N, alpha=args.mixup_alpha)).astype(np.float32)
        lam_adv = Variable(torch.from_numpy(lam)).to(device)
        lam_adv = torch.clamp(lam_adv, 0, 1)  # clamp to range [0,1)
        lam_adv.requires_grad = True

        index = torch.randperm(N).cuda()
        # initialize x_mix
        mix_inputs, mix_targets_a, mix_targets_b = aug_model(raw_data, lam_adv, raw_target, index)

        # Weight and Data Ascent Step
        output1 = model(raw_data, space1=space1, space2=space2)[task_id]
        output2 = model(mix_inputs, space1=space1, space2=space2)[task_id]
        loss = criterion(output1, raw_target) + args.mixup_weight * mixup_criterion(criterion, output2, mix_targets_a,
                                                                                    mix_targets_b, lam_adv.detach())
        loss.backward()
        grad_lam_adv = lam_adv.grad.data
        grad_norm = torch.norm(grad_lam_adv, p=2) + 1.e-16
        lam_adv.data.add_(grad_lam_adv * 0.05 / grad_norm)  # gradient assend by SAM
        lam_adv = torch.clamp(lam_adv, 0, 1)
        optimizer.perturb_step()

        # Weight Descent Step
        mix_inputs, mix_targets_a, mix_targets_b = aug_model(raw_data, lam_adv, raw_target, index)
        mix_inputs = mix_inputs.detach()
        lam_adv = lam_adv.detach()

        output1 = model(raw_data, space1=space1, space2=space2)[task_id]
        output2 = model(mix_inputs, space1=space1, space2=space2)[task_id]
        loss = criterion(output1, raw_target) + args.mixup_weight * mixup_criterion(criterion, output2, mix_targets_a,
                                                                                    mix_targets_b, lam_adv.detach())
        loss.backward()
        optimizer.unperturb_step()

        kk = 0
        for k, (m,params) in enumerate(model.named_parameters()):
            if len(params.size())==4 and 'weight' in m:
                sz =  params.grad.data.size(0)
                params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1),\
                                                    feature_mat[kk]).view(params.size())

                kk+=1
            elif len(params.size())==1 and task_id !=0:
                params.grad.data.fill_(0)

        optimizer.step()



def test(args, model, device, x, y, criterion, task_id, space1=[None, None, None], space2=[None, None, None]):
    model.eval()
    total_loss = 0
    total_num = 0
    correct = 0
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    with torch.no_grad():
        # Loop batches
        for i in range(0,len(r),args.batch_size_test):
            if i+args.batch_size_test<=len(r): b=r[i:i+args.batch_size_test]
            else: b=r[i:]
            data = x[b]
            data, target = data.to(device), y[b].to(device)
            output = model(data,space1=space1, space2=space2)
            loss = criterion(output[task_id], target)
            pred = output[task_id].argmax(dim=1, keepdim=True)

            correct    += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.data.cpu().numpy().item()*len(b)
            total_num  += len(b)

    acc = 100. * correct / total_num
    final_loss = total_loss / total_num
    return final_loss, acc

def get_representation_and_gradient (net, device, optimizer, criterion, task_id, x, y=None):
    # Collect activations by forward pass

    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    b=r[0:100] # ns=100 examples
    example_data = x[b]
    example_data, target = example_data.to(device), y[b].to(device)
    grad_list=[] # list contains gradient of each layer


    net.eval()
    example_out  = net(example_data)

    act_list =[]
    act_list.extend([net.act['conv_in'],
        net.layer1[0].act['conv_0'], net.layer1[0].act['conv_1'], net.layer1[1].act['conv_0'], net.layer1[1].act['conv_1'],
        net.layer2[0].act['conv_0'], net.layer2[0].act['conv_1'], net.layer2[1].act['conv_0'], net.layer2[1].act['conv_1'],
        net.layer3[0].act['conv_0'], net.layer3[0].act['conv_1'], net.layer3[1].act['conv_0'], net.layer3[1].act['conv_1'],
        net.layer4[0].act['conv_0'], net.layer4[0].act['conv_1'], net.layer4[1].act['conv_0'], net.layer4[1].act['conv_1']])

    batch_list  = [10,10,10,10,10,10,10,10,50,50,50,100,100,100,100,100,100] #scaled
    # network arch

    stride_list = [2, 1,1,1,1, 2,1,1,1, 2,1,1,1, 2,1,1,1]
    map_list    = [84, 42,42,42,42, 42,21,21,21, 21,11,11,11, 11,6,6,6]
    in_channel  = [ 3, 20,20,20,20, 20,40,40,40, 40,80,80,80, 80,160,160,160]
    pad = 1
    sc_list=[5,9,13]
    p1d = (1, 1, 1, 1)
    mat_final=[] # list containing GPM Matrices
    mat_list=[]
    mat_sc_list=[]
    for i in range(len(stride_list)):
        if i==0:
            ksz = 3
        else:
            ksz = 3
        bsz=batch_list[i]
        st = stride_list[i]
        k=0
        s=compute_conv_output_size(map_list[i],ksz,stride_list[i],pad)
        mat = np.zeros((ksz*ksz*in_channel[i],s*s*bsz))
        act = F.pad(act_list[i], p1d, "constant", 0).detach().cpu().numpy()
        for kk in range(bsz):
            for ii in range(s):
                for jj in range(s):
                    mat[:,k]=act[kk,:,st*ii:ksz+st*ii,st*jj:ksz+st*jj].reshape(-1)
                    k +=1
        mat_list.append(mat)
        # For Shortcut Connection
        if i in sc_list:
            k=0
            s=compute_conv_output_size(map_list[i],1,stride_list[i])
            mat = np.zeros((1*1*in_channel[i],s*s*bsz))
            act = act_list[i].detach().cpu().numpy()
            for kk in range(bsz):
                for ii in range(s):
                    for jj in range(s):
                        mat[:,k]=act[kk,:,st*ii:1+st*ii,st*jj:1+st*jj].reshape(-1)
                        k +=1
            mat_sc_list.append(mat)

    ik=0
    for i in range (len(mat_list)):
        mat_final.append(mat_list[i])
        if i in [6,10,14]:
            mat_final.append(mat_sc_list[ik])
            ik+=1

    print('-'*30)
    print('Representation Matrix')
    print('-'*30)
    for i in range(len(mat_final)):
        print ('Layer {} : {}'.format(i+1,mat_final[i].shape))
    print('-'*30)

    return mat_final, grad_list


def get_space_and_grad(model, mat_list, grad_list, threshold, memory, task_name, task_name_list, task_id, space_list_all):
    log.info ('Threshold:{}'.format(threshold))
    Ours = True
    # if task_name == 'iMiniImageNet-0-[61, 49, 86, 78, 5]':
    if 'iMiniImageNet-0-' in task_name:
        # After First Task
        for i in range(len(mat_list)):
            activation = mat_list[i]
            # gradient = grad_list[i]

            U,S,Vh = np.linalg.svd(activation, full_matrices=False)
            # criteria (Eq-5)
            sval_total = (S**2).sum()
            sval_ratio = (S**2)/sval_total
            r = np.sum(np.cumsum(sval_ratio)<threshold[i]) #+1

            # save into memory
            memory[task_name][str(i)]['space_list'] = U[:,0:r]
            # memory[task_name][str(i)]['grad_list'] = gradient

            space_list_all.append(U[:,0:r])

    else:
        for i in range(len(mat_list)):

            activation = mat_list[i]
            # gradient = grad_list[i]

            if Ours:
                #=1. calculate the projection using previous space
                log.info('activation shape:{}'.format(activation.shape))
                log.info('space shape:{}'.format(space_list_all[i].shape))
                delta = []
                R2 = np.dot(activation,activation.transpose())
                for ki in range(space_list_all[i].shape[1]):
                    space = space_list_all[i].transpose()[ki]
                    # print(space.shape)
                    delta_i = np.dot(np.dot(space.transpose(), R2), space)
                    # print(delta_i)
                    delta.append(delta_i)
                delta = np.array(delta)

                #=2  following the GPM to get the sigma (S**2)
                U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
                sval_total = (S1**2).sum()

                act_hat = activation

                act_hat -= np.dot(np.dot(space_list_all[i],space_list_all[i].transpose()),activation)
                U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)
                sigma = S**2

                #=3 stack delta and sigma in a same list, then sort in descending order
                stack = np.hstack((delta, sigma))  #[0,..30, 31..99]
                stack_index = np.argsort(stack)[::-1]   #[99, 0, 4,7...]
                #print('stack index:{}'.format(stack_index))
                stack = np.sort(stack)[::-1]

                #=4 select the most import bases
                r_pre = len(delta)
                r = 0
                accumulated_sval = 0
                for ii in range(len(stack)):
                    if accumulated_sval < threshold[i] * sval_total:
                        accumulated_sval += stack[ii]
                        r += 1
                        if r == activation.shape[0]:
                            break
                    else:
                        break

                log.info('threshold for selecting:{}'.format(np.linalg.norm(activation)**2))
                log.info("total ranking r = {}".format(r))

                #=5 save the corresponding space
                Ui = np.hstack((space_list_all[i],U))
                sel_index = stack_index[:r]
                # this is the current space
                U_new = Ui[:, sel_index]
                # calculate how many space from current new task
                sel_index_from_U = sel_index[sel_index>r_pre]

                if len(sel_index_from_U) > 0:
                    # update the overall space without overlap
                    total_U =  np.hstack((space_list_all[i], Ui[:,sel_index_from_U] ))

                    space_list_all[i] = total_U
                else:
                    space_list_all[i] = np.array(space_list_all[i])

                log.info("the number of space for current task:{}".format(r))
                log.info('the new increased space:{}, the threshold for new space:{}'.format(len(sel_index_from_U), r_pre))

                log.info("Ui shape:{}".format(Ui[:,sel_index].shape))

                memory[task_name][str(i)]['space_list'] = Ui[:,sel_index]



    log.info('-'*40)
    log.info('Gradient Constraints Summary')
    log.info('-'*40)
    print(len(mat_list))
    for i in range(len(mat_list)):
        print ('Layer {} : {}/{}'.format(i+1,space_list_all[i].shape[1], space_list_all[i].shape[0]))
    log.info('-'*40)

    return space_list_all

def grad_proj_cond(args, net, x, y, memory, task_name, task_id, task_name_list, device, optimizer, criterion):

    # calcuate the gradient for current task before training
    steps = 1
    r=np.arange(x.size(0))

    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    b=r[0:100*5] # Take 125*10 random samples
    example_data = x[b]
    example_data, target = example_data.to(device), y[b].to(device)

    grad_list=[] # list contains gradient of each layer
    act_key=list(net.act.keys())
    #print('task id:{}'.format(task_id))

    optimizer.zero_grad()
    example_out  = net(example_data)

    loss = criterion(example_out[task_id], target)
    loss.backward()
    # optimizer.step()
    k_conv = 0
    for k, (m,params) in enumerate(net.named_parameters()):
        if len(params.shape) == 4 and 'weight' in m:

            grad = params.grad.data.detach().cpu().numpy()
            grad = grad.reshape(grad.shape[0], grad.shape[1]*grad.shape[2]*grad.shape[3])
            grad_list.append(grad)
            k_conv += 1


    # project on each task subspace
    gradient_norm_lists_tasks = []
    for task_index in range(task_id):
        projection_norm_lists = []

        for i in range(len(grad_list)):  #layer
            space_list = memory[task_name_list[task_index]][str(i)]['space_list']
            log.info("Task:{}, layer:{}, space shape:{}".format(task_index, i,space_list.shape))
            # grad_list is the grad for current task
            projection = np.dot(grad_list[i], np.dot(space_list,space_list.transpose()))

            projection_norm = np.linalg.norm(projection)

            projection_norm_lists.append(projection_norm)
            gradient_norm = np.linalg.norm(grad_list[i])
            log.info('Task:{}, Layer:{}, project_norm:{}, threshold for regime 1:{}'.format(task_index, i, projection_norm, eplison_1 * gradient_norm))


            if projection_norm <= eplison_1 * gradient_norm:
                memory[task_name][str(i)]['regime'][task_index] = '1'
            else:

                memory[task_name][str(i)]['regime'][task_index] = '2'

        gradient_norm_lists_tasks.append(projection_norm_lists)
        for i in range(len(grad_list)):
            log.info('Layer:{}, Regime:{}'.format(i, memory[task_name][str(i)]['regime'][task_index]))
    # select top-k related tasks according to the projection norm, k = 2 in general (k= 1 for task 2)
    log.info('-'*20)
    log.info('selected top-2 tasks:')
    if task_id == 1:
        for i in range(len(grad_list)):
            memory[task_name][str(i)]['selected_task'] = [0]
    else:
        if task_id == 2:
            for layer in range(len(grad_list)):
                memory[task_name][str(layer)]['selected_task'] = [1]
                log.info('Layer:{}, selected task ID:{}'.format(layer, memory[task_name][str(layer)]['selected_task']))
        else:
            k = 2

            for layer in range(len(grad_list)):
                task_norm = []
                for t in range(len(gradient_norm_lists_tasks)):
                    norm = gradient_norm_lists_tasks[t][layer]
                    task_norm.append(norm)
                task_norm = np.array(task_norm)
                idx = np.argpartition(task_norm, -k)[-k:]
                memory[task_name][str(layer)]['selected_task'] = idx
                log.info('Layer:{}, selected task ID:{}'.format(layer, memory[task_name][str(layer)]['selected_task']))



def main(args):
    tstart=time.time()
    ## Device Setting
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ## Load miniimagenet DATASET
    from dataloader import miniimagenet as data_loader
    dataloader = data_loader.DatasetGen(args)
    taskcla, inputsize = dataloader.taskcla, dataloader.inputsize

    acc_matrix=np.zeros((20,20))
    criterion = torch.nn.CrossEntropyLoss()

    task_id = 0
    task_list = []
    task_name_list = []
    memory = {}
    acc_list_all = []

    epochs_back = []

    for k,ncla in taskcla:
        # specify threshold hyperparameter
        # threshold = np.array([0.985] * 20)
        threshold = np.array([args.gpm_thro] * 20)

        data = dataloader.get(k)
        task_name = data[k]['name']
        task_name_list.append(task_name)
        log.info('*'*100)
        log.info('Task {:2d} ({:s})'.format(k,data[k]['name']))
        log.info('*'*100)


        xtrain=data[k]['train']['x']
        ytrain=data[k]['train']['y']
        xvalid=data[k]['valid']['x']
        yvalid=data[k]['valid']['y']
        xtest =data[k]['test']['x']
        ytest =data[k]['test']['y']
        task_list.append(k)

        lr = args.lr
        best_loss=np.inf
        log.info ('-'*40)
        log.info ('Task ID :{} | Learning Rate : {}'.format(task_id, lr))
        log.info ('-'*40)

        if task_id==0:
            model = ResNet18(taskcla,20).to(device) # base filters: 20

            # for k_t, (m, param) in enumerate(model.named_parameters()):
            #     log.info ((k_t,m,param.shape))

            memory[task_name] = {}

            log.info ('Model parameters ---')
            kk = 0
            for k_t, (m, param) in enumerate(model.named_parameters()):
                if len(param.shape) == 4:
                    log.info ((k_t,m,param.shape))
                    memory[task_name][str(kk)] = {
                        'space_list': {},
                        'grad_list': {},
                        'regime':{},
                    }
                    kk += 1

            log.info ('-'*40)

            best_model=get_model(model)
            space_list_all =[]
            normal_param = [
                param for name, param in model.named_parameters()
                if not 'scale' in name
            ]
            base_optimizer = torch.optim.SGD([
                                        {'params': normal_param}
                                        ],
                                        lr=lr
                                        )
            optimizer = SAM(base_optimizer, model)

            acc_list = []
            for epoch in range(1, args.n_epochs+1):
                # Train
                clock0=time.time()
                train(args, model, device, xtrain, ytrain, optimizer, criterion, k)
                clock1=time.time()
                tr_loss,tr_acc = test(args, model, device, xtrain, ytrain, criterion, k)
                log.info('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,\
                                                            tr_loss,tr_acc, 1000*(clock1-clock0)))
                # Validate
                valid_loss,valid_acc = test(args, model, device, xvalid, yvalid,  criterion, k)
                acc_list.append(valid_acc)
                log.info(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc))
                # Adapt lr
                if valid_loss<best_loss:
                    best_loss=valid_loss
                    best_model=get_model(model)
                    patience=args.lr_patience
                else:
                    patience-=1
                    if patience<=0:
                        lr/=args.lr_factor
                        log.info(' lr={:.1e}'.format(lr))
                        if lr<args.lr_min:
                            break
                        patience=args.lr_patience
                        adjust_learning_rate(optimizer.optimizer, epoch, args)
            print(acc_list)
            acc_list_all.append(acc_list)
            set_model_(model,best_model)
            # Test
            log.info ('-'*40)
            test_loss, test_acc = test(args, model, device, xtest, ytest,  criterion, k)
            log.info('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))
            # Memory Update
            mat_list, grad_list = get_representation_and_gradient (model, device, optimizer, criterion, k, xtrain, ytrain)
            space_list_all = get_space_and_grad (model, mat_list, grad_list, threshold, memory, task_name, task_name_list, task_id, space_list_all)

        else:
            memory[task_name] = {}

            kk = 0
            log.info("reinit the scale for each task")
            for k_t, (m, params) in enumerate(model.named_parameters()):
                # create the saved memory
                if 'weight' in m and 'bn' not in m:

                    memory[task_name][str(kk)] = {
                        'space_list': {},
                        'grad_list': {},
                        'space_mat_list':{},
                        'scale1':{},
                        'scale2':{},
                        'regime':{},
                        'selected_task':{},
                    }
                    kk += 1
                #reinitialize the scale
                if 'scale' in m:
                    mask = torch.eye(params.size(0), params.size(1)).to(device)

                    params.data = mask
            # print("-----------------")
            normal_param = [
                param for name, param in model.named_parameters()
                if not 'scale' in name
            ]

            scale_param = [
                param for name, param in model.named_parameters()
                if 'scale' in name
            ]
            base_optimizer = torch.optim.SGD([
                                        {'params': normal_param},
                                        {'params': scale_param, 'weight_decay': 0, 'lr':lr}
                                        ],
                                        lr=lr
                                        )
            optimizer = SAM(base_optimizer, model)

            feature_mat = []
            # Projection Matrix Precomputation
            for i in range(len(space_list_all)):
                 Uf=torch.Tensor(np.dot(space_list_all[i],space_list_all[i].transpose())).to(device)
                 log.info('Layer {} - Projection Matrix shape: {}'.format(i+1,Uf.shape))
                 feature_mat.append(Uf)


            #==1 gradient projection condition
            log.info('excute gradient projection condition')
            grad_proj_cond(args, model, xtrain, ytrain, memory, task_name, task_id, task_name_list, device, optimizer, criterion)
            # select the regime 2, which need to learn scale
            space1 = [None] * 20
            space2 = [None] * 20
            for i in range(20):
                for k, task_sel in enumerate(memory[task_name][str(i)]['selected_task']):
                    if memory[task_name][str(i)]['regime'][task_sel] == '2' or memory[task_name][str(i)]['regime'][task_sel] == '3':
                        if k == 0:
                            # change the np array to torch tensor
                            space1[i] = torch.FloatTensor(memory[task_name_list[task_sel]][str(i)]['space_list']).to(device)
                        else:

                            space2[i] = torch.FloatTensor(memory[task_name_list[task_sel]][str(i)]['space_list']).to(device)
            if space1[0] is not None:
                log.info('space1 is not None!')
            if space2[0] is not None:
                log.info('space2 is not None!')

            log.info ('-'*40)
            acc_list = []
            for epoch in range(1, args.n_epochs+1):
                # Train

                clock0=time.time()
                train_projected_regime(args, model, device,xtrain, ytrain,optimizer,criterion,memory, task_name, task_name_list, space_list_all, task_id, feature_mat, space1=space1, space2=space2)
                clock1=time.time()
                tr_loss, tr_acc = test(args, model, device, xtrain, ytrain,criterion,task_id, space1=space1, space2=space2)
                log.info('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,\
                                                        tr_loss, tr_acc, 1000*(clock1-clock0)))
                # Validate
                valid_loss,valid_acc = test(args, model, device, xvalid, yvalid, criterion, task_id, space1=space1, space2=space2)
                acc_list.append(valid_acc)

                log.info(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc))
                # Adapt lr
                if valid_loss<best_loss:
                    best_loss=valid_loss
                    best_model=get_model(model)
                    patience=args.lr_patience
                else:
                    patience-=1
                    if patience<=0:
                        lr/=args.lr_factor
                        log.info(' lr={:.1e}'.format(lr))
                        if lr<args.lr_min:
                            break
                        patience=args.lr_patience
                        adjust_learning_rate(optimizer.optimizer, epoch, args)
            print(acc_list)
            acc_list_all.append(acc_list)
            set_model_(model,best_model)
            print(epochs_back)
            # Test
            test_acc_sum = 0
            for i in range(10):
                test_loss, test_acc = test(args, model, device, xtest, ytest,  criterion,task_id, space1=space1, space2=space2)
                log.info('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))
                test_acc_sum += test_acc
            test_acc_sum = test_acc_sum/10.
            log.info('Average acc={:5.1f}%'.format(test_acc_sum))
            # Memory Update
            mat_list, grad_list = get_representation_and_gradient (model, device, optimizer, criterion, task_id, xtrain, ytrain)
            space_list_all = get_space_and_grad (model, mat_list, grad_list, threshold, memory, task_name, task_name_list, task_id, space_list_all)
            # save the scale value to memory
            idx1 = 0
            idx2 = 0
            for m,params in model.named_parameters(): # layer
                if 'scale1' in m:
                    memory[task_name][str(idx1)]['scale1'] = params.data
                    idx1 += 1
                if 'scale2' in m:
                    memory[task_name][str(idx2)]['scale2'] = params.data
                    idx2 += 1

        # save accuracy
        jj = 0
        for ii in np.array(task_list)[0:task_id+1]:
            xtest =data[ii]['test']['x']
            ytest =data[ii]['test']['y']
            # select the regime 2, which need to learn scale
            space1 = [None] * 20
            space2 = [None] * 20
            task_test = data[ii]['name']
            log.info('current testing task:{}'.format(task_test))

            if ii > 0:
                for i in range(20):
                    for k, task_sel in enumerate(memory[task_test][str(i)]['selected_task']):
                        # print(memory[task_name]['regime'][task_sel])
                        if memory[task_test][str(i)]['regime'][task_sel] == '2' or memory[task_name][str(i)]['regime'][task_sel] == '3':
                            if k == 0:

                                space1[i] = torch.FloatTensor(memory[task_name_list[task_sel]][str(i)]['space_list']).to(device)
                                idx = 0
                                for m,params in model.named_parameters():
                                    if 'scale1' in m:
                                        params.data = memory[task_test][str(idx)]['scale1'].to(device)
                                        idx += 1
                            else:

                                space2[i] = torch.FloatTensor(memory[task_name_list[task_sel]][str(i)]['space_list']).to(device)
                                idx = 0
                                for m,params in model.named_parameters():
                                    if 'scale2' in m:
                                        params.data = memory[task_test][str(idx)]['scale2'].to(device)
                                        idx += 1

            test_acc_sum = 0
            for i in range(5):
                test_loss, test_acc = test(args, model, device, xtest, ytest,criterion,ii, space1=space1, space2=space2)
                log.info('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))
                test_acc_sum += test_acc
            acc_matrix[task_id,jj] = test_acc_sum/5.
            #_, acc_matrix[task_id,jj] = test(args, model, device, xtest, ytest,criterion,ii, space1=space1, space2=space2)
            jj +=1
        log.info('Accuracies =')
        for i_a in range(task_id + 1):
            acc_ = ''
            for j_a in range(acc_matrix.shape[1]):
                acc_ += '{:5.1f}% '.format(acc_matrix[i_a, j_a])
            log.info(acc_)
        # update task id
        task_id +=1
        save_model(model, memory,args.savename)
    np.save('five_ours.npy', acc_list_all)
    log.info('-'*50)
    # Simulation Results
    log.info ('Task Order : {}'.format(np.array(task_list)))
    log.info ('Final Avg Accuracy: {:5.2f}%'.format(acc_matrix[-1].mean()))
    bwt=np.mean((acc_matrix[-1]-np.diag(acc_matrix))[:-1])
    log.info ('Backward transfer: {:5.2f}%'.format(bwt))
    log.info('[Elapsed time = {:.1f} ms]'.format((time.time()-tstart)*1000))
    log.info('-'*50)
    # Plots
    # array = acc_matrix
    # df_cm = pd.DataFrame(array, index = [i for i in ["1","2","3","4","5","6","7",\
    #                                     "8","9","10","11","12","13","14","15","16","17","18","19","20"]],
    #                   columns = [i for i in ["1","2","3","4","5","6","7",\
    #                                     "8","9","10","11","12","13","14","15","16","17","18","19","20"]])
    # sn.set(font_scale=1.4)
    # sn.heatmap(df_cm, annot=True, annot_kws={"size": 10})
    # plt.show()
    return acc_matrix[-1].mean(), bwt

def create_log_dir(path, filename='log.txt'):
    import logging
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path+'/'+filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

if __name__ == "__main__":
    # Training parameters
    parser = argparse.ArgumentParser(description='5 datasets with GPM')
    parser.add_argument('--batch_size_train', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--batch_size_test', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=100, metavar='N',
                        help='number of training epochs/task (default: 200)')
    parser.add_argument('--seed', type=int, default=37, metavar='S',
                        help='random seed (default: 37)')
    parser.add_argument('--pc_valid',default=0.02,type=float,
                        help='fraction of training data used for validation')
    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--lr_min', type=float, default=1e-3, metavar='LRM',
                        help='minimum lr rate (default: 1e-5)')
    parser.add_argument('--lr_patience', type=int, default=5, metavar='LRP',
                        help='hold before decaying lr (default: 6)')
    parser.add_argument('--lr_factor', type=int, default=3, metavar='LRF',
                        help='lr decay factor (default: 2)')
    parser.add_argument('--gpm_thro', type=float, default=0.95,
                        metavar='gradient projection',
                        help='gpm_thro')
    parser.add_argument('--mixup_alpha', type=float, default=20, metavar='Alpha',
                        help='mixup_alpha')
    parser.add_argument('--mixup_weight', type=float, default=0.1, metavar='Weight',
                        help='mixup_weight')
    parser.add_argument('--savename', type=str, default='./log/DFTRGP/',
                        help='save path')

    args = parser.parse_args()
    str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    log = create_log_dir(args.savename, 'log_{}.txt'.format(str_time_))

    for mixup_weight in [0.01, 0.1, 0.001]:
        for thro_ in [0.95, 0.96, 0.97, 0.98]:
            args.gpm_thro = thro_
            args.mixup_weight = mixup_weight
            str_time = str_time_ + '_' + str(thro_)+ '_' + str(mixup_weight)

            accs, bwts = [], []
            for seed_ in [1, 2, 3]:
                try:
                    args.seed = seed_

                    log.info('=' * 100)
                    log.info('Arguments =')
                    log.info(str(args))
                    log.info('=' * 100)
                    train_begin_time = time.time()
                    acc, bwt = main(args)
                    print(time.time() - train_begin_time)
                    log.info('time cost =', str(time.time() - train_begin_time))

                    accs.append(acc)
                    bwts.append(bwt)
                except:
                    print("seed " +str (seed_) +"Error!!")

            log.info('mixup_weight: ' + str(mixup_weight))
            log.info('gpm_thro: ' + str(thro_))
            log.info('Accuracy: ' + str(accs))
            log.info('Backward transfer: ' + str(bwts))
            log.info('Final Avg Accuracy: {:5.2f}%, std:{:5.2f}'.format(np.mean(accs), np.std(accs)))
            log.info('Final Avg Backward transfer: {:5.2f}%, std:{:5.2f}'.format(np.mean(bwts), np.std(bwts)))
