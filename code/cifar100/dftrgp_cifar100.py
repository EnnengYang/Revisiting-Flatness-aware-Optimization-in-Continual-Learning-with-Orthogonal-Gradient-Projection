from calendar import firstweekday
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms

import sys
import os
import os.path
from collections import OrderedDict

import logging
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
import random
import pdb
import argparse,time
import math
from copy import deepcopy
from flatness_minima import SAM
from torch.autograd import Variable

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

Eplison_1 = 0.5
Eplison_2 = 0.
Lambda = 2.0
# Define specifc conv layer

class Pre_Model(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.model = None

    def add(self, model):
        self.model = model

    def update(self, model):
        self.model.load_state_dict(deepcopy(model.state_dict()))


class Conv2d(nn.Conv2d):

    def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                padding=0,
                stride=1,
                dilation=1,
                groups=1,
                bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels,
              kernel_size, stride=stride, padding=padding, bias=bias)
        # define the scale v
        size = self.weight.size(1) * self.weight.size(2) * self.weight.size(3)
        scale = self.weight.data.new(size, size)
        scale.fill_(0.)
        # initialize the diagonal as 1
        scale.fill_diagonal_(1.)
        # self.scale1 = scale.cuda()
        self.scale1 = nn.Parameter(scale, requires_grad=True)
        self.scale2 = nn.Parameter(scale, requires_grad=True)
        self.noise = False
        if self.noise:
            self.alpha_w1 = nn.Parameter(torch.ones(self.out_channels).view(-1,1,1,1)*0.02, requires_grad = True)
            self.alpha_w2 = nn.Parameter(torch.ones(self.out_channels).view(-1,1,1,1)*0.02, requires_grad = True)

    def forward(self, input, space1=None, space2=None):

        if self.noise:
            with torch.no_grad():
                std = self.weight.std().item()
                noise = self.weight.clone().normal_(0,std)
        if space1 is not None or space2 is not None:
            sz =  self.weight.grad.data.size(0)

            if space2 is None:
                real_scale1 = self.scale1[:space1.size(1), :space1.size(1)]
                norm_project = torch.mm(torch.mm(space1, real_scale1), space1.transpose(1, 0))
                #[chout, chinxkxk]  [chinxkxk, chinxkxk]
                proj_weight = torch.mm(self.weight.view(sz,-1),norm_project).view(self.weight.size())
                diag_weight = torch.mm(self.weight.view(sz,-1),torch.mm(space1, space1.transpose(1,0))).view(self.weight.size())
                if self.noise and self.training:
                    masked_weight = proj_weight + self.weight - diag_weight + self.alpha_w2 * noise * self.noise
                else:
                    masked_weight = proj_weight + self.weight - diag_weight

            if space1 is None:

                real_scale2 = self.scale2[:space2.size(1), :space2.size(1)]
                norm_project = torch.mm(torch.mm(space2, real_scale2), space2.transpose(1, 0))

                proj_weight = torch.mm(self.weight.view(sz,-1),norm_project).view(self.weight.size())
                diag_weight = torch.mm(self.weight.view(sz,-1),torch.mm(space2, space2.transpose(1,0))).view(self.weight.size())

                if self.noise:
                    masked_weight = proj_weight + self.weight - diag_weight + self.alpha_w2 * noise * self.noise
                else:
                    masked_weight = proj_weight + self.weight - diag_weight
            if space1 is not None and space2 is not None:
                real_scale1 = self.scale1[:space1.size(1), :space1.size(1)]
                norm_project1 = torch.mm(torch.mm(space1, real_scale1), space1.transpose(1, 0))
                proj_weight1 = torch.mm(self.weight.view(sz,-1),norm_project1).view(self.weight.size())
                diag_weight1 = torch.mm(self.weight.view(sz,-1),torch.mm(space1, space1.transpose(1,0))).view(self.weight.size())

                real_scale2 = self.scale2[:space2.size(1), :space2.size(1)]
                norm_project2 = torch.mm(torch.mm(space2, real_scale2), space2.transpose(1, 0))
                proj_weight2 = torch.mm(self.weight.view(sz,-1),norm_project2).view(self.weight.size())
                diag_weight2 = torch.mm(self.weight.view(sz,-1),torch.mm(space2, space2.transpose(1,0))).view(self.weight.size())

                if self.noise:
                    masked_weight = proj_weight1 - diag_weight1 + proj_weight2 - diag_weight2 + self.weight + ((self.alpha_w2 + self.alpha_w1)/2) * noise * self.noise
                else:
                    masked_weight = proj_weight1 - diag_weight1 + proj_weight2 - diag_weight2 + self.weight

        else:
            masked_weight = self.weight

        return F.conv2d(input, masked_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

# Define specific linear layer
class Linear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__(in_features, out_features, bias=bias)


        # define the scale v
        scale = self.weight.data.new(self.weight.size(1), self.weight.size(1))
        scale.fill_(0.)
        # initialize the diagonal as 1
        scale.fill_diagonal_(1.)
        # self.scale1 = scale.cuda()
        self.scale1 = nn.Parameter(scale, requires_grad=True)
        self.scale2 = nn.Parameter(scale, requires_grad=True)
        self.noise = False
        if self.noise:
            self.alpha_w1 = nn.Parameter(torch.ones(self.weight.size())*0.1, requires_grad = True)
            self.alpha_w2 = nn.Parameter(torch.ones(self.weight.size())*0.1, requires_grad = True)


        #self.fixed_scale = scale
    def forward(self, input, space1=None, space2=None):
        if self.noise:
            with torch.no_grad():
                std = self.weight.std().item()
                noise = self.weight.clone().normal_(0,std)
        if space1 is not None or space2 is not None:
            sz =  self.weight.grad.data.size(0)

            if space2 is None:

                real_scale1 = self.scale1[:space1.size(1), :space1.size(1)]
                norm_project = torch.mm(torch.mm(space1, real_scale1), space1.transpose(1, 0))

                proj_weight = torch.mm(self.weight,norm_project)

                diag_weight = torch.mm(self.weight,torch.mm(space1, space1.transpose(1,0)))
                # masked_weight = proj_weight + self.weight - diag_weight
                if self.noise and self.training:
                    masked_weight = proj_weight + self.weight - diag_weight + self.alpha_w2 * noise * self.noise
                else:
                    masked_weight = proj_weight + self.weight - diag_weight

            if space1 is None:

                real_scale2 = self.scale2[:space2.size(1), :space2.size(1)]
                norm_project = torch.mm(torch.mm(space2, real_scale2), space2.transpose(1, 0))

                proj_weight = torch.mm(self.weight,norm_project)
                diag_weight = torch.mm(self.weight,torch.mm(space2, space2.transpose(1,0)))

                #masked_weight = proj_weight + self.weight - diag_weight
                if self.noise and self.training:
                    masked_weight = proj_weight + self.weight - diag_weight + self.alpha_w2 * noise * self.noise
                else:
                    masked_weight = proj_weight + self.weight - diag_weight

            if space1 is not None and space2 is not None:
                real_scale1 = self.scale1[:space1.size(1), :space1.size(1)]
                norm_project1 = torch.mm(torch.mm(space1, real_scale1), space1.transpose(1, 0))
                proj_weight1 = torch.mm(self.weight,norm_project1)
                diag_weight1 = torch.mm(self.weight,torch.mm(space1, space1.transpose(1,0)))

                real_scale2 = self.scale2[:space2.size(1), :space2.size(1)]
                norm_project2 = torch.mm(torch.mm(space2, real_scale2), space2.transpose(1, 0))
                proj_weight2 = torch.mm(self.weight,norm_project2)
                diag_weight2 = torch.mm(self.weight,torch.mm(space2, space2.transpose(1,0)))

                #masked_weight = proj_weight1 - diag_weight1 + proj_weight2 - diag_weight2 + self.weight
                if self.noise and self.training:
                    masked_weight = proj_weight1 - diag_weight1 + proj_weight2 - diag_weight2 + self.weight + ((self.alpha_w2 + self.alpha_w1)/2) * noise * self.noise
                else:
                    masked_weight = proj_weight1 - diag_weight1 + proj_weight2 - diag_weight2 + self.weight

        else:
            masked_weight = self.weight
        return F.linear(input, masked_weight, self.bias)

## Define AlexNet model
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class AlexNet(nn.Module):
    def __init__(self,taskcla):
        super(AlexNet, self).__init__()
        self.act=OrderedDict()
        self.map =[]
        self.ksize=[]
        self.in_channel =[]
        self.map.append(32)
        self.conv1 = Conv2d(3, 64, 4, bias=False)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=False)
        s=compute_conv_output_size(32,4)
        s=s//2
        self.ksize.append(4)
        self.in_channel.append(3)
        self.map.append(s)
        self.conv2 = Conv2d(64, 128, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(128, track_running_stats=False)
        s=compute_conv_output_size(s,3)
        s=s//2
        self.ksize.append(3)
        self.in_channel.append(64)
        self.map.append(s)
        self.conv3 = Conv2d(128, 256, 2, bias=False)
        self.bn3 = nn.BatchNorm2d(256, track_running_stats=False)
        s=compute_conv_output_size(s,2)
        s=s//2
        self.smid=s
        self.ksize.append(2)
        self.in_channel.append(128)
        self.map.append(256*self.smid*self.smid)
        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()
        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)

        self.fc1 = Linear(256*self.smid*self.smid,2048, bias=False)
        self.bn4 = nn.BatchNorm1d(2048, track_running_stats=False)
        self.fc2 = Linear(2048,2048, bias=False)
        self.bn5 = nn.BatchNorm1d(2048, track_running_stats=False)
        self.map.extend([2048])

        self.taskcla = taskcla
        self.fc3=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.fc3.append(torch.nn.Linear(2048,n,bias=False))

    def forward(self, x, space1= [None, None, None, None, None], space2= [None, None, None, None, None]):
        bsz = deepcopy(x.size(0))
        if space1[0] is not None or space2[0] is not None:
            self.act['conv1']=x
            x = self.conv1(x, space1=space1[0], space2 = space2[0])
            x = self.maxpool(self.drop1(self.relu(self.bn1(x))))

            self.act['conv2']=x
            x = self.conv2(x, space1=space1[1], space2 = space2[1])
            x = self.maxpool(self.drop1(self.relu(self.bn2(x))))

            self.act['conv3']=x
            x = self.conv3(x, space1=space1[2], space2 = space2[2])
            x = self.maxpool(self.drop2(self.relu(self.bn3(x))))

            x=x.view(bsz,-1)
            self.act['fc1']=x
            x = self.fc1(x, space1=space1[3], space2 = space2[3])
            x = self.drop2(self.relu(self.bn4(x)))

            self.act['fc2']=x
            x = self.fc2(x, space1=space1[4], space2 = space2[4])
            x = self.drop2(self.relu(self.bn5(x)))
            y=[]
            for t,i in self.taskcla:
                y.append(self.fc3[t](x))
        else:
            self.act['conv1']=x
            x = self.conv1(x)
            x = self.maxpool(self.drop1(self.relu(self.bn1(x))))

            self.act['conv2']=x
            x = self.conv2(x)
            x = self.maxpool(self.drop1(self.relu(self.bn2(x))))

            self.act['conv3']=x
            x = self.conv3(x)
            x = self.maxpool(self.drop2(self.relu(self.bn3(x))))

            x=x.view(bsz,-1)
            self.act['fc1']=x
            x = self.fc1(x)
            x = self.drop2(self.relu(self.bn4(x)))

            self.act['fc2']=x
            x = self.fc2(x)
            x = self.drop2(self.relu(self.bn5(x)))
            y=[]
            for t,i in self.taskcla:
                y.append(self.fc3[t](x))

        return y

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



def train_projected_regime (args, model, device,x,y,optimizer,criterion,memory, task_name, task_name_list, space_list_all, task_id, feature_mat, epoch, space1=[None, None, None], space2=[None, None, None]):
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

        # data = x[b]
        # data, target = data.to(device), y[b].to(device)
        # optimizer.zero_grad()
        # output = model(data, space1=space1, space2=space2)
        # loss = criterion(output[task_id], target)
        #
        # loss.backward()

        # Gradient Projections
        kk = 0
        for k, (m,params) in enumerate(model.named_parameters()):
            if 'weight' in m:
                if k<21 and len(params.size())!=1:
                    sz =  params.grad.data.size(0)
                    params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1),\
                                                            feature_mat[kk]).view(params.size())


                    kk +=1
                elif (k<1 and len(params.size())==1) and task_id !=0 :
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

def get_representation_and_gradient (args, net, device, optimizer, criterion, task_id, x, y=None):
    '''
    aim to get the representation (activation) and gradient(optimal) of each layer
    '''

    # Collect activations by forward pass
    steps = 1
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    b=r[0:125] # Take 125 random samples
    example_data = x[b]
    example_data, target = example_data.to(device), y[b].to(device)

    batch_list=[2*12,100,100,125,125]
    mat_list=[]
    grad_list=[] # list contains gradient of each layer
    act_key=list(net.act.keys())
    grad_list_steps = []
    grad_list_avg = []


    net.eval()
    example_out  = net(example_data)

    for i in range(len(net.map)):
        bsz=batch_list[i]
        k=0
        if i<3:
            ksz= net.ksize[i]
            s=compute_conv_output_size(net.map[i],net.ksize[i])
            # logging.info("s:{}".format(s))
            mat = np.zeros((net.ksize[i]*net.ksize[i]*net.in_channel[i],s*s*bsz))
            act = net.act[act_key[i]].detach().cpu().numpy()
            for kk in range(bsz):
                for ii in range(s):
                    for jj in range(s):
                        mat[:,k]=act[kk,:,ii:ksz+ii,jj:ksz+jj].reshape(-1)
                        k +=1
            mat_list.append(mat)
        else:
            act = net.act[act_key[i]].detach().cpu().numpy()
            activation = act[0:bsz].transpose()
            mat_list.append(activation)

    log.info('-'*30)
    log.info('Representation Matrix')
    log.info('-'*30)
    for i in range(len(mat_list)):
        log.info ('Layer {} : {}'.format(i+1,mat_list[i].shape))
    log.info('-'*30)
    return mat_list, grad_list


def get_space_and_grad(args, model, mat_list, grad_list, threshold, memory, task_name, task_name_list, task_id, space_list_all):
    '''
    Get the space for each layer
    '''
    log.info ('Threshold:{}'.format(threshold))
    Ours = True
    if task_name == 'cifar100-0':
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


            if Ours:
                #=1. calculate the projection using previous space
                log.info('activation shape:{}'.format(activation.shape))
                log.info('space shape:{}'.format(space_list_all[i].shape))
                #delta = np.dot(np.dot(space_list_all[i],space_list_all[i].transpose()),activation)
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

                #=4 select the most import basis
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
                # if r == 0:
                #     print ('Skip Updating GPM for layer: {}'.format(i+1))
                #     continue
                log.info('threshold for selecting:{}'.format(np.linalg.norm(activation)**2))
                log.info("total ranking r = {}".format(r))

                #=5 save the corresponding space
                Ui = np.hstack((space_list_all[i],U))
                sel_index = stack_index[:r]
                #print('sel_index:{}'.format(sel_index))
                # this is the current space
                U_new = Ui[:, sel_index]
                # calculate how many space from current new task
                sel_index_from_U = sel_index[sel_index>r_pre]
                # print(sel_index)
                # print(sel_index_from_U)
                if len(sel_index_from_U) > 0:
                    # update the overall space without overlap
                    total_U =  np.hstack((space_list_all[i], Ui[:,sel_index_from_U] ))
                    space_list_all[i] = total_U
                else:
                    space_list_all[i] = np.array(space_list_all[i])
                # else:
                #     continue
                #print("Ui shape:{}".format(Ui.shape))
                log.info("the number of space for current task:{}".format(r))
                log.info('the new increased space:{}, the threshold for new space:{}'.format(len(sel_index_from_U), r_pre))

                log.info("Ui shape:{}".format(Ui[:,sel_index].shape))
                memory[task_name][str(i)]['space_list'] = Ui[:,sel_index]

            else:
                U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
                sval_total = (S1**2).sum()
                # Projected Representation (Eq-8)
                # Go through all the previous tasks
                act_hat = activation
                for task_index in range(task_id):
                    space_list = memory[task_name_list[task_index]][str(i)]['space_list']
                    act_hat -= np.dot(np.dot(space_list,space_list.transpose()),activation)

                U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)

                #update GPM
                # criteria (Eq-9)
                sval_hat = (S**2).sum()
                sval_ratio = (S**2)/sval_total
                accumulated_sval = (sval_total-sval_hat)/sval_total

                r = 0
                for ii in range (sval_ratio.shape[0]):
                    if accumulated_sval < threshold[i]:
                        accumulated_sval += sval_ratio[ii]
                        r += 1
                    else:
                        break
                if r == 0:
                    log.info ('Skip Updating GPM for layer: {}'.format(i+1))

                Ui=np.hstack((space_list_all[i],U[:,0:r]))
                log.info('Ui shape:{}'.format(Ui.shape))
                if Ui.shape[1] > Ui.shape[0] :
                    space_list_all[i]=Ui[:,0:Ui.shape[0]]
                else:
                    space_list_all[i]=Ui
                if r == 0:
                    memory[task_name][str(i)]['space_list'] = space_list
                else:
                    memory[task_name][str(i)]['space_list'] = U[:,0:r]


    log.info('-'*40)
    log.info('Gradient Constraints Summary')
    log.info('-'*40)

    for i in range(5):
        log.info ('Layer {} : {}/{}'.format(i+1,space_list_all[i].shape[1], space_list_all[i].shape[0]))
    log.info('-'*40)

    return space_list_all

def grad_proj_cond(args, net, x, y, memory, task_name, task_id, task_name_list, device, optimizer, criterion):
    '''
    get the regime descision
    '''

    # calcuate the gradient for current task before training
    steps = 1
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    b=r[0:125] # Take 125*10 random samples
    example_data = x[b]
    example_data, target = example_data.to(device), y[b].to(device)

    batch_list=[2*12,100,100,125,125]
    grad_list=[] # list contains gradient of each layer
    act_key=list(net.act.keys())
    #print('task id:{}'.format(task_id))
    for i in range(1):

        optimizer.zero_grad()
        example_out  = net(example_data)

        loss = criterion(example_out[task_id], target)
        loss.backward()

        k_linear = 0
        for k, (m,params) in enumerate(net.named_parameters()):
            if 'weight' in m and 'bn' not in m:
                if len(params.shape) == 4:

                    grad = params.grad.data.detach().cpu().numpy()
                    grad = grad.reshape(grad.shape[0], grad.shape[1]*grad.shape[2]*grad.shape[3])
                    grad_list.append(grad)
                else:
                    if 'fc3' in m and k_linear == task_id:
                        grad = params.grad.data.detach().cpu().numpy()
                        grad_list.append(grad)
                        k_linear += 1
                    elif 'fc3' not in m:
                        grad = params.grad.data.detach().cpu().numpy()
                        grad_list.append(grad)

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
            log.info('Task:{}, Layer:{}, project_norm:{}, threshold for regime 1:{}'.format(task_index, i, projection_norm, Eplison_1 * gradient_norm))

            # make decision if Regime 1
            # logging.info('project_norm:{}, threshold for regime 1:{}'.format(projection_norm, eplison_1 * gradient_norm))
            if projection_norm <= Eplison_1 * gradient_norm:
                memory[task_name][str(i)]['regime'][task_index] = '1'
            else:

                memory[task_name][str(i)]['regime'][task_index] = '2'
        gradient_norm_lists_tasks.append(projection_norm_lists)
        for i in range(len(grad_list)):
            log.info('Layer:{}, Regime:{}'.format(i, memory[task_name][str(i)]['regime'][task_index]))

    log.info('-'*20)
    log.info('selected top-2 tasks:')
    if task_id == 1:
        for i in range(len(grad_list)):
            memory[task_name][str(i)]['selected_task'] = [0]
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
    ## Load CIFAR100 DATASET
    from dataloader import cifar100 as cf100
    data,taskcla,inputsize=cf100.get(seed=args.seed, pc_valid=args.pc_valid)

    acc_matrix=np.zeros((10,10))
    criterion = torch.nn.CrossEntropyLoss()

    task_id = 0
    task_list = []
    task_name_list = []
    memory = {}

    epochs_back = []
    for k,ncla in taskcla:
        # specify threshold hyperparameter

        # threshold = np.array([0.97] * 5) + task_id*np.array([0.003] * 5)
        threshold = np.array([args.gpm_thro] * 5)
        log.info('thresholds: ' + str(threshold))

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
            model = AlexNet(taskcla).to(device)
            for k_t, (m, param) in enumerate(model.named_parameters()):
                print (k_t,m,param.shape)
            memory[task_name] = {}

            log.info ('Model parameters ---')
            kk = 0
            for k_t, (m, param) in enumerate(model.named_parameters()):
                if 'weight' in m and 'bn' not in m:
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
                                                            tr_loss,tr_acc, 1000*(clock1-clock0)),log)
                # Validate
                valid_loss,valid_acc = test(args, model, device, xvalid, yvalid,  criterion, k)
                acc_list.append(valid_acc)
                log.info(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc),log)

                # check the inner and threshold for regime 3

                # Adapt lr
                if valid_loss<best_loss:
                    best_loss=valid_loss
                    best_model=get_model(model)
                    patience=args.lr_patience
                    log.info(' *',log)
                else:
                    patience-=1
                    if patience<=0:
                        lr/=args.lr_factor
                        log.info(' lr={:.1e}'.format(lr))
                        if lr<args.lr_min:
                            # log.info("")
                            break
                        patience=args.lr_patience
                        adjust_learning_rate(optimizer.optimizer, epoch, args)
                # log.info("")
            log.info(acc_list)
            set_model_(model,best_model)
            # Test
            log.info ('-'*40)
            test_loss, test_acc = test(args, model, device, xtest, ytest,  criterion, k)
            log.info('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))
            # Memory Update
            mat_list, grad_list = get_representation_and_gradient (args, model, device, optimizer, criterion, k, xtrain, ytrain)
            space_list_all = get_space_and_grad (args, model, mat_list, grad_list, threshold, memory, task_name, task_name_list, task_id, space_list_all)

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
            for i in range(len(model.act)):
                Uf=torch.Tensor(np.dot(space_list_all[i],space_list_all[i].transpose())).to(device)
                log.info('Layer {} - Projection Matrix shape: {}'.format(i+1,Uf.shape))
                feature_mat.append(Uf)

            #==1 gradient projection condition
            log.info('=== excute gradient projection condition')
            grad_proj_cond(args, model, xtrain, ytrain, memory, task_name, task_id, task_name_list, device, optimizer, criterion)
            # select the regime 2, which need to learn scale
            space1 = [None, None, None, None, None]
            space2 = [None, None, None, None, None]
            for i in range(5):
                for k, task_sel in enumerate(memory[task_name][str(i)]['selected_task']):
                    # logging.info('regime:{}'.format(memory[task_name]['regime'][task_sel]))
                    if memory[task_name][str(i)]['regime'][task_sel] == '2' or memory[task_name][str(i)]['regime'][task_sel] == '3':
                        if k == 0:
                            # space1 = []
                            #for i in range(5):
                            # change the np array to torch tensor
                            space1[i] = torch.FloatTensor(memory[task_name_list[task_sel]][str(i)]['space_list']).to(device)
                        else:
                        # space2 = []
                        # for i in range(5):
                            space2[i] = torch.FloatTensor(memory[task_name_list[task_sel]][str(i)]['space_list']).to(device)
            if space1[0] is not None:
                log.info('space1 is not None!')
            if space2[0] is not None:
                log.info('space2 is not None!')

            log.info ('-'*40)
            acc_list = []
            pre_model = None
            for epoch in range(1, args.n_epochs+1):
                # Train

                clock0=time.time()
                train_projected_regime(args, model, device,xtrain, ytrain,optimizer,criterion,memory, task_name, task_name_list, space_list_all, task_id, feature_mat, epoch, space1=space1, space2=space2)
                clock1=time.time()
                tr_loss, tr_acc = test(args, model, device, xtrain, ytrain,criterion,task_id, space1=space1, space2=space2)
                log.info('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,\
                                                        tr_loss, tr_acc, 1000*(clock1-clock0)),log)
                # Validate
                valid_loss,valid_acc = test(args, model, device, xvalid, yvalid, criterion, task_id, space1=space1, space2=space2)
                log.info(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc),log)
                acc_list.append(valid_acc)

                # Adapt lr
                if valid_loss<best_loss:
                    best_loss=valid_loss
                    best_model=get_model(model)
                    patience=args.lr_patience
                    log.info(' *',log)
                else:
                    patience-=1
                    if patience<=0:
                        lr/=args.lr_factor
                        log.info(' lr={:.1e}'.format(lr))
                        if lr<args.lr_min:
                            # log.info("")
                            break
                        patience=args.lr_patience
                        adjust_learning_rate(optimizer.optimizer, epoch, args)
                # log.info("")
            set_model_(model,best_model)
            print(epochs_back)
            # Test
            log.info(acc_list)
            test_acc_sum = 0
            for i in range(10):
                test_loss, test_acc = test(args, model, device, xtest, ytest,  criterion,task_id, space1=space1, space2=space2)
                log.info('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))
                test_acc_sum += test_acc
            test_acc_sum = test_acc_sum/10.
            log.info('Average acc={:5.1f}%'.format(test_acc_sum))
            # Memory Update
            mat_list, grad_list = get_representation_and_gradient (args, model, device, optimizer, criterion, task_id, xtrain, ytrain)
            space_list_all = get_space_and_grad (args, model, mat_list, grad_list, threshold, memory, task_name, task_name_list, task_id, space_list_all)
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
            space1 = [None, None, None, None, None]
            space2 = [None, None, None, None, None]
            task_test = data[ii]['name']
            log.info('current testing task:{}'.format(task_test))


            if ii > 0:
                for i in range(5):
                    for k, task_sel in enumerate(memory[task_test][str(i)]['selected_task']):
                        # print(memory[task_name]['regime'][task_sel])
                        if memory[task_test][str(i)]['regime'][task_sel] == '2' or memory[task_name][str(i)]['regime'][task_sel] == '3':
                            if k == 0:
                            # space1 = []
                                # change the np array to torch tensor
                                space1[i] = torch.FloatTensor(memory[task_name_list[task_sel]][str(i)]['space_list']).to(device)
                                idx = 0
                                for m,params in model.named_parameters():
                                    if 'scale1' in m:
                                        params.data = memory[task_test][str(idx)]['scale1'].to(device)
                                        idx += 1
                            else:
                            #space2 = []
                                space2[i] = torch.FloatTensor(memory[task_name_list[task_sel]][str(i)]['space_list']).to(device)
                                idx = 0
                                for m,params in model.named_parameters():
                                    if 'scale2' in m:
                                        params.data = memory[task_test][str(idx)]['scale2'].to(device)
                                        idx += 1


            test_acc_sum = 0
            for i in range(10):
                test_loss, test_acc = test(args, model, device, xtest, ytest,criterion,ii, space1=space1, space2=space2)
                log.info('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))
                test_acc_sum += test_acc
            acc_matrix[task_id,jj] = test_acc_sum/10.
            #_, acc_matrix[task_id,jj] = test(args, model, device, xtest, ytest,criterion,ii, space1=space1, space2=space2)
            jj +=1

        # save_model(model, memory, args.savename)
        # save model
        if not os.path.exists(args.savename + '/' + str_time):
            os.makedirs(args.savename + '/' + str_time)
        torch.save(model, args.savename + '/' + str_time + '/model_random_' + str(args.seed) + '_task_' + str(task_id) + '.pkl')
        torch.save(memory, args.savename + '/' + str_time + '/memory_random_' + str(args.seed) + '_task_' + str(task_id) + '.pkl')

        log.info('Accuracies =')
        for i_a in range(task_id + 1):
            acc_ = ''
            for j_a in range(acc_matrix.shape[1]):
                acc_ += '{:5.1f}% '.format(acc_matrix[i_a, j_a])
            log.info(acc_)
        # update task id
        task_id +=1

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
    # df_cm = pd.DataFrame(array, index = [i for i in ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10"]],
    #                   columns = [i for i in ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10"]])
    # sn.set(font_scale=1.4)
    # ax = sn.heatmap(df_cm, annot=True, annot_kws={"size": 10})
    # ax.figure.savefig(args.savename+'fig.pdf')
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
    parser = argparse.ArgumentParser(description='Sequential CIFAR100 with GPM')
    parser.add_argument('--batch_size_train', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--batch_size_test', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=200, metavar='N',
                        help='number of training epochs/task (default: 200)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--pc_valid',default=0.05,type=float,
                        help='fraction of training data used for validation')
    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--lr_min', type=float, default=1e-5, metavar='LRM',
                        help='minimum lr rate (default: 1e-5)')
    parser.add_argument('--lr_patience', type=int, default=6, metavar='LRP',
                        help='hold before decaying lr (default: 6)')
    parser.add_argument('--lr_factor', type=int, default=2, metavar='LRF',
                        help='lr decay factor (default: 2)')
    parser.add_argument('--mixup_alpha', type=float, default=20, metavar='Alpha',
                        help='mixup_alpha')
    parser.add_argument('--mixup_weight', type=float, default=0.1, metavar='Weight',
                        help='mixup_weight')
    parser.add_argument('--savename', type=str, default='./log/DFTRGP/',
                        help='save path')

    args = parser.parse_args()
    str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    log = create_log_dir(args.savename, 'log_{}.txt'.format(str_time_))

    for mixup_weight in [0.01, 0.001, 0.0001]:
        for thro_ in [0.91, 0.93, 0.95, 0.97, 0.99]:
            args.mixup_weight = mixup_weight
            args.gpm_thro = thro_
            str_time = str_time_ + '/' + str(thro_) + '_' + str(mixup_weight)

            accs, bwts = [], []
            for seed_ in [1, 2, 3]:
                # try:
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
                # except:
                #     print("seed " + str(seed_) + "Error!!")

            log.info('mixup_weight: ' + str(mixup_weight))
            log.info('gpm_thro: ' + str(thro_))
            log.info('Accuracy: ' + str(accs))
            log.info('Backward transfer: ' + str(bwts))
            log.info('Final Avg Accuracy: {:5.2f}%, std:{:5.2f}'.format(np.mean(accs), np.std(accs)))
            log.info('Final Avg Backward transfer: {:5.2f}%, std:{:5.2f}'.format(np.mean(bwts), np.std(bwts)))
