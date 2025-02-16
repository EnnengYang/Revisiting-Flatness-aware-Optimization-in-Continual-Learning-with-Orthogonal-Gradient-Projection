import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms

import os
import os.path
import sys
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
import logging
from layers_trgp import Conv2d, Linear

from flatness_minima import SAM
from torch.autograd import Variable

os.environ["CUDA_VISIBLE_DEVICES"] = "1"



eplison_1 = 0.2
eplison_2 = 0.01
#  Define MLP model
class MLPNet(nn.Module):
    def __init__(self, n_hidden=100, n_outputs=10):
        super(MLPNet, self).__init__()
        self.act=OrderedDict()
        self.lin1 = Linear(784,n_hidden,bias=False)
        self.lin2 = Linear(n_hidden,n_hidden, bias=False)
        self.fc1  = Linear(n_hidden, n_outputs, bias=False)
        

    def forward(self, x, space1= [None, None, None], space2= [None, None, None]):
        # regime 2:
        if space1[0] is not None or space2[0] is not None:
            self.act['Lin1']=x
            x = self.lin1(x, space1=space1[0], space2 = space2[0])        
            x = F.relu(x)
            self.act['Lin2']=x
            x = self.lin2(x, space1=space1[1], space2 = space2[1])        
            x = F.relu(x)
            self.act['fc1']=x
            x = self.fc1(x,space1=space1[2], space2 = space2[2])
        else:
            self.act['Lin1']=x
            x = self.lin1(x)        
            x = F.relu(x)
            self.act['Lin2']=x
            x = self.lin2(x)        
            x = F.relu(x)
            self.act['fc1']=x
            x = self.fc1(x)           
        return x 


def get_model(model):
    return deepcopy(model.state_dict())

def set_model(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return

def beta_distributions(size, alpha=1):
    return np.random.beta(alpha, alpha, size=size)

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    loss_a = lam * criterion(pred, y_a)
    loss_b = (1 - lam) * criterion(pred, y_b)
    return loss_a.mean() + loss_b.mean()

class AugModule(nn.Module):
    def __init__(self):
        super(AugModule, self).__init__()
    def forward(self, xs, lam, y, index):
        x_ori = xs
        N = x_ori.size()[0]

        x_ori_perm = x_ori[index, :]

        lam = lam.view((N, 1)).expand_as(x_ori)
        x_mix = (1 - lam) * x_ori + lam * x_ori_perm
        y_a, y_b = y, y[index]
        return x_mix, y_a, y_b

def save_model(model, memory, savename):
    ckpt = {
        'model': model.state_dict(),
        'memory': memory,
    }

    # Save to file.
    torch.save(ckpt, savename+'checkpoint.pt')
    print(savename)

    return 

def train (args, model, device, x, y, optimizer,criterion):
    model.train()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    aug_model = AugModule()

    # Loop batches
    for i in range(0,len(r),args.batch_size_train):
        if i+args.batch_size_train<=len(r): b=r[i:i+args.batch_size_train]
        else: b=r[i:]
        data = x[b].view(-1, 28 * 28)
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
        output1 = model(raw_data)
        output2 = model(mix_inputs)
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
        output1 = model(raw_data)
        output2 = model(mix_inputs)
        loss = criterion(output1, raw_target) + args.mixup_weight * mixup_criterion(criterion, output2, mix_targets_a,
                                                                                    mix_targets_b, lam_adv.detach())
        loss.backward()
        optimizer.unperturb_step()

        optimizer.step()

def train_projected_regime (args, model,device,x,y,optimizer,criterion,memory, task_name, task_name_list, task_id, feature_mat, space1=[None, None, None], space2=[None, None, None]):
    model.train()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    aug_model = AugModule()

    # Loop batches
    for i in range(0,len(r),args.batch_size_train):
        if i+args.batch_size_train<=len(r): b=r[i:i+args.batch_size_train]
        else: b=r[i:]

        data = x[b].view(-1, 28 * 28)
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
        output1 = model(raw_data, space1=space1, space2=space2)
        output2 = model(mix_inputs, space1=space1, space2=space2)
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
        output1 = model(raw_data, space1=space1, space2=space2)
        output2 = model(mix_inputs, space1=space1, space2=space2)
        loss = criterion(output1, raw_target) + args.mixup_weight * mixup_criterion(criterion, output2, mix_targets_a,
                                                                                    mix_targets_b, lam_adv.detach())
        loss.backward()
        optimizer.unperturb_step()

        # data = x[b].view(-1,28*28)
        # data, target = data.to(device), y[b].to(device)
        # optimizer.zero_grad()
        # output = model(data, space1=space1, space2=space2)
        # loss = criterion(output, target)
        # loss.backward()

        kk = 0 
        for k, (m,params) in enumerate(model.named_parameters()):
            if 'weight' in m:
                sz =  params.grad.data.size(0)
                params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1),\
                                                    feature_mat[kk]).view(params.size())
                kk+=1 


        optimizer.step()
    
def test (args, model, device, x, y, criterion, space1=[None, None, None], space2=[None, None, None]):
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
            data = x[b].view(-1,28*28)
            data, target = data.to(device), y[b].to(device)
            output = model(data, space1=space1, space2=space2)
            loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True) 
            
            correct    += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.data.cpu().numpy().item()*len(b)
            total_num  += len(b)

    acc = 100. * correct / total_num
    final_loss = total_loss / total_num
    return final_loss, acc

def get_representation_and_gradient(net, device, optimizer, criterion, task_id, x, y=None):
    # Collect activations by forward pass
    # Collect gradient by backward pass
    # net.eval()
    steps = 1
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    b=r[0:300] # Take random training samples
    example_data = x[b].view(-1,28*28)
    example_data, target = example_data.to(device), y[b].to(device)
    
    batch_list=[300,300,300] 
    mat_list=[] # list contains representation matrix of each layer
    grad_list=[] # list contains gradient of each layer
    act_key=list(net.act.keys())
    # example_out  = net(example_data)

    net.eval()
    example_out  = net(example_data)

    for k in range(len(act_key)):
        bsz=batch_list[k]
        act = net.act[act_key[k]].detach().cpu().numpy()
        activation = act[0:bsz].transpose()
        mat_list.append(activation)
    
    print('-'*30)
    print('Representation Matrix')
    print('-'*30)
    for i in range(len(mat_list)):
        print ('Layer {} : {}'.format(i+1,mat_list[i].shape))
    print('-'*30)
    return mat_list, grad_list

def get_space_and_grad(model, mat_list, grad_list, threshold, memory, task_name, task_name_list, task_id, space_list_all):
    print ('Threshold: ', threshold) 
    Ours = True
    if task_name == 'pmnist-0':
        # After First Task 
        for i in range(len(mat_list)):
            activation = mat_list[i]

            U,S,Vh = np.linalg.svd(activation, full_matrices=False)
            # criteria (Eq-5)
            sval_total = (S**2).sum()
            sval_ratio = (S**2)/sval_total
            r = np.sum(np.cumsum(sval_ratio)<threshold[i]) #+1  

            # save into memory
            memory[task_name][str(i)]['space_list'] = U[:,0:r]

            space_list_all.append(U[:,0:r])


    else:
        for i in range(len(mat_list)):
            activation = mat_list[i]

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
 
                log.info("the number of space for current task:{}".format(r))
                log.info('the new increased space:{}, the threshold for new space:{}'.format(len(sel_index_from_U), r_pre))

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
                    log.info('Skip Updating GPM for layer: {}'.format(i+1)) 

                feature_list = []
                for task_index in range(task_id):
                    space_list = memory[task_name_list[task_index]][str(i)]['space_list']
                    feature_list.append(space_list)
                
                Ui=np.hstack((space_list_all[i],U[:,0:r]))  
                log.info('Ui shape:{}'.format(Ui.shape))
                if Ui.shape[1] > Ui.shape[0] :
                    space_list_all[i]=Ui[:,0:Ui.shape[0]]
                else:
                    space_list_all[i]=Ui
                                                                                                    
                if r == 0:
                    memory[task_name][str(i)]['space_list'] = space_list
                    # print(memory[task_name][str(i)]['space_list'])
                else:
                    memory[task_name][str(i)]['space_list'] = U[:,0:r]
 

    log.info('-'*40)
    log.info('Gradient Constraints Summary')
    log.info('-'*40)

    for i in range(3):
        print ('Layer {} : {}/{}'.format(i+1,space_list_all[i].shape[1], space_list_all[i].shape[0]))
    log.info('-'*40)
    
    return space_list_all       



def grad_proj_cond(args, net, x, y, memory, task_name, task_id, task_name_list, device, optimizer, criterion):

    # calcuate the gradient for current task before training
    steps = 1
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    b=r[0:300] # Take random training samples
    example_data = x[b].view(-1,28*28)
    example_data, target = example_data.to(device), y[b].to(device)
    
    batch_list=[300,300,300] 
    grad_list=[] # list contains gradient of each layer
    act_key=list(net.act.keys())

    for i in range(steps):
        optimizer.zero_grad()  
        example_out  = net(example_data)

        loss = criterion(example_out, target)         
        loss.backward()  

        for k, (m,params) in enumerate(net.named_parameters()):
            if 'weight' in m:
                grad = params.grad.data.detach().cpu().numpy()
                grad_list.append(grad)


    # project on each task subspace
    gradient_norm_lists_tasks = []
    # ratio_tasks = []
    for task_index in range(task_id):
        projection_norm_lists = []
        
        # ratio_layers = []
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
        # ratio_tasks.append(ratio_layers)

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
    
    # return ratio_tasks 


def main(args):
    tstart=time.time()
    ## Device Setting 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ## Load PMNIST DATASET
    from dataloader import pmnist as pmd
    data,taskcla,inputsize=pmd.get(seed=args.seed, pc_valid=args.pc_valid)

    acc_matrix=np.zeros((10,10))
    criterion = torch.nn.CrossEntropyLoss()

    task_id = 0
    task_list = []
    task_name_list = []
    memory = {}

    acc_list_all = []

    # ratios = []
    epochs_back = []
    for k,ncla in taskcla:
        
        # specify threshold hyperparameter
        threshold = np.array([args.gpm_thro,0.99,0.99])
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
        log.info ('-'*40)
        log.info ('Task ID :{} | Learning Rate : {}'.format(task_id, lr))
        log.info ('-'*40)
        
        if task_id==0:
            model = MLPNet(args.n_hidden, args.n_outputs).to(device)
            memory[task_name] = {}
            #memory[task_name]['regime'] = 10 * [0]
            log.info ('Model parameters ---')
            k = 0
            for k_t, (m, param) in enumerate(model.named_parameters()):
                if 'weight' in m:
                    print(k,m,param.shape)
                    # create the saved memory
                    memory[task_name][str(k)] = {
                        'space_list': {},
                        'grad_list': {},
                        'regime':{},
                    }
                    k += 1
            log.info ('-'*40)

            space_list_all =[]
            # coord_list = []
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
                train(args, model, device, xtrain, ytrain, optimizer, criterion)
                clock1=time.time()
                tr_loss,tr_acc = test(args, model, device, xtrain, ytrain, criterion)
                log.info('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,\
                                                            tr_loss,tr_acc, 1000*(clock1-clock0)))
                # Validate
                valid_loss,valid_acc = test(args, model, device, xvalid, yvalid,  criterion)
                acc_list.append(valid_acc)
                log.info(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc))
                print()
            print(acc_list)
            acc_list_all.append(acc_list)
            # Test
            log.info ('-'*40)
            test_loss, test_acc = test(args, model, device, xtest, ytest,  criterion)
            log.info('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))
            # Memory Update  
            mat_list, grad_list = get_representation_and_gradient (model, device, optimizer, criterion, task_id,  xtrain, ytrain)
            space_list_all = get_space_and_grad (model, mat_list, grad_list, threshold, memory, task_name, task_name_list, task_id, space_list_all)
            
        else:
            memory[task_name] = {}

            k = 0
            for k_t, (m, params) in enumerate(model.named_parameters()):
                # create the saved memory
                if 'weight' in m:
                    
                    memory[task_name][str(k)] = {
                        'space_list': {},
                        'grad_list': {},
                        'space_mat_list':{},
                        'scale1':{},
                        'scale2':{},
                        'regime':{},
                        'selected_task':{},
                        # 'ratio':{},
                    }
                    k += 1
                #reinitialize the scale
                if 'scale' in m:
                    mask = torch.eye(params.size(0), params.size(1)).to(device)
                    params.data = mask
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

            # optimizer = optim.SGD(model.parameters(), lr=args.lr)
            log.info('-'*40)

            # select the regime 2, which need to learn scale
            space1 = [None, None, None]
            space2 = [None, None, None]
            
      
            for i in range(3): #layer
                for k, task_sel in enumerate(memory[task_name][str(i)]['selected_task']):  #task loop
                    if memory[task_name][str(i)]['regime'][task_sel] == '2' or memory[task_name][str(i)]['regime'][task_sel] == '3':
                        if k == 0:
                            # change the np array to torch tensor
                            space1[i]=torch.tensor(memory[task_name_list[task_sel]][str(i)]['space_list']).to(device)
                        else:
                            space2[i]=torch.tensor(memory[task_name_list[task_sel]][str(i)]['space_list']).to(device)

            if space1[0] is not None:
                log.info('space1 is not None!')
            if space2[1] is not None:
                log.info('space2 is not None!') 

            log.info ('-'*40)
            acc_list = []

            for epoch in range(1, args.n_epochs+1):

                clock0=time.time()
                train_projected_regime(args, model,device,xtrain, ytrain,optimizer,criterion,memory, task_name, task_name_list, task_id, feature_mat, space1=space1, space2=space2)
                clock1=time.time()
                tr_loss, tr_acc = test(args, model, device, xtrain, ytrain,  criterion, space1=space1, space2=space2)
                log.info('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,\
                                                        tr_loss, tr_acc, 1000*(clock1-clock0)))
                # Validate
                valid_loss,valid_acc = test(args, model, device, xvalid, yvalid,  criterion, space1=space1, space2=space2)
                acc_list.append(valid_acc)
      
                log.info(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc))
                print()

            print(acc_list)
            acc_list_all.append(acc_list)
            print(epochs_back)

            # Test 
            test_loss, test_acc = test(args, model, device, xtest, ytest,  criterion, space1=space1, space2=space2)
            log.info('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))  

            # Memory Update  
            mat_list, grad_list = get_representation_and_gradient (model, device, optimizer, criterion, task_id, xtrain, ytrain)
            space_list_all = get_space_and_grad (model, mat_list, grad_list, threshold, memory, task_name, task_name_list, task_id, space_list_all)
            # save the scale value to memory
            idx1 = 0
            idx2 = 0
            for m,params in model.named_parameters():
                if 'scale1' in m:
                    memory[task_name][str(idx1)]['scale1'] = params.data
                    idx1 += 1
                if 'scale2' in m:
                    memory[task_name][str(idx2)]['scale2'] = params.data
                    idx2 += 1

        # save model
        if not os.path.exists(args.savename + '/' + str_time):
            os.makedirs(args.savename + '/' + str_time)
        torch.save(model.state_dict(), args.savename + '/' + str_time + '/model_random_' + str(args.seed) + '_task_' + str(task_id) + '.pkl')

        # save accuracy
        jj = 0 
        for ii in np.array(task_list)[0:task_id+1]:
            xtest =data[ii]['test']['x']
            ytest =data[ii]['test']['y'] 
            # select the regime 2, which need to learn scale
            space1 = [None, None, None]
            space2 = [None, None, None]
            task_test = data[ii]['name']
            log.info('current testing task:{}'.format(task_test))
            if ii > 0:
                          
                for i in range(3):
                    for k, task_sel in enumerate(memory[task_test][str(i)]['selected_task']):
                        if memory[task_test][str(i)]['regime'][task_sel] == '2' or memory[task_name][str(i)]['regime'][task_sel] == '3':
                            if k == 0:
                                # change the np array to torch tensor
                                space1[i] = torch.tensor(memory[task_name_list[task_sel]][str(i)]['space_list']).to(device)
                                idx = 0
                                for m,params in model.named_parameters():
                                    if 'scale1' in m:
                                        params.data = memory[task_test][str(idx)]['scale1'].to(device)
                                        idx += 1
                            else:
                                space2[i] = torch.tensor(memory[task_name_list[task_sel]][str(i)]['space_list']).to(device)
                                idx = 0
                                for m,params in model.named_parameters():                               
                                    if 'scale2' in m:
                                        params.data = memory[task_test][str(idx)]['scale2'].to(device)
                                        idx += 1                           

            _, acc_matrix[task_id,jj] = test(args, model, device, xtest, ytest,criterion, space1=space1, space2=space2) 
            jj +=1
        log.info('Accuracies =')
        for i_a in range(task_id + 1):
            # log.info('\t')
            acc_ = ''
            for j_a in range(acc_matrix.shape[1]):
                acc_ += '{:5.1f}% '.format(acc_matrix[i_a, j_a])
            log.info(acc_)
        # for i_a in range(task_id+1):
        #     print('\t',end='')
        #     for j_a in range(acc_matrix.shape[1]):
        #         print('{:5.1f}% '.format(acc_matrix[i_a,j_a]),end='')
        #     print()
        # update task id 
        task_id +=1
        save_model(model, memory, args.savename)

    log.info('-'*50)
    # Simulation Results 
    log.info ('Task Order : {}'.format(np.array(task_list)))
    log.info ('Final Avg Accuracy: {:5.2f}%'.format(acc_matrix[-1].mean())) 
    bwt=np.mean((acc_matrix[-1]-np.diag(acc_matrix))[:-1]) 
    log.info ('Backward transfer: {:5.2f}%'.format(bwt))
    log.info('[Elapsed time = {:.1f} ms]'.format((time.time()-tstart)*1000))
    log.info('-'*50)
    # Plots
    array = acc_matrix
    df_cm = pd.DataFrame(array, index = [i for i in ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10"]],
                      columns = [i for i in ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10"]])
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 10})

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
    parser = argparse.ArgumentParser(description='Sequential PMNIST with GPM')
    parser.add_argument('--batch_size_train', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--batch_size_test', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=5, metavar='N',
                        help='number of training epochs/task (default: 5)')
    parser.add_argument('--seed', type=int, default=2, metavar='S',
                        help='random seed (default: 2)')
    parser.add_argument('--pc_valid',default=0.1,type=float,
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
    # Architecture
    parser.add_argument('--n_hidden', type=int, default=100, metavar='NH',
                        help='number of hidden units in MLP (default: 100)')
    parser.add_argument('--n_outputs', type=int, default=10, metavar='NO',
                        help='number of output units in MLP (default: 10)')
    parser.add_argument('--n_tasks', type=int, default=10, metavar='NT',
                        help='number of tasks (default: 10)')
    # parser.add_argument('--savename', type=str, default='save/P_MNIST/Ours/two_task_overlap',
    #                     help='save path')
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
        args.mixup_weight = mixup_weight

        for gpm_thro_ in [0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]:
            args.gpm_thro = gpm_thro_
            str_time = str_time_ + '_' + str(gpm_thro_) + '_0.99_0.99_mixup'+ str(mixup_weight)

            accs, bwts = [], []
            for seed_ in [1, 2, 3]:
                args.seed = seed_
                try:
                    log.info('=' * 100)
                    log.info('Arguments =')
                    log.info(str(args))
                    log.info('=' * 100)

                    acc, bwt = main(args)
                    accs.append(acc)
                    bwts.append(bwt)

                except:
                    print("seed " + str(seed_) + "Error!!")

            log.info('gpm_thro: ' + str(args.gpm_thro))
            log.info('Accuracy: ' + str(accs))
            log.info('Backward transfer: ' + str(bwts))
            log.info('Final Avg Accuracy: {:5.2f}%, std:{:5.2f}'.format(np.mean(accs), np.std(accs)))
            log.info('Final Avg Backward transfer: {:5.2f}%, std:{:5.2f}'.format(np.mean(bwts), np.std(bwts)))




