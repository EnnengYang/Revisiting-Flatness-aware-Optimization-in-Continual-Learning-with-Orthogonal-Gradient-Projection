import torch
import torch.nn as nn
import torch.nn.functional as F

# Define specifc conv layer
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
        size = self.weight.size(1) * self.weight.size(2) * self.weight.size(3)
        scale = self.weight.data.new(size, size)
        scale.fill_(0.)
        scale.fill_diagonal_(1.)
        self.scale = nn.Parameter(scale.float(), requires_grad=True)

    def forward(self, input, space=None):

        if space is not None:
            sz =  self.weight.grad.data.size(0)
            real_scale = self.scale[:space.size(1), :space.size(1)]
            norm_project = torch.mm(torch.mm(space, real_scale), space.transpose(1, 0))
            proj_weight = torch.mm(self.weight.view(sz,-1),norm_project).view(self.weight.size())

            diag_weight = torch.mm(self.weight.view(sz,-1),torch.mm(space, space.transpose(1,0))).view(self.weight.size())
            masked_weight = proj_weight + self.weight - diag_weight 
       
        else:
            masked_weight = self.weight

        return F.conv2d(input, masked_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        
    def consolidate(self, space=None):

        if space is not None:
            sz =  self.weight.grad.data.size(0)
            real_scale = self.scale[:space.size(1), :space.size(1)].float()
            norm_project = torch.mm(torch.mm(space, real_scale), space.transpose(1, 0))
            proj_weight = torch.mm(self.weight.view(sz,-1),norm_project).view(self.weight.size())

            diag_weight = torch.mm(self.weight.view(sz,-1),torch.mm(space, space.transpose(1,0))).view(self.weight.size())
            masked_weight = proj_weight + self.weight - diag_weight 

        else:
            masked_weight = self.weight

        self.weight.data = masked_weight.data


# Define specific linear layer
class Linear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__(in_features, out_features, bias=bias)


        scale = self.weight.data.new(self.weight.size(1), self.weight.size(1))
        scale.fill_(0.)
        scale.fill_diagonal_(1.)
        self.scale = nn.Parameter(scale, requires_grad=True)

    def forward(self, input, space=None):

        if space is not None:
            sz =  self.weight.grad.data.size(0)
            real_scale = self.scale[:space.size(1), :space.size(1)]
            norm_project = torch.mm(torch.mm(space, real_scale), space.transpose(1, 0))
            proj_weight = torch.mm(self.weight.view(sz,-1),norm_project).view(self.weight.size())

            diag_weight = torch.mm(self.weight.view(sz,-1),torch.mm(space, space.transpose(1,0))).view(self.weight.size())
            masked_weight = proj_weight + self.weight - diag_weight 
       
        else:
            masked_weight = self.weight
        
        return F.linear(input, masked_weight, self.bias)
    
    def consolidate(self, space=None):

        if space is not None:
            sz =  self.weight.grad.data.size(0)
            real_scale = self.scale[:space.size(1), :space.size(1)]
            norm_project = torch.mm(torch.mm(space, real_scale), space.transpose(1, 0))
            proj_weight = torch.mm(self.weight.view(sz,-1),norm_project).view(self.weight.size())

            diag_weight = torch.mm(self.weight.view(sz,-1),torch.mm(space, space.transpose(1,0))).view(self.weight.size())
            masked_weight = proj_weight + self.weight - diag_weight 
       
        else:
            masked_weight = self.weight

        self.weight.data = masked_weight.data