"""
Author: Vaishakh Patil
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import kornia

EPSILON = 1e-6

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def get_coords(batch_size, H, W, fix_axis=False):
    U_coord = torch.arange(start=0, end=W).unsqueeze(0).repeat(H, 1).float()
    V_coord = torch.arange(start=0, end=H).unsqueeze(1).repeat(1, W).float()
    if not fix_axis:
        U_coord = (U_coord - ((W - 1) / 2)) / max(W, H)
        V_coord = (V_coord - ((H - 1) / 2)) / max(W, H)
    coords = torch.stack([U_coord, V_coord], dim=0)
    coords = coords.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    coords = coords.permute(0, 2, 3, 1).cuda()
    coords[..., 0] /= W - 1
    coords[..., 1] /= H - 1
    # coords = (coords - 0.5) * 2
    return coords

def depth2pqrs(depth): # , upratio

    depth = torch.clamp(depth, min=0.001)
    disp = 1./(depth)

    batch_size, H, W = disp.size()[0], disp.size()[2], disp.size()[3]

    coords = get_coords(batch_size, H, W)
    U_coord = coords[..., 0]
    V_coord = coords[..., 1]

    disp_blurred = kornia.filters.gaussian_blur2d(disp, (3, 3), (1.5, 1.5))
    grad = kornia.filters.spatial_gradient(disp_blurred, mode='sobel', order=1, normalized=False)
    param_p = grad[ :, :, 0, :, :]
    param_q = grad[ :, :, 1, :, :]

    pu = torch.mul(param_p, U_coord.unsqueeze(1))
    qv = torch.mul(param_q, V_coord.unsqueeze(1))
    param_r = disp - pu - qv

    param_s = torch.sqrt(param_p ** 2 + param_q ** 2 + param_r ** 2) + EPSILON

    norm_param_p = torch.div(param_p, param_s)
    norm_param_q = torch.div(param_q, param_s)
    norm_param_r = torch.div(param_r, param_s)

    return norm_param_p, norm_param_q, norm_param_r, param_s


class pqrs2depth(nn.Module):
    def __init__(self, max_depth):
        super(pqrs2depth, self).__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.get_coords = get_coords
        self.max_depth = max_depth

    def forward(self, x, upsample_size = None):

        if upsample_size != None:
            x = F.interpolate(x,(upsample_size[2], upsample_size[3]), mode='bilinear')

        p = x[:, 0, :, :]
        q = x[:, 1, :, :]
        r = x[:, 2, :, :]
        s = x[:, 3, :, :]

        batch_size, H, W = p.size()[0], p.size()[1], p.size()[2]

        coords = self.get_coords(batch_size, H, W)
        U_coord = coords[..., 0]
        V_coord = coords[..., 1]

        U_coord = nn.Parameter(U_coord, requires_grad=False)
        V_coord = nn.Parameter(V_coord, requires_grad=False)

        pu = p * U_coord
        qv = q * V_coord

        disp = (pu + qv + r) * s

        # disp = self.relu(disp) + 0.01
        disp = torch.clamp(disp, min=(1/self.max_depth))

        return disp.unsqueeze(1)
    
class custom_pqrs2depth(nn.Module):
    def __init__(self, max_depth):
        super(custom_pqrs2depth, self).__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.get_coords = get_coords
        self.max_depth = max_depth

    def forward(self, x, H, W, upsample_size = None):
        # x.shape = ba, num_q, 4 (6, 64, 4)
        batch_size = x.size()[0]
        x = x.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, H, W)
        if upsample_size != None:
            x = F.interpolate(x,(upsample_size[2], upsample_size[3]), mode='bilinear')

        p = x[:, :,0, :, :]
        q = x[:, :,1, :, :]
        r = x[:, :,2, :, :]
        s = x[:, :,3, :, :]
        
        norm_factor = torch.sqrt((p ** 2 + q ** 2 + r ** 2) + EPSILON)
        p = torch.div(p, norm_factor)
        q = torch.div(q, norm_factor)
        r = torch.div(r, norm_factor)
        # s = s * norm_factor
        
        batch_size, num_query, _ = p.size()[0], p.size()[1], p.size()[2]

        coords = self.get_coords(batch_size, H, W, fix_axis = True)
        U_coord = coords[..., 0]
        V_coord = coords[..., 1]

        U_coord = nn.Parameter(U_coord, requires_grad=False)
        V_coord = nn.Parameter(V_coord, requires_grad=False)
        U_coord = U_coord.unsqueeze(1).repeat(1,num_query,1,1)
        V_coord = V_coord.unsqueeze(1).repeat(1,num_query,1,1)
        pu = p * U_coord
        qv = q * V_coord

        disp = (pu + qv + r) * s

        # disp = self.relu(disp) + 0.01
        disp = torch.clamp(disp, min=(1/self.max_depth))

        return disp

class parameterized_disparity(nn.Module):
    def __init__(self, max_depth):
        super(parameterized_disparity, self).__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.max_depth = max_depth
        self.get_coords = get_coords

    def forward(self, x, epoch=0):


        p = x[:, 0, :, :]
        q = x[:, 1, :, :]
        r = x[:, 2, :, :]
        s = x[:, 3, :, :] # * self.max_depth
        #s = x[:, 3, :, :]

        # TODO: refer to dispnetPQRS
        norm_factor = torch.sqrt((p ** 2 + q ** 2 + r ** 2) + EPSILON)
        p = torch.div(p, norm_factor)
        q = torch.div(q, norm_factor)
        r = torch.div(r, norm_factor)
        # s = s * norm_factor

        batch_size, H, W = x.size()[0], x.size()[2], x.size()[3]

        coords = self.get_coords(batch_size, H, W)
        U_coord = coords[..., 0]
        V_coord = coords[..., 1]

        U_coord = nn.Parameter(U_coord, requires_grad=False)
        V_coord = nn.Parameter(V_coord, requires_grad=False)

        pu = p * U_coord
        qv = q * V_coord

        disp = (pu + qv + r) * s
        disp = torch.clamp(disp, min=(1 / self.max_depth))

        return p.unsqueeze(1), q.unsqueeze(1), r.unsqueeze(1), s.unsqueeze(1), disp.unsqueeze(1)


class local_planar_guidance(nn.Module):
    def __init__(self, num_in_filters, upratio, max_depth):
        super(local_planar_guidance, self).__init__()

        self.upratio = upratio
        self.u = torch.arange(self.upratio).reshape([1, 1, self.upratio]).float()
        self.v = torch.arange(int(self.upratio)).reshape([1, self.upratio, 1]).float()
        self.upratio = float(upratio)
        self.relu = nn.ReLU()
        self.max_depth = max_depth
        self.sigmoid = nn.Sigmoid()

    def forward(self, plane_eq):
        plane_eq_expanded = torch.repeat_interleave(plane_eq, int(self.upratio), 2)
        plane_eq_expanded = torch.repeat_interleave(plane_eq_expanded, int(self.upratio), 3)

        p = plane_eq_expanded[:, 0, :, :]
        q = plane_eq_expanded[:, 1, :, :]
        r = plane_eq_expanded[:, 2, :, :]
        s = plane_eq_expanded[:, 3, :, :]

        u = self.u.repeat(plane_eq.size(0), plane_eq.size(2) * int(self.upratio), plane_eq.size(3)).cuda()
        u = (u - (self.upratio - 1) * 0.5) / self.upratio

        v = self.v.repeat(plane_eq.size(0), plane_eq.size(2), plane_eq.size(3) * int(self.upratio)).cuda()
        v = (v - (self.upratio - 1) * 0.5) / self.upratio

        norm_factor = torch.sqrt((p ** 2 + q ** 2 + r ** 2) + EPSILON)
        p = torch.div(p, norm_factor)
        q = torch.div(q, norm_factor)
        r = torch.div(r, norm_factor)
        s = s * norm_factor

        disp = (p * u + q * v + r) * s
        disp = torch.clamp(disp, min=(1 / self.max_depth))

        return disp


class AO(nn.Module):
    # Adaptive output module
    def __init__(self, inchannels=132, outchannels=4, upfactor=4):
        super(AO, self).__init__()
        self.inchannels = inchannels
        self.outchannels = outchannels
        self.upfactor = upfactor

        self.adapt_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.inchannels, out_channels=self.inchannels // 2, kernel_size=3, padding=1,
                      stride=1, bias=True), \
            nn.BatchNorm2d(num_features=self.inchannels // 2), \
            nn.ELU(inplace=True), \
            nn.Conv2d(in_channels=self.inchannels // 2, out_channels=self.outchannels, kernel_size=3, padding=1,
                      stride=1, bias=True), \
            nn.Upsample(scale_factor=self.upfactor, mode='bilinear', align_corners=True))

    def forward(self, x):
        x = self.adapt_conv(x)
        return x



# class reduction_1x1(nn.Sequential):
#     def __init__(self, num_in_filters, max_depth):
#         super(reduction_1x1, self).__init__()        
#         self.max_depth = max_depth
#         self.sigmoid = nn.Sigmoid()
#         self.reduc = torch.nn.Sequential()

#         self.reduc.add_module('plane_params', Mlp(num_in_filters, 1024, 3))
    
#     def forward(self, net):
#         net = self.reduc.forward(net).permute(0, 2, 1).contiguous()
#         theta = self.sigmoid(net[:, 0, :]) * math.pi / 3
#         phi = self.sigmoid(net[:, 1, :]) * math.pi * 2
#         dist = self.sigmoid(net[:, 2, :]) * self.max_depth
#         n1 = torch.mul(torch.sin(theta), torch.cos(phi)).unsqueeze(1) 
#         n2 = torch.mul(torch.sin(theta), torch.sin(phi)).unsqueeze(1) 
#         n3 = torch.cos(theta).unsqueeze(1) 
#         n4 = dist.unsqueeze(1) 
#         net = torch.cat([n1, n2, n3, n4], dim=1)
        
#         return net.permute(0, 2, 1).contiguous()

class reduction_1x1(nn.Sequential):
    def __init__(self, num_in_filters, num_out_filters, max_depth, is_final=False):
        super(reduction_1x1, self).__init__()        
        self.max_depth = max_depth
        self.is_final = is_final
        self.sigmoid = nn.Sigmoid()
        self.reduc = torch.nn.Sequential()
        
        while num_out_filters >= 4:
            if num_out_filters < 8:
                if self.is_final:
                    self.reduc.add_module('final', torch.nn.Sequential(nn.Conv2d(num_in_filters, out_channels=1, bias=False,
                                                                                 kernel_size=1, stride=1, padding=0),
                                                                       nn.Sigmoid()))
                else:
                    self.reduc.add_module('plane_params', torch.nn.Conv2d(num_in_filters, out_channels=3, bias=False,
                                                                          kernel_size=1, stride=1, padding=0))
                break
            else:
                self.reduc.add_module('inter_{}_{}'.format(num_in_filters, num_out_filters),
                                      torch.nn.Sequential(nn.Conv2d(in_channels=num_in_filters, out_channels=num_out_filters,
                                                                    bias=False, kernel_size=1, stride=1, padding=0),
                                                          nn.ELU()))

            num_in_filters = num_out_filters
            num_out_filters = num_out_filters // 2
    
    def forward(self, net):
        net = self.reduc.forward(net)
        if not self.is_final:
            theta = self.sigmoid(net[:, 0, :, :]) * math.pi / 3
            phi = self.sigmoid(net[:, 1, :, :]) * math.pi * 2
            dist = self.sigmoid(net[:, 2, :, :]) * self.max_depth
            n1 = torch.mul(torch.sin(theta), torch.cos(phi)).unsqueeze(1)
            n2 = torch.mul(torch.sin(theta), torch.sin(phi)).unsqueeze(1)
            n3 = torch.cos(theta).unsqueeze(1)
            n4 = dist.unsqueeze(1)
            net = torch.cat([n1, n2, n3, n4], dim=1)
        
        return net
