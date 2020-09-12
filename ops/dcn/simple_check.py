import numpy as np
import torch
import torch.nn as nn

from deform_conv import DeformConv


deform_conv = DeformConv(2, 1, kernel_size=3, padding=1, deformable_groups=2).cuda()
nn.init.constant_(deform_conv.weight, 1)

offset = torch.tensor([
    1, 1, 1, 0, 1, -1, 0, 1, 0, 0, 0, -1, -1, 1, -1, 0, -1, -1], dtype=torch.float32).cuda()
offset = offset.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(1, 2, 3, 3)
input = torch.arange(18, dtype=torch.float32).view(1, 2, 3, 3).cuda()


gt = torch.FloatTensor([81, 99, 117, 135, 153, 171, 189, 207, 225])
pd = deform_conv(input, offset)
diff = gt - pd.detach().cpu().flatten()
eps = 1e-8

print('Check Passed!' if diff.abs().sum().item() < eps else 'Check Failed.')


