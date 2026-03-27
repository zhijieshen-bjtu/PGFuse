import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .GridGenerator import GridGenerator

def calculate_berhu_loss(pred, gt, mask, weights):
    bs = pred.shape[0]
    diff = gt - pred
    abs_diff = torch.abs(diff)
    c = torch.max(abs_diff).item() / 5
    leq = (abs_diff <= c).float()
    l2_losses = (diff**2 + c**2) / (2 * c)
    loss = leq * abs_diff + (1 - leq) * l2_losses
    #_, c, __, ___ = loss.size()
    loss = loss.reshape(bs, -1)
    mask = mask.reshape(bs, -1)
    weights = weights.reshape(bs, -1)
    count = torch.sum(mask, dim=[1], keepdim=True).float()
    masked_loss = loss * mask.float()
    weighted_loss = masked_loss * weights
    return torch.mean(torch.sum(weighted_loss, dim=[1], keepdim=True) / count)
    
    
class BerhuLoss(nn.Module):
    def __init__(self, threshold=0.2):
        super(BerhuLoss, self).__init__()
        self.threshold = threshold

    def forward(self, target, pred, mask=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        if mask is not None:
            valid_mask *= mask.detach()

        diff = torch.abs(target - pred)
        diff = diff[valid_mask]
        delta = self.threshold * torch.max(diff).data.cpu().numpy()

        part1 = -F.threshold(-diff, -delta, 0.)
        part2 = F.threshold(diff ** 2 + delta ** 2, 2.0*delta ** 2, 0.)
        part2 = part2 / (2. * delta)
        diff = part1 + part2
        loss = diff.mean()
        return loss


class SphereConv(nn.Module):
  """
  kernel_size: (H, W)
  """

  def __init__(self):
    super(SphereConv, self).__init__()
    self.kernel_size=(3,3)
    self.stride=(1,1)
    self.grid_shape = None
    self.grid = None
    self.B = 8

  def genSamplingPattern(self, h, w):
    gridGenerator = GridGenerator(h, w, self.kernel_size, self.stride)
    LonLatSamplingPattern = gridGenerator.createSamplingPattern()

    # generate grid to use `F.grid_sample`
    lat_grid = (LonLatSamplingPattern[:, :, :,:, 0] / h) * 2 - 1
    lon_grid = (LonLatSamplingPattern[:, :, :,:, 1] / w) * 2 - 1

    grid = np.stack((lon_grid, lat_grid), axis=-1)
    with torch.no_grad():
      self.grid = torch.FloatTensor(grid)
      self.grid.requires_grad = False

  def forward(self, x, y):
    # Generate Sampling Pattern
    B, C, H, W = x.shape
    #print(x.shape)
    with torch.no_grad():
      if (self.grid_shape is None) or (self.grid_shape != (H, W) or (B != self.B)):
        self.grid_shape = (H, W)
        self.genSamplingPattern(H, W)
        self.grid = self.grid.repeat((B, 1, 1, 1, 1)).to(x.device)
        self.grid.requires_grad = False
        
      # (B, H*Kh, W*Kw, 2) 
    #print(grid.shape)
    #grid = torch.rand(B, H, W, num_samples, 2) * 2 - 1
        
    B, C, H, W = x.shape
    num_samples = 9
    
    if B != self.B:
      grid = self.grid[0,:,:,:,:].unsqueeze(0).repeat((B, 1, 1, 1, 1)).to(x.device).permute(0, 3, 1, 2, 4).reshape(B * num_samples, H, W, 2)
      self.B=B
    else:
      # 重新排列采样点
      grid = self.grid.permute(0, 3, 1, 2, 4).reshape(B * num_samples, H, W, 2)

    # 使用grid_sample进行采样
    x = F.grid_sample(x.repeat(num_samples,1,  1, 1), grid, mode='bilinear', align_corners=True)
    
    # 重新排列采样结果
    x = x.view(B, num_samples, C, H, W)
    x = x.permute(0, 2, 1, 3, 4)
    x = x.reshape(B, H*W, num_samples)
    #print(y.shape)
    #print(x.shape)
    x = torch.matmul(x,y.to(x.device)).view(B, 1, H, W)

    #x = F.grid_sample(x, grid, align_corners=True, mode='nearest')  # (B, in_c, H*Kh, W*Kw)
    
    #print(x.shape)

    # self.weight -> (out_c, in_c, Kh, Kw)
    #x = F.conv1d(x.squeeze(1), self.weight.to(x.device), self.bias.to(x.device))
    #print(x.shape)

    return x  # (B, out_c, H/stride_h, W/stride_w)
    
    
class SphericalGradientLoss(nn.Module):
    def __init__(self):
        super(SphericalGradientLoss, self).__init__()
        
        # Sobel水平和垂直算子
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(-1,1)
        self.sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(-1,1)
        # 初始化Sobel算子权重
        self.sobel_x = nn.Parameter(self.sobel_x, requires_grad=False)
        self.sobel_y = nn.Parameter(self.sobel_y, requires_grad=False)
        self.spherical_conv = SphereConv()
        self.b = BerhuLoss()
        #self.d = nn.Parameter(torch.tensor(1.), requires_grad=True)

    def forward(self, predicted, target, mask):
        predicted, target = predicted.float(), target.float()
        #not_nan_mask = ~torch.isnan(target)
        #mask = mask * not_nan_mask
        #predicted, target = not_nan_mask * predicted, not_nan_mask * target
        
        grad_x_predicted = self.spherical_conv(predicted, self.sobel_x)
        grad_y_predicted = self.spherical_conv(predicted, self.sobel_y)
        grad_x_target = self.spherical_conv(target, self.sobel_x)
        grad_y_target = self.spherical_conv(target, self.sobel_y)

        # 计算损失
        loss_x = self.b(torch.abs(grad_x_predicted), torch.abs(grad_x_target), mask)
        loss_y = self.b(torch.abs(grad_y_predicted), torch.abs(grad_y_target), mask)

        return loss_x + loss_y
        

class Gradient_Net(nn.Module):
  def __init__(self):
    super(Gradient_Net, self).__init__()
    kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    self.kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).cuda()

    kernel_y = [[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]
    self.kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).cuda()

    self.weight_x = nn.Parameter(data=self.kernel_x, requires_grad=False)
    self.weight_y = nn.Parameter(data=self.kernel_y, requires_grad=False)
    self.berhu = BerhuLoss()
    #self.d = nn.Parameter(torch.tensor(1.), requires_grad=True)

  def forward(self, x, y, mask):
    x, y = x.float(), y.float()
    grad_x = F.conv2d(x, self.weight_x, padding=1)
    grad_y = F.conv2d(x, self.weight_y, padding=1)
    grad_xt = F.conv2d(y, self.weight_x, padding=1)
    grad_yt = F.conv2d(y, self.weight_y, padding=1)
    x = self.berhu(torch.abs(grad_x), torch.abs(grad_xt), mask)
    y = self.berhu(torch.abs(grad_y), torch.abs(grad_yt), mask)
    
    return (x+y)