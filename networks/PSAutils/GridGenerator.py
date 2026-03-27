import torch
import math

class GridGenerator(torch.nn.Module):
    def __init__(self, height: int, width: int, kernel_size, stride=1, device='cpu'):
        super().__init__()
        self.height = height
        self.width = width
        self.kernel_size = kernel_size  # (Kh, Kw)
        self.stride = stride
        self.device = torch.device(device)  # Éè±¸

    def createKernel(self):
        Kh, Kw = self.kernel_size
        delta_lat = math.pi / self.height
        delta_lon = 2 * math.pi / self.width

        range_y = torch.linspace(-(Kh // 2), Kh // 2, Kh, dtype=torch.float64, device=self.device)
        range_x = torch.linspace(-(Kw // 2), Kw // 2, Kw, dtype=torch.float64, device=self.device)

        kerY = torch.tan(range_y * delta_lat) / torch.cos(range_y * delta_lon)
        kerX = torch.tan(range_x * delta_lon)

        return torch.meshgrid(kerY, kerX)  # Assumes PyTorch version >= 1.10 for `indexing` argument compatibility

    def createSamplingPattern(self):
        kerY, kerX = self.createKernel()

        rho = torch.sqrt(kerX ** 2 + kerY ** 2)
        Kh, Kw = self.kernel_size
        if Kh % 2 == 1 and Kw % 2 == 1:
            rho[Kh // 2, Kw // 2] = 1e-8

        nu = torch.atan(rho)
        cos_nu = torch.cos(nu)
        sin_nu = torch.sin(nu)

        stride_h, stride_w = self.stride, self.stride
        h_range = torch.arange(0, self.height, stride_h, dtype=torch.float64, device=self.device).view(-1, 1, 1, 1)
        w_range = torch.arange(0, self.width, stride_w, dtype=torch.float64, device=self.device).view(1, -1, 1, 1)

        lat_range = ((h_range / self.height) - 0.5) * math.pi
        lon_range = ((w_range / self.width) - 0.5) * (2 * math.pi)

        lat = torch.arcsin(cos_nu * torch.sin(lat_range) + (kerY * sin_nu * torch.cos(lat_range)) / rho)
        lon = lon_range + torch.atan(kerX * sin_nu/(rho * torch.cos(lat_range) * cos_nu - kerY * sin_nu * torch.sin(lat_range)))

        lat = (lat / math.pi + 0.5) * self.height
        lon = ((lon / (2 * math.pi) + 0.5) * self.width) % self.width

        LatLon = torch.stack((lat.repeat(1, self.width, 1, 1), lon), dim=-1)
        H, W, Kh, Kw, _ = LatLon.shape
        LatLon = LatLon.view(1, H, W, Kw * Kh, 2)

        return LatLon