class DropBlock2d(nn.Module):
    """
    Args:
        p (float): Probability of an element to be dropped.
        block_size (int): Size of the block to drop.
        inplace (bool): If set to ``True``, will do this operation in-place.
    """
    def __init__(self, block_size = 7, p = 0.1, inplace = True):
        super(DropBlock2d, self).__init__()
        if p < 0.0 or p > 1.0:
            raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
        self.p = p
        self.block_size = block_size
        self.inplace = inplace

    def forward(self, input):
        """
        Args:
            input (Tensor): Input feature map on which some areas will be randomly dropped.
        Returns:
            Tensor: The tensor after DropBlock layer.
        """
        if not self.training:
            return input

        N, C, H, W = input.size()
        # compute the gamma of Bernoulli distribution
        gamma = (self.p * H * W) / ((self.block_size ** 2) * ((H - self.block_size + 1) * (W - self.block_size + 1)))
        mask_shape = (N, C, H - self.block_size + 1, W - self.block_size + 1)
        mask = torch.bernoulli(torch.full(mask_shape, gamma, device=input.device))

        mask = F.pad(mask, [self.block_size // 2] * 4, value=0)
        mask = F.max_pool2d(mask, stride=(1, 1), kernel_size=(self.block_size, self.block_size), padding=self.block_size // 2)
        mask = 1 - mask
        normalize_scale = mask.numel() / (1e-6 + mask.sum())
        if self.inplace:
            input.mul_(mask * normalize_scale)
        else:
            input = input * mask * normalize_scale
        return input

    def __repr__(self):
        s = f"{self.__class__.__name__}(p={self.p}, block_size={self.block_size}, inplace={self.inplace})"
        return s

class NonLocal(nn.Module):                                                                                 
    def __init__(self, in_ch = 1024, init_weights = True):
        super(NonLocal, self).__init__()
        self.hidden_ch = in_ch // 2
        #function g C: 1024 to 512
        self.g = nn.Conv2d(in_ch, self.hidden_ch, kernel_size = 1)
        #function theta & phi  C: 1024 to 512
        self.theta = nn.Conv2d(in_ch, self.hidden_ch, kernel_size = 1)
        self.phi = nn.Conv2d(in_ch, self.hidden_ch, kernel_size = 1)
        #C 
        self.C = nn.Softmax(dim = 1)
        #the last conv(=W)  C: 512 to 1024
        self.W = nn.Sequential(nn.Conv2d(self.hidden_ch, in_ch, kernel_size = 1), nn.BatchNorm2d(in_ch))
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        #input B*C*H*W
        batch_size = x.shape[0]

        theta_x = self.theta(x).view(batch_size, self.hidden_ch, int(x.shape[2]*x.shape[3]))
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(x.shape[0], self.hidden_ch, int(x.shape[2]*x.shape[3]))

        g_x = self.g(x).view(batch_size, self.hidden_ch, int(x.shape[2]*x.shape[3]))
        g_x = g_x.permute(0, 2, 1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = self.C(f)
        y = torch.matmul(f_div_C, g_x)    #B*HW*C
        y = y.permute(0, 2, 1).view(batch_size, self.hidden_ch, x.shape[2], x.shape[3])
        w_y = self.W(y)
        Z = w_y + x
        return Z
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity = 'relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

from torchvision.models import mobilenet_v2
mobilenet = mobilenet_v2(weights='DEFAULT')
mobilenet.features[0][0] = nn.Conv2d(1, 32, 3, 2, 1, bias = False)
mobilenet.features.insert(5, NonLocal(32))
mobilenet.features.insert(14, DropBlock2d(5, 0.1))
mobilenet.features.insert(16, DropBlock2d(5, 0.1))
mobilenet.features.insert(18, DropBlock2d(5, 0.1))
mobilenet.features.insert(20, DropBlock2d(5, 0.1))
mobilenet.features.insert(22, DropBlock2d(5, 0.1))
mobilenet.classifier[1] = nn.Linear(1280, 8, bias = True)        
