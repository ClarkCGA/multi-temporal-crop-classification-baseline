import torch
from torch import nn


class ConvBlock(nn.Module):
    """This module creates a user-defined number of conv+BN+ReLU layers.
    Args:
        in_channels (int)-- number of input features.
        out_channels (int) -- number of output features.
        kernel_size (int) -- Size of convolution kernel.
        stride (int) -- decides how jumpy kernel moves along the spatial dimensions.
        padding (int) -- how much the input should be padded on the borders with zero.
        dilation (int) -- dilation ratio for enlarging the receptive field.
        num_conv_layers (int) -- Number of conv+BN+ReLU layers in the block.
        drop_rate (float) -- dropout rate at the end of the block.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, num_conv_layers=2, drop_rate=0):
        super(ConvBlock, self).__init__()

        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                            stride=stride, padding=padding, dilation=dilation, bias=False),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(inplace=True), ]

        if num_conv_layers > 1:
            if drop_rate > 0:
                layers += [nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, dilation=dilation, bias=False),
                           nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
                           nn.Dropout(drop_rate), ] * (num_conv_layers - 1)
            else:
                layers += [nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                     padding=padding, dilation=dilation, bias=False),
                           nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), ] * (num_conv_layers - 1)

        self.block = nn.Sequential(*layers)

    def forward(self, inputs):
        outputs = self.block(inputs)
        return outputs


class DUC(nn.Module):
    """
    Dense Upscaling Convolution (DUC) layer.
        
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        upscale (int): Upscaling factor.
    
    Returns:
        torch.Tensor: Output tensor after applying DUC.
    """
    def __init__(self, in_channels, out_channles, upscale):
        super(DUC, self).__init__()
        out_channles = out_channles * (upscale ** 2)
        self.conv = nn.Conv2d(in_channels, out_channles, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channles)
        self.relu = nn.ReLU(inplace=True)
        self.pixl_shf = nn.PixelShuffle(upscale_factor=upscale)

        kernel = self.icnr(self.conv.weight, scale=upscale)
        self.conv.weight.data.copy_(kernel)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.pixl_shf(x)
        return x

    def icnr(self, x, scale=2, init=nn.init.kaiming_normal):
        """
        ICNR (Initialization from Corresponding Normalized Response) function.
        
        Args:
            x (torch.Tensor): Input tensor.
            scale (int): Upscaling factor.
            init (function): Initialization function.
            
        Returns:
            torch.Tensor: Initialized kernel.
        Note:
            Even with pixel shuffle we still have check board artifacts,
            the solution is to initialize the d**2 feature maps with the same
            radom weights: https://arxiv.org/pdf/1707.02937.pdf
        """

        new_shape = [int(x.shape[0] / (scale ** 2))] + list(x.shape[1:])
        subkernel = torch.zeros(new_shape)
        subkernel = init(subkernel)
        subkernel = subkernel.transpose(0, 1)
        subkernel = subkernel.contiguous().view(subkernel.shape[0],
                                                subkernel.shape[1], -1)
        kernel = subkernel.repeat(1, 1, scale ** 2)
        transposed_shape = [x.shape[1]] + [x.shape[0]] + list(x.shape[2:])
        kernel = kernel.contiguous().view(transposed_shape)
        kernel = kernel.transpose(0, 1)
        return kernel

class UpconvBlock(nn.Module):
    """
    Decoder layer decodes the features along the expansive path.
    Args:
        in_channels (int) -- number of input features.
        out_channels (int) -- number of output features.
        upmode (str) -- Upsampling type. If "fixed" then a linear upsampling with scale factor
                        of two will be applied using bi-linear as interpolation method.
                        If deconv_1 is chosen then a non-overlapping transposed convolution will
                        be applied to upsample the feature maps. If deconv_1 is chosen then an
                        overlapping transposed convolution will be applied to upsample the feature maps.
    """

    def __init__(self, in_channels, out_channels, upmode="deconv_1"):
        super(UpconvBlock, self).__init__()

        if upmode == "fixed":
            layers = [nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True), ]
            layers += [nn.BatchNorm2d(in_channels),
                       nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False), ]

        elif upmode == "deconv_1":
            layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, dilation=1), ]

        elif upmode == "deconv_2":
            layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, dilation=1), ]

        # Dense Upscaling Convolution
        elif upmode == "DUC":
            up_factor = 2
            upsample_dim = (up_factor ** 2) * out_channels
            layers = [nn.Conv2d(in_channels, upsample_dim, kernel_size=3, padding=1),
                      nn.BatchNorm2d(upsample_dim),
                      nn.ReLU(inplace=True),
                      nn.PixelShuffle(up_factor), ]
            
            #layers = [DUC(in_channels, out_channels, upscale=2)]

        else:
            raise ValueError("Provided upsampling mode is not recognized.")

        self.block = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.block(inputs)


class AdditiveAttentionBlock(nn.Module):
    r"""
    additive attention gate (AG) to merge feature maps extracted at multiple scales through skip connection.

    Args:
        f_g (int) -- number of feature maps collected from the higher resolution in encoder path.
        f_x (int) -- number of feature maps in layer "x" in the decoder.
        f_inter (int) -- number of feature maps after summation equal to the number of
                       learnable multidimensional attention coefficients.

    Note: Unlike the original paper we upsample
    """

    def __init__(self, F_g, F_x, F_inter):
        super(AdditiveAttentionBlock, self).__init__()

        # Decoder
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_inter, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_inter)
        )
        # Encoder
        self.W_x = nn.Sequential(
            nn.Conv2d(F_x, F_inter, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_inter)
        )

        # Fused
        self.psi = nn.Sequential(
            nn.Conv2d(F_inter, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # set_trace()
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        merge = self.relu(g1 + x1)
        psi = self.psi(merge)

        return x * psi


class Unet(nn.Module):
    def __init__(self, n_classes, in_channels, filter_config=None, use_skipAtt=False, dropout_rate=0):
        """
        UNet model with optional additive attention between skip connections 
        for semantic segmentation of multispectral satellite images.

        Args:
            n_classes (int): Number of output classes.
            in_channels (int): Number of input channels.
            filter_config (tuple, optional): Configuration of filters in the contracting path.
                        Default is None, which uses the configuration (64, 128, 256, 512, 1024, 2048).
            use_skipAtt (bool, optional): Flag indicating whether to use skip connections with attention.
                        Default is False.
            dropout_rate (float, optional): Dropout rate applied to the convolutional layers.
                        Default is 0.

        """
        super(Unet, self).__init__()

        self.in_channels = in_channels
        self.use_skipAtt = use_skipAtt

        if not filter_config:
            filter_config = (64, 128, 256, 512, 1024, 2048)

        assert len(filter_config) == 6

        # Contraction Path
        self.encoder_1 = ConvBlock(self.in_channels, filter_config[0], num_conv_layers=2,
                                   drop_rate=dropout_rate)  # 64x224x224
        self.encoder_2 = ConvBlock(filter_config[0], filter_config[1], num_conv_layers=2,
                                   drop_rate=dropout_rate)  # 128x112x112
        self.encoder_3 = ConvBlock(filter_config[1], filter_config[2], num_conv_layers=2,
                                   drop_rate=dropout_rate)  # 256x56x56
        self.encoder_4 = ConvBlock(filter_config[2], filter_config[3], num_conv_layers=2,
                                   drop_rate=dropout_rate)  # 512x28x28
        self.encoder_5 = ConvBlock(filter_config[3], filter_config[4], num_conv_layers=2,
                                   drop_rate=dropout_rate)  # 1024x14x14
        self.encoder_6 = ConvBlock(filter_config[4], filter_config[5], num_conv_layers=2,
                                   drop_rate=dropout_rate)  # 2048x7x7
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Expansion Path
        self.decoder_1 = UpconvBlock(filter_config[5], filter_config[4], upmode="deconv_2")  # 1024x14x14
        self.conv1 = ConvBlock(filter_config[4] * 2, filter_config[4], num_conv_layers=2, drop_rate=dropout_rate)

        self.decoder_2 = UpconvBlock(filter_config[4], filter_config[3], upmode="deconv_2")  # 512x28x28
        self.conv2 = ConvBlock(filter_config[3] * 2, filter_config[3], num_conv_layers=2, drop_rate=dropout_rate)

        self.decoder_3 = UpconvBlock(filter_config[3], filter_config[2], upmode="deconv_2")  # 256x56x56
        self.conv3 = ConvBlock(filter_config[2] * 2, filter_config[2], num_conv_layers=2, drop_rate=dropout_rate)

        self.decoder_4 = UpconvBlock(filter_config[2], filter_config[1], upmode="deconv_2")  # 128x112x112
        self.conv4 = ConvBlock(filter_config[1] * 2, filter_config[1], num_conv_layers=2, drop_rate=dropout_rate)

        self.decoder_5 = UpconvBlock(filter_config[1], filter_config[0], upmode="deconv_2")  # 64x224x224
        self.conv5 = ConvBlock(filter_config[0] * 2, filter_config[0], num_conv_layers=2, drop_rate=dropout_rate)

        if self.use_skipAtt:
            self.Att1 = AdditiveAttentionBlock(F_g=filter_config[4], F_x=filter_config[4], F_inter=filter_config[3])
            self.Att2 = AdditiveAttentionBlock(F_g=filter_config[3], F_x=filter_config[3], F_inter=filter_config[2])
            self.Att3 = AdditiveAttentionBlock(F_g=filter_config[2], F_x=filter_config[2], F_inter=filter_config[1])
            self.Att4 = AdditiveAttentionBlock(F_g=filter_config[1], F_x=filter_config[1], F_inter=filter_config[0])
            self.Att5 = AdditiveAttentionBlock(F_g=filter_config[0], F_x=filter_config[0],
                                               F_inter=int(filter_config[0] / 2))

        self.classifier = nn.Conv2d(filter_config[0], n_classes, kernel_size=1, stride=1, padding=0)  # classNumx224x224

    def forward(self, inputs):
        """
        Forward pass of the UNet model.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_classes, height, width).

        """
        e1 = self.encoder_1(inputs)  # batch size x 64 x 224 x 224
        p1 = self.pool(e1)  # batch size x 64 x 112 x 112

        e2 = self.encoder_2(p1)  # batch size x 128 x 112 x 112
        p2 = self.pool(e2)  # batch size x 128 x 56 x 56

        e3 = self.encoder_3(p2)  # batch size x 256 x 56 x 56
        p3 = self.pool(e3)  # batch size x 256 x 28 x 28

        e4 = self.encoder_4(p3)  # batch size x 512 x 28 x 28
        p4 = self.pool(e4)  # batch size x 1024 x 14 x 14

        e5 = self.encoder_5(p4)  # batch size x 1024 x 14 x 14
        p5 = self.pool(e5)  # batch size x 1024 x 7 x 7

        e6 = self.encoder_6(p5)  # batch size x 2048 x 7 x 7

        d6 = self.decoder_1(e6)  # batch size x 1024 x 14 x 14

        if self.use_skipAtt:
            x5 = self.Att1(g=d6, x=e5)  # batch size x 1024 x 14 x 14
            skip1 = torch.cat((x5, d6), dim=1)  # batch size x 2048 x 14 x 14
        else:
            skip1 = torch.cat((e5, d6), dim=1)  # batch size x 2048 x 14 x 14

        d6_proper = self.conv1(skip1)  # batch size x 1024 x 14 x 14

        d5 = self.decoder_2(d6_proper)  # batch size x 512 x 28 x 28

        if self.use_skipAtt:
            x4 = self.Att2(g=d5, x=e4)  # batch size x 512 x 28 x 28
            skip2 = torch.cat((x4, d5), dim=1)  # batch size x 1024 x 28 x 28
        else:
            skip2 = torch.cat((e4, d5), dim=1)  # batch size x 1024 x 28 x 28

        d5_proper = self.conv2(skip2)  # batch size x 512 x 28 x 28

        d4 = self.decoder_3(d5_proper)  # batch size x 256 x 56 x 56

        if self.use_skipAtt:
            x3 = self.Att3(g=d4, x=e3)  # batch size x 256 x 56 x 56
            skip3 = torch.cat((x3, d4), dim=1)  # batch size x 512 x 56 x 56
        else:
            skip3 = torch.cat((e3, d4), dim=1)  # batch size x 512 x 56 x 56

        d4_proper = self.conv3(skip3)  # batch size x 256 x 56 x 56

        d3 = self.decoder_4(d4_proper)  # batch size x 128 x 112 x 112

        if self.use_skipAtt:
            x2 = self.Att4(g=d3, x=e2)  # batch size x 128 x 112 x 112
            skip4 = torch.cat((x2, d3), dim=1)  # batch size x 256 x 112 x 112
        else:
            skip4 = torch.cat((e2, d3), dim=1)  # batch size x 256 x 112 x 112

        d3_proper = self.conv4(skip4)  # batch size x 128 x 112 x 112

        d2 = self.decoder_5(d3_proper)  # batch size x 64 x 224 x 224

        if self.use_skipAtt:
            x1 = self.Att5(g=d2, x=e1)  # batch size x 64 x 224 x 224
            skip5 = torch.cat((x1, d2), dim=1)  # batch size x 128 x 224 x 224
        else:
            skip5 = torch.cat((e1, d2), dim=1)  # batch size x 128 x 224 x 224

        d2_proper = self.conv5(skip5)  # batch size x 64 x 224 x 224

        d1 = self.classifier(d2_proper)  # batch size x classNum x 224 x 224

        return d1