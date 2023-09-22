from __future__ import print_function


from torchsummary import summary
affine_par = True

import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, group=1):
        super(DecoderBlock, self).__init__()

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1, groups=group)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        # B, C/4, H, W -> B, C/4, H, W
        self.deconv2 = nn.ConvTranspose2d(
            in_channels // 4,
            in_channels // 4,
            3,
            stride=2,
            padding=1,
            output_padding=1,
            groups=group,
        )
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1, groups=group)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class acf_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_channels, out_channels):
        super(acf_Module, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.init_weight()

    def forward(self, feat_ffm, coarse_x):
        """
            inputs :
                feat_ffm : input feature maps( B X C X H X W), C is channel
                coarse_x : input feature maps( B X N X H X W), N is class
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, N, height, width = coarse_x.size()

        # CCB: Class Center Block start...
        # 1x1conv -> F'
        feat_ffm = self.conv1(feat_ffm)
        b, C, h, w = feat_ffm.size()

        # P_coarse reshape ->(B, N, W*H)
        proj_query = coarse_x.view(m_batchsize, N, -1)

        # F' reshape and transpose -> (B, W*H, C')
        proj_key = feat_ffm.view(b, C, -1).permute(0, 2, 1)

        # multiply & normalize ->(B, N, C')
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        # CCB: Class Center Block end...

        # CAB: Class Attention Block start...
        # transpose ->(B, C', N)
        attention = attention.permute(0, 2, 1)

        # (B, N, W*H)
        proj_value = coarse_x.view(m_batchsize, N, -1)

        # # multiply (B, C', N)(B, N, W*H)-->(B, C, W*H)
        out = torch.bmm(attention, proj_value)

        out = out.view(m_batchsize, C, height, width)

        # 1x1conv
        out = self.conv2(out)
        # CAB: Class Attention Block end...

        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class ACFModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ACFModule, self).__init__()

        self.acf = acf_Module(in_channels, out_channels)


    def forward(self, x, coarse_x):
        class_output = self.acf(x, coarse_x)
        return class_output

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(4, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


def conv_block(in_channel, out_channel): # 一个卷积块
    layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(),
        nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False)
    )
    return layer


class dense_block(nn.Module):
    def __init__(self, in_channel, growth_rate, num_layers):
        super().__init__()  # growth_rate => k => out_channel
        block = []
        channel = in_channel  # channel => in_channel
        for i in range(num_layers):
            block.append(conv_block(channel, growth_rate))
            channel += growth_rate  # 连接每层的特征
        self.net = nn.Sequential(*block)  # 实现简单的顺序连接模型
        # 必须确保前一个模块的输出大小和下一个模块的输入大小是一致的

    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = torch.cat((out, x), dim=1)  # contact同维度拼接特征，stack(是把list扩维连接
            # torch.cat()是为了把多个tensor进行拼接，在给定维度上对输入的张量序列seq 进行连接操作
            # inputs : 待连接的张量序列，可以是任意相同Tensor类型的python 序列
            # dim : 选择的扩维, 必须在0到len(inputs[0])之间，沿着此维连接张量序列

        return x


def transition(in_channel, out_channel):
    trans_layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(),
        nn.Conv2d(in_channel, out_channel, 1),  # kernel_size = 1 1x1 conv
        # nn.AvgPool2d(2, 2)  # 2x2 pool
    )
    return trans_layer


class BasicResnetBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, padding=1, downsample=None):
        super(BasicResnetBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # out = self.ca(out) * out
        # out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class HourglassModuleMTL(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        super(HourglassModuleMTL, self).__init__()
        self.depth = depth
        self.block = block
        self.upsample = nn.Upsample(scale_factor=2)
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)
        self.ca = self._make_coordinate_attention_layer(planes,depth)

    def _make_residual1(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes * block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(4):
                res.append(self._make_residual1(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual1(block, num_blocks, planes))
                res.append(self._make_residual1(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _make_coordinate_attention_layer( self,planes,depth):
        ca = []
        for i in range(depth):
            ca.append(CoordAtt(planes,planes))
        return nn.ModuleList(ca)

    def _hour_glass_forward(self, n, x):
        rows = x.size(2)
        cols = x.size(3)
        up1 = self.ca[n - 1](x)
        up1 = self.hg[n - 1][0](up1)
        low1 = F.max_pool2d(x, 2, stride=2, ceil_mode=True)
        low1 = self.hg[n - 1][1](low1)

        if n > 1:
            low2_1, low2_2 = self._hour_glass_forward(n - 1, low1)
        else:
            low2_1 = self.hg[n - 1][4](low1)
            low2_2 = self.hg[n - 1][5](low1)
        low3_1 = self.hg[n - 1][2](low2_1)
        low3_2 = self.hg[n - 1][3](low2_2)
        up2_1 = self.upsample(low3_1)
        up2_2 = self.upsample(low3_2)
        out_1 = up1 + up2_1[:, :, :rows, :cols]
        out_2 = up1 + up2_2[:, :, :rows, :cols]

        return out_1, out_2

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


class MCTNNet(nn.Module):
    def __init__(
        self,
        task1_classes=5,
        task2_classes=37,
        block=BasicResnetBlock,
        in_channels=3,
        num_stacks=2,
        num_blocks=1,
        hg_num_blocks=3,
        depth=3,
    ):
        super(MCTNNet, self).__init__()

        self.inplanes = 64
        self.midplanes = 128
        self.num_feats = 256
        self.num_stacks = num_stacks

        self.conv1 = nn.Conv2d(
            in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=True
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.DB1 = self._make_dense_block(64,growth_rate=32,num=6)
        self.TL1 = self._make_transition_layer(256)
        self.DB2 = self._make_dense_block(128, growth_rate=32, num=12)
        self.TL2 = self._make_transition_layer(512)

        self.layer3 = self._make_residual(block, self.num_feats, num_blocks)


        # build hourglass modules
        ch = self.num_feats * block.expansion
        hg = []
        res_1, fc_1, score_1, _fc_1, _score_1 = [], [], [], [], []
        res_2, fc_2, score_2, _fc_2, _score_2 = [], [], [], [], []

        for i in range(num_stacks):
            hg.append(HourglassModuleMTL(block, hg_num_blocks, self.num_feats, depth))

            res_1.append(self._make_residual(block, self.num_feats, hg_num_blocks))
            res_2.append(self._make_residual(block, self.num_feats, hg_num_blocks))

            if i < num_stacks - 1:
                fc_1.append(self._make_fc(ch, ch))
                fc_2.append(self._make_fc(ch, ch))
                score_1.append(nn.Conv2d(ch, task1_classes, kernel_size=1, bias=True))
                score_2.append(nn.Conv2d(ch, task2_classes, kernel_size=1, bias=True))

                _fc_1.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                _score_1.append(nn.Conv2d(task1_classes, ch, kernel_size=1, bias=True))
                _fc_2.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                _score_2.append(nn.Conv2d(task2_classes, ch, kernel_size=1, bias=True))
            else:
                fc_1.append(self._make_fc(ch, ch//2))
                score_1.append(nn.Conv2d(ch//2, task1_classes, kernel_size=1, bias=True))

                fc_2.append(self._make_fc(ch, ch))
                score_2.append(nn.Conv2d(ch, task2_classes, kernel_size=1, bias=True))

        self.hg = nn.ModuleList(hg)
        self.res_1 = nn.ModuleList(res_1)
        self.fc_1 = nn.ModuleList(fc_1)
        self.score_1 = nn.ModuleList(score_1)
        self._fc_1 = nn.ModuleList(_fc_1)
        self._score_1 = nn.ModuleList(_score_1)

        self.acfhead1 = ACFModule(32, 32)

        self.res_2 = nn.ModuleList(res_2)
        self.fc_2 = nn.ModuleList(fc_2)
        self.score_2 = nn.ModuleList(score_2)
        self._fc_2 = nn.ModuleList(_fc_2)
        self._score_2 = nn.ModuleList(_score_2)

        # Deep supervised module
        self.deconv_1 = self._make_fc(256, 128)
        self.conv_1 = self._make_conv3x3(256, 128)
        self.deconv_2 = self._make_ConvTranspose(128,64)
        self.conv_2 = self._make_conv3x3(128, 64)
        self.deconv_3 = self._make_ConvTranspose(64, 32)
        self.conv_3 = self._make_conv3x3(32, 5)
        self.deconv_4 = self._make_ConvTranspose(32, 5)
        self.conv_4 = self._make_conv3x3(64, 32)

        # Final Classifier
        self.decoder1 = DecoderBlock(128, 64)
        self.decoder1_score = nn.Conv2d(
            64, task1_classes, kernel_size=1, bias=True
        )
        self.finaldeconv1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3,padding=1)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, task1_classes, 1)

        # Final Classifier
        self.angle_decoder1 = DecoderBlock(self.num_feats, self.inplanes)
        self.angle_decoder1_score = nn.Conv2d(
            self.inplanes, task2_classes, kernel_size=1, bias=True
        )
        self.angle_finaldeconv1 = nn.ConvTranspose2d(self.inplanes, 32, 3, stride=2)
        self.angle_finalrelu1 = nn.ReLU(inplace=True)
        self.angle_finalconv2 = nn.Conv2d(32, 32, 3)
        self.angle_finalrelu2 = nn.ReLU(inplace=True)
        self.angle_finalconv3 = nn.Conv2d(32, task2_classes, 2, padding=1)

    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=True,
                )
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_dense_block(self, channels, growth_rate, num):  # num是块的个数
        block = []
        block.append(dense_block(channels, growth_rate, num))
        channels += num * growth_rate  # 特征变化 # 这里记录下即可，生成时dense_block()中也做了变化
        return nn.Sequential(*block)

    def _make_transition_layer(self, channels):
        block = []
        block.append(transition(channels, channels // 2))  # channels // 2就是为了降低复杂度 θ = 0.5
        return nn.Sequential(*block)



    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(outplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(conv, bn, self.relu)

    def _make_conv3x3(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(outplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=3, padding=1,bias=True)
        return nn.Sequential(conv, bn, self.relu)

    def _make_ConvTranspose(self, in_channels, outchanels):
        deconv = nn.ConvTranspose2d(
            in_channels,
            outchanels,
            3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        norm = nn.BatchNorm2d(outchanels)
        return nn.Sequential(deconv, norm, self.relu)

    def forward(self, x):

        out_1 = []
        out_2 = []

        rows = x.size(2)
        cols = x.size(3)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.DB1(x)
        x = self.TL1(x)
        x = self.maxpool(x)
        x = self.DB2(x)
        x = self.TL2(x)

        for i in range(self.num_stacks):
            y1, y2 = self.hg[i](x)


            if i < self.num_stacks - 1:

                y1, y2 = self.res_1[i](y1), self.res_2[i](y2)
                y1, y2 = self.fc_1[i](y1), self.fc_2[i](y2)
                score1, score2 = self.score_1[i](y1), self.score_2[i](y2)

            else:
                s1 = self.deconv_1(y1)
                y1, y2 = self.res_1[i](y1), self.res_2[i](y2)
                y1, y2 = self.fc_1[i](y1), self.fc_2[i](y2)


                s2 = self.conv_1(torch.cat((s1,y1),dim=1))
                score1 = self.score_1[i](s2)
                score2 = self.score_2[i](y2)

            out_1.append(
                score1[:, :, : int(math.ceil(rows / 4.0)), : int(math.ceil(cols / 4.0))]
            )

            out_2.append(
                score2[:, :, : int(math.ceil(rows / 4.0)), : int(math.ceil(cols / 4.0))]
            )
            if i < self.num_stacks - 1:
                _fc_1, _fc_2 = self._fc_1[i](y1), self._fc_2[i](y2)
                _score_1, _score_2 = self._score_1[i](score1), self._score_2[i](score2)
                x = x + _fc_1 + _score_1 + _fc_2 + _score_2

        # Final Classification
        d1 = self.decoder1(y1)[
             :, :, : int(math.ceil(rows / 2.0)), : int(math.ceil(cols / 2.0))
             ]

        s3 = self.conv_2(torch.cat((self.deconv_2(s2),d1),dim=1))
        d1_score = self.decoder1_score(s3)

        out_1.append(d1_score)

        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)

        s4 = self.deconv_3(s3)
        s5 = self.conv_3(s4)
        out_1.append(s5)
        s6 = self.acfhead1(s4,s5)
        s7 = self.conv_4(torch.cat((s6,f4),dim=1))
        f5 = self.finalconv3(s7)
        out_1.append(f5)

        # Final Classification
        a_d1 = self.angle_decoder1(y2)[
               :, :, : int(math.ceil(rows / 2.0)), : int(math.ceil(cols / 2.0))
               ]
        a_d1_score = self.angle_decoder1_score(a_d1)
        out_2.append(a_d1_score)
        a_f1 = self.angle_finaldeconv1(a_d1)
        a_f2 = self.angle_finalrelu1(a_f1)
        a_f3 = self.angle_finalconv2(a_f2)
        a_f4 = self.angle_finalrelu2(a_f3)
        a_f5 = self.angle_finalconv3(a_f4)
        out_2.append(a_f5)

        return out_1, out_2

if __name__ == "__main__":
    model = MCTNNet()
    model.eval()
    image = torch.randn(4, 3, 256, 256)
    with torch.no_grad():
        output1 = model.forward(image)
    print(model)
    summary(model, input_size=(3,512,512),device="cpu")