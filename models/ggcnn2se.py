import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) 注意力模块
    可以帮助网络在遮挡环境下聚焦有效通道特征
    """
    def __init__(self, channel, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # squeeze：对每个通道做全局平均池化
        y = self.avg_pool(x).view(b, c)
        # excitation：通过全连接层得到通道权重
        y = self.fc(y).view(b, c, 1, 1)
        # scale：逐通道乘以权重
        return x * y


class GGCNN2SE(nn.Module):
    """
    改进版 GGCNN2，在 dilated conv 后添加一个 SEBlock 注意力模块
    以在遮挡环境下聚焦有效特征通道
    """
    def __init__(self, input_channels=1, filter_sizes=None, l3_k_size=5, dilations=None):
        super().__init__()

        if filter_sizes is None:
            filter_sizes = [16,  # 第一组卷积层通道数
                            16,  # 第二组卷积层通道数
                            32,  # 膨胀卷积层通道数
                            16]  # 转置/上采样卷积层通道数

        if dilations is None:
            dilations = [2, 4]

        # 块1：两次卷积 + ReLU + MaxPool
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, filter_sizes[0], kernel_size=11, stride=1, padding=5, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_sizes[0], filter_sizes[0], kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 块2：再次卷积 + ReLU + MaxPool
        self.block2 = nn.Sequential(
            nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_sizes[1], filter_sizes[1], kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 块3：膨胀卷积 + ReLU + SE 注意力模块
        self.block3 = nn.Sequential(
            nn.Conv2d(filter_sizes[1], filter_sizes[2], kernel_size=l3_k_size,
                      dilation=dilations[0], stride=1,
                      padding=(l3_k_size // 2) * dilations[0], bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_sizes[2], filter_sizes[2], kernel_size=l3_k_size,
                      dilation=dilations[1], stride=1,
                      padding=(l3_k_size // 2) * dilations[1], bias=True),
            nn.ReLU(inplace=True),
            SEBlock(filter_sizes[2], reduction=4)  # 在这里插入 SEBlock
        )

        # 块4：上采样并细化输出特征图
        self.block4 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(filter_sizes[2], filter_sizes[3], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(filter_sizes[3], filter_sizes[3], 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 最终输出层：分别预测 pos, cos, sin, width
        self.pos_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)
        self.cos_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)
        self.sin_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)
        self.width_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)

        # 初始化卷积权重
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        pos_output = self.pos_output(x)
        cos_output = self.cos_output(x)
        sin_output = self.sin_output(x)
        width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output

    def compute_loss(self, xc, yc):
        """
        计算损失，可根据需要加入加权或平滑等改进
        """
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)

        p_loss = F.mse_loss(pos_pred, y_pos)
        cos_loss = F.mse_loss(cos_pred, y_cos)
        sin_loss = F.mse_loss(sin_pred, y_sin)
        width_loss = F.mse_loss(width_pred, y_width)

        total_loss = p_loss + cos_loss + sin_loss + width_loss

        return {
            'loss': total_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }