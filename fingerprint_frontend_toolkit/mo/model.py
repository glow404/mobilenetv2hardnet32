import torch
import torch.nn as nn
import torch.nn.functional as F


# class Encoder3D(nn.Module):
#
#     def __init__(self):
#
#         super().__init__()
#
#         self.backbone = nn.Sequential(
#
#             nn.Conv3d(
#                 1,
#                 16,
#                 kernel_size=(5,3,3),
#                 padding=(2,1,1)
#             ),
#
#             nn.BatchNorm3d(16),
#             nn.ReLU(),
#
#             nn.MaxPool3d(2),
#
#             nn.Conv3d(
#                 16,
#                 32,
#                 kernel_size=3,
#                 padding=1
#             ),
#
#             nn.BatchNorm3d(32),
#             nn.ReLU(),
#
#             nn.MaxPool3d(2),
#
#             nn.Conv3d(
#                 32,
#                 64,
#                 kernel_size=3,
#                 padding=1
#             ),
#
#             nn.BatchNorm3d(64),
#             nn.ReLU(),
#
#             nn.AdaptiveAvgPool3d(1)
#         )
#
#         self.fc = nn.Linear(
#             64,
#             128
#         )
#
#     def forward(self, x):
#
#         x = self.backbone(x)
#
#         x = x.flatten(1)
#
#         x = self.fc(x)
#
#         x = F.normalize(
#             x,
#             p=2,
#             dim=1
#         )
#
#         return x


import torch
import torch.nn as nn
import torch.nn.functional as F


class DWConv3D(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
    ):

        super().__init__()

        self.depthwise = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size,
            padding=padding,
            groups=in_channels
        )

        self.pointwise = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=1
        )


        self.bn = nn.BatchNorm3d(
            out_channels
        )

        self.act = nn.ReLU(inplace=True)


    def forward(self,x):

        x=self.depthwise(x)

        x=self.pointwise(x)

        x=self.bn(x)

        x=self.act(x)

        return x



class Encoder3D(nn.Module):

    def __init__(self):

        super().__init__()


        self.backbone = nn.Sequential(

            # 1,16,100,110
            DWConv3D(
                1,
                16,
                kernel_size=(5,3,3),
                padding=(2,1,1)
            ),


            nn.MaxPool3d(2),


            # 16
            DWConv3D(
                16,
                32
            ),


            nn.MaxPool3d(2),


            # 32
            DWConv3D(
                32,
                48
            ),


            nn.AdaptiveAvgPool3d(1)

        )


        self.fc = nn.Linear(
            48,
            128
        )


    def forward(self,x):

        x=self.backbone(x)

        x=x.flatten(1)

        x=self.fc(x)


        x=F.normalize(
            x,
            p=2,
            dim=1
        )

        return x