import torch
import torch.nn as nn

class DepthEstimationModel(nn.Module):
    def __init__(self):
        super(DepthEstimationModel, self).__init__()

        #encoder: feature extraction layers with more complexity
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  #128x128 -> 64x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  #64x64 -> 32x32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  #32x32 -> 16x16
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  #16x16 -> 8x8
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        #bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),  #8x8 -> 4x4
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

        #decoder: upsampling layers with skip connections
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),  #4x4 -> 8x8
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  #8x8 -> 16x16
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  #16x16 -> 32x32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  #32x32 -> 64x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.final_layer = nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1)  #64x64 -> 128x128

    def forward(self, x):
        #encoder
        e1 = self.encoder1(x)  #128x128 -> 64x64
        e2 = self.encoder2(e1)  #64x64 -> 32x32
        e3 = self.encoder3(e2)  #32x32 -> 16x16
        e4 = self.encoder4(e3)  #16x16 -> 8x8

        #bottleneck
        b = self.bottleneck(e4)  #8x8 -> 4x4

        #decoder with skip connections
        d4 = self.decoder4(torch.cat([b, e4], dim=1))  #4x4 -> 8x8
        d3 = self.decoder3(torch.cat([d4, e3], dim=1))  #8x8 -> 16x16
        d2 = self.decoder2(torch.cat([d3, e2], dim=1))  #16x16 -> 32x32
        d1 = self.decoder1(torch.cat([d2, e1], dim=1))  #32x32 -> 64x64
        output = self.final_layer(d1)  #64x64 -> 128x128

        return output
