import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # I could enclose the Encoder and Decoder within loops to make them fewer lines. However, I find it more readable to leave the steps exapnded

        # This model is a direct implementation of UNet as described in "U-Net: Convolutional Networks for Biomedical Image Segmentation" by Olaf Ronneberger, Philipp Fischer, and Thomas Brox

        # "Encoder" -- 4 consecutive double convolutions+ReLU and max pooling
        # Input: 512x512x1 Note: PyTorch tensors have backwards indexing: (channels, height, width), I'm listing output dims: (height, width, channels)
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(),     # output: 512x512x64
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU()     # output: 512x512x64
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(),   # output: 256x256x128
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU()   # output: 256x256x128
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(),  # output: 128x128x256
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU()   # output: 128x128x256
        )

        self.down4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU(),  # output: 64x64x512
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU()   # output: 64x64x512
        )

        self.mpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottom part of the U ("bridge" perhaps)
        self.bridge = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1), nn.ReLU(), # output: 32x32x1024
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1), nn.ReLU() # output: 32x32x1024
        )

        # "Decoder" -- During the encoding step, a copy of the output of the double convolution is saved. In each layer, this copy is concatenated to an upscaled output of the previous layer. Next, another double convolution+ReLU is performed
        self.upcon1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up1 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU(), # output: 64x64x512
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU()   # output: 64x64x512
        )

        self.upcon2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(),  # output: 128x128x256
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU()   # output: 128x128x256
        )

        self.upcon3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(),  # output: 256x256x128
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU()   # output: 256x256x128
        )

        self.upcon4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(),   # output: 512x512x64
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU()     # output: 512x512x64
        )

        # Finally, a convolution is perfomed with a kernel of 1 to provide an output of identical size to the input
        self.final = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)        # output: 512x512x1

    def forward(self, x):

        # Encoder
        x = self.down1(x)
        skip1 = x
        x = self.mpool(x)

        x = self.down2(x)
        skip2 = x
        x = self.mpool(x)
        
        x = self.down3(x)
        skip3 = x
        x = self.mpool(x)

        x = self.down4(x)
        skip4 = x
        x = self.mpool(x)
        
        # Bridge
        x = self.bridge(x)

        # Decoder
        x = self.upcon1(x)
        x = torch.cat((x, skip4), dim=1)
        x = self.up1(x)

        x = self.upcon2(x)
        x = torch.cat((x, skip3), dim=1)
        x = self.up2(x)

        x = self.upcon3(x)
        x = torch.cat((x, skip2), dim=1)
        x = self.up3(x)

        x = self.upcon4(x)
        x = torch.cat((x, skip1), dim=1)
        x = self.up4(x)

        # Perform final convolution to return image of identical size to the input
        return self.final(x)
    


if __name__ == '__main__':
    def testShape():
        device = torch.device('cude' if torch.cuda.is_available() else 'cpu')
        x = torch.randn(1, 1, 512, 512).to(device)
        model = UNet().to(device)
        predictions = model(x)
        print('\n', 'Out shape:', predictions.shape, '\n',
            'Expected shape: torch.size([1, 1, 512, 512])', '\n')
    
    testShape()