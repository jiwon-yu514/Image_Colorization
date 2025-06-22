import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os


class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super(UNetGenerator, self).__init__()

        def down_block(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        def up_block(in_feat, out_feat, dropout=0.0):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, kernel_size=4, stride=2, padding=1),
                      nn.BatchNorm2d(out_feat),
                      nn.ReLU(inplace=True)]
            if dropout:
                layers.append(nn.Dropout(dropout))
            return nn.Sequential(*layers)

        self.down1 = down_block(in_channels, 64, normalize=False)
        self.down2 = down_block(64, 128)
        self.down3 = down_block(128, 256)
        self.down4 = down_block(256, 512)
        self.down5 = down_block(512, 512)
        self.down6 = down_block(512, 512)
        self.down7 = down_block(512, 512)
        self.down8 = down_block(512, 512, normalize=False)

        self.up1 = up_block(512, 512, dropout=0.5)
        self.up2 = up_block(1024, 512, dropout=0.5)
        self.up3 = up_block(1024, 512, dropout=0.5)
        self.up4 = up_block(1024, 512)
        self.up5 = up_block(1024, 256)
        self.up6 = up_block(512, 128)
        self.up7 = up_block(256, 64)
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8)
        u2 = self.up2(torch.cat([u1, d7], 1))
        u3 = self.up3(torch.cat([u2, d6], 1))
        u4 = self.up4(torch.cat([u3, d5], 1))
        u5 = self.up5(torch.cat([u4, d4], 1))
        u6 = self.up6(torch.cat([u5, d3], 1))
        u7 = self.up7(torch.cat([u6, d2], 1))
        u8 = self.up8(torch.cat([u7, d1], 1))
        return u8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_paths = [
    "saved_models/best_generator_1.pth",
    "saved_models/best_generator_2.pth",
    "saved_models/best_generator_3.pth"
]

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def colorize_image(input_path, output_dir, filename):
    img = Image.open(input_path).convert("L")
    input_tensor = transform(img).unsqueeze(0).to(device)

    os.makedirs(output_dir, exist_ok=True)
    result_files = []

    for idx, model_path in enumerate(model_paths, start=1):
        generator = UNetGenerator().to(device)
        generator.load_state_dict(torch.load(model_path, map_location=device))
        generator.eval()

        with torch.no_grad():
            output = generator(input_tensor).cpu()[0]
            output = (output * 0.5 + 0.5).clamp(0, 1)
            output_img = transforms.ToPILImage()(output)
            result_filename = f"{os.path.splitext(filename)[0]}{idx}.png"
            output_img.save(os.path.join(output_dir, result_filename))
            result_files.append(result_filename)

    return result_files
