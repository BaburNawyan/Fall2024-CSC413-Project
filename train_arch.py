from data_augmentations import get_augmentations
from dataset import CloudDataset
import torch
import torch.nn as nn
import torch.nn.functional as F

# Modify as necessary
train_image_paths = {}
train_gt_mask_paths = []
val_image_paths = {}
val_gt_mask_paths = []


train_dataset = CloudDataset(
    image_paths=train_image_paths,
    gt_mask_paths=train_gt_mask_paths,
    augmentations=get_augmentations()
)

val_dataset = CloudDataset(
    image_paths=val_image_paths,
    gt_mask_paths=val_gt_mask_paths,
    augmentations=None  # No augmentations for validation
)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(gating_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU()

    def forward(self, x, g):
        # Resize x to match g's spatial dimensions
        x_resized = F.interpolate(x, size=g.shape[2:], mode="bilinear", align_corners=True)
        
        # Transform features
        g1 = self.W_g(g)  # Context from decoder
        x1 = self.W_x(x_resized)  # Resized skip connection
        
        # Compute attention map
        psi = self.relu(g1 + x1)  # Element-wise addition and ReLU
        psi = self.psi(psi)  # Sigmoid activation for attention map
        
        # Apply attention to resized skip connection
        return x_resized * psi
    
def crop_and_concat(encoder_features, decoder_features):
    # Crop encoder_features to match decoder_features
    _, _, h, w = decoder_features.shape
    encoder_features = F.interpolate(encoder_features, size=(h, w), mode="bilinear", align_corners=True)
    return encoder_features


class UNetWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetWithAttention, self).__init__()
        # Encoder
        self.encoder1 = ConvBlock(in_channels, 32)
        self.encoder2 = ConvBlock(32, 64)
        self.encoder3 = ConvBlock(64, 128)
        self.encoder4 = ConvBlock(128, 256)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ConvBlock(256, 512)

        # Attention and Decoder
        self.att4 = AttentionBlock(256, 512, 128)
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = ConvBlock(512, 256)

        self.att3 = AttentionBlock(128, 256, 64)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = ConvBlock(256, 128)

        self.att2 = AttentionBlock(64, 128, 32)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(128, 64)

        self.att1 = AttentionBlock(32, 64, 16)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(64, 32)

        # Final Convolution
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)  # 384x384 -> 32x380x380
        e2 = self.encoder2(self.pool(e1))  # 190x190 -> 64x186x186
        e3 = self.encoder3(self.pool(e2))  # 93x93 -> 128x89x89
        e4 = self.encoder4(self.pool(e3))  # 45x45 -> 256x41x41

        # Bottleneck
        b = self.bottleneck(self.pool(e4))  # 41x41 -> 512x41x41

        # Decoder with Attention Gates
        up4 = self.upconv4(b)
        att4 = self.att4(e4, b)
        d4 = self.decoder4(torch.cat([crop_and_concat(att4, up4), up4], dim=1))  # 256x41x41 -> 128x82x82

        up3 = self.upconv3(d4)
        att3 = self.att3(e3, d4)
        d3 = self.decoder3(torch.cat([crop_and_concat(att3, up3), up3], dim=1))  # 128x82x82 -> 64x166x166

        up2 = self.upconv2(d3)
        att2 = self.att2(e2, d3)
        d2 = self.decoder2(torch.cat([crop_and_concat(att2, up2), up2], dim=1))  # 64x166x166 -> 32x324x324

        up1 = self.upconv1(d2)
        att1 = self.att1(e1, d2)
        d1 = self.decoder1(torch.cat([crop_and_concat(att1, up1), up1], dim=1))  # 32x324x324

        # Output
        return torch.sigmoid(self.final_conv(d1))  # 1x324x324


# Instantiate the model
model = UNetWithAttention(in_channels=4, out_channels=1)
input_image = torch.rand((1, 4, 384, 384))  # Example input (batch_size=1, channels=4, height=384, width=384)
output = model(input_image)
print(output.shape)  # Should be (1, 1, 324, 324) # Note calculations are of due to padding, this might turn out better though since we get a better final shape.


from torch.utils.data import DataLoader
import torch.optim as optim

# Create a small synthetic dataset
class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=10, image_size=(4, 384, 384)):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Random synthetic images and binary masks
        image = torch.rand(self.image_size)  # Random values in [0, 1]
        mask = (torch.rand(1, self.image_size[1], self.image_size[2]) > 0.5).float()  # Binary mask
        return image, mask

# Initialize synthetic dataset and dataloader
synthetic_dataset = SyntheticDataset()
dataloader = DataLoader(synthetic_dataset, batch_size=2, shuffle=True)

# Initialize model, loss, and optimizer
model = UNetWithAttention(in_channels=4, out_channels=1)
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Test training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.train()

# Training loop for a few iterations
print("Starting sanity check...")
for epoch in range(2):  # Train for 2 epochs
    for batch_idx, (images, masks) in enumerate(dataloader):
        images, masks = images.to(device), masks.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/2], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item()}")

print("Sanity check completed!")

