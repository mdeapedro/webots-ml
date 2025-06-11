import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np

IMAGES_DIR = 'dataset/images'
LABELS_DIR = 'dataset/labels'
INPUT_SIZE = 224
S = 7  # Grid size
B = 1  # Boxes per cell
C = 3  # Classes (0: objective, 1: box, 2: ball)
NUM_EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 0.001

class SimpleYOLO(nn.Module):
    def __init__(self):
        super(SimpleYOLO, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * (INPUT_SIZE // 32)**2, 512),
            nn.ReLU(),
            nn.Linear(512, S * S * (C + B * 5)),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x.view(-1, S, S, C + B * 5)

def yolo_loss(pred, target):
    # Simples loss: MSE para bbox, BCE para conf e class
    lambda_coord = 5
    lambda_noobj = 0.5
    obj_mask = target[..., 4] > 0
    noobj_mask = ~obj_mask

    loss_coord = lambda_coord * torch.sum((pred[..., :4] - target[..., :4])**2 * obj_mask.unsqueeze(-1))
    loss_conf_obj = torch.sum((pred[..., 4] - target[..., 4])**2 * obj_mask)
    loss_conf_noobj = lambda_noobj * torch.sum((pred[..., 4] - target[..., 4])**2 * noobj_mask)
    loss_class = torch.sum((pred[..., 5:] - target[..., 5:])**2 * obj_mask.unsqueeze(-1))

    return loss_coord + loss_conf_obj + loss_conf_noobj + loss_class

class YOLODataset(Dataset):
    def __init__(self):
        self.images = [f for f in os.listdir(IMAGES_DIR) if f.endswith('.png')]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(IMAGES_DIR, self.images[idx])
        label_path = os.path.join(LABELS_DIR, self.images[idx].replace('.png', '.txt'))
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE)) / 255.0
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        
        target = torch.zeros((S, S, C + B * 5))
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    cls, cx, cy, nw, nh = map(float, line.strip().split())
                    cls = int(cls)
                    i = int(cy * S)
                    j = int(cx * S)
                    if target[i, j, 4] == 0:  # Apenas um box por célula
                        target[i, j, 0:4] = torch.tensor([cx * S - j, cy * S - i, nw, nh])
                        target[i, j, 4] = 1.0  # Conf
                        target[i, j, 5 + cls] = 1.0
        
        return img, target

dataset = YOLODataset()
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = SimpleYOLO()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
    for imgs, targets in dataloader:
        preds = model(imgs)
        loss = yolo_loss(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item()}')

torch.save(model.state_dict(), 'model.pth')
