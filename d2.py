import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% 1. 加载并检查数据
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True,
    download=True
)
test_dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=False,
    download=True
)

# 检查数据维度
print(f"训练集图像形状: {train_dataset.data.shape}")  # (50000, 32, 32, 3)
print(f"测试集图像形状: {test_dataset.data.shape}")   # (10000, 32, 32, 3)
print(f"类别标签: {train_dataset.classes}")  # 10个类别

# 可视化样本图像
def show_images(dataset, num_samples=6):
    fig, axes = plt.subplots(1, num_samples, figsize=(12, 3))
    for i in range(num_samples):
        img, label = dataset[i]
        axes[i].imshow(img)
        axes[i].set_title(dataset.classes[label])
        axes[i].axis('off')
    plt.show()

print("训练集样本示例:")
show_images(train_dataset)

# %% 2. 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1,1]
])

# 应用预处理
train_dataset.transform = transform
test_dataset.transform = transform

# 创建DataLoader
BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# %% 3. 创建CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(self.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = x.view(-1, 64 * 8 * 8)  # 展平
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN().to(device)
print(model)

# %% 4. 训练模型
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 15
train_losses, val_accuracies = [], []

print("开始训练...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # 计算验证集准确率
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    train_losses.append(epoch_loss)
    val_accuracies.append(accuracy)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"训练损失: {epoch_loss:.4f} | "
          f"验证准确率: {accuracy:.2f}%")

# 绘制训练曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, 'b-o')
plt.title("训练损失")
plt.xlabel("Epochs")
plt.subplot(1, 2, 2)
plt.plot(val_accuracies, 'r-o')
plt.title("验证准确率")
plt.xlabel("Epochs")
plt.tight_layout()
plt.show()

# %% 5. 评估模型
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    correct, total = 0, 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 收集结果用于可视化
        if len(all_preds) < 100:  # 只保存部分样本
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

print(f"最终测试准确率: {100 * correct / total:.2f}%")

# 可视化预测结果
def visualize_predictions(images, labels, preds, class_names):
    fig, axes = plt.subplots(4, 5, figsize=(15, 10))
    axes = axes.ravel()
    for i in range(20):
        img = images[i].permute(1, 2, 0).numpy()
        img = img * 0.5 + 0.5  # 反归一化
        axes[i].imshow(np.clip(img, 0, 1))
        axes[i].set_title(f"True: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

# 获取测试样本
test_images, test_labels = next(iter(test_loader))
visualize_predictions(
    test_images[:20].cpu(),
    test_labels[:20].cpu().numpy(),
    all_preds[:20],
    train_dataset.classes
)
