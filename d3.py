# %%
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import tkinter as tk
from tkinter import filedialog

# %%
# 加载数据 - 交互式文件选择
def load_data():
    # 尝试自动查找文件
    possible_paths = [
        'data/household_power_consumption.txt',
        './data/household_power_consumption.txt',
        'household_power_consumption.txt',
        '../data/household_power_consumption.txt',
        'household_power_consumption/household_power_consumption.txt',
        'electricity/household_power_consumption.txt',
        'dataset/household_power_consumption.txt'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"找到数据文件: {os.path.abspath(path)}")
            return pd.read_csv(path, sep=';', low_memory=False)
    
    # 如果自动查找失败，提示用户选择文件
    print("无法自动找到数据文件，请手动选择...")
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    file_path = filedialog.askopenfilename(
        title="选择家庭用电量数据文件",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )
    
    if not file_path:
        raise FileNotFoundError("未选择任何文件，程序终止")
    
    print(f"您选择的文件: {file_path}")
    return pd.read_csv(file_path, sep=';', low_memory=False)

# 加载数据
df = load_data()
print("\n原始数据预览:")
print(df.head())

# %%
# 检查数据
print("\n数据信息:")
print(f"数据形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")
print("\n数据类型:")
print(df.dtypes)

# %%
# 处理缺失值：将'?'替换为NaN并删除
print("\n处理缺失值...")
df.replace('?', np.nan, inplace=True)

# 识别数值列和非数值列
print("\n识别数值列和非数值列...")
# 显式列出数值列（根据数据集描述）
numeric_cols = [
    'Global_active_power', 'Global_reactive_power', 'Voltage', 
    'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
]

# 只转换数值列为浮点数
print("转换数值列为浮点数...")
for col in numeric_cols:
    # 检查列是否存在于DataFrame中
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    else:
        print(f"警告: 列 '{col}' 在数据集中不存在")

# 处理日期和时间列
print("\n处理日期和时间列...")
# 创建日期时间列 - 更健壮的处理方式
try:
    df['datetime'] = pd.to_datetime(
        df['Date'] + " " + df['Time'], 
        format='%d/%m/%Y %H:%M:%S',  # 指定日期格式
        errors='coerce'  # 转换错误时设为NaT
    )
except Exception as e:
    print(f"日期转换错误: {e}")
    # 备选方法
    df['datetime'] = df['Date'] + " " + df['Time']
    df['datetime'] = pd.to_numeric(df['datetime'], errors='coerce')

# 检查日期转换结果
date_na_count = df['datetime'].isna().sum()
if date_na_count > 0:
    print(f"警告: {date_na_count} 行日期转换失败")

# 删除原始日期和时间列
df.drop(['Date', 'Time'], axis=1, inplace=True)

# 删除包含缺失值的行
initial_count = len(df)
df.dropna(inplace=True)
final_count = len(df)
print(f"删除了 {initial_count - final_count} 行缺失数据")
print(f"处理后数据形状: {df.shape}")

# 打印前几行以验证
print("\n处理后的数据预览:")
print(df.head())
print("\n数据类型:")
print(df.dtypes)

# %%
# 打印日期范围
print("\n数据概览:")
print("开始日期: ", df['datetime'].min())
print("结束日期: ", df['datetime'].max())
print("数据点数量: ", len(df))

# %%
# 划分训练集和测试集（按时间划分）
train = df.loc[df['datetime'] <= '2009-12-31']
test = df.loc[df['datetime'] > '2009-12-31']

print(f"\n训练集大小: {len(train)} 条记录 ({(len(train)/len(df)*100):.1f}%)")
print(f"测试集大小: {len(test)} 条记录 ({(len(test)/len(df)*100):.1f}%)")

# 检查划分后的数据
print("\n训练集预览:")
print(train.head())
print("\n测试集预览:")
print(test.head())

# %%
# 数据归一化（仅使用训练集拟合）
scaler = MinMaxScaler()
# 获取除日期时间外的所有列名
cols = train.columns.drop('datetime')
print(f"\n归一化列: {cols.tolist()}")

# 对训练集和测试集进行归一化
train[cols] = scaler.fit_transform(train[cols])
test[cols] = scaler.transform(test[cols])

print("\n归一化后的训练集预览:")
print(train.head())

# %%
# 划分特征X和目标y
def create_sequences(data, seq_length, target_col_idx=0):
    """创建时间序列数据"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        # 取seq_length个时间步作为特征
        X.append(data[i:i+seq_length])
        # 取下一个时间步的目标值
        y.append(data[i+seq_length, target_col_idx])
    return np.array(X), np.array(y)

# 设置序列长度（60个时间步=1小时）
seq_length = 60
# 预测目标列
target_col = 'Global_active_power'

# 准备训练数据
train_data = train[cols].values
# 创建训练序列
X_train, y_train = create_sequences(train_data, seq_length, 
                                   target_col_idx=cols.get_loc(target_col))

# 准备测试数据
test_data = test[cols].values
# 创建测试序列
X_test, y_test = create_sequences(test_data, seq_length,
                                 target_col_idx=cols.get_loc(target_col))

print(f"\n序列数据准备完成:")
print(f"训练序列形状: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"测试序列形状: X_test={X_test.shape}, y_test={y_test.shape}")

# %%
# 创建数据加载器
class TimeSeriesDataset(Dataset):
    """时间序列数据集类"""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 创建训练和测试数据集
train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

# 设置批大小
batch_size = 64
# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"\n数据加载器创建完成:")
print(f"训练批次数量: {len(train_loader)} (每批 {batch_size} 个样本)")
print(f"测试批次数量: {len(test_loader)}")

# %%
# 构建LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True  # 输入格式为(batch, seq, feature)
        )
        # 全连接输出层
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # LSTM前向传播
        out, _ = self.lstm(x)  # 输出形状: (batch, seq_len, hidden_size)
        # 只取最后一个时间步的输出
        out = out[:, -1, :]
        # 通过全连接层
        out = self.fc(out)
        return out

# 模型参数设置
input_size = len(cols)      # 特征数量
hidden_size = 50            # LSTM隐藏单元数
num_layers = 2              # LSTM层数
output_size = 1             # 预测1个值（有功功率）

# 检测设备（GPU或CPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n使用设备: {device}")
# 创建模型并移动到设备
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)

# 打印模型结构
print("\n模型结构:")
print(model)
print(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# %%
# 训练模型
criterion = nn.MSELoss()  # 均方误差损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam优化器
num_epochs = 10  # 训练轮数

print("\n开始训练模型...")
train_losses = []  # 记录每轮训练损失

for epoch in range(num_epochs):
    model.train()  # 训练模式
    total_loss = 0
    batch_count = 0
    
    for batch_X, batch_y in train_loader:
        # 将数据移到设备
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        # 前向传播
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
        
        # 每10个批次打印一次进度
        if batch_count % 10 == 0:
            print(f"轮次 {epoch+1}/{num_epochs}, 批次 {batch_count}/{len(train_loader)}, 当前批次损失: {loss.item():.6f}")
    
    # 计算平均损失
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    # 打印每轮的平均损失
    print(f'轮次 [{epoch+1}/{num_epochs}], 平均损失: {avg_loss:.6f}')

print("训练完成!")

# 绘制训练损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, 'o-', label='训练损失')
plt.title('训练过程损失变化')
plt.xlabel('轮次')
plt.ylabel('损失')
plt.legend()
plt.grid(True)
plt.savefig('training_loss.png', dpi=300)
plt.show()

# %%
# 在测试集上评估模型
model.eval()  # 评估模式
test_preds = []  # 存储预测值
test_targets = []  # 存储真实值

print("\n在测试集上评估模型...")
with torch.no_grad():  # 不计算梯度
    for i, (batch_X, batch_y) in enumerate(test_loader):
        batch_X = batch_X.to(device)
        # 模型预测
        outputs = model(batch_X).cpu().numpy().squeeze()
        test_preds.extend(outputs)
        test_targets.extend(batch_y.numpy())
        
        # 每10个批次打印一次进度
        if i % 10 == 0:
            print(f"处理测试批次 {i+1}/{len(test_loader)}")

# 计算均方根误差(RMSE)
rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
print(f'归一化尺度测试集RMSE: {rmse:.4f}')

# %%
# 绘制预测结果与真实值的对比图
plt.figure(figsize=(15, 6))
# 绘制前500个样本的真实值
plt.plot(test_targets[:500], label='真实值', alpha=0.8, linewidth=1.5)
# 绘制前500个样本的预测值
plt.plot(test_preds[:500], label='预测值', alpha=0.7, linestyle='--')
plt.title('家庭用电量预测 (LSTM模型)')
plt.xlabel('时间步 (每步=1分钟)')
plt.ylabel('归一化有功功率')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('normalized_prediction.png', dpi=300)
plt.show()

# %%
# 反归一化结果以查看原始尺度
print("\n反归一化结果...")
# 创建用于反归一化的假数据
dummy_data = np.zeros((len(test_preds), len(cols)))
dummy_data[:, cols.get_loc(target_col)] = test_preds
inverse_preds = scaler.inverse_transform(dummy_data)[:, cols.get_loc(target_col)]

dummy_data[:, cols.get_loc(target_col)] = test_targets
inverse_targets = scaler.inverse_transform(dummy_data)[:, cols.get_loc(target_col)]

# 计算原始尺度的RMSE
original_rmse = np.sqrt(mean_squared_error(inverse_targets, inverse_preds))
print(f'原始尺度测试集RMSE: {original_rmse:.4f} 千瓦')

# 绘制原始尺度的预测结果
plt.figure(figsize=(15, 6))
plt.plot(inverse_targets[:500], label='真实值', alpha=0.8, linewidth=1.5)
plt.plot(inverse_preds[:500], label='预测值', alpha=0.7, linestyle='--')
plt.title('家庭用电量预测 (原始尺度)')
plt.xlabel('时间步 (每步=1分钟)')
plt.ylabel('有功功率 (千瓦)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('original_scale_prediction.png', dpi=300)
plt.show()

# %%
# 保存模型和结果
torch.save(model.state_dict(), 'power_consumption_lstm.pth')
print("模型已保存为 'power_consumption_lstm.pth'")

# 保存预测结果
results = pd.DataFrame({
    'timestamp': test['datetime'].iloc[seq_length:seq_length+len(test_preds)].values,
    'actual': inverse_targets,
    'predicted': inverse_preds
})
results.to_csv('power_consumption_predictions.csv', index=False)
print("预测结果已保存为 'power_consumption_predictions.csv'")

# %%
# 额外分析：计算并打印关键统计指标
print("\n预测性能分析:")
mae = np.mean(np.abs(inverse_targets - inverse_preds))
print(f"平均绝对误差 (MAE): {mae:.4f} 千瓦")

mape = np.mean(np.abs((inverse_targets - inverse_preds) / inverse_targets)) * 100
print(f"平均绝对百分比误差 (MAPE): {mape:.2f}%")

r2 = 1 - (np.sum((inverse_targets - inverse_preds)**2) / 
          np.sum((inverse_targets - np.mean(inverse_targets))**2))
print(f"决定系数 (R²): {r2:.4f}")

# 绘制误差分布
errors = inverse_targets - inverse_preds
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=50, alpha=0.7)
plt.title('预测误差分布')
plt.xlabel('误差 (千瓦)')
plt.ylabel('频率')
plt.grid(True)
plt.savefig('error_distribution.png', dpi=300)
plt.show()