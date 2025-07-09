# -*- coding: utf-8 -*-
# %% [markdown]
"""
泰坦尼克号生存预测作业
使用机器学习模型预测乘客生存情况
"""

# %%
# 1. 加载并探索数据
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 加载数据
data = pd.read_csv('data/train.csv')  # 注意路径可能需要调整
df = data.copy()
print("数据集形状:", df.shape)
df.sample(5)

# %%
# 2. 数据预处理
# 删除无关特征
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

# 处理缺失值
print("\n缺失值统计:")
print(df.isnull().sum())
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# 转换分类变量
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
print("\n处理后的数据集信息:")
df.info()

# %%
# 3. 特征工程
# 创建新特征：家庭规模
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
# 创建新特征：是否独自旅行
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
# 年龄分组
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                        labels=['Child', 'Teen', 'Adult', 'MidAge', 'Senior'])

# 将年龄分组转换为虚拟变量
df = pd.get_dummies(df, columns=['AgeGroup'], drop_first=True)

# 显示处理后的数据
print("\n特征工程后的数据集样本:")
df.sample(5)

# %%
# 4. 划分特征和标签
X = df.drop('Survived', axis=1)
y = df['Survived']

# 5. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("\n训练集形状:", X_train.shape)
print("测试集形状:", X_test.shape)

# 6. 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
# 7. 构建和训练模型
models = {
    "SVM": SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# 训练模型并评估
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # 评估指标
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm
    }
    
    print(f"\n{name} 模型性能:")
    print(f"准确率: {accuracy:.4f}")
    print("分类报告:\n", report)

# %%
# 8. 模型比较和可视化
# 比较准确率
accuracies = [results[name]['accuracy'] for name in models]
plt.figure(figsize=(10, 6))
plt.bar(models.keys(), accuracies, color=['skyblue', 'lightgreen', 'salmon'])
plt.title('模型准确率比较')
plt.xlabel('模型')
plt.ylabel('准确率')
plt.ylim(0.7, 0.9)
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
plt.show()

# 绘制混淆矩阵
plt.figure(figsize=(15, 4))
for i, (name, result) in enumerate(results.items()):
    plt.subplot(1, 3, i+1)
    sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['未存活', '存活'], yticklabels=['未存活', '存活'])
    plt.title(f'{name} 混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
plt.tight_layout()
plt.show()

# %%
# 9. 特征重要性分析 (仅随机森林)
rf_model = results['Random Forest']['model']
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importances = feature_importances.sort_values(ascending=False)

plt.figure(figsize=(12, 6))
feature_importances.plot(kind='bar')
plt.title('随机森林特征重要性')
plt.xlabel('特征')
plt.ylabel('重要性得分')
plt.xticks(rotation=45)
plt.show()

# %%
# 10. 最佳模型选择和应用
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']
print(f"\n最佳模型: {best_model_name} (准确率: {results[best_model_name]['accuracy']:.4f})")

# 使用最佳模型进行预测示例
sample_passenger = X_test.iloc[0:1]
sample_scaled = scaler.transform(sample_passenger)
prediction = best_model.predict(sample_scaled)
probabilities = best_model.predict_proba(sample_scaled)

print("\n示例乘客预测:")
print(f"特征值:\n{sample_passenger}")
print(f"预测结果: {'存活' if prediction[0] == 1 else '未存活'}")
print(f"预测概率: 未存活: {probabilities[0][0]:.4f}, 存活: {probabilities[0][1]:.4f}")