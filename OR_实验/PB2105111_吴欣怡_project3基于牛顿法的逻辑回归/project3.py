import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc("font",family='MicroSoft YaHei',weight="bold")

class LogisticRegressionNewton():
    def __init__(self, X, y, alpha, beta, regularization=1.0, max_iter=1000, tol=1e-6):
        # 初始化函数，包括添加偏置项，设定参数等
        self.X = np.insert(X, 0, values=1.0, axis=1)
        self.y = y.reshape(-1, 1)
        self.m, self.n = self.X.shape
        self.regularization = regularization
        self.max_iter = max_iter
        self.tol = tol
        self.theta = np.zeros((self.n, 1))
        self.loss = self.calculate_loss(self.theta) 
        self.alpha = alpha
        self.beta = beta
        self.loss_history = []
        self.gradient_norm_history = [] 

    # sigmoid激活函数
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))
    
    # 计算Hessian矩阵
    def hessian(self):
        diag_elements = self.sigmoid(self.X.dot(self.theta)) * self.sigmoid(-self.X.dot(self.theta))
        W = np.diag(diag_elements.flatten())
        hessian = (self.X.T @ W @ self.X) / self.m + 2 * self.regularization * np.identity(self.n)
        return hessian

    # 计算梯度
    def gradient(self):
        return self.X.T @ (self.sigmoid(self.X.dot(self.theta)) - self.y) / self.m + 2 * self.regularization * self.theta

    # 回溯线性搜索
    def backtracking_line_search(self, update, alpha, beta):
        t = 1.0
        while True:
            new_theta = self.theta + t * update
            new_loss = self.calculate_loss(new_theta)
            if new_loss < self.loss + alpha * t * self.gradient().T @ update:
                break
            t *= beta
        return t

    # 牛顿法迭代步骤
    def newton_step(self, alpha, beta):
        update = -np.linalg.inv(self.hessian()) @ self.gradient()
        step_size = self.backtracking_line_search(update, alpha, beta)
        self.theta += step_size * update
        self.loss = self.calculate_loss(self.theta)

        # 记录收敛信息
        self.loss_history.append(self.loss)
        gradient_norm = np.linalg.norm(self.gradient())
        self.gradient_norm_history.append(gradient_norm)

        return step_size

    # 计算损失函数
    def calculate_loss(self, theta):
        H = self.sigmoid(self.X.dot(theta))
        loss = -np.sum(self.y * np.log(H) + (1 - self.y) * np.log(1 - H)) / self.m
        return loss + self.regularization * np.sum(theta[1:]**2)  # 添加正则化项

    # 训练模型
    def train(self):
        for iteration in range(self.max_iter):
            old_theta = np.copy(self.theta)
            step_size = self.newton_step(self.alpha, self.beta)
            if np.linalg.norm(self.theta - old_theta) < self.tol:
                print(f"在 {iteration+1} 次迭代中收敛.")
                break

    # 预测函数
    def predict(self, X):
        X_with_bias = np.insert(X, 0, values=1.0, axis=1)
        probabilities = self.sigmoid(X_with_bias.dot(self.theta))
        predictions = np.sign(probabilities - 0.3)  # 设置阈值为0.5
        return predictions
    
    # 绘制收敛信息图
    def plot_convergence(self, iterations):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.semilogy(iterations, self.loss_history, label='损失')
        plt.axhline(y=self.calculate_loss(self.theta), color='r', linestyle='--', label='最优值')
        plt.xlabel('迭代次数')
        plt.ylabel('对数损失')
        plt.legend()
        plt.title('损失函数收敛情况 (对数坐标)')

        plt.subplot(1, 2, 2)
        plt.semilogy(iterations, self.gradient_norm_history, label='梯度范数')
        plt.xlabel('迭代次数')
        plt.ylabel('对数梯度范数')
        plt.legend()
        plt.title('梯度范数收敛情况 (对数坐标)')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # 读取数据
    features_data = pd.read_csv('a9a_features.csv', header=None)
    labels_data = pd.read_csv('a9a_labels.csv', header=None)
    X = features_data.T.values
    y = labels_data.values

    # 数据处理，随机抽样保持类别平衡
    class_0_indices = np.where(y == 1)[0]
    class_1_indices = np.where(y == -1)[0]
    min_samples = min(len(class_0_indices), len(class_1_indices))
    balanced_class_0_indices = np.random.choice(class_0_indices, min_samples, replace=False)
    balanced_class_1_indices = np.random.choice(class_1_indices, min_samples, replace=False)
    balanced_indices = np.concatenate([balanced_class_0_indices, balanced_class_1_indices])
    np.random.shuffle(balanced_indices)
    X_balanced=X[balanced_indices]
    y_balanced=y[balanced_indices]

    # 创建模型实例，可以修改为其它参数
    model = LogisticRegressionNewton(X_balanced, y_balanced, 0.1, 0.5)
    
    # 训练模型
    model.train()
    print(f'线搜索中的下降速率:{0.1}')
    print(f'线搜索的衰减率:{0.5}')

    # 绘制收敛信息图
    iterations = np.arange(1, len(model.loss_history) + 1)
    model.plot_convergence(iterations)