import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import backend as K 
import json
from datetime import datetime

class Visualizer:
    """可视化工具类：用于绘制训练过程和结果的各种图表"""
    
    @staticmethod
    def plot_training_history(history, save_path=None):
        """
        绘制训练历史曲线
        :param history: 训练历史对象
        :param save_path: 图表保存路径
        """
        # 使用内置的样式
        plt.style.use('default')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        fig.patch.set_facecolor('white')
        
        # 绘制准确率曲线
        ax1.plot(history.history['accuracy'], label='Training Accuracy', 
                color='#2ecc71', marker='o', markersize=4, linewidth=2)
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy',
                color='#e74c3c', marker='o', markersize=4, linewidth=2)
        ax1.set_title('Model Accuracy', pad=15, fontsize=12)
        ax1.set_xlabel('Epochs', fontsize=10)
        ax1.set_ylabel('Accuracy', fontsize=10)
        ax1.legend(loc='lower right', fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_facecolor('#f8f9fa')
        
        # 绘制损失曲线
        ax2.plot(history.history['loss'], label='Training Loss',
                color='#3498db', marker='o', markersize=4, linewidth=2)
        ax2.plot(history.history['val_loss'], label='Validation Loss',
                color='#e67e22', marker='o', markersize=4, linewidth=2)
        ax2.set_title('Model Loss', pad=15, fontsize=12)
        ax2.set_xlabel('Epochs', fontsize=10)
        ax2.set_ylabel('Loss', fontsize=10)
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_facecolor('#f8f9fa')
        # 调整布局
        plt.tight_layout()
        
        if save_path:
            # 确保保存路径存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"[信息] 训练历史图表已保存至: {save_path}")
        
        plt.show()
        plt.close()

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, classes, save_path=None):
        """
        绘制混淆矩阵
        :param y_true: 真实标签
        :param y_pred: 预测标签
        :param classes: 类别名称列表
        :param save_path: 保存路径
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 8))
        
        # 创建热力图
        im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar(im)
        
        # 设置标签
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45, ha='right')
        plt.yticks(tick_marks, classes)
        
        # 添加数值
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[信息] 混淆矩阵已保存至: {save_path}")
        
        plt.show()
        plt.close()

class ModelAnalyzer:
    """模型分析工具类：用于分析和评估模型性能"""
    
    @staticmethod
    def print_model_summary(model):
        """
        打印模型详细信息
        :param model: Keras模型对象
        """
        print("\n" + "="*50)
        print("模型架构摘要:")
        print("="*50)
        model.summary()
        
        # 使用tf.keras.backend的函数计算参数
        total_params = model.count_params()
        trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
        non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
        
        print("\n" + "="*50)
        print("模型参数统计:")
        print("="*50)
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        print(f"非训练参数: {non_trainable_params:,}")
        print(f"模型大小估计: {total_params * 4 / (1024*1024):.2f} MB\n")

class Logger:
    """日志工具类：用于记录训练过程和结果"""
    
    def __init__(self, log_dir):
        """
        初始化日志记录器
        :param log_dir: 日志保存目录
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, 
                                    f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
    def log(self, message):
        """
        记录日志信息
        :param message: 日志信息
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
            
    def log_metrics(self, metrics):
        """
        记录训练指标
        :param metrics: 指标字典
        """
        self.log("\n训练指标:")
        for key, value in metrics.items():
            self.log(f"{key}: {value}")
