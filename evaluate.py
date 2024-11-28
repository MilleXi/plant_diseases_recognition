import tensorflow as tf
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from config.config import Config
from data.data_generator import DataGenerator
from utils.utils import Visualizer, Logger
from models.model import DiseaseClassifier
from models.callbacks import TrainingCallbacks

def setup_visualization():
    """设置可视化环境"""
    # 创建可视化目录
    vis_dir = os.path.join(Config.OUTPUT_DIR, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    return vis_dir

def get_predictions(model, data_generator):
    """
    获取模型预测结果和真实标签
    """
    # 重置数据生成器
    data_generator.reset()
    
    # 获取预测结果
    predictions = model.predict(data_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # 获取真实标签和类别名称
    true_classes = data_generator.classes
    print(true_classes)
    class_names = list(data_generator.class_indices.keys())
    
    return true_classes, predicted_classes, class_names

def evaluate():
    # 创建日志记录器
    logger = Logger(Config.LOG_DIR)
    
    # 设置可视化
    vis_dir = setup_visualization()
    
    logger.log(f"[信息] 可视化输出目录: {vis_dir}")
    
    try:
        # 加载模型
        logger.log("[信息] 加载训练好的模型...")
        model_path = os.path.join(Config.MODEL_DIR, 'final_model.keras')
        model = tf.keras.models.load_model(model_path)
        
        # 获取数据生成器
        logger.log("[信息] 准备数据生成器...")
        data_generator = DataGenerator(Config)
        _, valid_generator = data_generator.create_generators()
        
        # 获取预测结果
        y_true, y_pred, class_names = get_predictions(model, valid_generator)
        
        # 计算分类指标
        logger.log("[信息] 计算分类指标...")
        
        # 准确率
        accuracy = accuracy_score(y_true, y_pred)
        logger.log(f"Overall Accuracy: {accuracy:.4f}")
        
        # 精确度、召回率、F1 score
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
        
        # 输出每个类别的精确度、召回率、F1 score
        for i, class_name in enumerate(class_names):
            logger.log(f"Class {class_name}:")
            logger.log(f"  精确度: {precision[i]:.4f}")
            logger.log(f"  召回率: {recall[i]:.4f}")
            logger.log(f"  F1 Score: {f1[i]:.4f}")
        
        # 加权平均精确度、召回率、F1 score
        weighted_precision = np.average(precision, weights=np.bincount(y_true))
        weighted_recall = np.average(recall, weights=np.bincount(y_true))
        weighted_f1 = np.average(f1, weights=np.bincount(y_true))
        
        logger.log(f"\n加权平均精确度: {weighted_precision:.4f}")
        logger.log(f"加权召回率: {weighted_recall:.4f}")
        logger.log(f"加权 F1 Score: {weighted_f1:.4f}")
        
        # 生成混淆矩阵
        logger.log("[信息] 生成混淆矩阵...")
        y_true, y_pred, class_names = get_predictions(model, valid_generator)
        confusion_matrix_path = os.path.join(vis_dir, 'confusion_matrix.png')
        Visualizer.plot_confusion_matrix(
            y_true,
            y_pred,
            class_names,
            confusion_matrix_path
        )
        
        # 总体加权 F1 score
        overall_f1 = f1_score(y_true, y_pred, average='weighted')
        logger.log(f"总体加权 F1 Score: {overall_f1:.4f}")
         
    except Exception as e:
        logger.log(f"[错误] 评估过程中出现错误: {str(e)}")
        raise

if __name__ == "__main__":
    evaluate()
