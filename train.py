import tensorflow as tf
import os
import datetime
import gc
import numpy as np
from tensorflow.keras import backend as K
from config.config import Config
from data.data_generator import DataGenerator
from models.model import DiseaseClassifier
from models.callbacks import TrainingCallbacks
from utils.utils import Visualizer, ModelAnalyzer, Logger

def setup_visualization():
    """设置可视化环境"""
    # 创建可视化目录
    vis_dir = os.path.join(Config.OUTPUT_DIR, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 配置TensorBoard
    log_dir = os.path.join(Config.LOG_DIR, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    return log_dir, vis_dir

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
    class_names = list(data_generator.class_indices.keys())
    
    return true_classes, predicted_classes, class_names

def train():
    # 创建日志记录器
    logger = Logger(Config.LOG_DIR)
    
    # 设置可视化
    log_dir, vis_dir = setup_visualization()
    logger.log(f"[信息] TensorBoard日志目录: {log_dir}")
    logger.log(f"[信息] 可视化输出目录: {vis_dir}")
    
    # 获取分布式策略
    strategy = tf.distribute.get_strategy()
    logger.log(f"[信息] 使用 {strategy.num_replicas_in_sync} 个计算设备进行训练")
    
    try:
        with strategy.scope():
            # 准备数据生成器
            logger.log("[信息] 准备数据生成器...")
            data_generator = DataGenerator(Config)
            train_generator, valid_generator = data_generator.create_generators()
            
            # 构建模型
            logger.log("[信息] 构建模型...")
            classifier = DiseaseClassifier(Config)
            model = classifier.build_model(train_generator.num_classes)
            
            # 设置回调函数，包括可视化回调
            callbacks = TrainingCallbacks(Config).get_callbacks()
            
            # 打印模型信息
            ModelAnalyzer.print_model_summary(model)
            
            # 将模型结构保存为图像
            tf.keras.utils.plot_model(
                model,
                to_file=os.path.join(vis_dir, 'model_architecture.png'),
                show_shapes=True,
                show_layer_names=True,
                rankdir='TB'
            )
        
        # 训练模型
        logger.log("[信息] 开始训练模型...")
        history = model.fit(
            train_generator,
            validation_data=valid_generator,
            epochs=Config.EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
        
        # 保存最终模型
        model_dir = os.path.join(Config.MODEL_DIR)
        os.makedirs(model_dir, exist_ok=True)
        final_model_path = os.path.join(Config.MODEL_DIR, 'final_model.keras')
        model.save(final_model_path)
        logger.log(f"[信息] 最终模型已保存至: {final_model_path}")
        
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
        logger.log(f"[信息] 混淆矩阵已保存至: {confusion_matrix_path}")
        
        # 保存训练历史
        history_path = os.path.join(vis_dir, 'training_history.png')
        Visualizer.plot_training_history(history, history_path)
        
        # 打印训练总结
        logger.log("\n训练总结:")
        logger.log(f"最佳验证准确率: {max(history.history['val_accuracy']):.4f}")
        logger.log(f"最佳训练准确率: {max(history.history['accuracy']):.4f}")
        logger.log(f"最终验证损失: {min(history.history['val_loss']):.4f}")
        
        # 计算每个类别的准确率
        logger.log("\n各类别准确率:")
        for i, class_name in enumerate(class_names):
            mask = (y_true == i)
            class_accuracy = np.mean(y_pred[mask] == y_true[mask])
            logger.log(f"{class_name}: {class_accuracy:.4f}")
        
        # 提示TensorBoard使用方法
        logger.log("\n要查看详细的训练可视化，请在命令行运行:")
        logger.log(f"tensorboard --logdir={log_dir}")
        
    except Exception as e:
        logger.log(f"[错误] 训练过程中出现错误: {str(e)}")
        raise
    finally:
        # 清理资源
        K.clear_session()
        gc.collect()

if __name__ == "__main__":
    train()