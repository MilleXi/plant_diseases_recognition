import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, TensorBoard, 
    ReduceLROnPlateau, CSVLogger, LambdaCallback
)
import os
import datetime
import json
import numpy as np
from tensorflow.keras import backend as K

class TrainingProgressCallback(tf.keras.callbacks.Callback):
    """自定义训练进度回调"""
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir
    
    def on_epoch_end(self, epoch, logs=None):
        """记录每个epoch的训练进度"""
        logs = logs or {}
        progress_file = os.path.join(self.log_dir, 'training_progress.json')
        
        # 读取现有进度
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                progress = json.load(f)
        else:
            progress = []
            
        # 获取当前学习率
        if hasattr(self.model.optimizer, 'learning_rate'):
            lr = float(K.eval(self.model.optimizer.learning_rate))
        else:
            lr = float(K.eval(self.model.optimizer.lr))
            
        # 添加新的epoch数据
        progress.append({
            'epoch': epoch + 1,
            'train_accuracy': float(logs.get('accuracy', 0)),
            'train_loss': float(logs.get('loss', 0)),
            'val_accuracy': float(logs.get('val_accuracy', 0)),
            'val_loss': float(logs.get('val_loss', 0)),
            'learning_rate': lr
        })
        
        # 保存进度
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=4)

class PerformanceMonitorCallback(tf.keras.callbacks.Callback):
    """性能监控回调"""
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir
        self.batch_times = []
        self.epoch_times = []
        self.start_time = None
        
    def on_train_begin(self, logs=None):
        self.start_time = datetime.datetime.now()
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = datetime.datetime.now()
        
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = (datetime.datetime.now() - self.epoch_start_time).total_seconds()
        self.epoch_times.append(epoch_time)
        
        # 记录性能指标
        metrics = {
            'epoch': epoch + 1,
            'epoch_time': epoch_time,
            'average_epoch_time': np.mean(self.epoch_times),
            'total_time': (datetime.datetime.now() - self.start_time).total_seconds(),
        }
        
        # 保存性能指标
        performance_file = os.path.join(self.log_dir, 'performance_metrics.json')
        with open(performance_file, 'w') as f:
            json.dump(metrics, f, indent=4)

class TrainingCallbacks:
    def __init__(self, config):
        self.config = config
        
    def get_callbacks(self):
        """设置训练回调函数"""
        # 创建日志目录
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(self.config.LOG_DIR, current_time)
        os.makedirs(log_dir, exist_ok=True)
        
        callbacks = []
        
        # TensorBoard回调
        tensorboard = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch',
            profile_batch=2
        )
        callbacks.append(tensorboard)
        
        # CSV日志记录
        csv_logger = CSVLogger(
            os.path.join(log_dir, 'training_log.csv'),
            separator=',',
            append=False
        )
        callbacks.append(csv_logger)
        
        # 模型检查点
        checkpoint = ModelCheckpoint(
            filepath=os.path.join(self.config.CHECKPOINT_DIR, 'model-{epoch:02d}-{val_accuracy:.4f}.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        callbacks.append(checkpoint)

        # 早停
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)

        # 学习率调整
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=self.config.MIN_LEARNING_RATE,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # 训练进度回调
        training_progress = TrainingProgressCallback(log_dir)
        callbacks.append(training_progress)
        
        # 性能监控回调
        performance_monitor = PerformanceMonitorCallback(log_dir)
        callbacks.append(performance_monitor)
        
        return callbacks