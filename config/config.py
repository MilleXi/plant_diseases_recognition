import os
import tensorflow as tf

class Config:

    # 基础路径
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 数据相关配置
    DATA_DIR = 'Plantvillage-Dataset'
    IMG_SIZE = (128, 128)
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2
    
    # 训练相关配置
    EPOCHS = 50
    INITIAL_LEARNING_RATE = 1e-3
    MIN_LEARNING_RATE = 1e-6
    
    # 输出目录配置
    OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
    MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
    LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
    