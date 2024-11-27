import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array
import os
import tensorflow as tf

class DataGenerator:
    def __init__(self, config):
        self.config = config
        self.train_generator = None
        self.valid_generator = None
        
    def create_generators(self):
        """创建内存优化的数据生成器"""
        # 训练数据增强
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=self.config.VALIDATION_SPLIT
        )

        # 验证数据只需要缩放
        valid_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=self.config.VALIDATION_SPLIT
        )

        # 创建训练数据生成器
        self.train_generator = train_datagen.flow_from_directory(
            self.config.DATA_DIR,
            target_size=self.config.IMG_SIZE,
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )

        # 创建验证数据生成器
        self.valid_generator = valid_datagen.flow_from_directory(
            self.config.DATA_DIR,
            target_size=self.config.IMG_SIZE,
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )

        return self.train_generator, self.valid_generator

    def _create_tf_dataset(self, generator):
        """创建tf.data.Dataset以优化内存使用"""
        return tf.data.Dataset.from_generator(
            lambda: generator,
            output_signature=(
                tf.TensorSpec(shape=(None, *self.config.IMG_SIZE, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, generator.num_classes), dtype=tf.float32)
            )
        ).prefetch(tf.data.AUTOTUNE)