from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, MaxPooling2D,
    Dropout, GlobalAveragePooling2D, Dense, Add, Concatenate,
    DepthwiseConv2D, SeparableConv2D, LayerNormalization,
    MultiHeadAttention
)
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

class DiseaseClassifier:
    def __init__(self, config):
        self.config = config
        self.model = None
        
    def _create_residual_block(self, x, filters, kernel_size=3):
        """创建残差块"""
        shortcut = x
        
        # 第一个卷积层
        x = Conv2D(filters, kernel_size, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        # 第二个卷积层
        x = Conv2D(filters, kernel_size, padding='same')(x)
        x = BatchNormalization()(x)
        
        # 如果输入和输出维度不同，需要调整shortcut
        if shortcut.shape[-1] != filters:
            shortcut = Conv2D(filters, 1, padding='same')(shortcut)
            shortcut = BatchNormalization()(shortcut)
        
        # 添加残差连接
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x
    
    def _create_attention_block(self, x, filters):
        """创建注意力块"""
        # 空间注意力
        mha = MultiHeadAttention(
            num_heads=4,
            key_dim=filters // 4
        )(x, x)
        x = LayerNormalization()(Add()([x, mha]))
        
        return x
    
    def _create_inception_block(self, x, filters):
        """创建Inception模块"""
        # 1x1 卷积
        conv1 = Conv2D(filters//4, 1, padding='same', activation='relu')(x)
        
        # 1x1 -> 3x3 卷积
        conv3 = Conv2D(filters//4, 1, padding='same', activation='relu')(x)
        conv3 = Conv2D(filters//4, 3, padding='same', activation='relu')(conv3)
        
        # 1x1 -> 5x5 卷积
        conv5 = Conv2D(filters//4, 1, padding='same', activation='relu')(x)
        conv5 = Conv2D(filters//4, 5, padding='same', activation='relu')(conv5)
        
        # 3x3池化 -> 1x1卷积
        pool = MaxPooling2D(3, strides=1, padding='same')(x)
        pool = Conv2D(filters//4, 1, padding='same', activation='relu')(pool)
        
        # 合并所有分支
        return Concatenate()([conv1, conv3, conv5, pool])
    
    def build_model(self, n_classes):
        """构建增强版CNN模型"""
        inputs = Input(shape=(*self.config.IMG_SIZE, 3))
        
        # 初始卷积块
        x = Conv2D(32, 3, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D()(x)
        
        # 残差块 1
        x = self._create_residual_block(x, 64)
        x = MaxPooling2D()(x)
        x = Dropout(0.25)(x)
        
        # Inception块
        x = self._create_inception_block(x, 128)
        x = MaxPooling2D()(x)
        x = Dropout(0.25)(x)
        
        # 残差块 2
        x = self._create_residual_block(x, 256)
        x = MaxPooling2D()(x)
        x = Dropout(0.25)(x)
        
        # 注意力块
        x = self._create_attention_block(x, 256)
        
        # 全局平均池化
        x = GlobalAveragePooling2D()(x)
        
        # 全连接层
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        # 输出层
        outputs = Dense(n_classes, activation='softmax')(x)
        
        # 创建模型
        model = Model(inputs, outputs)
        
        # 编译模型
        optimizer = Adam(learning_rate=self.config.INITIAL_LEARNING_RATE)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def load_model(self, model_path):
        """加载已训练的模型"""
        self.model = tf.keras.models.load_model(model_path)
        return self.model
    
    def get_intermediate_model(self, layer_names):
        """获取一个新的模型，返回中间层的输出"""
        # 获取中间层的输出
        layer_outputs = [self.model.get_layer(name).output for name in layer_names]
        intermediate_model = Model(inputs=self.model.input, outputs=layer_outputs)
        return intermediate_model