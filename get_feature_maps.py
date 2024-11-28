import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from config.config import Config
from data.data_generator import DataGenerator
from models.model import DiseaseClassifier
from utils.utils import Logger


def visualize_feature_maps(model, img_path, layer_names=None):
    """
    可视化模型每一层的特征映射图。

    :param model: 训练好的模型
    :param img_path: 图像路径
    :param layer_names: 要可视化的层名称列表（如果为 None，将显示所有卷积层）
    """
    # 加载并预处理图像
    img = image.load_img(img_path, target_size=Config.IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # 添加批次维度
    img_array = img_array / 255.0  # 如果模型是用归一化数据训练的

    # 如果没有指定层名称，则默认选择所有卷积层
    if layer_names is None:
        layer_names = [layer.name for layer in model.layers if 'conv' in layer.name]

    # 创建一个新的模型，输出每层的特征图
    layer_outputs = [model.get_layer(name).output for name in layer_names]
    feature_map_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)

    # 获取特征映射
    feature_maps = feature_map_model.predict(img_array)

    # 可视化每层的特征图
    for layer_name, feature_map in zip(layer_names, feature_maps):
        folder = f"output/feature_maps/feature_map_{layer_name}.png"
        num_feature_maps = feature_map.shape[-1]  # 获取通道数（特征图数量）

        # 创建一个子图，用来显示所有的特征图
        size = int(np.ceil(np.sqrt(num_feature_maps)))  # 每行显示的特征图数量
        fig, axes = plt.subplots(size, size, figsize=(10, 10))

        for i in range(num_feature_maps):
            ax = axes[i // size, i % size]
            ax.imshow(feature_map[0, :, :, i], cmap='viridis')
            ax.axis('off')

        plt.suptitle(f"Feature Maps of Layer: {layer_name}", size=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.imsave(folder, feature_map)


def get_feature_maps():
    # 创建日志记录器
    logger = Logger(Config.LOG_DIR)

    try:
        # 加载模型
        logger.log("[信息] 加载训练好的模型...")

        # 从 checkpoint 加载模型
        model_path = os.path.join(Config.MODEL_DIR, 'final_model.keras')
        model = load_model(model_path)

        # 加载数据生成器
        logger.log("[信息] 加载数据生成器...")
        data_generator = DataGenerator(Config)
        _, valid_generator = data_generator.create_generators()

        # 获取一张图像进行可视化
        img_path = valid_generator.filepaths[0]  # 取验证集的第一张图像

        # 可视化特征图
        logger.log("[信息] 可视化特征映射图...")
        visualize_feature_maps(model, img_path)

    except Exception as e:
        logger.log(f"[错误] 获取特征图过程中出现错误: {str(e)}")
        raise


if __name__ == "__main__":
    get_feature_maps()
