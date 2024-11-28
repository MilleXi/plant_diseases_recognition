import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
import io
import os

class Config:
    MODEL_DIR = 'models'
    IMG_SIZE = (128, 128)
    CLASS_NAMES = [
        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 
        'Apple___healthy', 'Background_without_leaves', 'Blueberry___healthy',
        'Cherry___Powdery_mildew', 'Cherry___healthy', 
        'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust',
        'Corn___Northern_Leaf_Blight', 'Corn___healthy', 'Grape___Black_rot',
        'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
        'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)',
        'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot',
        'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
        'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
        'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
        'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
        'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy'
    ]

_model = None

def get_model():
    global _model
    if _model is None:
        _model = load_model('output/models/final_model.keras')
    return _model

def preprocess_image(img):
    img_resized = img.resize(Config.IMG_SIZE)
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return Image.open(buf)

def visualize_feature_maps(model, img_array):
    conv_layers = [layer.name for layer in model.layers if 'conv2d' in layer.name][:4]
    outputs = [model.get_layer(name).output for name in conv_layers]
    vis_model = tf.keras.models.Model(model.input, outputs)
    
    feature_maps = vis_model.predict(img_array)
    images = []
    
    for idx, fmap in enumerate(feature_maps):
        fig = plt.figure(figsize=(8, 8))
        plt.title(f"Conv Layer {idx+1}")
        
        n_features = min(16, fmap.shape[-1])
        grid_size = int(np.ceil(np.sqrt(n_features)))
        
        for i in range(n_features):
            plt.subplot(grid_size, grid_size, i + 1)
            plt.imshow(fmap[0, :, :, i], cmap='viridis')
            plt.axis('off')
        
        img = fig_to_image(fig)
        images.append(img)
        plt.close(fig)
    
    return images

def predict_and_visualize(img):
    if img is None:
        return None, "请上传图片", None
    
    try:
        model = get_model()
        img_array = preprocess_image(img)
        
        pred = model.predict(img_array)
        pred_class = np.argmax(pred[0])
        confidence = pred[0][pred_class]
        
        img_with_text = img.copy()
        draw = ImageDraw.Draw(img_with_text)
        font = ImageFont.load_default()
        
        class_name = Config.CLASS_NAMES[pred_class]
        prediction_text = f"{class_name}\n{confidence:.2%}"
        draw.text((10, 10), prediction_text, fill='red', font=font)
        
        feature_maps = visualize_feature_maps(model, img_array)
        
        result_text = f"预测类别: {class_name}\n置信度: {confidence:.2%}"
        
        return img_with_text, result_text, feature_maps
        
    except Exception as e:
        return img, f"处理出错: {str(e)}", None

def create_interface():
    with gr.Blocks() as interface:
        gr.Markdown("# 植物病害识别系统")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="上传图片")
                prediction_text = gr.Textbox(label="预测结果")
            
            output_image = gr.Image(type="pil", label="处理后图片")
        
        feature_maps = gr.Gallery(label="特征图可视化")
        
        input_image.change(
            fn=predict_and_visualize,
            inputs=input_image,
            outputs=[output_image, prediction_text, feature_maps]
        )
        
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch()