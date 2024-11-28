# plant_diseases_recognition
Here’s a draft for the README based on the information you provided:

---

# Plant Diseases Recognition

This project aims to develop a model for recognizing plant diseases using deep learning. The model is trained on the PlantVillage dataset, and it classifies different plant diseases based on images of leaves.

## Requirements

- Python 3.x
- TensorFlow 2.17.0
- Other dependencies can be installed via `requirements.txt`.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MilleXi/plant_diseases_recognition.git
   ```

2. Clone the PlantVillage dataset:
   ```bash
   git clone https://github.com/gabrieldgf4/PlantVillage-Dataset.git
   ```
   - **Note:** After downloading the dataset, delete the `x_Removed_from_Healthy_leaves` folder inside the `PlantVillage-Dataset` directory.

3. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

You can modify the configuration settings by editing the `config.py` file located in the `config` folder. This file contains various parameters related to model training and dataset paths.

## Training

To train the model, run the following command in the terminal:
```bash
python train.py
```

The model will be trained on the PlantVillage dataset, and the training output, including logs and model checkpoints, can be found in the `output` folder.

## Output

After training, you can find the following in the `output` folder:
- Model weights
- Training logs
- Any other generated files during training

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---