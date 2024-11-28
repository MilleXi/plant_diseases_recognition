# plant_diseases_recognition

This project aims to develop a model (CNN) for recognizing plant diseases using deep learning. The model is trained on the PlantVillage dataset, and it classifies different plant diseases based on images of leaves.

## Requirements

- Python 3.12.3
- TensorFlow 2.17.0
- Poetry for dependency management.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MilleXi/plant_diseases_recognition.git
   ```

2. Clone the PlantVillage dataset:
   ```bash
   git clone https://github.com/gabrieldgf4/PlantVillage-Dataset.git
   ```
   - **Note:** After downloading the dataset, delete the `x_Removed_from_Healthy_leaves` folder and the `.git` folder inside the `PlantVillage-Dataset` directory.

3. Install the required Python dependencies:
   ```bash
   poetry install
   ```

## Configuration

You can modify the configuration settings by editing the `config.py` file located in the `config` folder. This file contains various parameters related to model training and dataset paths.

## Training

To train the model, run the following command in the terminal:
```bash
python train.py
```

## Evaluating

To evaluate the model, run the following command in the terminal:
```bash
python evaluate.py
```

The model will be trained and evaluated on the PlantVillage dataset, and the training output, including logs and model checkpoints, can be found in the `output` folder.

## Feature maps

If you want to see the feature maps, run the following command in the terminal:
```bash
python get_feature_maps.py
```

You can find the pictures in the `output` folder.

## Visual Interface

To open the visual interface, run the following command in the terminal:
```bash
python gradio_interface.py
```

## Output

After you run the above code, you can find the following in the `output` folder:
- Checkpoints
- Feature maps
- Logs
- Best model in the 'models' folder
- Other Visualizations

## License

This project is licensed under the Apache-2.0 license - see the [LICENSE](LICENSE) file for details.

---