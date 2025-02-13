# 🚀 Satellite Component Segmentation

## 📌 Project Overview
This project focuses on **segmenting spacecraft components** using deep learning. We employ a **U-Net architecture with a ResNet-34 backbone** for semantic segmentation of spacecraft images. The dataset consists of labeled images with four segmentation classes:
- **Background**
- **Solar Panel**
- **Body**
- **Antenna**

We use **PyTorch Lightning** for training and **Albumentations** for data augmentation.

## 🎮 Play the Game
You can interact with the **Satellite Masking Game** live on **Hugging Face**:
👉 [Play Now](https://huggingface.co/spaces/ShubhamJagtap/Satellite_Segmentation_API)

## 📂 Dataset
The dataset was used from **Kaggle**: [Spacecraft Component Segmentation](https://www.kaggle.com/code/dkudryavtsev/spacecraft-component-segmentation).

This project was developed on Kaggle: [Satellite Image Segmentation](https://www.kaggle.com/code/jagtapshubham17/satellite-image-segmentation/).

The dataset is structured as follows:
/kaggle/input/spacecraft-dataset ├── images/ │ ├── train/ │ └── val/ ├── masks/ │ ├── train/ │ └── val/

Each image in the dataset has a corresponding mask where different spacecraft components are color-coded.

## 🔧 Setup & Installation
Run the following command to install required dependencies:
```sh
pip install -r requirements.txt
Ensure you have a requirements.txt file listing all necessary dependencies for running this project.

📊 How It Works
Draw on the image using the interactive canvas.
Select a color corresponding to different satellite parts.
Submit your mask, and the AI will compare it to the ground truth.
Get feedback on your accuracy using IoU scores.
Achieve 80%+ accuracy to win the game! 🚀
💾 Saving the Model
Trained models are saved as follows:
torch.save(model.state_dict(), "satellite_segmentation.pth")
📜 Acknowledgments
Dataset from Kaggle: Spacecraft Component Segmentation
Project developed on Kaggle: Satellite Image Segmentation
Albumentations for image augmentations
PyTorch Lightning for structured training
🌟 Future Improvements
Experimenting with DeepLabV3+ and U-Net++ for better segmentation.
Implement self-supervised learning to improve generalization.
Implement real-time inference using Streamlit.
