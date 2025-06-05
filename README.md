# 🗑️ TRASHPRED: AI-Powered Waste Classification Model

## 🌟 Project Summary

TRASHPRED is a deep learning-based image classification model designed to automate the identification of waste types for smarter recycling and environmental management. The model classifies waste images into categories like **plastic**, **metal**, **glass**, **paper**, and **organic** using a custom-trained **Convolutional Neural Network (CNN)** architecture built with **TensorFlow/Keras**.

## 🔗 Model on Hugging Face

The trained model is deployed and publicly available on Hugging Face's Model Hub:

➡️ [Explore the Model on Hugging Face](https://huggingface.co/sriram7737/TRASHPRED)

You can use the hosted model for inference, evaluation, or as a base for fine-tuning with your own waste classification datasets.

## 🧠 Model Architecture

- **Framework**: TensorFlow with Keras API
- **Input**: RGB images resized to 224x224 pixels
- **Architecture**: Multi-layer CNN with ReLU activations, MaxPooling, Dropout for regularization
- **Output**: Softmax-activated prediction over 5 waste classes
- **Training**: 25 epochs, batch size of 32, Adam optimizer
- **Dataset**: Balanced, manually labeled dataset of common household and industrial waste items

## 🖥️ Example Inference (Python)

```python
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = tf.keras.models.load_model('path_to_trashpred_model.h5')

# Load and preprocess image
img = image.load_img('test_image.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predict class
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)

print(f"Predicted Waste Category: {predicted_class}")
```

## 📁 Repository Contents

```
TRASHPRED/
├── trashpred_model.h5       # Trained model file (optional upload)
├── example_images/          # Sample test images (optional)
├── scripts/
│   ├── train.py             # Script to train the model
│   ├── evaluate.py          # Evaluate model accuracy
├── README.md                # Project documentation
```

## 🚀 Use Cases

- Smart recycling kiosks and IoT trash sorters
- Environmental education tools
- Waste detection in industrial automation systems
- Robotics and sustainability research projects

## ⚖️ License

This project is licensed under the **MIT License** — free for personal, educational, or commercial use with attribution.

## 🙋‍♂️ About the Author

- **Name**: Sriram Rampelli  
- **GitHub**: [@sriram7737](https://github.com/sriram7737)  
- **Hugging Face**: [sriram7737](https://huggingface.co/sriram7737)  
- **Portfolio**: [sriram7737.github.io](https://sriram7737.github.io)

---

Feel free to fork this repository, test the model, or integrate it into your environmental AI solutions!
