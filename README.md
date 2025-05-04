# 🖖 Rock Paper Scissors AI Game

An interactive Rock Paper Scissors game powered by a Convolutional Neural Network (CNN) trained on webcam images. Play against the computer using real-time hand gesture recognition! First to 3 points wins.

## 🎮 Features

* **Real-time hand gesture recognition** using a trained CNN.
* **Countdown timer** before prediction.
* **Visual scoreboard** and results display.
* **Random computer choice** with representative images.
* **Easy restart functionality**.

## 📁 Folder Structure


project/
├── rps_cnn_model.h5          # Trained CNN model
├── rps_webcam_dataset_resized/  # Folder containing images for computer's move
│   ├── Rock/
│   ├── Paper/
│   └── Scissors/
├── main.py                   # Main game script
└── README.md                 # This file


## 🛠️ Requirements

* Python 3.x
* OpenCV
* TensorFlow / Keras
* NumPy

**Installation**

```bash
pip install opencv-python tensorflow numpy

🧠 Model
The CNN model (rps_cnn_model.h5) is trained on a custom webcam dataset with the following classes:

Rock

Paper

Scissors

Important: Ensure your model's input dimensions match the webcam frame size used during prediction.

🚀 How to Run
Ensure your webcam is connected.

Place the trained model (rps_cnn_model.h5) in the project directory.

Place the computer's move images in the respective folders under rps_webcam_dataset_resized/.

Run the game:

python main.py

🕹️ Game Controls
Spacebar: Start the countdown and make your move.

R: Restart the game after a winner is determined.

Q: Quit the game.

👀 Game Interface
The game screen is divided into two panels:

Left panel: Displays your webcam feed.

Right panel: Displays