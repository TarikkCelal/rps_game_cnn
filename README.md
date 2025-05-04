🖖 Rock Paper Scissors AI Game
This is an interactive computer vision-based Rock-Paper-Scissors game powered by a Convolutional Neural Network (CNN) trained on webcam images. The game uses your webcam to detect your hand gesture (rock, paper, or scissors) and plays against the computer. First to 3 points wins!

🎮 Features
Real-time hand gesture recognition using a trained CNN.

Countdown timer before prediction.

Visual scoreboard and results display.

Random computer choice with representative images.

Easy restart functionality.

📁 Folder Structure
bash
Copy
Edit
project/
│
├── rps_cnn_model.h5                     # Trained CNN model
├── rps_webcam_dataset_resized/         # Folder containing images for computer choice
│   ├── Rock/
│   ├── Paper/
│   └── Scissors/
├── main.py                              # Main game script
└── README.md                            # This file
🛠 Requirements
Python 3.x

OpenCV

TensorFlow / Keras

NumPy

Install dependencies:

bash
Copy
Edit
pip install opencv-python tensorflow numpy
🧠 Model
The CNN model (rps_cnn_model.h5) was trained on a custom webcam dataset with three classes:

Rock

Paper

Scissors

Ensure that your model input dimensions match your webcam frame size used in prediction.

🚀 How to Run
Ensure your webcam is connected.

Place your trained model (rps_cnn_model.h5) in the project directory.

Place your computer move images in the respective folders under rps_webcam_dataset_resized/.

Run the game:

bash
Copy
Edit
python main.py
Press Space to start the countdown and make your move.

Press R to restart when someone wins.

Press Q to quit the game.

👀 Example
The screen is divided into:

Left panel: Your webcam feed.

Right panel: The computer's move.

Score and round results shown at the top.