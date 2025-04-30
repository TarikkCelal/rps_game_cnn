import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dropout
import os

dataset_path = 'rps_webcam_dataset_resized' 
img_width, img_height = 320, 240
batch_size = 32

datagen = ImageDataGenerator( 
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    brightness_range=[0.8, 1.2],
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2, 2),
    Dropout(0.3),

    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    MaxPooling2D(2, 2),
    Dropout(0.3),

    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    MaxPooling2D(2, 2),
    Dropout(0.3),
    Flatten(),

    Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.5), 

    Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

epochs = 20

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[early_stopping]
)

model.save('rps_cnn_model.h5')

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
import random
import os
from PIL import Image

# Load your trained model
model = load_model('rps_cnn_model.h5')

# Class labels
class_labels = ['Paper', 'Rock', 'Scissors']

# Webcam
cap = cv2.VideoCapture(1)  # or VideoCapture(0) depending on your setup
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Dataset path for computer random pick
dataset_path = 'rps_webcam_dataset_resized'

# Scores
player_score = 0
computer_score = 0

# Game state
waiting_for_move = True
countdown = 0
start_countdown_time = None
player_move = None
computer_move = None
computer_img = None  # Store the computer image here
winner_text = ''
game_over = False

# Preprocessing function
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (320, 240))  # Same size you trained with
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_normalized = frame_rgb / 255.0
    frame_expanded = np.expand_dims(frame_normalized, axis=0)
    return frame_expanded

# Function to get random computer image
def get_computer_choice():
    choice = random.choice(class_labels)
    folder = os.path.join(dataset_path, choice)
    image_file = random.choice(os.listdir(folder))
    img_path = os.path.join(folder, image_file)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (640, 480))  # Resize computer image to 640x480
    return choice, img

# Function to decide winner
def decide_winner(player, computer):
    if player == computer:
        return 'Tie'
    elif (player == 'Rock' and computer == 'Scissors') or \
            (player == 'Paper' and computer == 'Rock') or \
            (player == 'Scissors' and computer == 'Paper'):
        return 'Player'
    else:
        return 'Computer'

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_flipped = cv2.flip(frame, 1)  # Mirror effect

    # Resize webcam frame to 640x480 (left panel)
    frame_resized = cv2.resize(frame_flipped, (640, 480))

    # Create a black background (double size panel: 1280x480)
    game_screen = np.zeros((480, 1280, 3), dtype=np.uint8)

    # Place webcam frame on the left panel (640x480 area)
    game_screen[0:480, 0:640] = frame_resized

    # Handle countdown and move
    if not waiting_for_move and not game_over:
        if countdown > 0:
            elapsed = time.time() - start_countdown_time
            if elapsed >= 1:
                countdown -= 1
                start_countdown_time = time.time()
        elif countdown == 0:
            # Predict player's move
            processed_frame = preprocess_frame(frame_flipped)
            prediction = model.predict(processed_frame, verbose=0)
            player_move_idx = np.argmax(prediction)
            player_move = class_labels[player_move_idx]

            # Get computer's move
            computer_move, computer_img = get_computer_choice()

            # <---- IMPORTANT CHANGE:  moved this block here
            # Place computer image on the right panel (640x480 area)
            game_screen[0:480, 640:1280] = computer_img
            # ----->

            # Decide winner
            round_winner = decide_winner(player_move, computer_move)
            if round_winner == 'Player':
                player_score += 1
            elif round_winner == 'Computer':
                computer_score += 1

            if player_score == 3 or computer_score == 3:
                game_over = True
                if player_score == 3:
                    winner_text = 'You Win!'
                else:
                    winner_text = 'Computer Wins!'

            waiting_for_move = True

    # Place computer image on the right panel (640x480 area)
    if computer_img is not None:  # <---- Added this condition
        game_screen[0:480, 640:1280] = computer_img
    # ----->

    # Draw scoreboard (adjusted font size)
    score_text = f"You: {player_score}  |  Computer: {computer_score}"
    cv2.putText(game_screen, score_text, (500, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Draw countdown
    if not waiting_for_move and countdown > 0:
        cv2.putText(game_screen, str(countdown), (580, 240), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 5)

    # Draw winner
    if game_over:
        cv2.putText(game_screen, winner_text, (440, 220), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
        cv2.putText(game_screen, "Press R to Restart", (480, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show frame
    cv2.imshow('Rock Paper Scissors Game', game_screen)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord(' '):
        if waiting_for_move and not game_over:
            countdown = 3
            start_countdown_time = time.time()
            waiting_for_move = False
            player_move = None
            computer_move = None
            # computer_img = None  #  <---- Removed this line

    if key == ord('r') and game_over:
        # Reset game
        player_score = 0
        computer_score = 0
        winner_text = ''
        game_over = False
        waiting_for_move = True
        player_move = None
        computer_move = None
        computer_img = None  # Reset computer image

cap.release()
cv2.destroyAllWindows()
