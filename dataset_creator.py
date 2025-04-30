import cv2
import os

dataset_dir = 'rps_webcam_dataset_resized'
rock_dir = os.path.join(dataset_dir, 'rock')
paper_dir = os.path.join(dataset_dir, 'paper')
scissors_dir = os.path.join(dataset_dir, 'scissors')

os.makedirs(rock_dir, exist_ok=True)
os.makedirs(paper_dir, exist_ok=True)
os.makedirs(scissors_dir, exist_ok=True)

cap = cv2.VideoCapture(1)

new_width, new_height = 320, 240

count = {'rock': len(os.listdir(rock_dir)),
         'paper': len(os.listdir(paper_dir)),
         'scissors': len(os.listdir(scissors_dir))}

capture_limit = 500  # Number of images per class

current_class = None

print(f"Capturing data and resizing to {new_width}x{new_height}")
print("Press 'r' to capture Rock, 'p' for Paper, 's' for Scissors, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, (new_width, new_height))
    cv2.imshow('Capturing Data (Resized)', resized_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        current_class = 'rock'
    elif key == ord('p'):
        current_class = 'paper'
    elif key == ord('s'):
        current_class = 'scissors'
    elif key == ord('q'):
        break

    if current_class and count[current_class] < capture_limit:
        img_name = os.path.join(eval(f'{current_class}_dir'), f'{count[current_class]}.jpg')
        cv2.imwrite(img_name, resized_frame)
        print(f"Captured and resized {count[current_class]} images for {current_class}")
        count[current_class] += 1
        current_class = None

    elif current_class and count[current_class] >= capture_limit:
        print(f"Reached capture limit for {current_class}")
        current_class = None

cap.release()
cv2.destroyAllWindows()

print("Dataset collection complete!")
print(f"Rock images: {len(os.listdir(rock_dir))}")
print(f"Paper images: {len(os.listdir(paper_dir))}")
print(f"Scissors images: {len(os.listdir(scissors_dir))}")