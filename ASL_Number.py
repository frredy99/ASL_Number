import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from NumberDetector import NumberDetector
from HandAngle import HandAngle
from time import time
from datetime import datetime

start = time()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Load model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = NumberDetector().to(device)
if torch.cuda.is_available():
    model.load_state_dict(torch.load('./NumberDetector_state_dict.pt'))
else:
    model.load_state_dict(torch.load('./NumberDetector_state_dict.pt',
                            map_location=torch.device('cpu')))
model.eval()

# For Video File input:
path = 0 # Input video path here
cap = cv2.VideoCapture(path)
fps = cap.get(cv2.CAP_PROP_FPS)
full_frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
original_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
original_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
output_height = 720
output_width = int(output_height*original_width//original_height)

# Define the codec and create VideoWriter Object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter('result.mp4', fourcc, fps, (output_width, output_height))
# Create result.txt
f = open("./result.txt", 'w')

if path == 0:
    print("Using web cam")

else:
    print("Using recorded video")

with mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    count = 1
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            # If loading a video, use 'break' instead of 'continue'.
            print(f"\nIgnoring empty camera frame\n")
            break
        
        if not path == 0:
            print(f"frame : {count:4d} / {full_frame_num}, {count/full_frame_num*100:.2f}%", end='\r')

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.resize(image, (output_width, output_height), interpolation=cv2.INTER_LINEAR)        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        pred = 0
        if results.multi_hand_landmarks:
            landmarks = []
            for result in results.multi_hand_landmarks:
                for _, lm in enumerate(result.landmark):
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)
                    landmarks.append(lm.z)

            # Draw the hand annotation on the image
                mp_drawing.draw_landmarks(
                        image,
                        result,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            
            # Detects number from extracted landmarks, and writes result on the image
            # There were some frames that outputs 126 landmarks, causing error.
            # To prevent the error, restrict to only when mediapipe outputs 63 landmarks.
            landmarks = torch.tensor(landmarks)            
            if landmarks.size()[0] == 63:
                output = model(HandAngle(landmarks))
                pred = output.max(1, keepdim=True)[1].item() + 1
                cv2.putText(
                        image, text=f"{pred}", org=(40, 60),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2,
                        color=(0, 255, 0), thickness=3)
                cv2.putText(
                        image, text="FPS : {:4f}".format(fps), org=(40, 100),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=(0, 255, 0), thickness=2)

        # Save Image Frames to Output Video
        cv2.flip(image, 1)
        if not path == 0:
            cv2.imshow("ASL number", image)
            out.write(image)
            f.write(f"Frame : {count}, Time : {count/fps:.4f}, Result : {pred}\n")
        else:
            cv2.imshow("ASL number", image)
            out.write(image)
            f.write(f"Frame : {count}, Time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}, Result : {pred}\n")

        if cv2.waitKey(5) & 0xFF == ord('e'):
            break
        
        count += 1

f.close()
# Close all the opened frames            
cap.release()
cv2.destroyAllWindows()

end = time()
print('time elapsed : ', end - start)

if __name__ == "__main__":
    print("ASL_Number is running.")