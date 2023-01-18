import cv2
import mediapipe as mp
import os
from tqdm import tqdm
import pandas as pd
pd.options.display.float_format = '{:.15f}'.format

input_path = " " # input path
IMAGE_FILES = os.listdir(input_path)

output_path = " " # output path
try:
    if not os.path.exists(output_path):
        os.makedirs(output_path)
except OSError:
    print ('Error: Creating directory. ' +  output_path)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

csv_path = " " # csv path should be at the parent directory of input and output path
f = open(csv_path + "Output Images - Sign \{label_number\}.csv", 'w')
f.write('Single Image Frame,')
for i in range(21):
    f.write(f'x{i:02d},')
    f.write(f'y{i:02d},')
    f.write(f'z{i:02d},')
f.write('Label\n')

# For static images:
count = 0
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5) as hands:
    for idx, file in enumerate(tqdm(IMAGE_FILES)):
        # Read an image, flip it around y-axis for correct handedness output (see
        # above).
        image = cv2.flip(cv2.imread(input_path + file), 1)
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.multi_hand_landmarks:
            continue
        image_height, image_width, _ = image.shape
        annotated_image = image.copy()
        f.write(f'{idx}, ')
        for result in results.multi_hand_landmarks:
            for _, lm in enumerate(result.landmark):
                f.write(f"{lm.x:.15f}" + ',')
                f.write(f"{lm.y:.15f}" + ',')
                f.write(f"{lm.z:.15f}" + ',')
            f.write('10\n')
            mp_drawing.draw_landmarks(
                annotated_image,
                result,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        cv2.imwrite(
            output_path + str(idx) + '.png', cv2.flip(annotated_image, 1))
        if count == 499: # Use only 500 images
            break
        count += 1

f.close()