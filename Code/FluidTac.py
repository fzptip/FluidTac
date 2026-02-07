# Main Code
# fzp-2025.12.16
from lib import *
import cv2
import os
import numpy as np

# Folder containing videos
data_dir = 'data1211_3'

for filename in os.listdir(data_dir):
    if not filename.lower().endswith('.mp4'):
        continue

    video_path = os.path.join(data_dir, filename)
    print(f"processing:{video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")

    skip_frames = int(fps * 0)
    print(f"will skip {skip_frames} frame")

    base_name = os.path.splitext(filename)[0]
    log_filename = f'{base_name}_angle_log.txt'
    log_path = os.path.join(data_dir, log_filename)
    log_file = open(log_path, 'w', encoding='utf-8')

    frame_idx = 0
    processed_count = 0

    while True:
        ret, image = cap.read()
        if not ret:
            break

        frame_idx += 1

        if frame_idx <= skip_frames:
            continue

        ############## Image Processing ##############
        alpha = 1.3  # Contrast (1.0-3.0)
        beta = 30    # Brightness (0-100)

        # Linear transformation
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        ############## End of Image Processing ##############

        h, w = image.shape[:2]  # Height, Width
        width = 350

        # center = np.array([
        #     [539, 516],
        #     [221, 443],
        #     [392, 214],
        #     [689, 226],
        #     [866, 453],
        #     [801, 728],
        #     [530, 850],
        #     [265, 731]
        # ])

        # Experiment 2 center points
        # center = np.array([
        #         [537, 516],
        #         [212, 431],
        #         [392, 209],
        #         [675, 221],
        #         [869, 434],
        #         [789, 724],
        #         [527, 841],
        #         [254, 731]
        #     ])

        # Experiment 3_3 center points
        # center = np.array([
        #     [534, 516],
        #     [216, 434],
        #     [396, 207],
        #     [693, 223],
        #     [869, 462],
        #     [794, 728],
        #     [530, 850],
        #     [256, 735]
        # ])

        # Experiment 3_2 center points
        center = np.array([
            [537, 511],
            [214, 424],
            [396, 205],
            [689, 216],
            [866, 445],
            [791, 738],
            [532, 850],
            [249, 731]
        ])


        img, angle, found = angle_cal(image, center, w, h, width)

        # Record angle
        log_message = f"frame_{frame_idx}: {angle}\n"
        log_file.write(log_message)
        processed_count += 1

    log_file.close()
    cap.release()
    print(f"result: {log_path}\n")

    output_filename = 'diff_result.xlsx'
    process_frame_data(log_path, output_filename)