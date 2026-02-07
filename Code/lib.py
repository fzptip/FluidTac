# Library functions
# fzp-2025.12.16
import cv2
import numpy as np
import math
import re
import pandas as pd


def calculate_angle_image_coords(x1, y1, x2, y2, image_height):
    y1_standard = image_height - y1
    y2_standard = image_height - y2

    dx = x2 - x1
    dy = y2 - y1

    angle_rad = math.atan2(dy, dx)

    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360

    return angle_deg


def contours_detect(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    height, width = img.shape[:2]
    center = (width // 2, height // 2)

    min_distance = float('inf')
    closest_triangle = None
    closest_triangle_points = None

    # Iterate through contours and detect triangles
    for i, contour in enumerate(contours):
        cnt_area = cv2.contourArea(contour)
        if cnt_area > 100:  # Filter out small area contours

            epsilon = 0.07 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 3:
                avg_point = np.mean(approx[:, 0, :], axis=0)
                distance_to_center = np.linalg.norm(avg_point - center)

                if distance_to_center < min_distance:
                    min_distance = distance_to_center
                    closest_triangle = approx
                    closest_triangle_points = avg_point

    # Draw the closest triangle
    if closest_triangle is not None:

        cv2.drawContours(img, [closest_triangle], 0, (0, 255, 0), 3)

        for point in closest_triangle:
            x, y = point[0]
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(img, f"({x}, {y})", (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    min_dis = 0
    dis_point = None

    if closest_triangle is not None:
        for point in closest_triangle:
            dis = np.linalg.norm(point - center)

            if dis > min_dis:
                dis_point = point
                min_dis = dis

        avg_x, avg_y = dis_point[0]
        cv2.circle(img, (avg_x, avg_y), 7, (255, 0, 0), -1)

        start = (width // 2, height // 2)
        end = (avg_x, avg_y)
        color = (255, 0, 255)
        thickness = 3
        cv2.arrowedLine(img, start, end, color, thickness, tipLength=0.2)

        angle = calculate_angle_image_coords(width // 2, height // 2, avg_x, avg_y, height)
    else:
        angle = 1000
    # print(angle)
    # cv2.imshow(f": {id}", thresh)

    return angle, closest_triangle


def find_farthest_black_pixel(binary_img, ref_point):
    if binary_img is None or binary_img.size == 0:
        raise ValueError("Image is empty!")

    black_pixels = np.argwhere(binary_img == 0)

    if len(black_pixels) == 0:
        raise ValueError("No black pixels in the image!")

    black_pixels_xy = black_pixels[:, [1, 0]]

    ref_x, ref_y = ref_point
    distances = np.sqrt((black_pixels_xy[:, 0] - ref_x) ** 2 +
                        (black_pixels_xy[:, 1] - ref_y) ** 2)

    farthest_idx = np.argmax(distances)
    farthest_point = tuple(black_pixels_xy[farthest_idx])

    return farthest_point


def remove_small_black_regions(binary_img, min_area=100):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        255 - binary_img, connectivity=8)

    # Filter regions with area greater than min_area
    mask = np.zeros_like(binary_img, dtype=np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            mask[labels == i] = 255

    cleaned_img = 255 - mask
    return cleaned_img


def angle_cal(img_, center_, w, h, width_):
    angle_ = []
    if_found = 1

    hsv = cv2.cvtColor(img_, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 40, 30])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([155, 55, 45])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2

    binary = np.zeros_like(red_mask)
    binary[red_mask > 0] = 0
    binary[red_mask == 0] = 255

    frame = binary

    frame_vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    cv2.imshow('Processed Frame', frame)
    cv2.waitKey(0)

    for center in center_:

        center_x = int(center[0])
        center_y = int(center[1])

        x1 = max(0, center_x - width_ // 2)
        y1 = max(0, center_y - width_ // 2)
        x2 = min(frame.shape[1], center_x + width_ // 2)
        y2 = min(frame.shape[0], center_y + width_ // 2)

        img = frame[y1:y2, x1:x2]

        angle, closest_triangle = contours_detect(img)

        if angle == 1000:

            _, img_thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

            img_thresh = remove_small_black_regions(img_thresh, min_area=100)

            img_h, img_w = img_thresh.shape
            ref_point = (img_w // 2, img_h // 2)

            try:
                farthest_point = find_farthest_black_pixel(img_thresh, ref_point)
                angle = calculate_angle_image_coords(img_w // 2, img_h // 2,
                                                     farthest_point[0], farthest_point[1], img_h)

            except ValueError as e:
                print(f"Propeller not found: {e}")

            if_found = 0

        if closest_triangle is not None:

            closest_triangle = closest_triangle.squeeze()
            closest_triangle[:, 0] += center_x - width_ // 2
            closest_triangle[:, 1] += center_y - width_ // 2

            min_dis = 0
            dis_point = None

            cv2.drawContours(frame_vis, [closest_triangle], 0, (0, 255, 0), 3)

            for point in closest_triangle:
                x = point[0]
                y = point[1]
                cv2.circle(frame_vis, (x, y), 5, (0, 0, 255), -1)

            for point in closest_triangle:
                dis = np.linalg.norm(point - center)

                if dis > min_dis:
                    dis_point = point
                    min_dis = dis

            avg_x = dis_point[0]
            avg_y = dis_point[1]

            cv2.circle(frame_vis, (avg_x, avg_y), 7, (255, 0, 0), -1)

            start = (center_x, center_y)
            end = (avg_x, avg_y)
            color = (255, 0, 255)
            thickness = 3
            # cv2.arrowedLine(frame_vis, start, end, color, thickness, tipLength=0.2)

        angle_.append(round(angle, 2))

    return frame_vis, angle_, if_found


def interpolate_sentinel_1000(raw_data, sentinel=1000.0, tol=1e-9):
    if not raw_data:
        return raw_data

    n_rows = len(raw_data)
    n_cols = len(raw_data[0])

    def is_sentinel(v):
        return abs(v - sentinel) <= tol

    for col in range(n_cols):
        i = 0
        while i < n_rows:
            if not is_sentinel(raw_data[i][col]):
                i += 1
                continue

            start = i
            while i < n_rows and is_sentinel(raw_data[i][col]):
                i += 1
            end = i - 1

            prev_idx = start - 1
            while prev_idx >= 0 and is_sentinel(raw_data[prev_idx][col]):
                prev_idx -= 1

            next_idx = end + 1
            while next_idx < n_rows and is_sentinel(raw_data[next_idx][col]):
                next_idx += 1

            prev_ok = prev_idx >= 0
            next_ok = next_idx < n_rows

            if prev_ok and next_ok:
                prev_val = raw_data[prev_idx][col]
                next_val = raw_data[next_idx][col]
                run_len = end - start + 1

                for k in range(run_len):
                    t = (k + 1) / (run_len + 1)
                    raw_data[start + k][col] = prev_val + (next_val - prev_val) * t

            elif prev_ok and not next_ok:
                fill_val = raw_data[prev_idx][col]
                for r in range(start, end + 1):
                    raw_data[r][col] = fill_val

            elif (not prev_ok) and next_ok:
                fill_val = raw_data[next_idx][col]
                for r in range(start, end + 1):
                    raw_data[r][col] = fill_val

            else:
                pass

    return raw_data


def process_frame_data(input_file, output_file):
    raw_data = []

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            match = re.search(r'\[(.*?)\]', line)
            if match:
                content = match.group(1)
                values = [float(x.strip()) for x in content.split(',')]
                raw_data.append(values)

    except FileNotFoundError:
        print(f"Error: File {input_file} not found")
        return

    if len(raw_data) < 2:
        print("Insufficient data rows, cannot perform difference calculation.")
        return

    raw_data = interpolate_sentinel_1000(raw_data, sentinel=1000.0)

    processed_rows = []

    for i in range(1, len(raw_data)):
        current_frame = raw_data[i]
        prev_frame = raw_data[i - 1]

        diff_row = []

        for curr_val, prev_val in zip(current_frame, prev_frame):
            diff = curr_val - prev_val

            if diff > 180:
                diff -= 360
            elif diff < -180:
                diff += 360

            if abs(diff) > 80:
                diff = 0.0

            diff_row.append(round(diff, 2))

        processed_rows.append(diff_row)

    num_columns = len(processed_rows[0])
    columns = [f'Diff_Val_{k + 1}' for k in range(num_columns)]

    df = pd.DataFrame(processed_rows, columns=columns)

    frame_indices = [f'Frame_{k + 1}-{k + 2}' for k in range(len(processed_rows))]
    df.insert(0, 'Frame_Diff_Index', frame_indices)

    df.to_excel(output_file, index=False)
    print(f"Processing complete!")
    print(f"Original row count: {len(raw_data)}")
    print(f"Row count after difference: {len(processed_rows)}")
    print(f"Results saved to: {output_file}")