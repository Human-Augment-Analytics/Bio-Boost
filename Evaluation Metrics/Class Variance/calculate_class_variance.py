# Imports
import pandas as pd
import cv2
import numpy as np
import os
import warnings
import random

# Variables
csv_folder = 'experiment file names'
results = []
image_counter = 0
save_folder = '15. Run 2 contour_examples'
os.makedirs(save_folder, exist_ok=True)
csv_counter = 0 

# Process Each CSV File
for csv_file in os.listdir(csv_folder):

    if csv_file.endswith('.csv'):

        csv_path = os.path.join(csv_folder, csv_file)
        df = pd.read_csv(csv_path)

        hue_differences = {0: [], 1: []}

        for _, row in df.iterrows():

            # Get Frame
            img_path = row['filepath']
            true_label = row['true_label']
            root_dir = r'D:\...\TrainingFrames2'
            img_path = os.path.join(root_dir, img_path)
            frame = cv2.imread(img_path)

            if frame is None:
                print(f"Warning: Could not read image {img_path}")
                continue

            # Convert to HSV, Apply Blur, Convert to Gray for Contouring
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            blurred_img = cv2.GaussianBlur(hsv_frame, (3, 3), 0)
            gray_frame = cv2.cvtColor(blurred_img, cv2.COLOR_HSV2BGR)
            gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_BGR2GRAY)

            # Dilate, Erode, and Invert Edges so Fish are Contoured Instead of the Sand
            edges = cv2.Canny(gray_frame, 50, 100)
            kernel = np.ones((4, 4), np.uint8)
            dilated_edges = cv2.dilate(edges, kernel, iterations=2)
            eroded_dilated_edges = cv2.erode(dilated_edges, kernel, iterations=1)
            inverted_edges = cv2.bitwise_not(eroded_dilated_edges)

            # Find and Filter Contours
            contours, _ = cv2.findContours(inverted_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [c for c in contours if 300 < cv2.contourArea(c) < 4500]

            # Filter Out Extremely Round Contours, Like Contours of Sand
            selected_contours = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h != 0 else 0
                if aspect_ratio < 1.5:
                    selected_contours.append(contour)

            if selected_contours:
                h, w = frame.shape[:2]
                image_center = np.array([w / 2, h / 2])
            
                min_dist = float('inf')
                best_contour = None
                best_head = None
                best_tail = None
            
                # Smooth Contours for Less Jagged Edges
                for contour in selected_contours:
                    epsilon = 0.004 * cv2.arcLength(contour, True)
                    approx_contour = cv2.approxPolyDP(contour, epsilon, True)
                    points = approx_contour.reshape(-1, 2).astype(np.float32)
            
                    # Use PCA to Find Head and Tail
                    if len(points) > 1:

                        # Compute Eigenvectors, Primary is First One; Project to Principal Axis
                        mean, eigenvectors = cv2.PCACompute(points, mean=None)
                        principal_axis = eigenvectors[0]
                        projections = np.dot(points - mean, principal_axis)
            
                        max_idx = np.argmax(projections)
                        min_idx = np.argmin(projections)
                        head = points[max_idx]
                        tail = points[min_idx]
                        midpoint = (head + tail) / 2
            
                        dist = np.linalg.norm(midpoint - image_center)
                        if dist < min_dist:
                            min_dist = dist
                            best_contour = approx_contour
                            best_head = head
                            best_tail = tail
            
                if best_contour is not None:
                    head_point = tuple(best_head.astype(int))
                    tail_point = tuple(best_tail.astype(int))
                contour = best_contour  
                
                # Smooth Contours for Less Jagged Edges
                epsilon = 0.004 * cv2.arcLength(contour, True)
                approx_contour = cv2.approxPolyDP(contour, epsilon, True)
                points = approx_contour.reshape(-1, 2).astype(np.float32)

                if len(points) > 1:

                    mean, eigenvectors = cv2.PCACompute(points, mean=None)
                    principal_axis = eigenvectors[0]
                    projections = np.dot(points - mean, principal_axis)

                    max_idx = np.argmax(projections)
                    min_idx = np.argmin(projections)
                    head_point = tuple(points[max_idx].astype(int))
                    tail_point = tuple(points[min_idx].astype(int))

                    head_hue = hsv_frame[head_point[1], head_point[0], 0]
                    tail_hue = hsv_frame[tail_point[1], tail_point[0], 0]

                    warnings.filterwarnings("ignore", category=RuntimeWarning,
                                            message="overflow encountered in scalar subtract")

                    # Calculate Hue Difference between Head and Tail
                    hue_dif = min(abs(head_hue - tail_hue), abs(tail_hue - head_hue))
                    hue_differences[true_label].append(hue_dif)

                    # Draw Contours and Points for Visual Inspection
                    image_counter += 1
                    if image_counter % 1000 == 0:
                        drawn_frame = frame.copy()
                        cv2.drawContours(drawn_frame, [approx_contour], -1, (0, 255, 0), 2)
                        cv2.circle(drawn_frame, head_point, 5, (0, 0, 255), -1)
                        cv2.circle(drawn_frame, tail_point, 5, (255, 0, 0), -1)
                        save_path = os.path.join(save_folder, f'{os.path.splitext(csv_file)[0]}_{image_counter}.jpg')
                        cv2.imwrite(save_path, drawn_frame)

        # Data to Write Results to File
        avg_0 = np.mean(hue_differences[0]) if hue_differences[0] else np.nan
        avg_1 = np.mean(hue_differences[1]) if hue_differences[1] else np.nan
        results.append({
            'experiment': csv_file.replace('.csv', ''),
            'avg_hue_diff_label_0': avg_0,
            'avg_hue_diff_label_1': avg_1
        })

    csv_counter = csv_counter + 1
    print(f"{csv_counter} CSV complete.")

# Save Results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('15. hue_diff_by_experiment_run2.csv', index=False)