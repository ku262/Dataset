import pandas as pd
import numpy as np
from Evaluation import str2contrix

def fixation_points(csv_path):

    # 1920x1080
    width, height = 1920, 1080
    # time window = 80ms
    time_window = 80
    # fixation points radius = 30 pixel
    radius = 30

    df = pd.read_csv(csv_path)

    df = df[(df['left_gaze_point_validity'] == 1) & (df['right_gaze_point_validity'] == 1)]
    df = df.reset_index(drop=True)
    df = df.sort_index()

    date_time, time_stamp, left_gaze_point_on_display_area, right_gaze_point_on_display_area = \
        df['date_time'], df['timestamp'], df['left_gaze_point_on_display_area'], df['right_gaze_point_on_display_area']
    left_gaze_point_on_display_area, right_gaze_point_on_display_area = str2contrix(
        left_gaze_point_on_display_area), str2contrix(right_gaze_point_on_display_area)

    point_contrix = []
    for i in range(left_gaze_point_on_display_area.shape[0]):
        point_contrix.append(
            [(left_gaze_point_on_display_area[i, 0] + right_gaze_point_on_display_area[i, 0]) * width / 2,
             (left_gaze_point_on_display_area[i, 1] + right_gaze_point_on_display_area[i, 1]) * height / 2])
    point_contrix = np.array(point_contrix)
    time_stamp = np.array(time_stamp).astype(float) * 1000  # ms
    point_contrix = np.insert(point_contrix, 0, time_stamp, axis=1)

    # init
    fixed_points = []
    # fixation points
    current_points = []
    current_time = None

    idx = 0
    head_idx = 0
    while idx < point_contrix.shape[0]:

        point = point_contrix[idx, :]
        if current_time is None:
            current_time = point[0]
            head_idx = idx
        if point[0] - current_time <= time_window:
            # Check if the point is within the radius range of the current fixed point
            if len(current_points) == 0:
                current_points.append(point)
            else:
                all_in_radius = True
                for fixed_point in current_points:
                    if np.sqrt((point[1] - fixed_point[1]) ** 2 + (point[2] - fixed_point[2]) ** 2) > radius:
                        all_in_radius = False
                        break
                if all_in_radius:
                    current_points.append(point)
                else:
                    # If a point does not meet the distance condition, end the construction of the current fixed point
                    idx = head_idx + 1
                    current_points = [point_contrix[idx, :]]
                    current_time = point_contrix[idx, 0]
                    head_idx = idx
        else:
            all_in_radius = True
            for fixed_point in current_points:
                if np.sqrt((point[1] - fixed_point[1]) ** 2 + (point[2] - fixed_point[2]) ** 2) > radius:
                    all_in_radius = False
                    break
            if all_in_radius:
                current_points.append(point)
            else:
                # Calculate the position and time of fixed points
                mean_position = np.mean(current_points, axis=0)
                mean_time = np.median([p[0] for p in current_points])
                fixed_points.append([mean_time, mean_position[1], mean_position[2]])

                # Reset the current point list and time
                current_points = [point]
                current_time = point[0]
                head_idx = idx

        idx += 1

    fixed_points = np.array(fixed_points)

    return fixed_points

def fixations_ratio(fixation_points, fields=1):

    image_width = 1920
    image_height = 1080
    # The x-coordinate of the endoscopic image on the screen image, with the left part in black.
    reset_coordinate_x = 650

    grid_rows = 5
    grid_cols = 5
    if fields == 1:
        region1_size = 1
        def calculate_region(x, y):
            row = y // (image_height // grid_rows)
            col = (x-reset_coordinate_x) // ((image_width - reset_coordinate_x) // grid_cols)

            # Calculate the boundary of the central block
            center_row_start = grid_rows // 2 - region1_size // 2
            center_row_end = center_row_start + region1_size
            center_col_start = grid_cols // 2 - region1_size // 2
            center_col_end = center_col_start + region1_size

            # Determine which area the point is in
            if center_row_start <= row < center_row_end and center_col_start <= col < center_col_end:
                return "Ring 1"
            elif center_row_start - 1 <= row <= center_row_end + 1 and center_col_start - 1 <= col <= center_col_end + 1:
                return "Ring 2"
            elif col < 0 or row < 0 or col > 5 or row > 5:
                return "OUT of Range"
            else:
                return "Ring 3"

        # Count the number of points in each region
        region_counts = {"Ring 1": 0, "Ring 2": 0, "Ring 3": 0, "OUT of Range": 0}
        for point in fixation_points:
            x, y = point[1:]
            region = calculate_region(x, y)
            region_counts[region] += 1

        region_counts = {key: value / len(fixation_points) for key, value in region_counts.items()}

        return region_counts

    else:
        def calculate_region(x, y):
            row = y // (image_height // grid_rows)
            col = (x - reset_coordinate_x) // ((image_width - reset_coordinate_x) // grid_cols)

            # Determine which area the point is in
            if 0 <= col < 2:
                return "Left"
            elif 2 <= col <= 3:
                return "Middle"
            elif col < 0 or row < 0 or col > 5 or row > 5:
                return "OUT of Range"
            else:
                return "Right"

        # Count the number of points in each region
        region_counts = {"Left": 0, "Middle": 0, "Right": 0, "OUT of Range": 0}
        for point in fixation_points:
            x, y = point[1:]
            region = calculate_region(x, y)
            region_counts[region] += 1

        region_counts = {key: value / len(fixation_points) for key, value in region_counts.items()}

        return region_counts
