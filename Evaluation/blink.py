import numpy as np
import pandas as pd

def blink(csv_path):
    # 1920x1080
    width, height = 1920, 1080
    # Blink to determine time window 70 ms
    time_window = 70

    df = pd.read_csv(csv_path)

    timestamp, left_gaze_point_validity, right_gaze_point_validity = \
        df['timestamp'], df['left_gaze_point_validity'], df['right_gaze_point_validity']
    timestamp = np.array(timestamp).astype(float) * 1000  #ms
    result = np.logical_or(np.array(left_gaze_point_validity), np.array(right_gaze_point_validity)).astype(int)
    result = np.column_stack((timestamp, result))

    # Closed eye frames
    count = 0
    queue = []
    for i in range(result.shape[0]):
        data = result[i, :]
        if data[1] == 0:
            queue.append(data)
        else:
            if len(queue) > 1:
                time_diff = queue[-1][0] - queue[0][0]
                if time_diff >= time_window:
                    count += 1
            queue = []
    # Check the time difference of the last set of events
    if len(queue) > 1:
        time_diff = queue[-1][0] - queue[0][0]
        if time_diff >= time_window:
            count += 1

    return count