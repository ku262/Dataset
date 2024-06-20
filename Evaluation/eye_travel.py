import pandas as pd
import numpy as np
from Evaluation import str2contrix


def eye_travel_distance(csv_path):

    # 1920x1080

    width, height = 1920, 1080
    df = pd.read_csv(csv_path)

    df = df[(df['left_gaze_point_validity'] == 1) & (df['right_gaze_point_validity'] == 1)]
    df = df.reset_index(drop=True)
    df = df.sort_index()

    left_gaze_point_on_display_area, right_gaze_point_on_display_area = \
        df['left_gaze_point_on_display_area'], df['right_gaze_point_on_display_area']
    left_gaze_point_on_display_area, right_gaze_point_on_display_area = str2contrix(left_gaze_point_on_display_area), str2contrix(right_gaze_point_on_display_area)
    point_contrix = []
    for i in range(left_gaze_point_on_display_area.shape[0]):
        point_contrix.append([(left_gaze_point_on_display_area[i, 0]+right_gaze_point_on_display_area[i, 0])*width/2,
                              (left_gaze_point_on_display_area[i, 1]+right_gaze_point_on_display_area[i, 1])*height/2])
    point_contrix = np.array(point_contrix)
    diff_distance = point_contrix[1:, :] - point_contrix[:-1, :]

    result = np.sqrt(np.sum(diff_distance[:, 0:2] ** 2, axis=1))

    final_result = np.sum(result)

    return final_result
