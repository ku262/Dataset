import os
import pandas as pd
import numpy as np
from datetime import datetime

def cal_reaction_time(obj):

    csv_file_path = os.path.join(obj, "match.csv")
    if os.path.exists(csv_file_path):
        df = pd.read_csv(csv_file_path)
        reaction_time = np.array(list(df['reaction time']))[1:].astype(float)
        mean_reaction_time = np.mean(reaction_time[reaction_time != 0])
        print(obj)
        print(mean_reaction_time)

def str2timestamp(timestamp_str):
    # 解析时间戳字符串
    timestamp_parts = timestamp_str.split(':')

    return int(timestamp_parts[0])*3600+int(timestamp_parts[1])*60+float(timestamp_parts[2])

def trans(total_seconds):
    # 使用 divmod 函数获取整数和小数部分
    seconds_integer, seconds_fraction = divmod(total_seconds, 1)

    # 将整数部分转换为小时、分钟和秒数，并确保输出是两位数
    hours = str(int(seconds_integer // 3600)).zfill(2)
    minutes = str(int((seconds_integer % 3600) // 60)).zfill(2)
    seconds = str(int(seconds_integer % 60)).zfill(2)

    # 将小数部分转换为毫秒，并确保输出是三位数（两位数字 + 补零）
    milliseconds = str(int(seconds_fraction * 1000)).zfill(3)

    res = hours+":"+minutes+":"+seconds+"."+milliseconds
    return res

def check(start_timestamp, end_timestamp, video_frames):

    # 初始化一个存储索引的列表
    result_indices = []
    bool_list = np.zeros(np.array(start_timestamp).shape)
    error_list = []
    # 遍历 press 数组
    for idx, p in enumerate(video_frames):
        found_index = -1  # 默认为 -1，表示没找到匹配的范围

        # 使用 zip() 同时遍历 start 和 end 数组
        for i, (s, e) in enumerate(zip(start_timestamp, end_timestamp)):
            if s <= p <= e:  # 如果 press 数值在当前范围内
                found_index = i  # 记录索引
                break  # 找到了就不用再继续搜索了，直接跳出循环
        if found_index in result_indices:
            found_index = -1
        if found_index == -1:
            error_list.append(idx)
        result_indices.append(found_index)  # 将结果添加到列表中
    for i in result_indices:
        if i != -1:
            bool_list[i] = 1

    return result_indices, bool_list, error_list

def readtxt(file_path):
    with open(file_path, 'r') as file:
        # 读取文件内容
        lines = file.readlines()
    press_time_list = []
    video_head_date = "None"
    for line in lines:
        if "Sent value e to COM2" in line:
            press_time_list.append(line.split(": ")[0])
        if "Sent value ' ' (space) to COM2" in line:
            video_head_date = line.split(": ")[0]
    return video_head_date, press_time_list

def get_match_file(obj, calibrate_csv_path, output):

    name = obj.split(os.sep)[-1]
    output = os.path.join(output, name)
    if not os.path.exists(output):
        os.mkdir(output)

    calibrate_df = pd.read_excel(calibrate_csv_path)
    calibrate_start, calibrate_end = list(calibrate_df.iloc[1:, 1]), list(calibrate_df.iloc[1:, 2])

    calibrate_start_timestamp, calibrate_end_timestamp = \
        [str2timestamp(i) for i in calibrate_start], \
        [str2timestamp(i) for i in calibrate_end]

    video_head_date, press_log = readtxt(os.path.join(obj, [i for i in os.listdir(obj) if ".txt" in i][0]))

    press_timestamp_log = [datetime.timestamp(datetime.strptime(i, "%Y-%m-%d %H:%M:%S.%f")) for i in press_log]

    df = pd.read_csv(os.path.join(obj, "time_gaze.csv"))
    video_head_timestamp = datetime.strptime(video_head_date, "%Y-%m-%d %H:%M:%S.%f").timestamp()
    real_timestamp, real_datetime = df['timestamp'], df['date_time']
    timestamp_interval = video_head_timestamp - real_timestamp[0]

    press_video_frames = [i for i in (np.array(press_timestamp_log) - real_timestamp[0])]
    calibrate_press_video_frames = [trans(i) for i in
                                    (np.array(press_timestamp_log) - (real_timestamp[0] + timestamp_interval))]

    index_list, bool_list, error_list = check(calibrate_start_timestamp, calibrate_end_timestamp, press_video_frames)
    for i in error_list:
        print(f"idx: {i} log: {press_log[i]}")

    save_match_press_frames, reaction_time, press_save_log = ["press"], ["reaction time"], ["press log"]

    for i in range(len(bool_list)):
        save_match_press_frames.append("0")
        reaction_time.append("0")
        press_save_log.append("0")

    for idx, value in enumerate(index_list):
        if value != -1:
            press_save_log[value+1] = press_log[idx]
            save_match_press_frames[value+1] = calibrate_press_video_frames[idx]
            reaction_time[value+1] = str2timestamp(calibrate_press_video_frames[idx]) - str2timestamp(calibrate_start[value])

    calibrate_df.insert(3, "press_video_frames", save_match_press_frames)
    calibrate_df.insert(4, 'reaction time', reaction_time)
    calibrate_df.insert(5, 'press_log', press_save_log)
    calibrate_df.to_csv(os.path.join(output, 'match.csv'), mode='w', encoding='utf-8', index=False)