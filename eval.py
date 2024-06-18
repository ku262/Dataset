import os
import cv2
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy.stats import gaussian_kde
import matplotlib.colors as mcolors
import glob
import csv
matplotlib.use('TKAgg')

def str2contrix(x):

    res = []
    for i in x:
        res.append([float(i.split(", ")[0].split("(")[-1]), float(i.split(", ")[1].split(")")[0])])
    res = np.array(res)
    return res

def read_txt(file_path):
    with open(file_path, 'r') as file:
        # 读取文件内容
        lines = file.readlines()
    video_head_date = "None"
    timestamp = 0
    head_datetime = "None"
    for line in lines:
        if "Sent value ' ' (space) to COM2" in line:
            video_head_date = line.split(": ")[0]
            timestamp = datetime.strptime(video_head_date, "%Y-%m-%d %H:%M:%S.%f").timestamp()
            head_datetime = datetime.strptime(video_head_date, "%Y-%m-%d %H:%M:%S.%f")
    return video_head_date, timestamp, head_datetime

def detection_rate(csv_path):

    df = pd.read_csv(csv_path)
    reaction_time = np.array(list(df['reaction time']))[1:].astype(float)
    ratio = np.count_nonzero(reaction_time) / reaction_time.shape[0]

    return ratio

def eye_travel_distance(csv_path, txt_path):

    # 1920x1080
    # 33 min 23 s
    total_video_len = timedelta(minutes=33, seconds=23)
    video_head_date, head_timestamp, head_datetime = read_txt(txt_path)
    if video_head_date == "None":
        print("error! no head date")
        return
    end_datetime = head_datetime + total_video_len
    end_timestamp = end_datetime.timestamp()

    width, height = 1920, 1080
    df = pd.read_csv(csv_path)

    df = df[(df['timestamp'] >= head_timestamp) & (df['timestamp'] <= end_timestamp)]
    df = df.reset_index(drop=True)
    df = df.sort_index()

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

    # 计算每行第一列与第二列元素的平方和再开方
    result = np.sqrt(np.sum(diff_distance[:, 0:2] ** 2, axis=1))

    # 计算所有行的结果相加
    final_result = np.sum(result)

    return final_result

def fixation_points(csv_path, txt_path):

    # 1920x1080
    # 33 min 23 s
    width, height = 1920, 1080
    total_video_len = timedelta(minutes=33, seconds=23)
    # 假设时间窗口为 80ms
    time_window = 80
    # 假设固定点的半径为 30 像素
    radius = 30

    video_head_date, head_timestamp, head_datetime = read_txt(txt_path)
    if video_head_date == "None":
        print("error! no head date")
        return
    end_datetime = head_datetime + total_video_len
    end_timestamp = end_datetime.timestamp()

    df = pd.read_csv(csv_path)

    df = df[(df['timestamp'] >= head_timestamp) & (df['timestamp'] <= end_timestamp)]
    df = df.reset_index(drop=True)
    df = df.sort_index()

    df = df[(df['left_gaze_point_validity'] == 1) & (df['right_gaze_point_validity'] == 1)]
    df = df.reset_index(drop=True)
    df = df.sort_index()

    date_time, time_stamp, device_timestamp, left_gaze_point_on_display_area, right_gaze_point_on_display_area = \
        df['date_time'], df['timestamp'], df['device_time_stamp'], df['left_gaze_point_on_display_area'], df['right_gaze_point_on_display_area']
    left_gaze_point_on_display_area, right_gaze_point_on_display_area = str2contrix(
        left_gaze_point_on_display_area), str2contrix(right_gaze_point_on_display_area)
    device_timestamp = np.array(device_timestamp).astype(float)/1000  #ms

    point_contrix = []
    for i in range(left_gaze_point_on_display_area.shape[0]):
        point_contrix.append(
            [(left_gaze_point_on_display_area[i, 0] + right_gaze_point_on_display_area[i, 0]) * width / 2,
             (left_gaze_point_on_display_area[i, 1] + right_gaze_point_on_display_area[i, 1]) * height / 2])
    point_contrix = np.array(point_contrix)
    point_contrix = np.insert(point_contrix, 0, device_timestamp, axis=1)
    # point_contrix = np.insert(point_contrix, 3, date_time, axis=1)
    # point_contrix = np.insert(point_contrix, 4, time_stamp, axis=1)

    # 初始化一个列表来存储固定点
    fixed_points = []
    # 初始化变量来跟踪连续出现超过 80ms 的点
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
            # 检查点是否在当前固定点的半径范围内
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
                    # 如果有一个点不满足距离条件，则结束当前固定点的构建
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
                # 计算固定点的位置和时间
                mean_position = np.mean(current_points, axis=0)
                mean_time = np.median([p[0] for p in current_points])
                fixed_points.append([mean_time, mean_position[1], mean_position[2]])

                # 重置当前点列表和时间
                current_points = [point]
                current_time = point[0]
                head_idx = idx

        idx += 1
    # 将结果转换为 NumPy 数组
    fixed_points = np.array(fixed_points)

    return fixed_points

def fixations_ratio(fixation_points, fields=1):


    # 定义图像大小
    image_width = 1920
    image_height = 1080
    reset_coordinate_x = 650

    # 定义网格大小和区域
    grid_rows = 5
    grid_cols = 5
    if fields == 1:
        region1_size = 1
        def calculate_region(x, y):
            row = y // (image_height // grid_rows)
            col = (x-reset_coordinate_x) // ((image_width - reset_coordinate_x) // grid_cols)

            # 计算中心方块的边界
            center_row_start = grid_rows // 2 - region1_size // 2
            center_row_end = center_row_start + region1_size
            center_col_start = grid_cols // 2 - region1_size // 2
            center_col_end = center_col_start + region1_size

            # 判断点在哪个区域
            if center_row_start <= row < center_row_end and center_col_start <= col < center_col_end:
                return "Ring 1"
            elif center_row_start - 1 <= row <= center_row_end + 1 and center_col_start - 1 <= col <= center_col_end + 1:
                return "Ring 2"
            elif col < 0 or row < 0 or col > 5 or row > 5:
                return "OUT of Range"
            else:
                return "Ring 3"

        # 统计每个区域中点的数量
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
            # col = x // (image_width // grid_cols)
            col = (x - reset_coordinate_x) // ((image_width - reset_coordinate_x) // grid_cols)

            # 判断点在哪个区域
            if 0 <= col < 2:
                return "Left"
            elif 2 <= col <= 3:
                return "Middle"
            elif col < 0 or row < 0 or col > 5 or row > 5:
                return "OUT of Range"
            else:
                return "Right"

        # 统计每个区域中点的数量
        region_counts = {"Left": 0, "Middle": 0, "Right": 0, "OUT of Range": 0}
        for point in fixation_points:
            x, y = point[1:]
            region = calculate_region(x, y)
            region_counts[region] += 1

        region_counts = {key: value / len(fixation_points) for key, value in region_counts.items()}

        return region_counts

def M_W_Utest(sample1, sample2):
    # 执行Mann-Whitney U检验
    statistic, p_value = mannwhitneyu(sample1, sample2)

    # 打印结果
    print("Mann-Whitney U检验统计量:", statistic)
    print("p值:", p_value)

    # 根据p值判断是否拒绝原假设
    alpha = 0.05
    if p_value < alpha:
        print("拒绝原假设，样本中位数不相等")
    else:
        print("接受原假设，样本中位数相等")

def blink(csv_path, txt_path):
    # 1920x1080
    # 33 min 23 s
    total_video_len = timedelta(minutes=33, seconds=23)
    width, height = 1920, 1080
    time_window = 70

    video_head_date, head_timestamp, head_datetime = read_txt(txt_path)
    if video_head_date == "None":
        print("error! no head date")
        return
    end_datetime = head_datetime + total_video_len
    end_timestamp = end_datetime.timestamp()

    df = pd.read_csv(csv_path)

    df = df[(df['timestamp'] >= head_timestamp) & (df['timestamp'] <= end_timestamp)]
    df = df.reset_index(drop=True)
    df = df.sort_index()

    device_timestamp, left_gaze_point_validity, right_gaze_point_validity = \
        df['device_time_stamp'], df['left_gaze_point_validity'], df['right_gaze_point_validity']
    device_timestamp = np.array(device_timestamp).astype(float)/1000  #ms
    result = np.logical_or(np.array(left_gaze_point_validity), np.array(right_gaze_point_validity)).astype(int)
    # result = np.logical_and(np.array(left_gaze_point_validity), np.array(right_gaze_point_validity)).astype(int)
    result = np.column_stack((device_timestamp, result))

    # 获取黑暗次数
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
    # 确保在循环结束时也检查最后一组事件的时间差
    if len(queue) > 1:
        time_diff = queue[-1][0] - queue[0][0]
        if time_diff >= time_window:
            count += 1

    return count

def generate_heatmap(points, width, height):
    x = [p[0] for p in points]
    y = [p[1] for p in points]

    # 计算核密度估计
    k = gaussian_kde(np.vstack([x, y]))
    xi, yi = np.mgrid[0:width:complex(width), 0:height:complex(height)]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    heatmap = zi.reshape(xi.shape)

    # 归一化到0-255
    min_value = np.min(heatmap)
    max_value = np.max(heatmap)
    heatmap_normalized = 255 * (heatmap - min_value) / (max_value - min_value)
    heatmap_normalized = np.uint8(np.clip(heatmap_normalized, 0, 255))  # 确保值在0-255之间

    return heatmap_normalized


def heat_map(point_contrix):
    # 1920x1080
    # 33 min 23 s
    width, height = 1920, 1080
    last_str = "AI_yrj"
    # 生成注视热图
    # print(f"point num: {point_contrix.shape[0]}")

    random_indices = np.random.choice(point_contrix.shape[0], size=100, replace=False)
    heatmap = generate_heatmap(point_contrix[random_indices, 1:], width, height)

    # min_value = np.min(heatmap)
    # max_value = np.max(heatmap)
    # print(f"Min value: {min_value}, Max value: {max_value}")

    # 绘制热图
    # 'Spectral_r' 'turbo' 'jet' 'rainbow'

    plt.figure(figsize=(6.4, 3.6), dpi=300)
    ax = plt.subplot()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    plt.imshow(heatmap.T, cmap="turbo", origin='lower', alpha=1)
    # plt.title('Fixation Heatmap')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    plt.savefig(f'heatmap_{last_str}.png', dpi=300)
    plt.show()

    image = cv2.imread('raw.png')
    image2 = cv2.imread(f'heatmap_{last_str}.png')

    result = cv2.addWeighted(image, 1.0, image2, 0.7, 0)  # 0.5是热图的透明度，可以根据需要调整

    # 保存或显示结果
    cv2.imwrite(f'result_{last_str}.png', result)


def main(parent_path):

    person_path = [os.path.join(parent_path, i) for i in os.listdir(parent_path)]
    person_path = [i for i in person_path if os.path.isdir(i)]
    fields = 1  # 1: Ring 2: Side
    regions = {1: "Ring", 2: "Side"}

    total_data = []
    for idx, i in enumerate(person_path):
        print(f"{idx}/{len(person_path)}")
        name = i.split("\\")[-1]

        if os.path.exists(os.path.join(i, "new_raw.csv")):
            raw_path = os.path.join(i, "new_raw.csv")
        else:
            raw_path = os.path.join(i, "raw.csv")
        match2_path = os.path.join(i, "match2.csv")
        txt_path = glob.glob(os.path.join(i, "*.txt"))[0]

        detection = detection_rate(match2_path)
        # print(f"detection rate: {detection}")
        total_distance = eye_travel_distance(raw_path, txt_path)
        # print(f"total distance: {total_distance}")
        fixed_points = fixation_points(raw_path, txt_path)
        region_counts_1 = fixations_ratio(fixed_points, fields=1)
        region_counts_2 = fixations_ratio(fixed_points, fields=2)
        # print(f"Region: {regions[fields]} Counts: {region_counts}")
        blink_count = blink(raw_path, txt_path)
        # print(f"Blink Count: {blink_count}")
        heat_map(fixed_points)

        total_data.append([name, detection, total_distance, len(fixed_points), blink_count, region_counts_1, region_counts_2])
        # exit(-1)

    # 指定要保存的 CSV 文件路径
    # csv_file_path = parent_path+"_eval.csv"
    # # 打开 CSV 文件进行写操作
    # with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    #     writer = csv.writer(file)
    #
    #     # 写入 CSV 文件的标题行
    #     writer.writerow(["姓名", "检出率", "眼动距离", "Fixation", "Blink frequencies", "Ring", "Side"])
    #
    #     # 写入每个 .txt 文件的路径
    #     for file_path in total_data:
    #         writer.writerow(file_path)

def combine(parent_path):
    # 读取 XLSX 文件
    xlsx_file_path = parent_path+"_反应时间.xlsx"
    df_xlsx = pd.read_excel(xlsx_file_path)

    # 读取 CSV 文件
    csv_file_path = parent_path+"_eval.csv"
    df_csv = pd.read_csv(csv_file_path)

    # 合并两个数据框，基于第一列（假设为姓名列）
    # 假设第一列的列名是 'Name'，根据实际文件列名调整
    merged_df = pd.merge(df_xlsx, df_csv, on='姓名', how='outer')

    # 按照 XLSX 文件中的行顺序排序
    # 这里假设 df_xlsx 中的索引可以用来排序
    merged_df['sort_order'] = pd.Categorical(merged_df['姓名'], categories=df_xlsx['姓名'], ordered=True)
    merged_df.sort_values('sort_order', inplace=True)
    merged_df.drop('sort_order', axis=1, inplace=True)

    # 保存合并后的数据框到一个新的 CSV 文件
    output_file_path = parent_path + '_merged.csv'
    merged_df.to_csv(output_file_path, index=False)

    print(f"save to {output_file_path}")

def handle(parent_path, dst_path):
    # 读取 CSV 文件
    csv_file_path = parent_path
    df_xlsx = pd.read_excel(csv_file_path)

    hans2eng = {"学生": "Graduate students ", "护士": "Endoscopy nurses", "低年": "Novice endoscopists", "高年": "Senior endoscopists"}

    level, Ring, Side = list(df_xlsx["熟练程度/身份判定"]), list(df_xlsx["Ring"]), list(df_xlsx["Side"])
    data = []
    columns = ["level", "Ring", "% of Total(Ring)", "Side", "% of Total(Side)"]
    for idx, _ in enumerate(level):
        ring_dict = eval(Ring[idx])
        ring_dict.pop("OUT of Range")
        side_dict = eval(Side[idx])
        side_dict.pop("OUT of Range")
        ring_keys, side_keys = list(ring_dict.keys()), list(side_dict.keys())

        for j in range(3):
            data.append([hans2eng[level[idx]], ring_keys[j].split(" ")[-1], float(ring_dict[ring_keys[j]])*100, side_keys[j], float(side_dict[side_keys[j]])*100])

    df = pd.DataFrame(data, columns=columns)
    # 绘制箱线图
    fields = "Side"
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=fields, y=f"% of Total({fields})", hue="level", data=df, palette="flare")
    plt.xlabel(fields)
    plt.ylabel('% of Total Fixations')
    plt.savefig(dst_path+rf'\Normal control_{fields}.png', dpi=300)
    plt.show()

if __name__ == "__main__":


    parent_path = r"XXX\眼动-回顾\CADe-assisted"  #or  "XXX\眼动-回顾\Normal control"
    main(parent_path)
    # combine(parent_path)

    # parent_path = r"XXX\Normal control.xlsx"
    # dst_path = "XXX"
    # handle(parent_path, dst_path)