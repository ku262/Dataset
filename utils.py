import numpy as np
from datetime import datetime
from scipy.stats import mannwhitneyu
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')
import seaborn as sns

def str2contrix(x):

    res = []
    for i in x:
        res.append([float(i.split(", ")[0].split("(")[-1]), float(i.split(", ")[1].split(")")[0])])
    res = np.array(res)
    return res

def read_txt(file_path):
    with open(file_path, 'r') as file:
        # read
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

def M_W_Utest(sample1, sample2):
    # Mann-Whitney U
    statistic, p_value = mannwhitneyu(sample1, sample2)

    print("Mann-Whitney U:", statistic)
    print("p:", p_value)

    alpha = 0.05
    if p_value < alpha:
        print("Rejecting the null hypothesis, the median of the sample is not equal")
    else:
        print("Accept the null hypothesis and ensure that the median of the sample is equal")

def combine(parent_path):

    xlsx_file_path = parent_path+"_reaction time.xlsx"
    df_xlsx = pd.read_excel(xlsx_file_path)

    csv_file_path = parent_path+"_eval.csv"
    df_csv = pd.read_csv(csv_file_path)

    merged_df = pd.merge(df_xlsx, df_csv, on='name', how='outer')

    merged_df['sort_order'] = pd.Categorical(merged_df['name'], categories=df_xlsx['name'], ordered=True)
    merged_df.sort_values('sort_order', inplace=True)
    merged_df.drop('sort_order', axis=1, inplace=True)

    output_file_path = parent_path + '_merged.csv'
    merged_df.to_csv(output_file_path, index=False)

    print(f"save to {output_file_path}")

def handle(parent_path, dst_path, fields = "Side"):

    csv_file_path = parent_path
    df = pd.read_csv(csv_file_path)

    hans2eng = {"student": "Graduate students ", "nurse": "Endoscopy nurses", "novice": "Novice endoscopists", "senior": "Senior endoscopists"}

    level, Ring, Side = list(df["name"]), list(df["Ring"]), list(df["Side"])
    level = [i.split("-")[0] for i in level]
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


    plt.figure(figsize=(8, 6))
    sns.boxplot(x=fields, y=f"% of Total({fields})", hue="level", data=df, palette="flare")
    plt.xlabel(fields)
    plt.ylabel('% of Total Fixations')
    plt.savefig(dst_path, dpi=300)
    plt.show()