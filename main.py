import os
import csv
from Evaluation import detection_rate, eye_travel_distance, fixation_points, fixations_ratio, blink, heat_map
from Evaluation import get_match_file, cal_reaction_time, handle

def main(parent_path, output_path, last_str="None", draw=False):

    person_path = [os.path.join(parent_path, i) for i in os.listdir(parent_path)]
    person_path = [i for i in person_path if os.path.isdir(i)]

    total_data = []
    for idx, i in enumerate(person_path):
        print(f"{idx}/{len(person_path)}")
        name = i.split(os.sep)[-1]

        raw_path = os.path.join(i, "time_gaze.csv")
        match_path = os.path.join(output_path, name, "match.csv")

        detection = detection_rate(match_path)
        # print(f"detection rate: {detection}")
        total_distance = eye_travel_distance(raw_path)
        # print(f"total distance: {total_distance}")
        fixed_points = fixation_points(raw_path)
        region_counts_1 = fixations_ratio(fixed_points, fields=1)  # regions = {1: "Ring", 2: "Side"}
        region_counts_2 = fixations_ratio(fixed_points, fields=2)
        # print(f"Region: {regions[fields]} Counts: {region_counts}")
        blink_count = blink(raw_path)
        # print(f"Blink Count: {blink_count}")
        if draw:
            heat_map(fixed_points, last_str)

        total_data.append([name, detection, total_distance, len(fixed_points), blink_count, region_counts_1, region_counts_2])

    csv_file_path = output_path+"_eval.csv"

    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        writer.writerow(["name", "Detection rate", "Eye movement distance", "Fixation", "Blink frequencies", "Ring", "Side"])

        for file_path in total_data:
            writer.writerow(file_path)

def get_file(parent_path, calibrate_csv_path, output_path):

    person_path = [os.path.join(parent_path, i) for i in os.listdir(parent_path)]
    person_path = [i for i in person_path if os.path.isdir(i)]
    for idx, i in enumerate(person_path):
        get_match_file(i, calibrate_csv_path, output_path)
        cal_reaction_time(output_path) #Calculate reaction time

if __name__ == "__main__":

    calibrate_csv_path = r"XXX\retrospective\polyp-presence-period.csv"
    parent_path = r"XXX\retrospective\CADe-assisted"  #or  "XXX\retrospective\Normal control"
    output_path = r".\outputs\retrospective\CADe-assisted"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    get_file(parent_path, calibrate_csv_path, output_path)
    main(parent_path=parent_path, output_path=output_path, draw=True)

    parent_path = r".\outputs\retrospective\CADe-assisted_eval.csv"
    dst_path = r'.\imgs\CADe-assisted_Side.png'
    handle(parent_path, dst_path, fields="Side") # Side or Ring