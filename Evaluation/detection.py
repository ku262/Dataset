import pandas as pd
import numpy as np

def detection_rate(csv_path):

    df = pd.read_csv(csv_path)
    reaction_time = np.array(list(df['reaction time']))[1:].astype(float)
    ratio = np.count_nonzero(reaction_time) / reaction_time.shape[0]

    return ratio