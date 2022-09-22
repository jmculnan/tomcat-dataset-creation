# concatenate files
import sys
import os
import pandas as pd


def combine_files(annotations_dir, base_dir, dropna=False):
    """
    Combine files of corrected/annotated data
    """
    all_files = None
    # go through items that exist for corrected transcriptions
    for item in os.listdir(annotations_dir):
        if item.endswith(".csv"):
            print(item)
            # get name of csv
            csv_name = item.rsplit("/", 1)[0]

            csv_name_info = csv_name.split("_")
            trial_id = csv_name_info[2].split("-")[-1]
            team_id = csv_name_info[3].split("-")[-1]

            data = pd.read_csv(f"{annotations_dir}/{item}")
            print(data.columns)

            data['trial_id'] = trial_id
            data['team_id'] = team_id

            data = fix_labels(data)

            # if we want to remove null sent/emo
            if dropna:
                data.dropna(subset=['sentiment', 'emotion'], inplace=True)

            if all_files is None:
                all_files = data
            else:
                all_files = pd.concat([all_files, data], axis=0)

    # todo: change to if saving required
    all_files.to_csv(f"{base_dir}/all_sent-emo.csv", index=False)

    return all_files


def fix_labels(df):
    df['emotion'] = df['emotion'].apply(lambda x: str(x).strip())
    df['sentiment'] = df['sentiment'].apply(lambda x: str(x).strip())

    errors = {"angry": "anger", 'neural': 'neutral', 'neutrl': 'neutral', "suprise": "surprise",
              "sadeness": "sadness", "netural": "neutral", "negatve": "negative",
              "positie": "positive", "negatie": "negative", "poitive": "positive",
              "nutral": "neutral", "neative": "negative", "postive": "positive",
              "postivie": "positive", "n": "neutral", "netral": "neutral", "negtive": "negative",
              "posiive": "positive"}

    df["emotion"] = df["emotion"].apply(lambda x: errors[x] if x in errors.keys() else x)
    df["sentiment"] = df["sentiment"].apply(lambda x: errors[x] if x in errors.keys() else x)

    return df


if __name__ == "__main__":
    base_dir = "/media/jculnan/backup/jculnan/datasets/asist_data2"
    annotations = "/media/jculnan/backup/jculnan/datasets/asist_data2/combined"

    combine_files(annotations, base_dir, dropna=True)
