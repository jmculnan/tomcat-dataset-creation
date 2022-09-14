# combine sent-emo with tipi scores

import pandas as pd
import os


def combine_group_of_sent_df_with_personality(sent_path, tipi_df):
    """
    Used to create separate combined files for each trial
    Saves the new files with names of sent-emo files
    """
    for f in os.listdir(sent_path):
        if f.endswith(".csv"):
            fname = f.split(".csv")[0]
            sent_df = pd.read_csv(f"{sent_path}/{f}")
            sent_df["trial_id"] = fname.split("_")[2][-7:]
            sent_df["team_id"] = fname.split("_")[3][-8:]
            sent_df['real_start'] = sent_df['start_timestamp'].apply(lambda x: get_start_in_sec(x))
            sent_df['real_end'] = sent_df['end_timestamp'].apply(lambda x: get_end_in_sec(x))

            this_combined = combine_sent_with_personality(sent_df, tipi_df)
            this_combined.dropna(subset=['sentiment', 'emotion'], inplace=True)

            this_combined.to_csv(f"{sent_path}/{fname.split('_')[2]}_{fname.split('_')[3]}_gold.csv", index=False)


def combine_sent_with_personality(sent_df, tipi_df):
    """
    Because the personality gold labels come from the
    participants themselves, you must add them to
    the csv with all sent/emo annotations
    """
    tipi_df.rename(columns={"participant_role": "participant"}, inplace=True)

    sent_df = sent_df.merge(tipi_df, on=["trial_id", "team_id", "participant"], sort=False)

    return sent_df


def get_start_in_sec(time_str):
    time = get_sec(time_str)
    return time - 0.500 - 120  # first 2 minutes are always missing


def get_end_in_sec(time_str):
    time = get_sec(time_str)
    return time + 0.500 - 120  # first 2 minutes are always missing


def get_sec(time_str):
    """Get seconds from time."""
    time_components = time_str.split(":")
    if len(time_components) < 2:
        return float(time_components[0])
    elif len(time_components) == 2:
        return int(time_components[0]) * 60 + float(time_components[1])
    else:
        # sometimes string has trailing 0s--ignore these
        return 60 * int(time_components[0]) + float(time_components[1])


if __name__ == "__main__":
    # contains columns:  team_id	trial_id	participantid	participant_role	role_name	max_trait
    tipi_path = "/media/jculnan/backup/jculnan/asist_data/personality/personality_traits_with_participant_info.csv"
    tipi = pd.read_csv(tipi_path)
    sent_path = "/home/jculnan/asist_data/sent-emo/for_PI_meeting_07.22"

    combine_group_of_sent_df_with_personality(sent_path, tipi)
