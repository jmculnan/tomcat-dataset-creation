import pandas as pd
import sys, os


class ToMCATSentEmoPrep:
    def __init__(self, base_dir):
        # the directory within which to work
        self.base_dir = base_dir

        # holder for tipi results
        self.tipi_dir = f"{base_dir}/tipi"
        # holder for sent-emo annotated files
        self.annotations_dir = f"{base_dir}/sent-emo"

        # holder for corrected transcriptions
        self.correct_times_dir = f"{base_dir}/corrected_transcriptions"

    def combine_corrected_and_annotated(self):
        # go through items that exist for corrected transcriptions
        for item in os.listdir(self.annotations_dir):
            if item.endswith(".csv"):
                # get name of csv
                csv_name = item.rsplit("/", 1)[0]

                # open as pandas df
                item_df = pd.read_csv(f"{self.annotations_dir}/{item}")
                item_df = item_df[["participant", "utt", "emotion", "sentiment", "start_timestamp"]]

                # check for equivalent item in corrected transcriptions
                if os.path.isfile(f"{self.correct_times_dir}/{csv_name}"):
                    print(f"{self.correct_times_dir}/{csv_name}")
                    sent_df = pd.read_csv(f"{self.correct_times_dir}/{csv_name}")
                    sent_df = sent_df[["participant", "utt", "corr_utt", "start_timestamp", "end_timestamp"]]
                    sent_df = sent_df[sent_df["utt"].notna()]

                    # combine items when there is an annotation for the utt
                    # join should probably be inner
                    combined_df = pd.merge(item_df, sent_df, on=["participant", "utt"], how="left")
                    # combined_df = item_df.join(sent_df, on=["participant", "utt"])

                    # either save combined df or keep it in memory to deal with later
                    combined_df.to_csv(f"{self.base_dir}/combined/{csv_name}", index=False)

    def combine_files(self):
        """
        Combine files of corrected/annotated data
        """
        all_files = None
        # go through items that exist for corrected transcriptions
        for item in os.listdir(self.annotations_dir):
            if item.endswith(".csv"):
                print(item)
                # get name of csv
                csv_name = item.rsplit("/", 1)[0]

                csv_name_info = csv_name.split("_")
                trial_id = csv_name_info[2][-7:]
                team_id = csv_name_info[3][-8:]

                data = pd.read_csv(f"{self.annotations_dir}/{item}")

                data['trial_id'] = trial_id
                data['team_id'] = team_id

                if all_files is None:
                    all_files = data
                else:
                    all_files = pd.concat([all_files, data], axis=0)

        # todo: change to if saving required
        all_files.to_csv(f"{self.base_dir}/all_sent-emo.csv", index=False)

        return all_files

    def read_in_tipi(self):
        all_tipi = None
        for item in os.listdir(self.tipi):
            tipi_df = pd.read_csv(f"{self.tipi}/{item}")
            # check if there is already a tipi df
            if all_tipi == None:
                all_tipi = tipi_df
            else:
                # join them together if needed
                all_tipi = pd.join(tipi_df, all_tipi)

    def count_all_emotions(self, all_data):
        """
        Get class counts for all emotion classes in all the data
        all_data : a pandas df containing the data
        """
        return all_data['emotion'].value_counts()

    def count_all_sentiments(self, all_data):
        """
        Get class counts for all sentiment classes in all the data
        """
        return all_data['sentiment'].value_counts()

    def count_all_traits(self, all_data):
        """
        Get class counts for all personality trait classes in the data
        """
        return all_data['trait'].value_counts()
