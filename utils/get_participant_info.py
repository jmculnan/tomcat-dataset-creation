# get participant information from the .metadata files
import json
import logging
from pathlib import Path


class MetadataToParticipantInfo:
    def __init__(self, base_path):
        # path to metadata files
        self.base_path = base_path

    def _get_participant_info(self, json_path):
        """
        Get the participant info out of a single json file
        :param json_file:
        :return:
        """
        participant_info = []
        print(json_path)
        with open(json_path) as json_file:
            # first line is a header line but doesn't contain
            # all the required information
            json_file.readline()

            for l in json_file:
                # if line is the metadata line with IDs
                theline = json.loads(l)
                if "header" in theline.keys() and 'name' in theline['data'].keys():
                    thedata = theline['data']

                    # get relevant info from this
                    try:
                        team, exp = thedata['name'].split("_")
                    except ValueError:
                        splitname = thedata['name'].split("_")
                        if len(splitname) > 2:
                            team = splitname[0]
                            exp = splitname[1]
                        else:
                            print(splitname)
                            exit()

                    players = thedata['client_info']
                    for player in players:
                        p_id = player['participant_id']
                        p_name = player['playername']

                        participant_info.append([team, exp, p_id, p_name])

                    break

        return participant_info

    def get_info_on_multiple_trials(self):
        """
        Get information about multiple trials
        All trials' metadata should be stored in the same
            directory
        :param metadata_dir_path:
        :return:
        """
        all_participant_info = []
        for f in Path(self.base_path).iterdir():
            if f.suffix == ".metadata":
                p_info = self._get_participant_info(str(f))
                all_participant_info.extend(p_info)
            else:
                print(f"{f} not added to participant info")

        return all_participant_info

    def return_trial_player_dict(self, participant_info):
        """
        Return a dict of Trial_ID: {Playername: Participant ID}
        Used in conjunction with data compilation utils
        To switch out player names for participant IDs
        :return:
        """
        tp_dict = {}
        if type(participant_info[0] == list):
            for trial in participant_info:
                if trial[1] not in tp_dict.keys():
                    tp_dict[trial[1]] = {trial[3]: trial[2]}
                else:
                    tp_dict[trial[1]][trial[3]] = trial[2]

        return tp_dict

    def save_participant_info(self, participant_info, save_location):
        """
        Save extracted participant information
        :param save_location: path + name of file to save participant info
        :return:
        """
        participant_info.to_csv(save_location, index=False)

        # print(participant_info)
        # with open(save_location, 'w') as f:
        #     # write header
        #     f.write("Team_ID,Trial_ID,participantid,playername\n")
        #     for p_info in participant_info:
        #         print(p_info)
        #         for player_info in p_info:
        #             print(player_info)
        #             f.write(",".join(player_info))
        #             f.write("\n")


def add_scores_to_participant_info(participant_df, scores_df):
    """
    Add scores from mission to the participant data
    :param scores_df:
    :return:
    """
    participant_df['Score'] = participant_df['Trial_ID'].map(scores_df.set_index('Trial_ID')['Score'])

    return participant_df

if __name__ == "__main__":
    import pandas as pd

    base_path = f"/media/jculnan/backup/jculnan/datasets/asist_data2"
    participant_info = f"{base_path}/participant_info.csv"
    scores = f"{base_path}/scores.csv"

    part = pd.read_csv(participant_info)
    sc = pd.read_csv(scores)

    print(add_scores_to_participant_info(part, sc))