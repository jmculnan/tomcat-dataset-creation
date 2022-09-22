import sys
sys.path.append("/home/jculnan/github/tomcat-dataset-creation")

from config import data_prep_config as config

from utils.get_participant_info import MetadataToParticipantInfo, add_scores_to_participant_info
from utils.combine_gold_labels import combine_group_of_sent_df_with_personality
from utils.data_compilation_utils import ToMCATDatasetPrep
from utils.preprocess_tipi_gold_labels import preprocess_tipi, add_traits_to_id_df

import pandas as pd



# get the participant info from metadata
partinfo = MetadataToParticipantInfo(config.metadata_path)

all_part_info = partinfo.get_info_on_multiple_trials()
all_part_df = pd.DataFrame(all_part_info, columns=["Team_ID", "Trial_ID", "participantid", "playername"])

# get tipi labels
gold_data = pd.read_csv(config.tipi_survey)
tipi_results = preprocess_tipi(gold_data)

# combine
participant_tipi_info = add_traits_to_id_df(all_part_df, tipi_results)

scores_df = pd.read_csv(config.scores)
participant_tipi_info = add_scores_to_participant_info(participant_tipi_info, scores_df)

partinfo.save_participant_info(participant_tipi_info, config.participant_save_path)
exit()

# get the dict to convert playername to participant ID
pid2namedict = partinfo.return_trial_player_dict(all_part_info)

# contains columns:  team_id	trial_id	participantid	participant_role	role_name	max_trait
tipi = pd.read_csv(config.tipi_file)

# combine sent with personality
combine_group_of_sent_df_with_personality(config.sent_path, tipi)

# create dataset / replace playername with participant ID
prep_obj = ToMCATDatasetPrep(config.base_path)
prep_obj.convert_values(pid2namedict)
prep_obj.save_merged()