# config file for data prep

import sys
# append to path if needed
sys.path.append("/home/jculnan/github/tomcat-dataset-creation")

# save base path
base_path = "/media/jculnan/backup/jculnan/datasets/asist_data2"

# location of metadata files
metadata_path = f"{base_path}/metadata"

# where to save participant information
participant_save_path = f"{base_path}/participant_info.csv"

# tipi results
tipi_survey = f"{base_path}/HSRData_Surveys0Numeric_Trial-na_Team-na_Member-na_CondBtwn-na_CondWin-na_Vers-07132022.csv"
tipi_file = f"{base_path}/personality_traits_for_all_participants.csv"
sent_path = f"{base_path}/sent-emo"

# file containing scores (from get_scores_from_metadata.py)
scores = f"{base_path}/scores.csv"