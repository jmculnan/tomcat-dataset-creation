# look at correlation between outcomes and annotations

# get required packages
require(ggplot2)
require(dplyr)

# set working directory
setwd("/media/jculnan/backup/jculnan/datasets/asist_data2")

# load data
utt_data <- read.csv("combined_data_CORRECT9.21.22.csv")
utt_data <- utt_data[utt_data$sentiment != "nan" & utt_data$emotion != "nan",]
colnames(utt_data) <- c('message_id', 'participantid', 'utt', 'start_time', 'end_time', 'corr_utt', 'label', 'AP', 'notes', 'sentiment', 'emotion', 'Trial_ID', 'Team_ID')
trial_data <- read.csv("participant_info.csv")

# combine 
data <- merge(utt_data, trial_data, on="Trial_ID")

# get proportion of items per class for each trial
sent_data <- data %>%
  group_by(Trial_ID, Score, sentiment) %>%
  summarise(n = n()) %>%
  mutate(freq = n/sum(n), logfreq = log(n/sum(n)))

emo_data <- data %>%
  group_by(Trial_ID, Score, emotion) %>%
  summarise(n = n()) %>%
  mutate(freq = n/sum(n), logfreq = log(n/sum(n)))

# visualize correlation between these
sent_score <- ggplot(data=sent_data, aes(x=freq, y=Score, color=sentiment)) + 
  geom_point(stat="identity")

log_sent_score <- ggplot(data=sent_data, aes(x=logfreq, y=Score, color=sentiment)) + 
  geom_point(stat="identity")

emo_score <- ggplot(data=emo_data, aes(x=freq, y=Score, color=emotion)) +
  geom_point(stat="identity")

# USE THESE --- THEY ARE HELPFUL
########################################################
emo_facet <- ggplot(data=emo_data, aes(x=freq, y=Score, color=emotion)) +
  geom_point(stat="identity") + 
  geom_smooth(method="lm") +
  facet_grid(rows= vars(emotion))

non_neutral_emo_facet <- ggplot(data=emo_data[emo_data$emotion != "neutral",], aes(x=freq, y=Score, color=emotion)) +
  geom_point(stat="identity") +
  geom_smooth(method="lm") +
  facet_grid(rows=vars(emotion))

sent_facet <- ggplot(data=sent_data, aes(x=freq, y=Score, color=sentiment)) + 
  geom_point(stat="identity") +
  geom_smooth(method="lm") +
  facet_grid(rows=vars(sentiment))
#########################################################

# joy
joy_log_score <- ggplot(data=emo_data[emo_data$emotion=="joy",], aes(x=logfreq, y=Score)) + 
  geom_point(stat="identity")

joy_score <- ggplot(data=emo_data[emo_data$emotion=="joy",], aes(x=freq, y=Score)) +
  geom_point(stat="identity")

# anger
anger_score <- ggplot(data=emo_data[emo_data$emotion=="anger",], aes(x=freq, y=Score)) + 
  geom_point(stat="identity")

# disgust 
disgust_score <- ggplot(data=emo_data[emo_data$emotion=="disgust",], aes(x=freq, y=Score)) +
  geom_point(stat="identity")

# fear
fear_score <- ggplot(data=emo_data[emo_data$emotion=="fear",], aes(x=freq, y=Score)) + 
  geom_point(stat="identity")

# sadness
sadness_score <- ggplot

# surprise


log_emo_score <- ggplot(data=emo_data, aes(x=logfreq, y=Score, color=emotion)) + 
  geom_point(stat="identity")
