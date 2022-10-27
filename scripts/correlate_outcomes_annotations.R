# look at correlation between outcomes and annotations

# get required packages
require(ggplot2)
require(GGally)
require(dplyr)
require(tidyr)
require(ggpubr)
require(stringr)

# set working directory
setwd("/media/jculnan/backup/jculnan/datasets/asist_data2")

# load data
utt_data <- read.csv("overall_sent-emo.csv")

# older version had string nan in text, cleaned up version does not 
utt_data <- utt_data[utt_data$sentiment != "nan" & utt_data$emotion != "nan",]
utt_data <- utt_data[utt_data$sentiment != "" & utt_data$emotion != "",]

colnames(utt_data) <- c('AP', 'corr_utt', 'emotion', 'end_time', 
                        'label', 'message_id', 'notes', 'participantid', 
                        'sentiment', 'start_time', 'Team_ID', 'Trial_ID', 'utt')

trial_data <- read.csv("participant_info.csv")


#####################################################
   ########### PREPARING DATAFRAMES  ##############
#####################################################

# combine the tipi + utt data
data <- merge(utt_data, trial_data, on="Trial_ID")

odd = c("1","3","5","7","9")
data$trial_num <- ifelse((str_sub(data$Trial_ID, start=-1) %in% odd), 1, 2)
data$trial_num <- as.factor(data$trial_num)

# get proportion of items per class for each trial
sent_data_by_participant <- data %>%
  group_by(Trial_ID, participantid, Score, sentiment, trial_num) %>%
  summarise(n = n()) %>%
  mutate(freq = n/sum(n), logfreq = log(n/sum(n)))

sent_data <- data %>%
  group_by(Trial_ID, Score, sentiment) %>%
  summarise(n = n()) %>%
  mutate(freq = n/sum(n), logfreq = log(n/sum(n)))

emo_data_by_participant <- data %>%
  group_by(Trial_ID, participantid, Score, emotion) %>%
  summarise(n = n()) %>%
  mutate(freq = n/sum(n), logfreq = log(n/sum(n)))

emo_data  <- data %>%
  group_by(Trial_ID, Score, emotion) %>%
  summarise(n = n()) %>%
  mutate(freq = n/sum(n), logfreq = log(n/sum(n)))

# convert wide to long 
trait_data <- trial_data %>%
  gather(key="trait", value="trait_score", extroversion, conscientiousness, agreeableness, neuroticism, openness)

all_trait_data <- trait_data %>%
  group_by(Trial_ID, trait, Score) %>%
  summarise(avg_score = mean(trait_score))

max_trait_data <- trial_data %>%
  group_by(Trial_ID, max_trait)


# prepare predictions data 
pred_data1 <- read.csv("~/github/tomcat-speech/output/ToMCAT_resultstest/results/predictions_1.csv")
pred_data2 <- read.csv("~/github/tomcat-speech/output/ToMCAT_resultstest/results/predictions_2.csv")
pred_data <- rbind(pred_data1, pred_data2)
colnames(pred_data) <- c("trait", "emotion", "sentiment", "message_id")

labs <- c("participantid", "Team_ID", "Trial_ID", "message_id", "Score")
important_labels <- data[labs]

pred_data <- merge(pred_data, important_labels, on="message_id")
pred_data$emotion <- as.factor(pred_data$emotion)
pred_data$sentiment <- as.factor(pred_data$sentiment)
pred_data$trait <- as.factor(pred_data$sentiment)

sent_pred_data <- pred_data %>%
  group_by(Trial_ID, Score, sentiment) %>%
  summarise(n = n()) %>%
  mutate(freq = n/sum(n), logfreq = log(n/sum(n)))

emo_pred_data <- pred_data %>%
  group_by(Trial_ID, Score, emotion) %>%
  summarise(n = n()) %>%
  mutate(freq = n/sum(n), logfreq = log(n/sum(n)))


#####################################################
########### NEWER ANALYSIS STARTS HERE ##############
#####################################################
# -----------first analysis: line XXX --------

emo_preds_facet <- ggplot(data=emo_pred_data, aes(x=logfreq, y=Score, color=emotion)) +
  geom_point(stat="identity") + 
  geom_smooth(method="lm") +
  facet_grid(rows= vars(emotion)) +
  stat_cor(aes(label=..rr.label..), color="black", label.x.npc = "center")

emo_counts_facet <- ggplot(data=emo_data, aes(x=n, y=Score, color=emotion)) +
  geom_point(stat="identity") + 
  geom_smooth(method="lm") +
  facet_grid(rows= vars(emotion)) +
  stat_cor(aes(label=..rr.label..), color="black", label.x.npc = "center")

emo_counts_facet_id <- ggplot(data=emo_data_by_participant, aes(x=n, y=Score, color=emotion)) +
  geom_point(stat="identity") + 
  geom_smooth(method="lm") +
  facet_grid(rows= vars(emotion)) +
  stat_cor(aes(label=..rr.label..), color="black", label.x.npc = "center")


require(lmerTest)

# not quite sure why we need to subtract different numbers here
# did manual checks to find when lowest value == 0
data$emo.f <- as.factor(as.numeric(data$emotion) - 3) 
data$sent.f <- as.factor(as.numeric(data$sentiment) - 2)
data$trait.f <- as.factor(as.numeric(data$max_trait) - 1)

outcomes <- lmer(Score ~ emo.f*trait.f*trial_num + (1 | emo.f) + 
                    (1 | trial_num) + (1 | trait.f), data=data,
                 control=lmerControl(optCtrl=list(maxfun=100000),
                                     optimizer="bobyqa"))

outcomes_tester <- lmer(Score ~ emo.f + trial_num + (1 | emo.f), data=data)
# THIS GIVES US A SIGNIFICANT CORRELATION
outcomes <- lm(Score ~ trial_num*sentiment*emotion*max_trait, data=data)


require(corrplot)
correlated_data <- select(data, c("emotion", "sentiment", "max_trait", "Score"))

corred <- corrplot(correlated_data)

corr_data <- data %>%
  group_by(Trial_ID, participantid, Score, max_trait) %>%
  summarize(utts=n(),
            positive_c=length(which(sentiment=="positive")),
            positive=positive_c/n(),
            negative_c=length(which(sentiment=="negative")),
            negative=negative_c/n(),
            neutral_sent_c=length(which(sentiment=="neutral")),
            neutral_sent=neutral_sent_c/n(),
            anger_c=length(which(emotion=="anger")),
            anger=anger_c/n(),
            disgust_c=length(which(emotion=="disgust")),
            disgust=disgust_c/n(),
            fear_c=length(which(emotion=="fear")),
            fear=fear_c/n(),
            joy_c=length(which(emotion=="joy")),
            joy=joy_c/n(),
            neutral_emo_c=length(which(emotion=="neutral")),
            neutral_emo=neutral_emo_c/n(),
            sadness_c=length(which(emotion=="sadness")),
            sadness=sadness_c/n(),
            surprise_c=length(which(emotion=="surprise")),
            surprise=surprise_c/n(),
            agree=length(which(max_trait=="agreeableness"))/ (3 * 100),
            consc=length(which(max_trait=="conscientiousness"))/300,
            extro=length(which(max_trait=="extroversion"))/300,
            neur=length(which(max_trait=="neuroticism"))/300,
            open=length(which(max_trait=="openness"))/300,
  )
corr_data$trial_num <- c(1,2)[(sub("(\d)$", "\\1", corr_data$Trial_ID) )]

require(stringr)

odd = c("1","3","5","7","9")
corr_data$trial_num <- ifelse((str_sub(corr_data$Trial_ID, start=-1) %in% odd), 1, 2)
corr_data$trial_num <- as.factor(corr_data$trial_num)

# utts vs score
utts_score <- ggplot(data=corr_data, aes(x=utts, y=Score)) +
  geom_point(stat="identity") + 
  geom_smooth(method="lm") +
  stat_cor(aes(label=..rr.label..), color="black", label.x.npc = "center") +
  facet_grid(rows=vars(trial_num))

#emosent v score
emosent_score <- ggplot(data=corr_data, aes(x=interaction(), y=Score)) +
  geom_point(stat="identity") + 
  geom_smooth(method="lm") +
  stat_cor(aes(label=..rr.label..), color="black", label.x.npc = "center")

joy_neutralsent_score <-  ggplot(data=corr_data, 
                                      aes(x=interaction(joy, neutral_sent), 
                                          y=Score, color=trial_num)) +
  geom_point(stat="identity") + 
  geom_smooth(method="lm") +
  facet_grid(vars(trial_num)) +
  stat_cor(aes(label=..rr.label..), color="black", label.x.npc = "center")

surprise_neutralsent_score <-  ggplot(data=corr_data, 
                                 aes(x=interaction(surprise_c, neutral_sent_c), 
                                     y=Score, color=trial_num)) +
  geom_point(stat="identity") + 
  geom_smooth(method="lm") +
  facet_grid(vars(trial_num)) +
  stat_cor(aes(label=..rr.label..), color="black", label.x.npc = "center")
  
#####################################################
########### FIRST ANALYSIS STARTS HERE ##############
#####################################################

trait_facet <- ggplot(data=trait_data, aes(x=avg_score, y=Score, color=trait)) +
  geom_point(stat="identity") + 
  geom_smooth(method="lm") +
  facet_grid(rows= vars(trait))

# FACET GRIDS OF SINGLE VARIABLE AGAINST SCORE
########################################################

emo_facet <- ggplot(data=emo_data, aes(x=freq, y=Score, color=emotion)) +
  geom_point(stat="identity") + 
  geom_smooth(method="lm") +
  facet_grid(rows= vars(emotion)) +
  stat_cor(aes(label=..rr.label..), color="black", label.x.npc = "center")




#######################################################
###           LOG VS NON-LOG DATA            ##########
#######################################################
##### BY PARTICIPANT ##################################

emo_facet_id <- ggplot(data=emo_data_by_participant, aes(x=freq, y=Score, color=emotion)) +
  geom_point(stat="identity") + 
  geom_smooth(method="lm") +
  facet_grid(rows= vars(emotion)) +
  stat_cor(aes(label=..rr.label..), color="black", label.x.npc = "center")

log_emo_facet_id <- ggplot(data=emo_data_by_participant, aes(x=log(freq), y=Score, color=emotion)) +
  geom_point(stat="identity") + 
  geom_smooth(method="lm") +
  facet_grid(rows= vars(emotion)) +
  stat_cor(aes(label=..rr.label..), color="black", label.x.npc = "center")


#######################################################
############### PAIRS PLOTS  ##########################
#######################################################

# get overall data and calculate percentages of each feature
this_data <- data %>%
  group_by(Trial_ID, participantid, Score, max_trait, trial_num) %>%
  summarize(positive=length(which(sentiment=="positive"))/n(),
            negative=length(which(sentiment=="negative"))/n(),
            neutral_sent=length(which(sentiment=="neutral"))/n(),
            anger=length(which(emotion=="anger"))/n(),
            disgust=length(which(emotion=="disgust"))/n(),
            fear=length(which(emotion=="fear"))/n(),
            joy=length(which(emotion=="joy"))/n(),
            neutral_emo=length(which(emotion=="neutral"))/n(),
            sadness=length(which(emotion=="sadness"))/n(),
            surprise=length(which(emotion=="surprise"))/n(),
            )
# bin scores for visualization purposes
this_data$score_bin = cut(this_data$Score, 
                          breaks=c(min(this_data$Score),
                                   quantile(this_data$Score, .33),
                                   quantile(this_data$Score, .67),
                                   max(this_data$Score)),
                          labels=c("Low", "Medium", "High"))

this_data$score_bin2 <- cut(this_data$Score,
                            breaks=c(min(this_data$Score),
                                         quantile(this_data$Score, .50),
                                         max(this_data$Score)),
                            labels=c("Low", "High"))

#####################################################
#### compare sentiments for 3 score bins   ##########
#####################################################
sentiment_pairs <- ggpairs(data=this_data,
        columns = c("positive", "negative", "neutral_sent", "Score"),
        aes(color=trial_num, alpha=0.5))

emotion_pairs <- ggpairs(data=this_data,
                         columns=c("anger", "disgust", "fear", "joy",
                                   "neutral_emo", "sadness", "surprise",
                                   "Score"),
                         aes(color=trial_num, alpha=0.5))

angerplt <- ggplot(data=this_data, aes(x=anger, y=Score, color=trial_num)) +
  geom_point(stat="identity") +
  geom_smooth(method="lm") + 
  #stat_cor(color="black", label.x.npc="center", p.accuracy=0.001) +
  stat_cor(aes(color=trial_num), label.x.npc="center", p.accuracy=0.001) +
  scale_color_manual(values=c("#d8b365", "#5ab4ac")) +
  xlab("Proportion of utterances produced with emotion 'Anger'")

disgustplt <- ggplot(data=this_data, aes(x=disgust, y=Score, color=trial_num)) +
  geom_point(stat="identity") +
  geom_smooth(method="lm") + 
  #stat_cor(color="black", label.x.npc="center", p.accuracy=0.001) +
  stat_cor(aes(color=trial_num), label.x.npc="center", p.accuracy=0.001) +
  scale_color_manual(values=c("#d8b365", "#5ab4ac")) +
  xlab("Proportion of utterances produced with emotion 'Disgust'")


# look at average proportions of utterances at a team level 
# max proportion of utterances for an individual on a team 

fearplt <- ggplot(data=this_data, aes(x=fear, y=Score)) +
  geom_point(stat="identity", size=3) +
  geom_smooth(method="lm", size=2) + 
  stat_cor(label.x.npc=0.4, p.accuracy=0.001, size=8) +
  theme(text = element_text(size=20)) +
  xlab("Proportion of utterances labeled 'Fear'")

joyplt <-ggplot(data=this_data, aes(x=joy, y=Score)) +
  geom_point(stat="identity", size=3) +
  geom_smooth(method="lm", size=2) + 
  stat_cor(label.x.npc="center", p.accuracy=0.001, size=8) +
  theme(text = element_text(size=20)) +
  xlab("Proportion of utterances labeled 'Joy'")

neutralplt <- ggplot(data=this_data, aes(x=neutral_emo, y=Score, color=trial_num)) +
  geom_point(stat="identity") +
  geom_smooth(method="lm") + 
  #stat_cor(color="black", label.x.npc="center", p.accuracy=0.001) +
  stat_cor(aes(color=trial_num), label.x.npc="center", p.accuracy=0.001) +
  scale_color_manual(values=c("#d8b365", "#5ab4ac")) +
  xlab("Proportion of items produced with emotion 'Neutral")

sadnessplt <- ggplot(data=this_data, aes(x=sadness, y=Score)) +
  geom_point(stat="identity", size=3) +
  geom_smooth(method="lm", size=2) + 
  stat_cor(label.x.npc=0.4, p.accuracy=0.001, size=8) +
  theme(text = element_text(size=20)) +
  xlab("Proportion of utterances labeled 'Sadness'")

surpriseplt <- ggplot(data=this_data, aes(x=surprise, y=Score)) +
  geom_point(stat="identity", size=3) +
  geom_smooth(method="lm", size=2) + 
  stat_cor(label.x.npc="center", p.accuracy=0.001, size=8) +
  theme(text = element_text(size=20)) +
  xlab("Proportion of utterances labeled 'Surprise'")

fearsadplt <- ggplot(data=this_data, 
                      aes(x=fear + sadness,
                          y=Score)) + 
  geom_point(stat="identity") + 
  geom_smooth(method="lm") + 
  stat_cor(label.x.npc="center", p.accuracy=0.001) +
  xlab("Proportion of items produced with emotions 'Fear', and 'Sadness'")

joysurpriseplt <- ggplot(data=this_data, 
                     aes(x=joy + surprise,
                         y=Score)) + 
  geom_point(stat="identity") + 
  geom_smooth(method="lm") + 
  stat_cor(label.x.npc="center", p.accuracy=0.001) +
  xlab("Proportion of items produced with emotions 'Joy', and 'Surprise'")

allfourplt <- ggplot(data=this_data,
                     aes(x=(joy+surprise) - (fear + sadness),
                         y=Score)) + 
  geom_point(stat="identity") + 
  geom_smooth(method="lm") + 
  stat_cor(label.x.npc="center", p.accuracy=0.001) +
  xlab("Proportion of items produced with emotions 'Fear', 'Joy', 'Sadness', and 'Surprise'")

allfour_bytrialplt <- ggplot(data=this_data,
                             aes(x=(joy+surprise) - (fear + sadness),
                                 y=Score, color=trial_num)) + 
  geom_point(stat="identity") + 
  geom_smooth(method="lm") + 
  facet_grid(rows=vars(trial_num)) +
  stat_cor(color="black", label.x.npc="center", p.accuracy=0.001) +
  xlab("Proportion of items produced with emotions 'Fear', 'Joy', 'Sadness', and 'Surprise'")

this_data_trials <- this_data[-c(2, 4)] %>%
  group_by(Trial_ID, trial_num) %>%
  summarise(Score=mean(Score), maxanger=max(anger), 
            maxdisgust=max(disgust),
            maxfear=max(fear), maxjoy=max(joy),
            maxneut=max(neutral_emo), maxsadness=max(sadness),
            maxsurprise=max(surprise), anger=mean(anger), 
            disgust=mean(disgust), fear=mean(fear),
            joy=mean(joy), neutral_emo=mean(neutral_emo),
            sadness=mean(sadness), surprise=mean(surprise),
            )

############################################################
############################################################

angerplt <- ggplot(data=this_data_trials, aes(x=anger, y=Score)) +
  geom_point(stat="identity") +
  geom_smooth(method="lm") + 
  stat_cor(label.x.npc="center", p.accuracy=0.001) +
  scale_color_manual(values=c("#d8b365", "#5ab4ac")) +
  xlab("Proportion of utterances produced with emotion 'Anger'")

disgustplt <- ggplot(data=this_data_trials, aes(x=disgust, y=Score)) +
  geom_point(stat="identity") +
  geom_smooth(method="lm") + 
  #stat_cor(color="black", label.x.npc="center", p.accuracy=0.001) +
  stat_cor(label.x.npc="center", p.accuracy=0.001) +
  scale_color_manual(values=c("#d8b365", "#5ab4ac")) +
  xlab("Proportion of utterances produced with emotion 'Disgust'")

fearplt <- ggplot(data=this_data_trials, aes(x=fear, y=Score)) +
  geom_point(stat="identity", size=3) +
  geom_smooth(method="lm", size=2) + 
  stat_cor(label.x.npc=0.4, p.accuracy=0.001, size=8) +
  theme(text = element_text(size=20)) +
  xlab("Proportion of utterances labeled 'Fear'")

maxfearplt <- ggplot(data=this_data_trials, aes(x=maxfear, y=Score)) +
  geom_point(stat="identity", size=3) +
  geom_smooth(method="lm", size=2) + 
  stat_cor(label.x.npc=0.4, p.accuracy=0.001, size=8) +
  theme(text = element_text(size=20)) +
  xlab("Proportion of utterances labeled 'Fear'")

joyplt <-ggplot(data=this_data_trials, aes(x=joy, y=Score)) +
  geom_point(stat="identity", size=3) +
  geom_smooth(method="lm", size=2) + 
  stat_cor(label.x.npc="center", p.accuracy=0.001, size=8) +
  theme(text = element_text(size=20)) +
  xlab("Proportion of utterances labeled 'Joy'")

maxjoyplt <-ggplot(data=this_data_trials, aes(x=maxjoy, y=Score)) +
  geom_point(stat="identity", size=3) +
  geom_smooth(method="lm", size=2) + 
  stat_cor(label.x.npc="center", p.accuracy=0.001, size=8) +
  theme(text = element_text(size=20)) +
  xlab("Proportion of utterances labeled 'Joy'")

sadnessplt <- ggplot(data=this_data_trials, aes(x=sadness, y=Score)) +
  geom_point(stat="identity", size=3) +
  geom_smooth(method="lm", size=2) + 
  stat_cor(label.x.npc=0.4, p.accuracy=0.001, size=8) +
  theme(text = element_text(size=20)) +
  xlab("Proportion of utterances labeled 'Sadness'")

maxsadnessplt <- ggplot(data=this_data_trials, aes(x=maxsadness, y=Score)) +
  geom_point(stat="identity", size=3) +
  geom_smooth(method="lm", size=2) + 
  stat_cor(label.x.npc=0.4, p.accuracy=0.001, size=8) +
  theme(text = element_text(size=20)) +
  xlab("Proportion of utterances labeled 'Sadness'")

surpriseplt <- ggplot(data=this_data_trials, aes(x=surprise, y=Score)) +
  geom_point(stat="identity", size=3) +
  geom_smooth(method="lm", size=2) + 
  stat_cor(label.x.npc="center", p.accuracy=0.001, size=8) +
  theme(text = element_text(size=20)) +
  xlab("Proportion of utterances labeled 'Surprise'")

maxsurpriseplt <- ggplot(data=this_data_trials, aes(x=maxsurprise, y=Score)) +
  geom_point(stat="identity", size=3) +
  geom_smooth(method="lm", size=2) + 
  stat_cor(label.x.npc="center", p.accuracy=0.001, size=8) +
  theme(text = element_text(size=20)) +
  xlab("Proportion of utterances labeled 'Surprise'")


############################################################
############################################################

allfour_bytrial_totalplt <- ggplot(data=this_data_trials,
                                   aes(x=(joy+surprise) - (fear + sadness),
                                       y=Score, color=trial_num)) + 
  geom_point(stat="identity") + 
  geom_smooth(method="lm") + 
  facet_grid(rows=vars(trial_num)) +
  stat_cor(color="black", label.x.npc="center", p.accuracy=0.001) +
  xlab("Proportion of items produced with emotions 'Fear', 'Joy', 'Sadness', and 'Surprise'")

allfour_totalplt <- ggplot(data=this_data_trials,
                          aes(x=(joy+surprise) - (fear + sadness),
                              y=Score)) + 
  geom_point(stat="identity", size=3) + 
  geom_smooth(method="lm", size=2) + 
  stat_cor(color="black", label.x.npc="left", p.accuracy=0.001, size=8) +
  theme(text = element_text(size=20)) +
  xlab("Proportion of items with 'Fear', 'Joy', 'Sadness', and 'Surprise'")


sentemo_pairs <- ggpairs(data=this_data,
                          columns=c("anger", "disgust", "fear", "joy",
                                    "neutral_emo", "sadness", "surprise",
                                    "positive", "neutral_sent", "negative",
                                    "Score"),
                          aes(color=trial_num, alpha=0.5))

relevant <- c("anger", "disgust", "fear", "joy",
              "neutral_emo", "sadness", "surprise",
              "positive", "neutral_sent", "negative")

high_scores <- this_data[this_data$score_bin=="High",]
avg_scores <- this_data[this_data$score_bin=="Medium",]
low_scores <- this_data[this_data$score_bin=="Low",]

t1 <- this_data[this_data$trial_num==1,]
t2 <- this_data[this_data$trial_num==2,]

m1_corr <- cor(t1[relevant], t1$Score)
m2_corr <- cor(t2[relevant], t2$Score)

high_corr <- cor(high_scores[relevant], high_scores$Score)
avg_corr <- cor(avg_scores[relevant], avg_scores$Score)
low_corr <- cor(low_scores[relevant], low_scores$Score)
overall_corr <- cor(this_data[relevant], this_data$Score)

# use ALL personality data for this 
pers_data <- trial_data
pers_data$score_bin <- cut(pers_data$Score, 
                           breaks=c(min(this_data$Score),
                                    quantile(this_data$Score, .33),
                                    quantile(this_data$Score, .67),
                                    max(this_data$Score)),
                           labels=c("Low", "Medium", "High"))

pers_pairs <- ggpairs(data=pers_data,
                      columns=c("extroversion", "agreeableness",
                                "conscientiousness", "openness",
                                "neuroticism", "max_trait", "Score"),
                      aes(color=trial_num, alpha=0.5))

#####################################################
#### compare sentiments for 2 score bins   ##########
#####################################################
sentiment_pairs2 <- ggpairs(data=this_data,
                           columns = c("positive", "negative", "neutral_sent", "Score"),
                           aes(color=score_bin2, alpha=0.5))

emotion_pairs2 <- ggpairs(data=this_data,
                         columns=c("anger", "disgust", "fear", "joy",
                                   "neutral_emo", "sadness", "surprise",
                                   "Score"),
                         aes(color=score_bin2, alpha=0.5))

sentemo_pairs2 <- ggpairs(data=this_data,
                          columns=c("anger", "disgust", "fear", "joy",
                                    "neutral_emo", "sadness", "surprise",
                                    "positive", "neutral_sent", "negative",
                                    "Score"),
                          aes(color=score_bin2, alpha=0.5))

pers_data$score_bin2 <- cut(pers_data$Score,
                            breaks=c(min(pers_data$Score),
                                     quantile(pers_data$Score, .50),
                                     max(pers_data$Score)),
                            labels=c("Low", "High"))

pers_pairs2 <- ggpairs(data=pers_data,
                      columns=c("extroversion", "agreeableness",
                                "conscientiousness", "openness",
                                "neuroticism", "Score"),
                      aes(color=score_bin2, alpha=0.5))


###########################################################
########### PCA  ##########################################
###########################################################
require(ggfortify)

relevant_plus <- c("score_bin", "anger", "disgust", "fear", "joy",
              "neutral_emo", "sadness", "surprise",
              "positive", "neutral_sent", "negative")

relevant_data <- this_data[relevant_plus]
pca <- prcomp(relevant_data[-1])

p <- autoplot(pca, data=relevant_data, colour="score_bin")

#######################################################

non_neutral_emo_facet <- ggplot(data=emo_data[emo_data$emotion != "neutral",], aes(x=freq, y=Score, color=emotion)) +
  geom_point(stat="identity") +
  geom_smooth(method="lm") +
  facet_grid(rows=vars(emotion)) +
  stat_cor(aes(label=..rr.label..), color="black", label.x.npc = "center")

sent_facet <- ggplot(data=sent_data, aes(x=freq, y=Score, color=sentiment)) + 
  geom_point(stat="identity") +
  geom_smooth(method="lm") +
  facet_grid(rows=vars(sentiment)) +
  stat_cor(aes(label=..rr.label..), color="black", label.x.npc = "center")
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
