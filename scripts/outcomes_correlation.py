# correlate the outcomes of the missions with different measures
import pandas as pd


def proportion_of_items_per_class(mission_annotations):
    """
    Calculate the proportion of each class in a mission
    :param mission_annotations: A pandas series
    :return:
    """
    items_per_class = {}

    num_items = mission_annotations.value_counts()

    classes = num_items.index.tolist()
    all_items = sum([item for item in num_items])

    for i, item in enumerate(num_items):
        items_per_class[classes[i]] = float(item) / all_items

    return items_per_class


def proportion_of_items_sent_emo(mission_df):
    """
    Get the proportion of each class for each annotation type
        in a mission
    :param mission_df:
    :return:
    """
    sent_prop = proportion_of_items_per_class(mission_df['sentiment'])
    emo_prop = proportion_of_items_per_class(mission_df['emotion'])

    return sent_prop, emo_prop


def get_series_of_proportions(data_path):
    """
    For a given directory, go through all mission files
    and get the proportion of items per class for sent and emo
    :param data_path:
    :return:
    """



if __name__ == "__main__":
    test_file = "/media/jculnan/backup/jculnan/datasets/asist_data2/combined_data_CORRECT9.21.22.csv"
    test = pd.read_csv(test_file)

    items_per_class = proportion_of_items_sent_emo(test)
    print(items_per_class)



