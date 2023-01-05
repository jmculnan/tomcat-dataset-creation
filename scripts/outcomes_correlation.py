# correlate the outcomes of the missions with different measures
import pandas as pd
import re

def get_items_per_class(mission_annotations):
    """
    Get number of items per class
    :param mission_annotations:
    :return:
    """
    return mission_annotations.value_counts()


def proportion_of_items_per_class(mission_annotations):
    """
    Calculate the proportion of each class in a mission
    :param mission_annotations: A pandas series
    :return:
    """
    items_per_class = {}

    num_items = get_items_per_class(mission_annotations)

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


def get_number_of_items_da(mission_df):
    """
    Get the number of items for each da class
    :param mission_df:
    :return:
    """
    da_labels = []
    labels = mission_df['label'].dropna().tolist()

    # separate labels at pipes
    for item in labels:
        lst = item.split(" | ")
        da_labels.extend(lst)

    print(set(da_labels))

    # count these
    da_labels = pd.Series(da_labels)
    da_counts = da_labels.value_counts()

    return da_counts


def get_series_of_proportions(data_path):
    """
    For a given directory, go through all mission files
    and get the proportion of items per class for sent and emo
    :param data_path:
    :return:
    """



if __name__ == "__main__":
    test_file = "/media/jculnan/datadrive/asist_data_copy/overall_sent-emo.csv"
    test = pd.read_csv(test_file)

    items_per_class = proportion_of_items_sent_emo(test)
    print(items_per_class)
    da_items_per_class = get_number_of_items_da(test)
    print(da_items_per_class)



