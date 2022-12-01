# tomcat-dataset-creation
Code used in the creation and organization of a dataset for ToMCAT 

## Dataset Creation

To create a dataset, alter the paths needed according to where the raw data is held using:

```config/data_prep_config.py```

Then run the script

```scripts/organize_dataset.py```

## Correlating outcomes with annotations

To correlate outcomes with annotations, use

```scripts/correlate_outcomes_annotations.R```

## 