# HPA
Breast cancer prediction using supervised and unsupervised machine learning models

The [dataset](http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29) used for modelling comes from University of Wisconsin.


### Data Visualizing
---
in order to use visualization
run the script 

```python3.6 src/visualization.py --dataset_type type_of_dataset --covid_type type_of_covid_data```

Replace type_of_dataset with either one of these options: 
- breast-cancer-wisconsin.data
- cleaned_covid-19_blood.xlsx
- wdbc.data
- wpbc.data

Replace type_of_covid_data with either one of these options: (Only available when dealing with the covid-19 dataset)
- covid_result
- intensive_result

The script will create a folder called visualization. The dataset folder would be created accordingly 
in the visualization folder. In order to run the visualization we can run this script in terminal:

```tensorboard --logdir=visualization/wdbc.data --host=localhost ```


### Running training
In order to run training, navigate to the root folder of Cancer-Stats-Miner, then run the following script:
``` 
python3.6 src/supervised_models.py --dataset_type wpbc.data --model_type network_second --num_samples 5 --covid_type covid_result
```
dataset_type options:
- breast-cancer-wisconsin.data
- wdbc.data
- cleaned_covid-19_blood.xlsx
- wpbc.data

model_type options:
- logistic_regression
- random_forest
- decision_tree
- gradient_boost
- svc
- linear_svc

covid_type options: (This is only available for covid-19 dataset)
- covid_result
- intensive_result

#### --- our approach ---
- network
- network_second
- network_third
- network_fourth

--num_samples argument limit the amount of data samples that the model can be trained on with labeled data.
For instance, when num_samples provided is 5, the model is only able to see 5 cancerous data samples and 5 
non-cancerous data samples to train on.

The output for running this command will be the f1 score on the model on the dataset.
