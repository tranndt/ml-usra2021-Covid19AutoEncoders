import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.decomposition import PCA
import scipy
from sklearn.metrics import f1_score, confusion_matrix
import sys

sys.path.insert(0, "./")

from src.data_preprocessing import dataset_dict
from src.network.network import Network, NetworkSecondApproach, NetworkThirdApproach, NetworkFourthApproach


model_dict = {
    "logistic_regression": lambda *args, **kwargs: BinaryRelevance(LogisticRegression(*args, **kwargs)),
    "random_forest": lambda *args, **kwargs: BinaryRelevance(RandomForestClassifier(*args, **kwargs)),
    "decision_tree": lambda *args, **kwargs: BinaryRelevance(DecisionTreeClassifier(*args, **kwargs)),
    "gradient_boost": lambda *args, **kwargs: BinaryRelevance(GradientBoostingClassifier(*args, **kwargs)),
    "svc": lambda *args, **kwargs: BinaryRelevance(SVC(*args, **kwargs)),
    "linear_svc": lambda *args, **kwargs: BinaryRelevance(LinearSVC(*args, **kwargs)),
    "network": Network,
    "network_second": NetworkSecondApproach,
    "network_third": NetworkThirdApproach,
    # "network_fourth": NetworkFourthApproach
}

network_model_dict = {
"network": Network,
    "network_second": NetworkSecondApproach,
    "network_third": NetworkThirdApproach,
}


def model_initializer(all_X_train, arg, model_type, random_state=42, X_test=None, Y_test=None):
    if "network" in model_type:
        model_selected = model_dict[model_type](all_X_train, arg, random_state=random_state, X_test=X_test, Y_test=Y_test)
    else:
        model_selected = model_dict[model_type](random_state=random_state)
    return model_selected


def limit_samples(inputs, targets, num_classes=2, num_samples_per_class=5):
    """
    Limit the number of samples per class

    :param inputs:
    :param targets:
    :param num_classes:             The total number of classes
    :param num_samples_per_class:   Number of samples to train for model for each class
    :return:
    """
    limited_inputs = []
    limited_targets = []
    inserted_index = []  # this shows the inserted index into the limited_inputs and limited_targets
                         # so that we can delete them from the inputs and targets and use the remaining
                         # inputs and targets as testing data

    def __limit_each_col(col_index):
        targets_count = {}
        total_samples = num_classes * 2 * num_samples_per_class
        num_samples_added = 0
        # loop through the dataset and add the samples for each class to targets_dict
        for index in range(len(targets)):
            current_target = targets[index][col_index]
            if targets_count.get(current_target, None) is None:
                targets_count[current_target] = 1
                limited_inputs.append(inputs[index])
                limited_targets.append(targets[index])
                inserted_index.append(index)
                num_samples_added += 1
            else:
                # if the length of this current target is less than the num_samples_per_class we append
                if targets_count[current_target] < num_samples_per_class:
                    targets_count[current_target] += 1
                    limited_inputs.append(inputs[index])
                    limited_targets.append(targets[index])
                    inserted_index.append(index)
                    num_samples_added += 1
                    # check if the total targets_dict reach the limit
                    if len(limited_inputs) >= total_samples:
                        break

    for col_index in range(len(targets[0])):
        __limit_each_col(col_index)

    remaining_inputs = np.delete(inputs, np.unique(inserted_index), axis=0)
    remaining_targets = np.delete(targets, np.unique(inserted_index), axis=0)

    limited_inputs = np.array(limited_inputs).astype(np.float32)
    limited_targets = np.array(limited_targets).astype(np.float32)
    return limited_inputs, limited_targets, remaining_inputs, remaining_targets


def evaluate_model(dataset, arg, model_name, score_text, prediction_only=False):
    total_f1_score = 0
    total_FP = 0
    total_FN = 0
    total_TP = 0
    total_TN = 0
    for i in range(arg.num_folds):
        X_train, X_test, y_train, y_test = dataset.get_next_kfold_data()
        # if num_samples is specify, then we limit the training samples
        limit_X_train = X_train
        limit_y_train = y_train
        if arg.num_samples:
            limit_X_train, limit_y_train, remaining_X, remaining_y = limit_samples(X_train, y_train,
                                                                                   num_classes=y_train.shape[1],
                                                                                   num_samples_per_class=arg.num_samples)
            # combine the unused samples with the testing data
            X_test = np.concatenate([X_test, remaining_X])
            y_test = np.concatenate([y_test, remaining_y])

        # train model
        model = model_initializer(dataset.X, arg, model_name, random_state=arg.random_seed, X_test=X_test,
                                  Y_test=y_test)
        # load model
        if prediction_only and 'network' in model_name:
            # initialize network weights first
            if len(limit_y_train.shape) < 2:
                limit_y_train = np.expand_dims(limit_y_train, 1)
            # hard code the value 1 for now, we are only predicting 2 values
            model._initialize_network(limit_X_train.shape[1], limit_y_train.shape[1])

            success_loading = model.load(f'model/{arg.more_labels} {arg.num_encoded_features}_{arg.group_num}_{arg.num_samples}_{model_name}_{i}.pth')
            if not success_loading:
                model.fit(limit_X_train, limit_y_train)
        else:
            model.fit(limit_X_train, limit_y_train)

        pred = model.predict(X_test)
        if type(pred) == scipy.sparse.csc.csc_matrix:
            pred = pred.toarray()
        score = f1_score(y_test.ravel(), pred.ravel(), average='micro')
        cm = confusion_matrix(y_test.ravel(), pred.ravel())
        TP = cm[0][0]
        FN = cm[0][1]
        FP = cm[1][0]
        TN = cm[1][1]
        total_f1_score += score
        total_TP += TP
        total_FN += FN
        total_FP += FP
        total_TN += TN

        if 'network' in model_name:
            model.save_best_weights(f'model/{arg.more_labels} {arg.num_encoded_features}_{arg.group_num}_{arg.num_samples}_{model_name}_{i}.pth')

    if 'network' in model_name and not arg.more_labels:
        encoded_features = model.get_encoded_features(np.vstack((X_train, X_test)))
        all_labels = np.vstack((y_train, y_test)).ravel().astype(np.object)
        all_labels[all_labels==0] = 'blue'
        all_labels[all_labels==1] = 'red'

        pca = PCA(2)
        pca_features = pca.fit_transform(encoded_features)

        fig = plt.figure()
        red_patch = mpatches.Patch(color='red', label='covid')
        blue_patch = mpatches.Patch(color='blue', label='non-covid')
        plt.legend(handles=[red_patch, blue_patch])
        plt.scatter(pca_features[:, 0], pca_features[:, 1], c=all_labels)
        title = f'{model_name} group_num: {arg.group_num} num_samples: {arg.num_samples} num_encoded_features: {arg.num_encoded_features}'
        plt.title(title)
        plt.savefig(f'plots/morelabels {arg.more_labels}/encoded features {arg.num_encoded_features}/group {arg.group_num}/{title}.png', dpi=fig.dpi)


    mean_f1_score = total_f1_score / arg.num_folds
    mean_tp_score = total_TP / arg.num_folds
    mean_fp_score = total_FP / arg.num_folds
    mean_tn_score = total_TN / arg.num_folds
    mean_fn_score = total_FN / arg.num_folds

    print(f"{model_name} == f1 score: {round(mean_f1_score, 2)}")
    score_text["f1"] += f"{round(mean_f1_score, 2)}\n"
    score_text["tp"] += f"{round(mean_tp_score, 2)}\n"
    score_text["fp"] += f"{round(mean_fp_score, 2)}\n"
    score_text["tn"] += f"{round(mean_tn_score, 2)}\n"
    score_text["fn"] += f"{round(mean_fn_score, 2)}\n"


def compare_all_model(arg):
    # get dataset
    dataset_filename = os.path.join('data', arg.dataset_type)
    dataset = dataset_dict[arg.dataset_type](dataset_filename, arg)
    dataset.process_dataset()
    dataset.create_kfold_dataset()

    # score_text contains [f1 score, tp , tn, fp, fn]
    initial_text = f"num samples: {arg.num_samples} encoded features: {arg.num_encoded_features}\n====================\n"
    score_text = {"f1": initial_text, "fp": initial_text, "fn": initial_text, "tp": initial_text, "tn": initial_text}

    # use this if we want to train on the covid dataset with 4 groups (CovidKaggleGroups)
    # current_model_dict = network_model_dict if arg.num_encoded_features != 4 else model_dict

    current_model_dict = model_dict
    os.makedirs(f'plots/morelabels {arg.more_labels}/encoded features {arg.num_encoded_features}/group {arg.group_num}', exist_ok=True)


    for model_name in list(current_model_dict.keys()):
       evaluate_model(dataset, arg, model_name, score_text, prediction_only=arg.prediction_only)

    for key in score_text.keys():
        with open(f'group {arg.group_num}_{key}.txt', 'a') as f:
            f.write(f'{score_text[key]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", default=4, help="Set the random state so we can reproduce the result")
    parser.add_argument('--dataset_type', required=True, help="automatically looks into data directory"
                                                              "example would be breast-cancer-wisconsin.data")
    parser.add_argument('--prediction_only', action="store_true", help="only run evaluation instead of prediction "
                                                                       "(specifically for the deep learning networks")

    # for covidKaggleGroups argument only
    parser.add_argument('--more_labels', default=False, action='store_true')

    parser.add_argument('--num_encoded_features', default=16, type=int)
    parser.add_argument('--covid_type', choices=['covid_result', 'intensive_result'])
    parser.add_argument('--model_type', required=True, help="choose the model to train on the dataset")
    parser.add_argument('--num_samples', type=int, help="specify the number of samples for training the model")
    parser.add_argument('--num_folds', type=int, default=5, help="specify the number of samples for training the model")

    # for CovidKaggleGroups
    parser.add_argument('--group_num', type=int, default=0)
    arg = parser.parse_args()

    # get dataset
    dataset_filename = os.path.join('data', arg.dataset_type)
    dataset = dataset_dict[arg.dataset_type](dataset_filename, arg)
    dataset.process_dataset()
    dataset.create_kfold_dataset()

    os.makedirs('model', exist_ok=True)
    os.makedirs(f'plots/morelabels {arg.more_labels}/encoded features {arg.num_encoded_features}/group {arg.group_num}', exist_ok=True)

    # evaluate_model(dataset, arg, arg.model_type)
    num_encoded_features_list = [16]#[4, 8, 16]
    group_num_list = [1]#[1, 2, 3]
    num_sample_list = [None]#[1, 5, 10]
    for group_num in group_num_list:
        for num_samples in num_sample_list:
            for encoded_features in num_encoded_features_list:
                arg.group_num = group_num
                arg.num_encoded_features = encoded_features
                arg.num_samples = num_samples
                compare_all_model(arg)
