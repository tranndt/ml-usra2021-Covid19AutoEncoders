import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.linear_model import BayesianRidge
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold


class Dataset:
    def __init__(self, path_to_file, arg):
        self.filename = path_to_file
        self._X = None
        self._Y = None
        self._kfold_dataset = None
        self.seed = arg.random_seed
        self.scale = StandardScaler()
        # kfold
        self.num_kfold_splits = arg.num_folds
        self.current_kfold_index = 0
        self.arg = arg

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @property
    def kfold_dataset(self):
        return self._kfold_dataset

    def process_dataset(self):
        # normalize X
        self._X = self.scale.fit_transform(self._X)
        
        # using Label Encoder to convert categorical data into number so the
        # model can understand better
        labelencoder_Y = LabelEncoder()
        self._Y = labelencoder_Y.fit_transform(self._Y)
        self._Y = np.expand_dims(self._Y, 1).astype(np.float32)

    def create_kfold_dataset(self):
        if self.arg.num_folds > 1:
            kfold = KFold(self.num_kfold_splits, shuffle=True, random_state=self.seed)
            self._kfold_dataset = list(kfold.split(self._X, self._Y))  # get the index

    def get_next_kfold_data(self):
        assert self.arg.num_folds > 1

        current_kfold_data = self._kfold_dataset[self.current_kfold_index]
        self.current_kfold_index += 1
        # goes back if current fold is more than the number of folds
        if self.current_kfold_index >= self.num_kfold_splits:
            self.current_kfold_index = 0

        current_X_train = self._X[current_kfold_data[0]]
        current_y_train = self._Y[current_kfold_data[0]]
        current_X_test = self._X[current_kfold_data[1]]
        current_y_test = self._Y[current_kfold_data[1]]

        return current_X_train, current_X_test, current_y_train, current_y_test


class BCWDataset(Dataset):
    def __init__(self, path_to_file, arg):
        super(BCWDataset, self).__init__(path_to_file, arg)

    def process_dataset(self):
        # Reading the data set
        data = pd.read_csv(self.filename)
        X = data.iloc[:, 1:10].values
        Y = data.iloc[:, 10].values
        # replace missing value with 0
        X[np.where(X == '?')] = 0
        X = X.astype(np.float32)
        self._X = X
        self._Y = Y

        print("breast cancer wisconsin cancer dataset dimensions : {}".format(data.shape))
        super(BCWDataset, self).process_dataset()


class WDBCDataset(Dataset):
    def __init__(self, path_to_file, arg):
        super(WDBCDataset, self).__init__(path_to_file, arg)

    def process_dataset(self):
        # Reading the data set
        data = pd.read_csv(self.filename)
        X = data.iloc[:, 2:].values
        Y = data.iloc[:, 1].values
        Y[np.where(Y == 'B')] = 0
        Y[np.where(Y == 'M')] = 1
        X = X.astype(np.float32)
        self._X = X
        self._Y = Y

        print("WDBC cancer dataset dimensions : {}".format(data.shape))
        super(WDBCDataset, self).process_dataset()


class WPBCDataset(Dataset):
    def __init__(self, path_to_file, arg):
        super(WPBCDataset, self).__init__(path_to_file, arg)

    def process_dataset(self):
        # Reading the data set
        data = pd.read_csv(self.filename)
        X = data.iloc[:, 2:].values
        Y = data.iloc[:, 1].values
        X[np.where(X == '?')] = 0
        Y[np.where(Y == 'N')] = 0
        Y[np.where(Y == 'R')] = 1
        X = X.astype(np.float32)
        self._X = X
        self._Y = Y

        print("WPBC cancer dataset dimensions : {}".format(data.shape))
        super(WPBCDataset, self).process_dataset()


class COVIDBloodDataset(Dataset):
    def __init__(self, path_to_file, arg):
        super(COVIDBloodDataset, self).__init__(path_to_file, arg)
        self.folder_name = self.filename
        self.arg = arg

    def get_next_kfold_data(self):
        if self.arg.num_folds > 1:
            return super(COVIDBloodDataset, self).get_next_kfold_data()
        else:
            return self._X, self._X_test, self._Y, self._y_test

    def process_dataset(self):
        x = pd.read_csv(os.path.join(self.folder_name, 'X_train.csv')).values.astype(np.float32)
        y = pd.read_csv(os.path.join(self.folder_name, 'y_train.csv')).values.astype(np.float32)
        x_test = pd.read_csv(os.path.join(self.folder_name, 'X_test.csv')).values.astype(np.float32)
        y_test = pd.read_csv(os.path.join(self.folder_name, 'y_test.csv')).values.astype(np.float32)

        if self.arg.num_folds > 1:
            self._X = self.scale.fit_transform(np.concatenate([x, x_test]))
            self._Y = np.concatenate([y, y_test])
        else:
            self._X = self.scale.fit_transform(x)
            self._Y = y
            self._X_test = self.scale.inverse_transform(x_test)
            self._y_test = y_test


class COVIDCalvinDataset(Dataset):
    def __init__(self, path_to_file, arg):
        super(COVIDCalvinDataset, self).__init__(path_to_file, arg)
        self.folder_name = self.filename

    def _process_column_nan(self, csv_reader, total_num_samples):
        """
        remove columns that have more than 50% NaN

        :param csv_reader:
        :param total_num_samples:  the total number of rows (number of samples)
        :return: processed csv reader
        """
        columns_to_drop = []
        # remove columns that have more than 50% NAN
        for column in csv_reader.columns:
            nan_count = csv_reader[column].isna().sum()
            if nan_count > total_num_samples // 2:
                columns_to_drop.append(column)
        processed_csv_reader = csv_reader.drop(columns=columns_to_drop)
        return processed_csv_reader

    def _process_all_data(self):
        all_csv_reader = None
        for i in range(self.arg.group_num):
            csv_reader = pd.read_csv(os.path.join(self.folder_name, f'group{i}.csv'))
            if all_csv_reader is None:
                all_csv_reader = csv_reader
            else:
                all_csv_reader = pd.concat([all_csv_reader, csv_reader])

        return all_csv_reader

    def process_dataset(self):
        if self.arg.group_num == 4:
            csv_reader = self._process_all_data()
        else:
            csv_reader = pd.read_csv(os.path.join(self.folder_name, f'group{self.arg.group_num}.csv'))

        total_num_samples = csv_reader.shape[0]
        processed_csv_reader = self._process_column_nan(csv_reader, total_num_samples)

        if self.arg.num_folds > 1:

            if self.arg.more_labels:
                self._X = np.concatenate([processed_csv_reader.values[:, 6:], processed_csv_reader.values[:, 1:2]], axis=1)
            else:
                self._X = np.concatenate([processed_csv_reader.values[:, 3:], processed_csv_reader.values[:, 1:2]], axis=1)

            self._X[np.where(self._X == 'not_detected')] = 0
            self._X[np.where(self._X == 'detected')] = 1
            self._X = self._X.astype(np.float32)

            iterative_imputer = IterativeImputer(estimator=BayesianRidge())
            self._X = iterative_imputer.fit_transform(self._X)

            self._X = self.scale.fit_transform(self._X)
            if self.arg.more_labels:
                self._Y = processed_csv_reader.values[:, 2:6]
            else:
                self._Y = processed_csv_reader.values[:, 2:3]
            self._Y[np.where(self._Y == 'negative')] = 0
            self._Y[np.where(self._Y == 'positive')] = 1
            self._Y = self._Y.astype(np.float32)

        # else:
        #     self._X = np.concatenate([processed_csv_reader.values[:, 3:], processed_csv_reader.values[:, 1:2]], axis=1)
        #     self._Y = processed_csv_reader.values[:, 2:3]
        #     self._X = self.scale.fit_transform(self._X)
        #
        #     self._Y = y
        #     self._X_test = self.scale.inverse_transform(x_test)
        #     self._y_test = y_test


dataset_dict = {
    'breast-cancer-wisconsin.data': BCWDataset,
    'wdbc.data': WDBCDataset,
    'wpbc.data': WPBCDataset,
    'covid-19': COVIDBloodDataset,
    'CovidKaggleGroups': COVIDCalvinDataset,
}

