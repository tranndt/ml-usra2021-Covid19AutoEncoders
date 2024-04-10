from copy import deepcopy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from torch.optim import Adam
import numpy as np
import os
import wandb

batch_size = 32
reconstruct_epoch = 300
epoch = 500


class TorchDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.from_numpy(inputs)
        self.targets = torch.from_numpy(targets)

        assert self.inputs.shape[0] == self.targets.shape[0]

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        X = self.inputs[index]
        y = self.targets[index]

        return X, y


class OnlyXDataset(Dataset):
    def __init__(self, inputs):
        self.inputs = torch.from_numpy(inputs)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        X = self.inputs[index]
        return X


class Network(nn.Module):
    def __init__(self, all_X_train, arg, random_state=42, X_test=None, Y_test=None):
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        super(Network, self).__init__()
        self.all_X_train = all_X_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.arg = arg

        self.output_criterion = nn.BCELoss()
        self.optimizer = None
        self.best_state_dict = None
        self.best_total_loss = 100000

        wandb.init(name=f'{self.__class__.__name__}_{self.arg.num_samples}_{self.arg.covid_type}')

    def _initialize_network(self, input_shape, output_shape):

        self.layer1 = nn.Linear(input_shape, 64)
        self.reconstruct_output = nn.Linear(64, input_shape)

        self.encode_layer = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.ReLU(),
            nn.Linear(64, self.arg.num_encoded_features),
            nn.ReLU()
        )

        self.decode_layer = nn.Sequential(
            nn.Linear(self.arg.num_encoded_features, 64),
            nn.ReLU(),
            nn.Linear(64, input_shape)
        )

        self.output_layer = nn.Sequential(
            nn.Linear(self.arg.num_encoded_features, 16),
            nn.Linear(16, output_shape),
        )
        self.optimizer = Adam(self.parameters(), lr=0.0001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)

    def forward(self, inputs):
        encoded_feature = self.encode_layer(inputs)

        # output
        output = torch.sigmoid(self.output_layer(encoded_feature))

        # reconstruction
        reconstruct_output = self.decode_layer(encoded_feature)

        return output, encoded_feature, reconstruct_output

    def freeze_encoder_layer(self):
        for layer in list(self.encode_layer.parameters()):
            layer.requires_grad = False

    def unfreeze_encoder_layer(self):
        for layer in list(self.encode_layer.parameters()):
            layer.requires_grad = True

    def get_encoded_features(self, inputs):
        torch_inputs = torch.from_numpy(inputs)
        return self.encode_layer(torch_inputs).detach().numpy()

    def fit(self, limited_inputs, limited_targets):
        wandb.watch(self)

        if len(limited_targets.shape) < 2:
            limited_targets = np.expand_dims(limited_targets, 1)

        # hard code the value 1 for now, we are only predicting 2 values
        self._initialize_network(limited_inputs.shape[1], limited_targets.shape[1])
        self.train(True)
        # make dataset to torch type
        dataset = OnlyXDataset(self.all_X_train)
        # shuffle false because data already shuffled
        data_loader = DataLoader(dataset, batch_size, shuffle=False)
        limited_dataset = TorchDataset(limited_inputs, limited_targets)
        # shuffle false because data already shuffled
        limited_data_loader = DataLoader(limited_dataset, batch_size, shuffle=False)

        # create test dataset
        test_dataset = TorchDataset(self.X_test, self.Y_test)
        test_data_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        for i in range(epoch):
            mean_output_loss = []
            mean_recon_loss = []
            for batch_limit_x, batch_limit_y in limited_data_loader:
                variable_batch_limit_x = Variable(batch_limit_x)
                output, _, _ = self.forward(variable_batch_limit_x)
                output_loss = self.output_criterion(output, batch_limit_y)

                batch_all_x = next(iter(data_loader))
                variable_batch_all_x = Variable(batch_all_x)
                _, _, reconstruction = self.forward(batch_all_x)

                reconstruction_loss = torch.abs(reconstruction - variable_batch_all_x).mean()

                total_loss = output_loss + 0.5 * reconstruction_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                mean_output_loss.append(output_loss)
                mean_recon_loss.append(reconstruction_loss)
            wandb.log({'training output loss': sum(mean_output_loss)/len(mean_output_loss),
                       'training reconstruction loss': sum(mean_recon_loss)/len(mean_recon_loss)})
            mean_test_output_loss = []
            mean_test_recon_loss = []

            for x_test, y_test in test_data_loader:
                variable_x_test = Variable(x_test)
                output, _, reconstruction = self.forward(variable_x_test)
                test_output_loss = self.output_criterion(output, y_test)
                test_reconstruction_loss = torch.abs(reconstruction - x_test).mean()

                mean_test_output_loss.append(test_output_loss)
                mean_test_recon_loss.append(test_reconstruction_loss)

            wandb.log({'testing output loss': sum(mean_test_output_loss)/len(mean_test_output_loss),
                       'testing reconstruction loss': sum(mean_test_recon_loss)/len(mean_test_recon_loss)})

            self.scheduler.step(sum(mean_test_output_loss)/len(mean_test_output_loss))

            if sum(mean_test_output_loss)/len(mean_test_output_loss) < self.best_total_loss:
                self.best_state_dict = deepcopy(self.state_dict())
                self.best_total_loss = sum(mean_test_output_loss) / len(mean_test_output_loss)

    def predict(self, inputs):
        self.train(False)
        self.eval()

        # error will occur when we are only doing evaluation and no training so we put a check here
        # in case if we are only doing evaluation, we will have already loaded in a file and don't need to
        # check what's is the best state when during training
        if self.best_state_dict is not None:
            self.load_state_dict(self.best_state_dict)

        variable_inputs = Variable(torch.from_numpy(inputs))
        output, _, _ = self.forward(variable_inputs)

        # if sigmoid
        output[output >= 0.5] = 1.
        output[output < 0.5] = 0.

        return output.detach().numpy()

    def score(self, inputs, targets):
        self.train(False)
        self.eval()

        test_dataset = TorchDataset(inputs, targets)
        data_loader = DataLoader(test_dataset, batch_size, shuffle=False)  # shuffle false because data already shuffled

        for batch_x, batch_y in data_loader:
            variable_batch_x = Variable(batch_x)
            variable_batch_y = Variable(batch_y)

            output, reconstruction = self.forward(variable_batch_x)
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            accuracy = (output.shape[0] - torch.abs(output - variable_batch_y).sum()) / output.shape[0]

            return accuracy

    def save_best_weights(self, filename):
        # can be None if we just use to evaluate the model
        if self.best_state_dict is not None:
            torch.save(self.best_state_dict, filename)

    def load(self, filename):
        if os.path.exists(filename):
            loaded_state_dict = torch.load(filename)

            if loaded_state_dict is None:
                return False
            self.load_state_dict(loaded_state_dict)
            print('loaded model')
            return True
        else:
            print('not loaded model, using initial weights instead')
            return False


class NetworkSecondApproach(Network):
    def __init__(self, all_X_train, arg, random_state=42, X_test=None, Y_test=None):
        super(NetworkSecondApproach, self).__init__(all_X_train, arg, random_state=random_state, X_test=X_test, Y_test=Y_test)
        self.best_recon_loss = 100000

    def fit(self, limited_inputs, limited_targets):
        if len(limited_targets.shape) < 2:
            limited_targets = np.expand_dims(limited_targets, 1)

        # hard code the value 1 for now, we are only predicting 2 values
        self._initialize_network(limited_inputs.shape[1], limited_targets.shape[1])
        self.train(True)
        # make dataset to torch type
        dataset = OnlyXDataset(self.all_X_train)
        # shuffle false because data already shuffled
        data_loader = DataLoader(dataset, batch_size, shuffle=False)
        limited_dataset = TorchDataset(limited_inputs, limited_targets)
        # shuffle false because data already shuffled
        limited_data_loader = DataLoader(limited_dataset, batch_size, shuffle=False)

        # create test dataset
        test_dataset = TorchDataset(self.X_test, self.Y_test)
        test_data_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        self.unfreeze_encoder_layer()
        # train autoencoder first
        for j in range(reconstruct_epoch):
            mean_recon_loss = []
            for all_x in data_loader:
                variable_all_x = Variable(all_x)
                _, _, reconstruction = self.forward(variable_all_x)

                reconstruction_loss = torch.abs(reconstruction - variable_all_x).mean()
                self.optimizer.zero_grad()
                reconstruction_loss.backward()
                self.optimizer.step()
                mean_recon_loss.append(reconstruction_loss)
            wandb.log({'training reconstruction loss': sum(mean_recon_loss) / len(mean_recon_loss)})
            if self.best_recon_loss > sum(mean_recon_loss) / len(mean_recon_loss):
                self.best_recon_loss = sum(mean_recon_loss) / len(mean_recon_loss)
                self.best_state_dict = deepcopy(self.state_dict())

        # after training autoencoder freeze the previous layers
        self.freeze_encoder_layer()
        self.load_state_dict(self.best_state_dict)

        for i in range(epoch):
            mean_output_loss = []
            for batch_limit_x, batch_limit_y in limited_data_loader:
                variable_batch_limit_x = Variable(batch_limit_x)
                output, _, _ = self.forward(variable_batch_limit_x)
                output_loss = self.output_criterion(output, batch_limit_y)

                self.optimizer.zero_grad()
                output_loss.backward()
                self.optimizer.step()
                mean_output_loss.append(output_loss)

            wandb.log({'training output loss': sum(mean_output_loss) / len(mean_output_loss)})
            mean_test_output_loss = []
            mean_test_recon_loss = []
            for x_test, y_test in test_data_loader:
                variable_x_test = Variable(x_test)
                output, _, reconstruction = self.forward(variable_x_test)
                test_output_loss = self.output_criterion(output, y_test)
                test_reconstruction_loss = torch.abs(reconstruction - x_test).mean()

                mean_test_output_loss.append(test_output_loss)
                mean_test_recon_loss.append(test_reconstruction_loss)

            wandb.log({'testing output loss': sum(mean_test_output_loss) / len(mean_test_output_loss),
                       'testing reconstruction loss': sum(mean_test_recon_loss) / len(mean_test_recon_loss)})
            if sum(mean_test_output_loss) / len(mean_test_output_loss) < self.best_total_loss:
                self.best_total_loss = sum(mean_test_output_loss) / len(mean_test_output_loss)
                self.best_state_dict = deepcopy(self.state_dict())


class NetworkThirdApproach(Network):
    def __init__(self, all_X_train, arg, random_state=42, X_test=None, Y_test=None):
        super(NetworkThirdApproach, self).__init__(all_X_train, arg, random_state=random_state, X_test=X_test, Y_test=Y_test)

    def forward_get_encode_features(self, inputs):
        encode_features = self.encode_layer(inputs)
        return encode_features

    def fit(self, limited_inputs, limited_targets):
        if len(limited_targets.shape) < 2:
            limited_targets = np.expand_dims(limited_targets, 1)

        self._initialize_network(limited_inputs.shape[1], limited_targets.shape[1])
        self.train(True)
        # make dataset to torch type
        dataset = OnlyXDataset(self.all_X_train)
        # shuffle false because data already shuffled
        data_loader = DataLoader(dataset, batch_size, shuffle=False)
        limited_dataset = TorchDataset(limited_inputs, limited_targets)
        # shuffle false because data already shuffled
        limited_data_loader = DataLoader(limited_dataset, batch_size, shuffle=False)

        self.unfreeze_encoder_layer()
        # train autoencoder first
        for j in range(reconstruct_epoch):
            for all_x in data_loader:
                variable_all_x = Variable(all_x)
                _, encode_features, reconstruction = self.forward(variable_all_x)
                _, recon_encode_features, _ = self.forward(reconstruction)

                reconstruction_loss = torch.abs(reconstruction - variable_all_x).mean()
                encode_reconstruction_loss = (encode_features - recon_encode_features).pow(2).mean()

                total_recons_loss = reconstruction_loss + 0.01 * encode_reconstruction_loss

                self.optimizer.zero_grad()
                total_recons_loss.backward()
                self.optimizer.step()

        # after training autoencoder freeze the previous layers
        self.freeze_encoder_layer()

        # create test dataset
        test_dataset = TorchDataset(self.X_test, self.Y_test)
        test_data_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        for i in range(epoch):
            mean_output_loss = []
            mean_recon_loss = []
            for batch_limit_x, batch_limit_y in limited_data_loader:
                variable_batch_limit_x = Variable(batch_limit_x)
                output, _, _ = self.forward(variable_batch_limit_x)
                output_loss = self.output_criterion(output, batch_limit_y)

                self.optimizer.zero_grad()
                output_loss.backward()
                self.optimizer.step()
                mean_output_loss.append(output_loss)
                mean_recon_loss.append(reconstruction_loss)

            wandb.log({'training output loss': sum(mean_output_loss) / len(mean_output_loss),
                       'training reconstruction loss': sum(mean_recon_loss) / len(mean_recon_loss)})

            mean_test_output_loss = []
            mean_test_recon_loss = []
            for x_test, y_test in test_data_loader:
                variable_x_test = Variable(x_test)
                output, _, reconstruction = self.forward(variable_x_test)
                test_output_loss = self.output_criterion(output, y_test)
                test_reconstruction_loss = torch.abs(reconstruction - x_test).mean()

                mean_test_output_loss.append(test_output_loss)
                mean_test_recon_loss.append(test_reconstruction_loss)

            wandb.log({'testing output loss': sum(mean_test_output_loss) / len(mean_test_output_loss),
                       'testing reconstruction loss': sum(mean_test_recon_loss) / len(mean_test_recon_loss)})
            self.scheduler.step(sum(mean_test_output_loss)/len(mean_test_output_loss))
            if sum(mean_test_output_loss) / len(mean_test_output_loss) < self.best_total_loss:
                self.best_state_dict = self.state_dict()
                self.best_total_loss = sum(mean_test_output_loss) / len(mean_test_output_loss)


class NetworkFourthApproach(Network):
    def __init__(self, all_X_train, arg, random_state=42, X_test=None, Y_test=None):
        super(NetworkFourthApproach, self).__init__(all_X_train, arg, random_state=random_state, X_test=X_test, Y_test=Y_test)
        self.similarity_criterion = nn.BCELoss()
        self.limited_data_loader = None  # this stores the available training samples so we can use to predict and
                                         # compare unseen samples

    def _initialize_network(self, input_shape, output_shape):
        super(NetworkFourthApproach, self)._initialize_network(input_shape, output_shape)
        self.similarity_layer = nn.Sequential(
            nn.Linear(self.arg.num_encoded_features*2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        encoded_feature = self.encode_layer(inputs)

        # output
        output = torch.sigmoid(self.output_layer(encoded_feature))

        # reconstruction
        reconstruct_output = self.decode_layer(encoded_feature)

        return output, encoded_feature, reconstruct_output

    def fit(self, limited_inputs, limited_targets):
        if len(limited_targets.shape) < 2:
            limited_targets = np.expand_dims(limited_targets, 1)

        # hard code the value 1 for now, we are only predicting 2 values
        self._initialize_network(limited_inputs.shape[1], limited_targets.shape[1])
        self.train(True)
        # make dataset to torch type
        dataset = OnlyXDataset(self.all_X_train)
        # shuffle false because data already shuffled
        data_loader = DataLoader(dataset, batch_size, shuffle=False)
        limited_dataset = TorchDataset(limited_inputs, limited_targets)
        # shuffle false because data already shuffled
        self.limited_data_loader = DataLoader(limited_dataset, batch_size, shuffle=False)

        # setup query samples
        self.query_samples = []  # use to calculate similarity during prediction stage, query samples
                                 # is the known data samples
        # get the number of targets for each label
        num_targets = int(torch.sum(torch.max(limited_dataset.targets, 0)[0]).cpu().item()) * 2
        for _ in range(num_targets):
            self.query_samples.append([])
        # add the inputs and targets into query_samples
        for index, target in enumerate(limited_dataset.targets):
            target_index = int(target.cpu().item())
            self.query_samples[target_index].append(limited_dataset.inputs[index])

        # train autoencoder first
        for j in range(reconstruct_epoch):
            for all_x in data_loader:
                variable_all_x = Variable(all_x)
                _, encode_features, reconstruction = self.forward(variable_all_x)
                _, recon_encode_features, _ = self.forward(reconstruction)

                reconstruction_loss = torch.abs(reconstruction - variable_all_x).mean()
                encode_reconstruction_loss = (encode_features - recon_encode_features).pow(2).mean()

                total_recons_loss = reconstruction_loss + 0.01 * encode_reconstruction_loss

                self.optimizer.zero_grad()
                total_recons_loss.backward()
                self.optimizer.step()

        # train similarity
        for i in range(epoch):
            for batch_limit_x, batch_limit_y in self.limited_data_loader:
                variable_batch_limit_x = Variable(batch_limit_x)
                output, encoded_features, _ = self.forward(variable_batch_limit_x)
                # calculate similarity loss
                total_similarity_loss = None
                for i in range(encoded_features.shape[0]):
                    for j in range(encoded_features.shape[0]):
                        concatenated_encoded_features = torch.cat((encoded_features[i], encoded_features[j]))
                        concatenated_encoded_features = concatenated_encoded_features.unsqueeze(0)
                        truth_similarity = batch_limit_y[i] == batch_limit_y[j]  # check if label is similar
                        truth_similarity = truth_similarity.type(torch.FloatTensor).view(-1, 1)  # convert boolean to float

                        similarity_output = self.similarity_layer(concatenated_encoded_features)

                        if total_similarity_loss is None:
                            total_similarity_loss = self.similarity_criterion(similarity_output, truth_similarity)
                        else:
                            total_similarity_loss += self.similarity_criterion(similarity_output, truth_similarity)

                # calculate mean similarity
                similarity_loss = total_similarity_loss / (encoded_features.shape[0] ** 2)
                self.optimizer.zero_grad()
                similarity_loss.backward()
                self.optimizer.step()

    def predict(self, inputs):
        self.train(False)
        self.eval()
        variable_inputs = Variable(torch.from_numpy(inputs))
        _, encoded_input_features, _ = self.forward(variable_inputs)

        # hard-coded, might need to change this
        current_similarity = np.zeros((inputs.shape[0], len(self.query_samples), 1))

        for query_samples_index in range(len(self.query_samples)):
            for query in self.query_samples[query_samples_index]:
                query = query.repeat(inputs.shape[0], 1)
                _, encoded_query_features, _ = self.forward(query)
                concatenated_features = torch.cat((encoded_input_features, encoded_query_features), 1)
                similarity_output = self.similarity_layer(concatenated_features)
                current_similarity[:, query_samples_index] += similarity_output.detach().numpy()

        output = current_similarity.argmax(1).astype(np.float32)
        return output
