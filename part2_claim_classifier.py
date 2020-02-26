import numpy as np
import pickle
import matplotlib.pyplot as plt
import copy
import pandas as pd


import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import metrics
#from pytorchtools import EarlyStopping
from sklearn import preprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

def analyse(model, data_x, data_y):
    # data_x and data_y are numpy array-of-arrays matrices
    X = torch.Tensor(data_x)
    Y = torch.ByteTensor(data_y)   # a Tensor of 0s and 1s
    oupt = model(X)            # a Tensor of floats
    pred_y = oupt >= 0.5       # a Tensor of 0s and 1s
    num_correct = torch.sum(Y==pred_y)  # a Tensor
    acc = (num_correct.item() * 100.0 / len(data_y))  # scalar
    return (acc, pred_y)

"""
class Insurance_NN(nn.Module):
    def __init__(self):
        super(Insurance_NN, self).__init__()

        self.apply_layers = nn.Sequential(
            # 2 fully connected hidden layers of 8 neurons goes to 1
            # 9 - (100 - 10) - 1
            nn.Linear(9, 100),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(100, 10),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.apply_layers(x)
        return x.view(len(x))
"""

def weighted_binary_cross_entropy(sigmoid_x, targets, pos_weight, weight=None, size_average=True, reduce=True):
    """
    Args:
        sigmoid_x: predicted probability of size [N,C], N sample and C Class. Eg. Must be in range of [0,1], i.e. Output from Sigmoid.
        targets: true value, one-hot-like vector of size [N,C]
        pos_weight: Weight for postive sample
    """
    if not (targets.size() == sigmoid_x.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(targets.size(), sigmoid_x.size()))

    loss = -pos_weight* targets * sigmoid_x.log() - (1-targets)*(1-sigmoid_x).log()

    if weight is not None:
        loss = loss * weight

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=1, weight=None, PosWeightIsDynamic= False, WeightIsDynamic= False, size_average=True, reduce=True):
        """
        Args:
            pos_weight = Weight for postive samples. Size [1,C]
            weight = Weight for Each class. Size [1,C]
            PosWeightIsDynamic: If True, the pos_weight is computed on each batch. If pos_weight is None, then it remains None.
            WeightIsDynamic: If True, the weight is computed on each batch. If weight is None, then it remains None.
        """
        super().__init__()

        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)
        self.size_average = size_average
        self.reduce = reduce
        self.PosWeightIsDynamic = PosWeightIsDynamic

    def forward(self, input, target):
        # pos_weight = Variable(self.pos_weight) if not isinstance(self.pos_weight, Variable) else self.pos_weight
        if self.PosWeightIsDynamic:
            positive_counts = target.sum(dim=0)
            nBatch = len(target)
            self.pos_weight = (nBatch - positive_counts)/(positive_counts +1e-5)

        if self.weight is not None:
            # weight = Variable(self.weight) if not isinstance(self.weight, Variable) else self.weight
            return weighted_binary_cross_entropy(input, target,
                                                 self.pos_weight,
                                                 weight=self.weight,
                                                 size_average=self.size_average,
                                                 reduce=self.reduce)
        else:
            return weighted_binary_cross_entropy(input, target,
                                                 self.pos_weight,
                                                 weight=None,
                                                 size_average=self.size_average,
                                                 reduce=self.reduce)



    def __iter__(self):
        return self

    def __next__(self):
        if self.ptr + self.batch_size > self.num_items:
            self.rnd.shuffle(self.indices)
            self.ptr = 0
            raise StopIteration  # exit calling for-loop
        else:
            result = self.indices[self.ptr:self.ptr + self.batch_size]
            self.ptr += self.batch_size
            return result


class ClaimClassifier():

    def __init__(self,):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary. 
        """
        self.fitted_model = 0;

        self.train_data = 0;
        self.test_data = 0;
        self.val_data = 0;

    def load_data(self, filename):
        """
        Function to load data from file
        Args:
            filename (str) - name of .txt file you are loading data from
        Output:
            (x, y) (tuple) - x: 2D array of training data where each row
            corresponds to a different sample and each column corresponds to a
            different attribute.
                            y: 1D array where each index corresponds to the
            ground truth label of the sample x[index][]
        """
        # load data to single 2D array
        dat = pd.read_csv("part2_training_data.csv")
        x = dat.drop(columns=["claim_amount", "made_claim"])
        y = dat["made_claim"]

        return x, y

    def separate_pos_neg(self, x, y):

        # Separate into positive and negative samples
        pos_train_y = []
        pos_train_x = np.empty((0, x.shape[1]), np.float32)
        neg_train_y = []
        neg_train_x = np.empty((0, x.shape[1]), np.float32)
        for i in range(y.shape[0]):
            if y[i] == 1:
                pos_train_y.append(y[i])
                pos_train_x = np.vstack((pos_train_x, x[i]))
            else:
                neg_train_y.append(y[i])
                neg_train_x = np.vstack((neg_train_x, x[i]))

        neg_train_y = np.array(neg_train_y, dtype=np.float32)
        pos_train_y = np.array(pos_train_y, dtype=np.float32)

        return (neg_train_x, neg_train_y), (pos_train_x, pos_train_y)

    def _preprocessor(self, X_raw):
        """Data preprocessing function.

        This function prepares the features of the data for training,
        evaluation, and prediction.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        ndarray
            A clean data set that is used for training and prediction.
        """
        # YOUR CODE HERE

        if not isinstance(X_raw, np.ndarray):
            X_raw = X_raw.to_numpy(dtype=np.float)

        min_max_scaler = preprocessing.MinMaxScaler()
        X_raw = min_max_scaler.fit_transform(X_raw)

        return X_raw.astype(np.float32)

    def set_axis_style(self, ax, labels):
        ax.get_xaxis().set_tick_params(direction='out')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_xlim(0.25, len(labels) + 0.75)
        ax.set_xlabel('Sample name')

    def evaluate_input1(self, X_raw):
        """
        Function to evaluate data loaded from file

        """

        attributes = []
        for i in range(np.shape(X_raw)[1]):
            attributes.append(X_raw[:, i])


        fig, ax1 = plt.subplots(figsize=(11, 4))

        # type of plot
        ax1.boxplot(attributes)
        labels = ['drv_age1', 'vh_age', 'vh_cyl', 'vh_din', 'pol_bonus', 'vh_sl_b',
                  'vh_sl_e', 'vh_value', 'vh_speed']

        self.set_axis_style(ax1, labels)

        plt.subplots_adjust(bottom=0.15, wspace=0.05)
        # plt.show()
        plt.xlabel("Attribute Type")
        plt.ylabel("Attribute Value")

        plt.savefig("box.pdf", bbox_inches='tight')

        ####################

        plt.cla()
        ax1.violinplot(attributes)

        labels = ['drv_age1', 'vh_age', 'vh_cyl', 'vh_din', 'pol_bonus',
                  'vh_sl_b', 'vh_sl_e', 'vh_value', 'vh_speed']

        self.set_axis_style(ax1, labels)

        plt.subplots_adjust(bottom=0.15, wspace=0.05)
        plt.xlabel("Attribute Type")
        plt.ylabel("Attribute Value")


        plt.savefig("violin.pdf", bbox_inches='tight')

    def evaluate_input2(self, x, y):
        """
        Function to evaluate data loaded from file

        """

        # Separate positive and negative results

        (neg_x, neg_y), (pos_x, pos_y) = self.separate_pos_neg(x, y)
        attributes1 = []
        attributes2 = []
        for i in range(np.shape(neg_x)[1]):
            attributes1.append(neg_x[:, i])
            attributes2.append(pos_x[:, i])

        fig, axs = plt.subplots(2, figsize=(11, 11))

        # type of plot
        axs[0].boxplot(attributes1)
        axs[1].boxplot(attributes2)
        labels = ['drv_age1', 'vh_age', 'vh_cyl', 'vh_din', 'pol_bonus', 'vh_sl_b',
                  'vh_sl_e', 'vh_value', 'vh_speed']

        self.set_axis_style(axs[0], labels)
        self.set_axis_style(axs[1], labels)

        # plt.show()
        axs[0].set(xlabel="Attribute Type", ylabel="Attribute Value")
        axs[0].set_title("No Claim")
        axs[1].set(xlabel="Attribute Type", ylabel="Attribute Value")
        axs[1].set_title("Claim")

        plt.subplots_adjust(bottom=0.15, wspace=0.05)
        plt.savefig("compare_box.pdf", bbox_inches='tight')

    def evaluate_input3(self, x, y):
        """
        Function to evaluate data loaded from file

        """

        # Separate positive and negative results

        (neg_x, neg_y), (pos_x, pos_y) = self.separate_pos_neg(x, y)
        attributes1 = []
        attributes2 = []
        difference = []
        for i in range(np.shape(neg_x)[1]):
            attributes1.append(np.mean(neg_x[:, i]))
            attributes2.append(np.mean(pos_x[:, i]))
            difference.append(((attributes2[i]-attributes1[i])*100)/attributes1[i])


        print(attributes1)
        print(attributes2)
        print(difference)


    def WeightedTrain(self, model, train_x, train_y, val_x, val_y, with_weight = True):
        # Weighted version of train
        # https://discuss.pytorch.org/t/unclear-about-weighted-bce-loss/21486
        # https://github.com/pytorch/pytorch/issues/5660
        if with_weight:
            pos_weight = torch.Tensor([(900 / 100)])
            criterion = WeightedBCELoss(pos_weight)
        else:
            criterion = nn.BCELoss()

        #print(torch.sum(train_y)/train_y.shape[0])
        optimiser = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=0.001, steps_per_epoch=900,epochs=100)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=5)

        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = []

        early_stopping = EarlyStopping(patience=20, verbose=True)

        num_epochs = 100

        for epoch in range(num_epochs):
            model.train()
            shuffled_train_x, shuffled_train_y = shuffle(train_x, train_y,
                                                         random_state=0)
            shuff_val_x, shuff_val_y = shuffle(val_x, val_y, random_state=0)

            x_batches = torch.split(shuffled_train_x, 20, dim=0)
            y_batches = torch.split(shuffled_train_y, 20, dim=0)
            x_val_batches = torch.split(shuff_val_x, 20, dim=0)
            y_val_batches = torch.split(shuff_val_y, 20, dim=0)

            for param_group in optimiser.param_groups:
                print("\nLearning Rate = ",param_group['lr'])

            # TRAIN MODEL
            for batch_i in range(len(x_batches)):

                batch_data = x_batches[batch_i]
                batch_label = y_batches[batch_i]

                # Forward pass: compute predicted outputs
                optimiser.zero_grad()
                batch_output = model(batch_data)

                # Backward Pass: Calculate loss by comparing ground truth to predictions on batch
                batch_loss = criterion(batch_output, batch_label)
                batch_loss.backward()
                optimiser.step()
                # Signal OneCycleLR adaptive LR
                scheduler.step()
                train_losses.append(batch_loss.item())


            # VALIDATE MODEL
            model.eval() #prep for evaluation

            for val_i in range(len(x_val_batches)):
                # Predict on validiation batch
                output = model(x_val_batches[val_i])
                # Calculate loss and save in list
                loss = criterion(output, y_val_batches[val_i])
                valid_losses.append(loss.item())

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            # Signal ReduceLROnPlateau adapaptive LR with validation loss
            scheduler.step(valid_loss)

            train_losses = []
            valid_losses = []

            print("Epoch = %d, Loss = %f" % (epoch + 1, batch_loss.item()))
            acc = analyse(model, val_x, val_y.numpy())[0]
            print("Validation Accuracy = ", acc)

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, model)

            if early_stopping.early_stop:
                print("Early Stopping")
                break

            model.load_state_dict(torch.load('checkpoint.pt'))

        return model



    def DownsampleTrain(self, model, train_x, train_y, val_x, val_y):

        loss_list = []
        criterion = nn.BCELoss()

        optimiser = torch.optim.AdamW(model.parameters(), lr=0.1)
        #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=0.1,steps_per_epoch=140,epochs=500)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=25)

        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = []

        early_stopping = EarlyStopping(patience=100, verbose=True)

        # Separate into positive and negative samples
        (neg_train_x, neg_train_y), (pos_train_x, pos_train_y) = \
            self.separate_pos_neg(train_x, train_y)

        print(len(pos_train_y))
        print(pos_train_x.shape)

        print(len(neg_train_y))

        neg_train_x, neg_train_y = shuffle(neg_train_x, neg_train_y)
        num_epochs = 500

        for epoch in range(num_epochs):
            model.train()

            neg_train_x, neg_train_y = shuffle(neg_train_x, neg_train_y)

            # 2004, 2508, 3012
            train_x_new = np.concatenate((neg_train_x[:int(1.0*len(pos_train_x))], pos_train_x))
            train_y_new = np.concatenate((neg_train_y[:int(1.0*len(pos_train_x))], pos_train_y))

            # concat first 1668 of this matrix to pos vals then proceed as if theyr're train_x and train_y
            shuffled_train_x, shuffled_train_y = shuffle(train_x_new, train_y_new,
                                                         random_state=0)
            shuff_val_x, shuff_val_y = shuffle(val_x, val_y, random_state=0)

            shuffled_train_x = torch.from_numpy(shuffled_train_x)
            shuffled_train_y = torch.from_numpy(shuffled_train_y)

            x_batches = torch.split(shuffled_train_x, 20, dim=0)
            y_batches = torch.split(shuffled_train_y, 20, dim=0)
            x_val_batches = torch.split(shuff_val_x, 20, dim=0)
            y_val_batches = torch.split(shuff_val_y, 20, dim=0)

            for param_group in optimiser.param_groups:
                print("\nLearning Rate = ", param_group['lr'])

            for batch_i in range(len(x_batches)):

                batch_data = x_batches[batch_i]
                batch_label = y_batches[batch_i]

                # Perform gradient decent algorithm to reduce loss
                optimiser.zero_grad()
                batch_output = model(batch_data)

                # Calculate loss by comparing ground truth to predictions on batch
                batch_loss = criterion(batch_output, batch_label)
                batch_loss.backward()
                optimiser.step()
                # Signal OneCycleLR adaptive LR
                #scheduler.step()

            # VALIDATE MODEL
            model.eval()  # prep for evaluation

            for val_i in range(len(x_val_batches)):
                # Predict on validiation batch
                output = model(x_val_batches[val_i])
                # Calculate loss and save in list
                loss = criterion(output, y_val_batches[val_i])
                valid_losses.append(loss.item())

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            # Signal ReduceLROnPlateau adapaptive LR with validation loss
            scheduler.step(valid_loss)

            train_losses = []
            valid_losses = []

            print("Epoch = %d, Loss = %f" % (epoch + 1, batch_loss.item()))
            acc = analyse(model, val_x, val_y.numpy())[0]
            print("Validation Accuracy = ", acc)

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, model)

            if early_stopping.early_stop:
                print("Early Stopping")
                break

            model.load_state_dict(torch.load('checkpoint.pt'))

        return model


    def separate_data(self, X_raw, Y_raw):
        """
        Separate data into training and test data in 85:15 ratio. The training
        data is then further partitioned to make validation set in fit( )
        class method, resulting in 70:15:15 split of train:validation:test
        data.
        """

        train_x, test_x, train_y, test_y = train_test_split(X_raw, Y_raw,
                                                              test_size=0.15)
        # Save split for evaluation later
        if not isinstance(test_y, np.ndarray):
            test_y = test_y.to_numpy(dtype=np.float)
        self.test_data = (self._preprocessor(test_x), test_y)

        return (train_x, train_y), (test_x, test_y)



    def fit(self, X_raw, y_raw):
        """Classifier training function.

        Here you will implement the training function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded
        y_raw : ndarray (optional)
            A one dimensional array, this is the binary target variable

        Returns
        -------
        self: (optional)
            an instance of the fitted model
        """

        # REMEMBER TO HAVE THE FOLLOWING LINE SOMEWHERE IN THE CODE
        # X_clean = self._preprocessor(X_raw)
        # YOUR CODE HERE
        #X_raw = X_raw.to_numpy()
        if not isinstance(y_raw, np.ndarray):
            y_raw = y_raw.to_numpy(dtype=np.float32)

        if not isinstance(X_raw, np.ndarray):
            X_raw = X_raw.to_numpy(dtype=np.float)

        X_clean = self._preprocessor(X_raw)

        # Split data into training and val
        # making val 17.65% of original makes it a total of 15% of original
        # data --> see separate_data( ) class method
        train_x, val_x, train_y, val_y = train_test_split(X_clean, y_raw,
                                                            test_size = 0.17647)

        # Save split for later evaluation
        self.train_data = (train_x, train_y)
        self.val_data = (val_x, val_y)

        print((train_x.shape, train_y.shape), (val_x.shape, val_y.shape))

        train_x = torch.from_numpy(train_x)
        train_y = torch.from_numpy(train_y)
        val_x = torch.from_numpy(val_x)
        val_y = torch.from_numpy(val_y)

        model = Insurance_NN()
        print(model)
        print(model(train_x).shape)

        model.train()

        #self.fitted_model = self.WeightedTrain(model, train_x, train_y, val_x, val_y)
        self.fitted_model = self.DownsampleTrain(model, train_x, train_y, val_x, val_y)

        self.save_model()

        return

        #model = load_model()
        model.eval()

        # -------------------- EVALUATE -------------------
        acc1 = analyse(model, train_x, train_y.numpy())
        acc2 = analyse(model, val_x, val_y.numpy())
        acc3 = analyse(model, test_x, test_y)
        print("Train Accuracy = ", acc1[0])
        print("Validation Accuracy = ", acc2[0])
        print("Test Accuracy = ", acc3[0])

        labels = ['No Accident', 'Accident']

        f, (ax1, ax2, ax3) = plt.subplots(1, 3)
        confusion1 = metrics.confusion_matrix(train_y.numpy(), acc1[1].numpy(), normalize='true')
        print(confusion1)
        confusion2 = metrics.confusion_matrix(val_y.numpy(), acc2[1].numpy(), normalize='true')
        confusion3 = metrics.confusion_matrix(test_y, acc3[1].numpy(), normalize='true')

        metrics.ConfusionMatrixDisplay(confusion1, labels).plot(ax=ax1)
        ax1.set_title("Training Set")
        metrics.ConfusionMatrixDisplay(confusion2, labels).plot(ax=ax2)
        ax2.set_title("Validation Set")
        metrics.ConfusionMatrixDisplay(confusion3, labels).plot(ax=ax3)
        ax3.set_title("Test Set")

        plt.gcf().set_size_inches(15, 5)
        plt.show()

    def predict(self, X_raw):
        """Classifier probability prediction function.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """

        # REMEMBER TO HAVE THE FOLLOWING LINE SOMEWHERE IN THE CODE
        # X_clean = self._preprocessor(X_raw)

        # YOUR CODE HERE

        #X_raw = X_raw.to_numpy()

        if not isinstance(X_raw, np.ndarray):
            X_raw= X_raw.to_numpy(dtype=np.float)

        X_clean = self._preprocessor(X_raw)
        self.fitted_model = load_model().fitted_model

        X_test = torch.Tensor(X_clean)
        oupt = self.fitted_model(X_test)  # a Tensor of floats
        pred_y = oupt >= 0.5  # a Tensor of 0s and 1s

        return pred_y.numpy()

        # -------------------- EVALUATE -------------------
        acc1 = analyse(model, train_x, train_y.numpy())
        acc2 = analyse(model, val_x, val_y.numpy())
        acc3 = analyse(model, test_x, test_y)
        print("Train Accuracy = ", acc1[0])
        print("Validation Accuracy = ", acc2[0])
        print("Test Accuracy = ", acc3[0])

        labels = ['No Accident', 'Accident']

        f, (ax1, ax2, ax3) = plt.subplots(1, 3)
        confusion1 = metrics.confusion_matrix(train_y.numpy(), acc1[1].numpy(),
                                              normalize='true')
        print(confusion1)
        confusion2 = metrics.confusion_matrix(val_y.numpy(), acc2[1].numpy(),
                                              normalize='true')
        confusion3 = metrics.confusion_matrix(test_y, acc3[1].numpy(),
                                              normalize='true')

        metrics.ConfusionMatrixDisplay(confusion1, labels).plot(ax=ax1)
        ax1.set_title("Training Set")
        metrics.ConfusionMatrixDisplay(confusion2, labels).plot(ax=ax2)
        ax2.set_title("Validation Set")
        metrics.ConfusionMatrixDisplay(confusion3, labels).plot(ax=ax3)
        ax3.set_title("Test Set")

        plt.gcf().set_size_inches(15, 5)
        plt.show()

        return  # YOUR PREDICTED CLASS LABELS

    def evaluate_architecture(self, with_test = False):
        """Architecture evaluation utility.

        Populate this function with evaluation utilities for your
        neural network.

        You can use external libraries such as scikit-learn for this
        if necessary.
        """
        train_x, train_y = self.test_data
        val_x, val_y = self.val_data

        # Calculate and print accuracies based on model predictions
        acc1 = analyse(self.fitted_model, train_x, train_y)
        acc2 = analyse(self.fitted_model, val_x, val_y)
        print("Train Accuracy = ", acc1[0])
        print("Validation Accuracy = ", acc2[0])

        labels = ['No Claim', 'Claim']

        if with_test:
            test_x, test_y = self.test_data
            acc3 = analyse(self.fitted_model, test_x, test_y)
            print("Test Accuracy = ", acc3[0])

            f, (ax1, ax2, ax3) = plt.subplots(1, 3)
            confusion_test = metrics.confusion_matrix(test_y, acc3[1].numpy(),
                                                  normalize='true')
            # Plot confusion for test data
            metrics.ConfusionMatrixDisplay(confusion_test, labels).plot(ax=ax3)
            ax3.set_title("Test Set")
            plot_width = 15
        else:
            f, (ax1, ax2) = plt.subplots(1, 2)
            plot_width = 10

        # Construct training and validation normalised confusion matricies
        confusion_train = metrics.confusion_matrix(train_y, acc1[1].numpy(),
                                              normalize='true')
        confusion_val = metrics.confusion_matrix(val_y, acc2[1].numpy(),
                                              normalize='true')

        # Plot training and validation set confusion matricies
        metrics.ConfusionMatrixDisplay(confusion_train, labels).plot(ax=ax1)
        ax1.set_title("Training Set")
        metrics.ConfusionMatrixDisplay(confusion_val, labels).plot(ax=ax2)
        ax2.set_title("Validation Set")

        plt.gcf().set_size_inches(plot_width, 5)
        plt.show()

        return

    def save_model(self):
        # Please alter this file appropriately to work in tandem with your load_model function below
        with open('part2_claim_classifier.pickle', 'wb') as target:
            pickle.dump(self, target)


def load_model():
    # Please alter this section so that it works in tandem with the save_model method of your class
    with open('part2_claim_classifier.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    return trained_model

# ENSURE TO ADD IN WHATEVER INPUTS YOU DEEM NECESSARRY TO THIS FUNCTION
def ClaimClassifierHyperParameterSearch():
    """Performs a hyper-parameter for fine-tuning the classifier.

    Implement a function that performs a hyper-parameter search for your
    architecture as implemented in the ClaimClassifier class. 

    The function should return your optimised hyper-parameters. 
    """

    return  # Return the chosen hyper parameters


class Insurance_NN(nn.Module):
    def __init__(self):
        super(Insurance_NN, self).__init__()

        self.apply_layers = nn.Sequential(
            # 2 fully connected hidden layers of 8 neurons goes to 1
            # 9 - (100 - 10) - 1
            nn.Linear(9, 100),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(100, 10),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.apply_layers(x)
        return x.view(len(x))

if __name__ == "__main__":

    test = ClaimClassifier()
    x, y = test.load_data("part2_training_data.csv")
    print(test._preprocessor(x))
    """
    train_data, test_data = test.separate_data(x, y)
    test.fit(train_data[0], train_data[1])
    predictions_test = test.predict(pd.DataFrame(test_data[0]))


    confusion_test = metrics.confusion_matrix(test.test_data[1], predictions_test,
                                             normalize='true')

    labels = ['No Claim', 'Claim']
    metrics.ConfusionMatrixDisplay(confusion_test, labels).plot()

    test.evaluate_architecture(True)
    test.evaluate_architecture()
    """
    """
    #test.evaluate_input3(x, y)
    x_clean = test._preprocessor(x)
    #print(x_clean.shape)
    #test.fit(x, y)
    test.evaluate_input3(x, y)
    
    data_set = np.genfromtxt("part2_training_data.csv", dtype=float, delimiter=',', skip_header=1)
    num_att = len(data_set[0])  # number of parameters

    claims = np.array(data_set[:, (num_att - 1)], dtype=np.float32)
    claim_amount = np.array(data_set[:, (num_att - 2)], dtype=np.float32)
    print(max(claim_amount))

    amounts_list = []

    for i in range(len(claim_amount)):
        if claims[i] == 1:
            amounts_list.append(claim_amount[i])

    print(amounts_list)

    fig, ax1 = plt.subplots(figsize=(4, 4), sharey=True)

    # type of plot
    ax1.boxplot(amounts_list)

    labels = ['Claim Amount']
    test = ClaimClassifier()
    test.set_axis_style(ax1, labels)

    plt.subplots_adjust(bottom=0.15, wspace=0.05)
    # plt.show()
    plt.xlabel("")
    plt.ylabel("Amount")

    plt.show()
    """