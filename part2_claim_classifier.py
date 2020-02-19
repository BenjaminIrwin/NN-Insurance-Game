import numpy as np
import pickle
import matplotlib.pyplot as plt

import random
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class InsuranceNN(nn.Module):
    def __init__(self):
        super(InsuranceNN, self).__init__()

        self.apply_layers = nn.Sequential(
            # 2 fully connected hidden layers of 8 neurons goes to 1
            # 4 - (8 - 8) - 1
            nn.Linear(9, 8),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(8, 8),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.apply_layers(x)
        return x.view(len(x))


class Batch_Maker:
    def __init__(self, num_items, batch_size, seed = 0):
        self.indices = np.arange(num_items)
        self.num_items = num_items
        self.batch_size = batch_size
        self.rnd = np.random.RandomState(seed)
        self.rnd.shuffle(self.indices)
        self.ptr = 0

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
        pass

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
        data_set = np.genfromtxt(filename, dtype=float, delimiter=',', skip_header=1)

        num_att = len(data_set[0])  # number of parameters

        x = np.array(data_set[:, :(num_att-2)], dtype=np.float32)
        y = np.array(data_set[:, (num_att-1)], dtype=np.float)

        return x, y

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

        num_samples, num_att = X_raw.shape

        for att in range(num_att):
            max_att = np.amax(X_raw[:, att])
            min_att = np.amin(X_raw[:, att])

            for sample in range(num_samples):
               X_raw[sample, att] = (X_raw[sample, att] - min_att)/(max_att -
                                                                    min_att)


        return X_raw.astype(np.float32)

    def set_axis_style(self, ax, labels):
        ax.get_xaxis().set_tick_params(direction='out')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_xlim(0.25, len(labels) + 0.75)
        ax.set_xlabel('Sample name')

    def evaluate_input(self, X_raw):
        """
        Function to evaluate data loaded from file

        """

        attributes = []
        for i in range(np.shape(X_raw)[1]):
            attributes.append(X_raw[:, i])


        fig, ax1 = plt.subplots(figsize=(11, 4), sharey=True)

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
        X_clean = self._preprocessor(X_raw)

        # Split data into training and test data
        train_x, test_x, train_y, test_y = train_test_split(X_clean, y_raw,
                                                            test_size = 0.1)
        print((train_x.shape, train_y.shape), (test_x.shape, test_y.shape))

        train_x = torch.from_numpy(train_x)
        #train_x = train_x.view(-1, 9)
        train_y = torch.from_numpy(train_y)
        print(train_x.shape, train_y.shape)
        print(train_y.type())
        model = InsuranceNN()
        print(model)
        print(model(train_x).shape)

        model.train()
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            model = model.cuda()
            print('Using GPU...')
        else:
            print('Using CPU...')

        criterion = nn.BCELoss()
        optimiser = optim.AdamW(model.parameters(), lr=0.001)

        n_iterations = 30000
        batch_size = 64

        for iteration in range(n_iterations):
            batch_data = torch.empty(batch_size, 9, dtype=torch.float)
            batch_label = torch.empty(batch_size, dtype=torch.float)
            # Fill random batch
            for i in range(batch_size):
                index = random.randint(0, train_x.shape[0] - 1)
                batch_data[i] = train_x[index]
                batch_label[i] = train_y[index]

            batch_data = torch.autograd.Variable(batch_data)
            batch_label = torch.autograd.Variable(batch_label)
            batch_data = batch_data.type(torch.FloatTensor)
            batch_label = batch_label.type(torch.FloatTensor)

            if use_gpu:
                batch_data = batch_data.cuda()
                batch_label = batch_label.cuda()

            # Perform gradient decent algorithm to reduce loss
            optimiser.zero_grad()
            batch_output = model(batch_data)
            batch_label = batch_label.view(batch_size)
            # Calculate loss by comparing ground truth to predictions on batch
            batch_loss = criterion(batch_output, batch_label)
            batch_loss.backward()
            optimiser.step()

            if (iteration + 1) % 100 == 0:
                print("Iteration = %d, Loss = %f" % (
                iteration + 1, batch_loss.item()))


        # --------------------- TEST ----------------------

        model.eval()

        predictions = []
        test_x = torch.from_numpy(test_x)

        for sample_i in range(test_x.shape[0]):
            test_sample = torch.autograd.Variable(test_x[sample_i:sample_i + 1].clone())
            test_sample = test_sample.type(torch.FloatTensor)

            if use_gpu:
                sample_data = test_sample.cuda()

            sample_out = model(test_sample)
            pred = torch.round(sample_out)
            predictions.append(pred)
            if (sample_i + 1) % 100 == 0:
                print("Total tested = %d" % (sample_i + 1))

        # -------------------- EVALUATE -------------------

        count = 0
        for i in range(len(predictions)):
            if predictions[i] == test_y[i]:
                count += 1

        print("Test Accuracy = ", count * 100 / 10000, "%")

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

        return  # YOUR PREDICTED CLASS LABELS

    def evaluate_architecture(self):
        """Architecture evaluation utility.

        Populate this function with evaluation utilities for your
        neural network.

        You can use external libraries such as scikit-learn for this
        if necessary.
        """
        pass

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

test = ClaimClassifier()
x, y = test.load_data("part2_training_data.csv")

x_clean = test._preprocessor(x)
print(x_clean.shape)
test.fit(x, y)

