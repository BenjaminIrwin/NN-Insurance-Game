import numpy as np
import pickle
import matplotlib.pyplot as plt
import copy

import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def akkuracy(model, data_x, data_y):
    # data_x and data_y are numpy array-of-arrays matrices
    X = torch.Tensor(data_x)
    Y = torch.ByteTensor(data_y)   # a Tensor of 0s and 1s
    oupt = model(X)            # a Tensor of floats
    pred_y = oupt >= 0.5       # a Tensor of 0s and 1s
    num_correct = torch.sum(Y==pred_y)  # a Tensor
    acc = (num_correct.item() * 100.0 / len(data_y))  # scalar
    return acc


class InsuranceNN(nn.Module):
    def __init__(self):
        super(InsuranceNN, self).__init__()

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
        y = np.array(data_set[:, (num_att-1)], dtype=np.float32)

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

    def WeightedTrain(self, model, train_x, train_y, test_x, test_y, use_gpu, with_weight = True):
        # Weighted version of train
        # https://discuss.pytorch.org/t/unclear-about-weighted-bce-loss/21486
        # https://github.com/pytorch/pytorch/issues/5660
        if with_weight:
            pos_weight = torch.Tensor([(900 / 100)])
            criterion = WeightedBCELoss(pos_weight)
        else:
            criterion = nn.BCELoss()


        #optimiser = optim.AdamW(model.parameters(), lr=0.001)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser)

        optimiser = torch.optim.AdamW(model.parameters(), lr=0.001)
        #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=0.001, steps_per_epoch=900,epochs=50)
        num_epochs = 50

        for epoch in range(num_epochs):
            shuffled_train_x, shuffled_train_y = shuffle(train_x, train_y,
                                                         random_state=0)

            x_batches = torch.split(shuffled_train_x, 20, dim=0)
            y_batches = torch.split(shuffled_train_y, 20, dim=0)
            print(len(x_batches))

            for param_group in optimiser.param_groups:
                print(param_group['lr'])

            for batch_i in range(len(x_batches)):

                batch_data = x_batches[batch_i]
                batch_label = y_batches[batch_i]

                if use_gpu:
                    batch_data = batch_data.cuda()
                    batch_label = batch_label.cuda()

                # Perform gradient decent algorithm to reduce loss
                optimiser.zero_grad()
                batch_output = model(batch_data)

                # Calculate loss by comparing ground truth to predictions on batch
                batch_loss = criterion(batch_output, batch_label)
                batch_loss.backward()
                optimiser.step()
                #scheduler.step()

           # scheduler.step(batch_loss)
            print("Epoch = %d, Loss = %f" % (epoch + 1, batch_loss.item()))
            acc = akkuracy(model, test_x, test_y)
            print("Accuracy = ", acc)

        return model



    def UpsampleTrain(self, model, train_x, train_y, test_x, test_y, use_gpu):

        loss_list = []
        criterion = nn.BCELoss()
        # lambda2 = lambda epoch: 0.96**epoch
        #optimiser = optim.Adam(model.parameters(), lr=1)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser)
        # Separate into positive and negative samples

        optimiser = torch.optim.SGD(model.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=0.1,
                                                        steps_per_epoch=140,
                                                        epochs=500)


        pos_train_y = []
        pos_train_x = np.empty((0,9), np.float32)
        neg_train_y = []
        neg_train_x = np.empty((0,9), np.float32)
        for i in range(train_y.shape[0]):
            if train_y[i] == 1:
                pos_train_y.append(train_y[i])
                pos_train_x = np.vstack((pos_train_x, train_x[i]))
            else:
                neg_train_y.append(train_y[i])
                neg_train_x = np.vstack((neg_train_x, train_x[i]))

        neg_train_y = np.array(neg_train_y, dtype=np.float32)
        pos_train_y = np.array(pos_train_y, dtype=np.float32)
        print(len(pos_train_y))
        print(pos_train_x.shape)

        print(len(neg_train_y))

        neg_train_x, neg_train_y = shuffle(neg_train_x, neg_train_y)
        num_epochs = 500
        for epoch in range(num_epochs):
            acc = akkuracy(model, test_x, test_y)
            print("-2: ", acc)

            neg_train_x, neg_train_y = shuffle(neg_train_x, neg_train_y)

            # 2004, 2508, 3012
            train_x_new = np.concatenate((neg_train_x[:1668], pos_train_x))
            train_y_new = np.concatenate((neg_train_y[:1668], pos_train_y))

            # concat first 1668 of this matrix to pos vals then proceed as if theyr're train_x and train_y
            shuffled_train_x, shuffled_train_y = shuffle(train_x_new, train_y_new,
                                                         random_state=0)
            shuffled_train_x = torch.from_numpy(shuffled_train_x)
            shuffled_train_y = torch.from_numpy(shuffled_train_y)

            x_batches = torch.split(shuffled_train_x, 24, dim=0)
            y_batches = torch.split(shuffled_train_y, 24, dim=0)

            for param_group in optimiser.param_groups:
                print(param_group['lr'])

            for batch_i in range(len(x_batches)):

                batch_data = x_batches[batch_i]
                batch_label = y_batches[batch_i]

                if use_gpu:
                    batch_data = batch_data.cuda()
                    batch_label = batch_label.cuda()

                # Perform gradient decent algorithm to reduce loss
                optimiser.zero_grad()
                batch_output = model(batch_data)

                # Calculate loss by comparing ground truth to predictions on batch
                batch_loss = criterion(batch_output, batch_label)
                batch_loss.backward()
                optimiser.step()
                scheduler.step()
            loss_list.append(batch_loss.item())

            #scheduler.step(batch_loss)

            print("Epoch = %d, Loss = %f" % (epoch + 1, batch_loss.item()))
            acc = akkuracy(model, test_x, test_y)
            print("Accuracy1 = ", acc)
            acc = akkuracy(model, test_x, test_y)
            print("Accuracy2 = ", acc)

        acc = akkuracy(model, test_x, test_y)
        print("Accuracy3 = ", acc)
        return model


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
        #print(train_x.shape, train_y.shape)
        #print(train_y.type())
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

        model = self.WeightedTrain(model, train_x, train_y, test_x, test_y, use_gpu)
        #model = self.UpsampleTrain(model, train_x, train_y, test_x, test_y, use_gpu)

        acc = akkuracy(model, test_x, test_y)
        print("1: ", acc)
        # --------------------- TEST ----------------------
        self.save_model(model)
        print("1: ", acc)
        #model = load_model()
        model.eval()
        acc = akkuracy(model, test_x, test_y)
        print("1: ", acc)
        predictions = []


        test_x = torch.from_numpy(test_x)
        for sample_i in range(test_x.shape[0]):
            test_sample = torch.autograd.Variable(test_x[sample_i:sample_i + 1].clone())
            test_sample = test_sample.type(torch.FloatTensor)

            if use_gpu:
                test_sample = test_sample.cuda()

            sample_out = model(test_sample)
            #print(sample_out)
            if sample_out >= 0.5:
                predictions.append(1)
            else:
                predictions.append(0)

            if (sample_i + 1) % 500 == 0:
                print("Total tested = %d" % (sample_i + 1))

        # -------------------- EVALUATE -------------------
        acc = akkuracy(model, test_x, test_y)
        print("2: ", acc)
        count = 0
        test_y.astype(int)
        for i in range(len(predictions)):
            test_y[i] = int(test_y[i])
            predictions[i] = int(predictions[i])

            if predictions[i] == test_y[i]:
                count += 1

        print("Test Accuracy = ", count * 100 / 2000, "%")
        acc = akkuracy(model, test_x, test_y)
        print("Accuracy = ", acc)
        print(predictions)
        print(test_y)

        labels = ['No Accident', 'Accident']
        confusion = metrics.confusion_matrix(test_y, predictions, normalize='true')
        metrics.ConfusionMatrixDisplay(confusion, labels).plot()
        plt.gcf().set_size_inches(5, 5)
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

        return  # YOUR PREDICTED CLASS LABELS

    def evaluate_architecture(self):
        """Architecture evaluation utility.

        Populate this function with evaluation utilities for your
        neural network.

        You can use external libraries such as scikit-learn for this
        if necessary.
        """
        pass

    def save_model(self, model):
        # Please alter this file appropriately to work in tandem with your load_model function below
        with open('part2_claim_classifier.pickle', 'wb') as target:
            pickle.dump(model, target)


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



if __name__ == "__main__":
    test = ClaimClassifier()
    x, y = test.load_data("part2_training_data.csv")

    x_clean = test._preprocessor(x)
    print(x_clean.shape)
    test.fit(x, y)

