from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

import matplotlib.pyplot as plt

def fit_and_calibrate_classifier(classifier, X, y):
    # DO NOT ALTER THIS FUNCTION
    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, train_size=0.85, random_state=0)
    classifier = classifier.fit(X_train, y_train)

    # This line does the calibration for you
    calibrated_classifier = CalibratedClassifierCV(
        classifier, method='sigmoid', cv='prefit').fit(X_cal, y_cal)
    return calibrated_classifier


# class for part 3
class PricingModel():
    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY
    def __init__(self, calibrate_probabilities=False):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary.
        """
        self.y_mean = None
        self.calibrate = calibrate_probabilities
        # =============================================================
        # READ ONLY IF WANTING TO CALIBRATE
        # Place your base classifier here
        # NOTE: The base estimator must have:
        #    1. A .fit method that takes two arguments, X, y
        #    2. Either a .predict_proba method or a decision
        #       function method that returns classification scores
        #
        # Note that almost every classifier you can find has both.
        # If the one you wish to use does not then speak to one of the TAs
        #
        # If you wish to use the classifier in part 2, you will need
        # to implement a predict_proba for it before use
        # =============================================================
        self.base_classifier = None # ADD YOUR BASE CLASSIFIER HERE


    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY TO THE _preprocessor METHOD
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
        X: ndarray
            A clean data set that is used for training and prediction.
        """
        # =============================================================
        # YOUR CODE HERE

        return  # YOUR CLEAN DATA AS A NUMPY ARRAY

    def fit(self, X_raw, y_raw, claims_raw):
        """Classifier training function.

        Here you will use the fit function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            This is the raw data as downloaded
        y_raw : ndarray
            A one dimensional array, this is the binary target variable
        claims_raw: ndarray
            A one dimensional array which records the severity of claims

        Returns
        -------
        self: (optional)
            an instance of the fitted model

        """
        nnz = np.where(claims_raw != 0)[0]
        self.y_mean = np.mean(claims_raw[nnz])
        # =============================================================
        # REMEMBER TO A SIMILAR LINE TO THE FOLLOWING SOMEWHERE IN THE CODE
        X_clean = self._preprocessor(X_raw)

        # THE FOLLOWING GETS CALLED IF YOU WISH TO CALIBRATE YOUR PROBABILITES
        if self.calibrate:
            self.base_classifier = fit_and_calibrate_classifier(
                self.base_classifier, X_clean, y_raw)
        else:
            self.base_classifier = self.base_classifier.fit(X_clean, y_raw)
        return self.base_classifier

    def predict_claim_probability(self, X_raw):
        """Classifier probability prediction function.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            This is the raw data as downloaded

        Returns
        -------
        ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """
        # =============================================================
        # REMEMBER TO A SIMILAR LINE TO THE FOLLOWING SOMEWHERE IN THE CODE
        # X_clean = self._preprocessor(X_raw)


        return  # return probabilities for the positive class (label 1)

    def predict_premium(self, X_raw):
        """Predicts premiums based on the pricing model.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : numpy.ndarray
            A numpy array, this is the raw data as downloaded

        Returns
        -------
        numpy.ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """
        # =============================================================
        # REMEMBER TO INCLUDE ANY PRICING STRATEGY HERE.
        # For example you could scale all your prices down by a factor

        return self.predict_claim_probability(X_raw) * self.y_mean

    def save_model(self):
        """Saves the class instance as a pickle file."""
        # =============================================================
        with open('part3_pricing_model.pickle', 'wb') as target:
            pickle.dump(self, target)


    # -------- NEW FUNCTIONS -----------

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

        x = np.array(data_set[:, :(num_att-2)], dtype=str)
        y = np.array(data_set[:, (num_att-1)], dtype=np.float)

        return x, y


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


def load_model():
    # Please alter this section so that it works in tandem with the save_model method of your class
    with open('part3_pricing_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    return trained_model


if __name__ == "__main__":
    test = PricingModel()
