import sys
sys.path.append('..')
import numpy as np
import joblib

from sklearn.metrics import recall_score, accuracy_score, precision_score, roc_auc_score, confusion_matrix
from core.config import SVM_MODEL, SELECTED_FEATURES_IDX, SVM_CLASSIFIER_PATH
from code.data_preparation import DataPreparation


class TrainTestClassifier:
    
    def __init__(self, X_train: np.ndarray, 
                 y_train: np.ndarray, 
                 X_test: np.ndarray, 
                 y_test: np.ndarray, 
                 classifier_model):
        """
        Parameters:
            X_train (np.ndarray): train data
            y_train (np.ndarray): train labels
            X_test (np.ndarray): test data
            y_test (np.ndarray): test labels
            classifier_model: classifier used
            
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.classifier_model = classifier_model

    def train_model(self, classifier_save_path):
        """
        train classifier on data and save it
        Parameters: 
            classifier_save_path (str): saved classifier path

        """

        classifier = self.classifier_model
        # Train the classifier
        classifier.fit(self.X_train, self.y_train)

        # Save the classifier
        joblib.dump(classifier, classifier_save_path)


    def test_model(self, classifier_save_path):
        """
        test classifier performance
        Parameters: 
            classifier_save_path (str): saved classifier path

        Return:
            recall (float): recall metric
            accuracy (float): accuracy metric
            precision (float): precision metric
            roc (float): roc_auc metric
            confusion_mat (np.ndarray): confusion matrix

        """
        classifier = joblib.load(classifier_save_path)
        # Predict
        predicted_labels = classifier.predict(self.X_test)

        # Compute test metrics
        confusion_mat = confusion_matrix(self.y_test, predicted_labels)
        recall = recall_score(self.y_test, predicted_labels)
        accuracy = accuracy_score(self.y_test, predicted_labels)
        precision = precision_score(self.y_test, predicted_labels)
        roc = roc_auc_score(self.y_test, predicted_labels)

        return recall, accuracy, precision, roc, confusion_mat
    

def main():
    
    data_prep = DataPreparation(data_csv_path = "./data/breast-cancer.csv",
                                selected_features_idx=SELECTED_FEATURES_IDX, 
                                test_data_size=0.2)
    X_train, X_test, y_train, y_test = data_prep.split_data()
    print(y_test)
    
    train_test_classifier = TrainTestClassifier(X_train = X_train, 
                                                y_train = y_train, 
                                                X_test = X_test, 
                                                y_test = y_test,
                                                classifier_model = SVM_MODEL)
    
    train_test_classifier.train_model(classifier_save_path = SVM_CLASSIFIER_PATH)
    recall, accuracy, precision, roc, confusion_mat = train_test_classifier.test_model(classifier_save_path = SVM_CLASSIFIER_PATH)
    
    print(f"recall = {recall}")
    print(f"accuracy = {accuracy}")
    print(f"precision = {precision}")
    print(f"roc = {roc}")
    print(f"confusion matrix = {confusion_mat}")


if __name__ == "__main__":
    main()