from sklearn.svm import SVC

# Selected features index
SELECTED_FEATURES_IDX = [22, 23, 7, 6, 27, 20, 2, 0, 3, 13]

#SVM Model
SVM_MODEL = SVC(C=100, kernel='poly')

#SVM and SCALER models path
SCALER_PATH = "./model/scaler.save"
SVM_CLASSIFIER_PATH = "./model/classifier_model"

# Random state value

RANDOM_STATE = 42