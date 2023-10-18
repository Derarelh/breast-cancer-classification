import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from core.config import SELECTED_FEATURES_IDX, SCALER_PATH, RANDOM_STATE


class DataPreparation:
    """
    In this class data is preprocessed
    """
    def __init__(self, data_csv_path: str, 
                 selected_features_idx: list, 
                 test_data_size: float):
        """
        Parameters:
            data_csv_path (str): csv file path
            selected_features_idx(list): index list of best features
            test_data_size (float): percentage of test data
            
        """
        self.data_csv_path = data_csv_path
        self.selected_features_idx = selected_features_idx
        self.test_data_size = test_data_size
        
    def load_dataset(self):


        """
        Load dataset from csv file and keep useful features
        Returns:
            df_selected_features (pd.DataFrame): dataframe containing useful features
        """

        # Load dataset
        df = pd.read_csv(self.data_csv_path, index_col= "id")
        df_selected = df.iloc[:,self.selected_features_idx]
        df_selected = df_selected.assign(**{"diagnosis": df_selected.pop("diagnosis")})
        df_selected['diagnosis'] = df_selected['diagnosis'].map({'B': 0, 'M': 1})
        
        return df_selected

    def split_data(self):

        """
        Split data to train/test sets
        Returns:
            X_train (np.ndarray): train data
            X_test (np.ndarray): test data
            y_train (np.ndarray): train label
            y_test (np.ndarray): test label
        """
        
        df_selected_features = self.load_dataset()
        
        # Extract labels and data
        y = df_selected_features['diagnosis'].values
        df_vals = df_selected_features.drop(['diagnosis'],axis=1).values

        # Scale dataset
        minmaxscaler = MinMaxScaler()
        X = minmaxscaler.fit_transform(df_vals)
        
        # Save scaler
        joblib.dump(minmaxscaler, SCALER_PATH)
        
        # Split data to train/test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size=self.test_data_size, 
                                                            random_state=RANDOM_STATE)

        return X_train, X_test, y_train, y_test
    
    
def main():
    data_prep = DataPreparation(data_csv_path = "./data/breast-cancer.csv", selected_features_idx=SELECTED_FEATURES_IDX, test_data_size=0.2)
    df = data_prep.load_dataset()
    print(df)
    X_train, X_test, y_train, y_test = data_prep.split_data()
    print(X_train)
    print(X_test)
    print(y_train)
    print(y_test)
    
if __name__ == "__main__":
    main()
