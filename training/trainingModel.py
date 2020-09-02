import pandas as pd
from app_logger import logger
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

class Training:
    """
        This class shall be used to Re-Train the model from the new training data provided by the user, appended with full train Data.
    """

    def __init__(self):
        self.file_object = open("logs/Training_Log.txt", 'a+')
        self.logger_object = logger.App_Logger()

    def train_model(self):

        self.logger_object.log(self.file_object, 'Re-Training - Started the Re-Training of the model')

        try:

            #  reading the preprocessed file from the server
            df_X_scaled = pd.read_csv("Preprocessed_Files/Preprocessed_File.csv")
            self.logger_object.log(self.file_object, 'Re-Training - Successfully read the pre-processed df_X file received for Training.')

            df_Y = pd.read_csv("Training_Files/Good_Raw/df_Y.csv")
            self.logger_object.log(self.file_object, 'Re-Training - Successfully read the df_Y file received for Training.')

            # Splitting the dataset into train & test
            x_train, x_test, y_train, y_test = train_test_split(df_X_scaled, df_Y, test_size=0.15, random_state=0)
            #xtrain_up, ytrain_up = smt.fit_sample(x_train, y_train)
            self.logger_object.log(self.file_object, 'Re-Training - Train & Test Split of the preprocessed file done')

            # Initialising the object of Support Vector machine classifier
            svcmodel = SVC(kernel='rbf', gamma=0.01, C=1)

            # Retraining the model with the hyper parameters already tuned using RandomoizedserarchCV
            svcmodel = svcmodel.fit(x_train, y_train)
            self.logger_object.log(self.file_object, 'Re-Training - Successfully completed the Model Re-training')

            # Saving this new model to the file
            # open a file, where you ant to store the data
            file1 = open('SVC_rbf_model_New.pkl', 'wb')

            # dump information to that file
            pickle.dump(svcmodel , file1)
            
            self.logger_object.log(self.file_object, 'Re-Training - NewModel file successfully saved')

            return ("Re-Training successfully completed and newmodel file saved.")

        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in data pre-processing. Exception message:  '+str(e))
            return "Error during Re-training! Please check logs for details."
