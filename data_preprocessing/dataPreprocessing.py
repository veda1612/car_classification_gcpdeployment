import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from app_logger import logger

class Preprocessor:
    """
        This class shall  be used to clean and transform the data before Prediction or Training.
    """

    def __init__(self,filepath,process_type):
        self.file_object = open("logs/PreProcessing_Log.txt", 'a+')
        self.logger_object = logger.App_Logger()

    def data_preprocess(self,filepath,process_type):

        self.logger_object.log(self.file_object, 'Pre-Processing - Started the pre-processing of the Validated file')

        try:
            if process_type == "T":
                # Reading the inputs given by the user
                df = pd.read_csv(filepath)
                df.to_csv('Training_Files/Good_Raw/df_New.csv', index=False)
                self.logger_object.log(self.file_object, 'Pre-Processing - Successfully read the Re-Training file')

                # Reading the old train data
                df_old_traindata = pd.read_csv("Old_TrainData/Original_data.csv")
                df_old_traindata.to_csv('Training_Files/Good_Raw/df_Old.csv', index=False)
                self.logger_object.log(self.file_object, 'Re-Training - Successfully read the Old Train Data.')

                # Merging the Old Train Data with the inputs given by the user
                df_retrain = pd.concat([df, df_old_traindata], axis=0)
                
                #renaming the columns
                df_retrain.columns = ['Price', 'Maintenance Cost', 'Number of Doors', 'Capacity', 'Size of Luggage Boot', 'safety', 'Decision']
                
                # Label Encoding ,Convert categories into integers for each column.
                df_retrain["Decision"]=df_retrain["Decision"].replace(('unacc','acc','good','vgood'),(0,1,2,3))
                df_retrain['safety']=df_retrain['safety'].replace(('low', 'med', 'high'), (0, 1, 2))
                df_retrain['Price']=df_retrain['Price'].replace(('low', 'med', 'high', 'vhigh'), (0, 1, 2, 3))
                df_retrain['Maintenance Cost']=df_retrain['Maintenance Cost'].replace(('low', 'med', 'high', 'vhigh'), (0, 1, 2, 3))
                df_retrain['Number of Doors']=df_retrain['Number of Doors'].replace('5more', 5)
                df_retrain['Capacity']=df_retrain['Capacity'].replace('more', 5)
                df_retrain['Size of Luggage Boot']=df_retrain['Size of Luggage Boot'].replace(('small', 'med', 'big'), (0, 1, 2))
                
                # Splitting the dataset into X & Y
                df_X = df_retrain.drop(['Decision'], axis=1)
                df_Y = df_retrain.Decision
                
                #Balancing the imbalanced data - Over Sampling
                oversample = SMOTE()
                df_X, df_Y = oversample.fit_resample(df_X, df_Y)

                               
                df_Y.to_csv('Training_Files/Good_Raw/df_Y.csv', index=False)
                self.logger_object.log(self.file_object, 'Pre-Processing - Successfully saved the df_Y file desired location.')

            if process_type == "P":
                #  reading the inputs given by the user
                df_X = pd.read_csv(filepath)                
                self.logger_object.log(self.file_object, 'Pre-Processing - Successfully read the Prediction file.')
                # Renaming the columns
                df_X.columns = ['Price', 'Maintenance Cost', 'Number of Doors', 'Capacity', 'Size of Luggage Boot', 'safety']
                df_X.to_csv('Preprocessed_Files/CarDetailsForPrediction.csv', index=False)
                # Label Encoding ,Convert categories into integers for each column.
                df_X['safety']=df_X['safety'].replace(('low', 'med', 'high'), (0, 1, 2))
                df_X['Price']=df_X['Price'].replace(('low', 'med', 'high', 'vhigh'), (0, 1, 2, 3))
                df_X['Maintenance Cost']=df_X['Maintenance Cost'].replace(('low', 'med', 'high', 'vhigh'), (0, 1, 2, 3))
                df_X['Number of Doors']=df_X['Number of Doors'].replace('5more', 5)
                df_X['Capacity']=df_X['Capacity'].replace('more', 5)
                df_X['Size of Luggage Boot']=df_X['Size of Luggage Boot'].replace(('small', 'med', 'big'), (0, 1, 2))
            

            #  Scale the dataset using Satandard scaler
            scalar = StandardScaler()
            df_X_scaled = scalar.fit_transform(df_X)
            self.logger_object.log(self.file_object, 'Pre-Processing - Successfully scaled down the data.')

            df_X_scaled = pd.DataFrame(df_X_scaled, columns=df_X.columns)

            df_X_scaled.to_csv('Preprocessed_Files/Preprocessed_File.csv', index=False)

            self.logger_object.log(self.file_object, 'Pre-Processing - Successfully saved the pre-processed file at desired location.')

            return ("Pre-Processing Success")

        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in data pre-processing. Exception message:  '+str(e))
            return "Error during input file pre-processing!Please check logs for details."

