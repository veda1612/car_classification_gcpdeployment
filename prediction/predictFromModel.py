import pandas as pd
from app_logger import logger
import pickle

class Prediction:
    """
        This class shall be used to obtain the predictions from the based saved model.
    """

    def __init__(self):
        self.file_object = open("logs/Prediction_Log.txt", 'a+')
        self.logger_object = logger.App_Logger()

    def predict_model(self):

        self.logger_object.log(self.file_object, 'Prediction - Started the Predictions from the best model ')

        try:

            #  reading the preprocessed file from the server
            df_X_scale = pd.read_csv("Preprocessed_Files/Preprocessed_File.csv")
            self.logger_object.log(self.file_object, 'Prediction - Successfully read the pre-processed file for Prediction.')

            modelname = 'SVC_rbf_model_v1.pkl'

            # loading the model file from the storage
            model = pickle.load(open(modelname, 'rb'))
            self.logger_object.log(self.file_object, 'Prediction - Successfully loaded the model.')

            # predictions using the loaded model file
            prediction = model.predict(df_X_scale)
            self.logger_object.log(self.file_object, 'Prediction - Successfully completed the predictions.')

            df_result = pd.DataFrame({'Decision': prediction})

            pred_val = {0: "unaccounted",1: "accounted",2: "good",3: "vgood"}
            df_result['Decision'] = df_result['Decision'].map(pred_val)

            #  reading the car details file file from the server
            df_cars_id = pd.read_csv("Preprocessed_Files/CarDetailsForPrediction.csv")

            df_final_result = pd.concat([df_cars_id, df_result], axis=1)

            result_html = df_final_result.to_html()

            df_final_result.to_csv('Predicted_Files/Result.csv', index=None)
            self.logger_object.log(self.file_object,
                                   'Prediction - Successfully saved the Results file as desired location.')

            return result_html

        except Exception as e:
            self.logger_object.log(self.file_object,'Prediction - Exception occured in data pre-processing. Exception message:  '+str(e))
            return "Error during prediction! Please check logs for details."
