#Report
#Objective: The goal was to predict whether a person has heart disease based on input data using a pre-trained machine learning model (model_bagg). 

#Steps Taken:

Loading Necessary Libraries: We imported the numpy library for handling array operations.
Model Loading: Ensured the pre-trained model (model_bagg) was loaded. Although the actual loading step was commented out in the script, we assumed the model was already available in the environment.
Data Preparation: Prepared the input data in the form of a tuple and converted it to a NumPy array. The array was reshaped to match the expected input shape for the model.
Prediction: Used the model to predict whether the input data indicated the presence of heart disease.
Interpreting Results: Printed the prediction and interpreted the result, outputting whether the person has heart disease or not based on the model's prediction.
##Challenges Faced:

Model Prediction Consistency: Initially, there was confusion regarding the interpretation of the model's predictions. The prediction logic was misinterpreted, leading to the incorrect display of results.
Model Bias: There was an issue where the model always predicted the presence of heart disease (1), regardless of the input data. This raised concerns about potential model overfitting, class imbalance during training, or input data preprocessing issues.
Ensuring Correct Model Loading: The process assumed that the model was already loaded into the environment. This can lead to issues if the model is not properly loaded or if there's a discrepancy in the file path or model name.
Final Script:
The final script ensures that:

The input data is correctly formatted and reshaped.
The model's prediction is interpreted correctly, distinguishing between the presence (1) and absence (0) of heart disease.
Here is the final working script:

import numpy as np

 Assuming `model_bagg` is your trained model and it's already loaded
 Uncomment and modify the following line if you need to load your model
 import joblib
 model_bagg = joblib.load('path_to_your_model_file.pkl')

# Input data
datainput = (0, 1, 1, 120, 2, 0, 0, 0, 295, 0.0, 1, 42, 162)
datainput_array = np.asarray(datainput)
datainput_array_shape = datainput_array.reshape(1, -1)

# Prediction
prediction = model_bagg.predict(datainput_array_shape)
print("Prediction:", prediction)

# Interpreting Prediction
if prediction[0] == 1:
    print('The Person has Heart Disease')
else:
    print('The Person does not have Heart Disease')
Conclusion:
We successfully created a script to predict heart disease using a pre-trained model. The process highlighted the importance of correct data formatting, model loading, and interpretation of predictions. The main issue encountered was the model's apparent bias, which suggests a need for further evaluation of the model's training data and validation process. However, both KNN and Logistic Regression models still exhibited overfitting tendencies. Further optimization and regularization techniques may be necessary to achieve a balance between bias and variance in the models.

