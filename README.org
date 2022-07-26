#+TITLE: Red Wine Quality Classifier
#+AUTHOR: João Lucas Correia Barbosa de Farias
#+AUTHOR: Júlio Freire
#+EMAIL: joao.farias.080@ufrn.edu.br

* Deployment of a full decision tree pipeline
This project consists of classifying red wine quality based on several chemical proprieties of the wine. A full pipeline was created to handle processing of the features and the training of the model, which was performed using Decision Trees. Then, the model and target encoder were exported to [[https://wandb.ai/site][Weights & Biases]] (W&B) where they were later retrieved for the deployment of the model. The deployment was possible due to [[https://github.com/][GitHub]] CI/CD integration with [[https://www.heroku.com/][Heroku]]. The final product is available for testing at [[https://red-wine-quality-ml.herokuapp.com/][Red Wine Quality App]] where anyone can input the values for the wine features and get back a result for its quality: either 'good' or 'bad'. The dataset was taken from [[https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009][Kaggle]].

P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

* About the Authors
The authors of this project are [[https://github.com/jotafarias13][João Lucas Farias]] and [[https://github.com/juliofreire][Júlio Freire]]. The project was created as part of a Graduate Level course on Machine Learning at the Graduate Program in Electrical and Computer Engineering at the Federal University of Rio Grande do Norte (UFRN), Brasil. The course is taught by [[https://github.com/ivanovitchm][Professor Ivanovitch Silva]]. The goal of the project was to teach the students how to create a Pipeline for Machine Learning algorithms (along with many good practices in ML programming), train the model and deploy the exported model for production. This way, anyone can test and verify the performance of the model. By using [[https://wandb.ai/site][Weights & Biases]], [[https://github.com/][GitHub]] and [[https://www.heroku.com/][Heroku]], we were able to train the model and deploy it.

* From data fetch to full pipeline
In order to perform the proposed task in a clear and organized way, we split each stage of implementation in a single file. The files can be found at the [[file:source/pipeline/][pipeline directory]]. From data fetch to a full pipeline passing through preprocessing, data check and segretation, we implemented the code necessary to train and test the model. All coding was done on [[https://colab.research.google.com/][Google Colab]] so as to encourage and facilitate colaboration.

** Data Fetch
In this [[file:source/pipeline/1-fetch_data.ipynb][first file]], we only uploaded the raw dataset to our [[https://wandb.ai/ppgeec-ml-jj][colaborative team]] on W&B. In this link of our colaborative team, you can find the project [[https://wandb.ai/ppgeec-ml-jj/red_wine_quality][red_wine_quality]] and can see all the runs we created with W&B when working on this project. Also, all the artifacts that were exported to W&B can be found in the [[https://wandb.ai/ppgeec-ml-jj/red_wine_quality/artifacts/][Artifacts page]].

** Exploratory Data Analysis (EDA)
In this [[file:source/pipeline/2-eda.ipynb][second file]], we performed the exploratory data analysis, which consists in going over the whole dataset, looking for patterns, correlation between features and possible outliers. This analysis is based on statistical tools that helps us understand and visualize our data in a proper manner. Here, we found how many duplicated rows we had in our dataset. Also, we noticed we had an imbalanced target feature which later made us transform it to get better and more sound results.

** Preprocessing
In this [[file:source/pipeline/3-preprocessing.ipynb][third]] stage, we put into action some of the changes we conceived during the EDA step. That is, we removed the duplicated rows and transformed our target column. The 'quality' feature had only integer values that ranged from 3 to 8. By finding the median to be 6, we split the data from 3 to 6.5 ('bad') and from 6.5 to 8 ('good'). After this we still had an imbalanced target but less than it was before. Also, we changed the names of the columns to replace all spaces (blank values ' ') with underscores '_'. The preprocessed dataset was exported to W&B and can be seen in the [[https://wandb.ai/ppgeec-ml-jj/red_wine_quality/artifacts/preprocessed_data.csv/][Artifacts page]].

** Data Check
In this [[file:source/pipeline/4-data_check.ipynb][fourth]] step, we performed some tests on the preprocessed dataset to check if all the columns were in order and according to what we expected. Basically, we created some test functions to check datatypes and ranges of the features and execute the tests with pytest.

** Data Segretation
In this [[file:source/pipeline/5-data_segregation.ipynb][fifth file]], we segregated the data into train and test sets. 70% percent of our dataset was used for training and 30% for testing. Also, we used the /stratify/ option when segregating the data to make sure we had proportional amounts of 'good' wines in both train and test sets. This was done because the value 'good' for the quality column was the one with less rows when compared to the 'bad' value. The segregated dataset was exported to W&B and can be seen in the Artifacts page ([[https://wandb.ai/ppgeec-ml-jj/red_wine_quality/artifacts/segregated_data/train.csv/][train]] and [[https://wandb.ai/ppgeec-ml-jj/red_wine_quality/artifacts/segregated_data/test.csv/][test]]). 

** Train
In this [[file:source/pipeline/6-train.ipynb][sixth]] and longest step, we collected the segregated train data from W&B and split it into two new sets: train and validation sets. The validation set is 30% of the original train set and /stratify/ was used again to make sure we had proportional amounts of 'good' wines in both train and validation sets. The train set is used to train the model while the validation set is used to analyze the metrics and tune the hyperparameters. So, we first train the model with the train set, then validate the trained model with the validation set.

But, before training, we had to remove the outliers from the train set, encode our target variable and create our pipeline. Since we only had numerical features, we did not need to process categorical features. The numerical features were normalized with either the min-max method or the z-score method. We also added a simple imputer to fill any missing values with the respective median value. After that, we moved on to the training step. We used Decision Trees as the classifier and analyzed the Accuracy, Precision, Recall and F1 metrics. We got an accuracy of about 86% and Precision, Recall and F1 of about 50%. This difference in result is due to the fact the our target is imbalanced.

Next, we performed Hyperparameter Tuning with the help of W&B sweeps. We configured the sweep to test some configurations for our training, like using min-max or z-score and using different criteria and splitters for Decision Trees. After running these different configurations, the W&B [[https://wandb.ai/ppgeec-ml-jj/red_wine_quality/sweeps/3u0enxtm/][sweep]] showed us the [[https://wandb.ai/ppgeec-ml-jj/red_wine_quality/runs/59o4c0t3/overview][best]] result (the one with the highest accuracy). This configuration model was then used as our best model. Finally, the best model and target encoder were exported to W&B and can be seen in the Artifacts page ([[https://wandb.ai/ppgeec-ml-jj/red_wine_quality/artifacts/inference_artifact/model_export/][model]] and [[https://wandb.ai/ppgeec-ml-jj/red_wine_quality/artifacts/inference_artifact/target_encoder/][encoder]]). Also, the [[file:images/confusion_matrix_best_model.pdf][confusion matrix]] and [[file:images/feature_importance_best_model.pdf][feature importance]] for the validation of the best model were plotted.

** Test
In this [[file:source/pipeline/7-test.ipynb][seven and final]] step, we tested our model against the test set. Our final metrics were as shown below and can be seen in this W&B [[https://wandb.ai/ppgeec-ml-jj/red_wine_quality/runs/33eooynf/overview][run]].

29-05-2022 19:26:47 Test Accuracy: 0.8382352941176471

29-05-2022 19:26:47 Test Precision: 0.3829787234042553

29-05-2022 19:26:47 Test Recall: 0.32727272727272727

29-05-2022 19:26:47 Test F1: 0.35294117647058826

The confusion matrix for the test set with the best model can be seen [[file:images/confusion_matrix_test.pdf][here]].

* References
This project was based on a [[https://github.com/ivanovitchm/ppgeecmachinelearning/tree/main/lessons/week_02/sources][Decision Tree classifier model]] used to to predict whether individuals make less than or equal to 50k a year, or more than 50k a year using data from the [[https://archive.ics.uci.edu/ml/datasets/Adult][Census Bureau Data]]. Also, for the deployment stage of the project, we took inspiration from the [[https://github.com/ivanovitchm/colab2mlops][deployment]] of the classification model for the same dataset.

