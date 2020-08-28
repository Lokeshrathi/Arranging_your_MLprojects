# Machine_learning_basics

## Data source [link](https://www.kaggle.com/c/digit-recognizer)
- The data files train.csv and test.csv contain gray-scale images of hand-drawn digits, from zero through nine.
- Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.


## About the repo folders
- **models/**: This folder keeps all the trained models
- **src/**: All the python scripts asscociated with the projects are kept here. 
 - **create_folds.py** : Its the same as the train.csv though the difference is that *create_folds.py* creates a a CSV which is shuffled and has a new column called Kfold.
 - **config.py** : This python script is used to avoid hardcoding such as the path to the training data and saving of the models.
 - **model_dispatcher.py** : This script contains various models that will be used for training of our model example DecisionTree, RandomForest etc.
 
### This project is Licensed to Apache.
