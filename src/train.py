import joblib
import pandas as pd 
from sklearn import metrics
from sklearn import tree

import os
import argparse

import config
import model_dispatcher

def run(fold,model):
    #read training data with the folds
    df = pd.read_csv(config.Training_file)
    
    #training data is where kfold is not equal to the provided fold
    #we have reset the index
    df_train = df[df.kfold != fold].reset_index(drop = False)

    #validation data is where the kfold is equal to the provided fold
    df_valid = df[df.kfold == fold].reset_index(drop = False)

    #drop the label column and convert the same into numpy array by using .values
    #target is label column

    x_train = df_train.drop('label', axis = 1).values
    y_train = df_train.label.values

    # similarly for validation we have
    x_valid = df_valid.drop('label', axis =1).values
    y_valid = df_valid.label.values

    clf = model_dispatcher.models[model]

    clf.fit(x_train,y_train)

    preds = clf.predict(x_valid)
    #calculate and print accuracy
    accuracy = metrics.accuracy_score(y_valid,preds)
    print(f"Fold = {fold}, Accuracy ={accuracy}")

    #saving the model
    joblib.dump(
        clf,
        os.path.join(config.model_output,f'dt_{fold}.bin')
    )
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # add different arguments you need and their types
    #currently, we only need folds
    parser.add_argument("--fold", type = int)

    parser.add_argument("--model", type=str)

    #read the argument from the command line
    args = parser.parse_args()
    run(fold=args.fold, model = args.model)
    
