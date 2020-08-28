import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("/home/lokesh/Desktop/Projects/digit_recogniser/input/train.csv")
    # we create a new column called kfold and fill it with -1
    #print(df[:100,:100])
    df['kfold'] = -1 

    #randomising rows of the data
    df = df.sample(frac=1).reset_index(drop = True)

    kf = model_selection.KFold(n_splits = 5)

    for fold, (trn_,val_) in enumerate (kf.split(X=df)):
        df.loc[val_,'kfold'] = fold 

    df.to_csv('/home/lokesh/Desktop/Projects/digit_recogniser/input/mnist_train_folds.csv', index = False)

