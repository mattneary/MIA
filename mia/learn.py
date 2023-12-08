import pandas as pd
from sklearn.linear_model import LogisticRegression
import math

def learn():
    df = pd.read_csv('train.csv', header=None, nrows=1000000)
    num_total = df.shape[0]
    num_train = math.ceil(0.7 * num_total)
    num_test = math.floor(0.3 * num_total)

    X = df[df.columns[2:-1]][num_test:]
    Xt = df[df.columns[2:-1]][:num_test]
    Y = df[14][num_test:]
    Yt = df[14][:num_test]

    logreg = LogisticRegression(random_state=16)
    logreg.fit(X, Y)
    y_pred = logreg.predict(Xt)

    print(logreg.score(Xt, Yt))
    print(logreg.coef_)

# learn()
