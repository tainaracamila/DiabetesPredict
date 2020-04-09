import pandas as pd
import pickle
from plots import Plots
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def data_exploration(df):
    # df shape
    print(df.shape)
    # 0 - 4
    print(df.head(5))
    # 763 - 767
    print(df.tail(5))
    # statistic df overview
    print(df.describe())
    # columns
    print(list(df))
    # null values
    print("Are there null values?", df.isnull().values.any())

    # Data distribution
    num_true = len(df.loc[df['diabetes'] == True])
    num_false = len(df.loc[df['diabetes'] == False])

    print("True cases:", num_true, ((num_true / (num_true + num_false)) * 100))
    print("False cases:", num_false, ((num_false / (num_true + num_false)) * 100))


def split_analysis(df, x_train, x_test, y_train, y_test):
    print("{0:0.2f}% from data train".format((len(x_train) / len(df.index)) * 100))
    print("{0:0.2f}% from data test".format((len(x_test) / len(df.index)) * 100))

    print("\nOriginal True : {0} ({1:0.2f}%)".format(len(df.loc[df['diabetes'] == 1]),
                                                   (len(df.loc[df['diabetes'] == 1]) / len(df.index) * 100)))

    print("Original False : {0} ({1:0.2f}%)".format(len(df.loc[df['diabetes'] == 0]),
                                                    (len(df.loc[df['diabetes'] == 0]) / len(df.index) * 100)))

    print("\nTraining True : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 1]),
                                                   (len(y_train[y_train[:] == 1]) / len(y_train) * 100)))

    print("Training False : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 0]),
                                                    (len(y_train[y_train[:] == 0]) / len(y_train) * 100)))

    print("\nTest True : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 1]),
                                               (len(y_test[y_test[:] == 1]) / len(y_test) * 100)))

    print("Test False : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 0]),
                                                (len(y_test[y_test[:] == 0]) / len(y_test) * 100)))


def missing_values(df):
    print("# Data frame rows {0}".format(len(df)))
    print("# Rows missing glucose_conc: {0}".format(len(df.loc[df['glucose_conc'] == 0])))
    print("# Rows missing diastolic_bp: {0}".format(len(df.loc[df['diastolic_bp'] == 0])))
    print("# Rows missing thickness: {0}".format(len(df.loc[df['thickness'] == 0])))
    print("# Rows missing insulin: {0}".format(len(df.loc[df['insulin'] == 0])))
    print("# Rows missing bmi: {0}".format(len(df.loc[df['bmi'] == 0])))
    print("# Rows missing age: {0}".format(len(df.loc[df['age'] == 0])))


def naive_bayes_model(x_train, x_test, y_train, y_test):
    # Training model
    model = GaussianNB()
    # ravel = format shape
    model.fit(x_train, y_train.ravel())

    # Validating model
    nb_predict_test = model.predict(x_test)
    print("Accuracy Naive Bayes test data: {0:.4f}".format(metrics.accuracy_score(y_test, nb_predict_test)))

    # Confusion matrix and classification report
    print("\nConfusion Matrix")
    print("{0}".format(metrics.confusion_matrix(y_test, nb_predict_test, labels=[1, 0])))
    print("\nClassification Report")
    print(metrics.classification_report(y_test, nb_predict_test, labels=[1, 0]))


def random_forest_model(x_train, x_test, y_train, y_test):
    # Training model
    model = RandomForestClassifier(random_state=42)
    model.fit(x_train, y_train.ravel())

    # Validating model
    rf_predict_test = model.predict(x_test)
    print("Accuracy random forest test data: {0:.4f}".format(metrics.accuracy_score(y_test, rf_predict_test)))

    # Confusion matrix and classification report
    print("\nConfusion Matrix")
    print("{0}".format(metrics.confusion_matrix(y_test, rf_predict_test, labels=[1, 0])))
    print("\nClassification Report")
    print(metrics.classification_report(y_test, rf_predict_test, labels=[1, 0]))

    return model


def logistic_regression_model(x_train, x_test, y_train, y_test):
    # Training model
    model = LogisticRegression(C=0.7, random_state=42, max_iter=1000)
    model.fit(x_train, y_train.ravel())

    # Validation model and classification report
    lr_predict_test = model.predict(x_test)
    print("Accuracy logistic regression test data : {0:.4f}".format(metrics.accuracy_score(y_test, lr_predict_test)))
    print("\nClassification Report")
    print(metrics.classification_report(y_test, lr_predict_test, labels=[1, 0]))


if __name__ == '__main__':
    p = Plots()
    df = pd.read_csv('pima-data.csv')

    # data_exploration(df)

    # Correlation
    c = df.corr()
    # p.correlation(c)
    # print(c)

    # Format target
    diabetes_map = {True: 1, False: 0}
    df['diabetes'] = df['diabetes'].map(diabetes_map)
    # print(df.head())

    # Splitting data
    attributes = ['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin', 'bmi', 'diab_pred', 'age']
    attribute_target = ['diabetes']

    X = df[attributes].values
    Y = df[attribute_target].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

    # split_analysis(df, X_train, X_test, Y_train, Y_test)

    # Format missing values with mean value
    # missing_values(df)
    fill_0 = SimpleImputer(missing_values=0, strategy="mean")

    X_train = fill_0.fit_transform(X_train)
    X_test = fill_0.fit_transform(X_test)

    # Models and accuracy
    naive_bayes_model(X_train, X_test, Y_train, Y_test)
    random_forest_model(X_train, X_test, Y_train, Y_test)
    logistic_regression_model(X_train, X_test, Y_train, Y_test)

    # Choosing model

    m = random_forest_model(X_train, X_test, Y_train, Y_test)

    filename = 'model_trained.sav'
    pickle.dump(m, open(filename, 'wb'))

    loaded_model = pickle.load(open(filename, 'rb'))

    # Results model

    result = loaded_model.predict(X_test[15].reshape(1, -1))
    result_two = loaded_model.predict(X_test[18].reshape(1, -1))

    print(result)
    print(result_two)






