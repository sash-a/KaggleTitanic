import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics

import warnings

warnings.filterwarnings('ignore')

x_train = pd.read_csv('files/train.csv')
y_train = x_train['Survived']

x_test = pd.read_csv('files/test.csv')

all_data = [x_train, x_test]
pd_all_data = pd.concat(all_data)


# print(train.info())  # this tells me that: age, cabin and embarked have null values
def clean():
    global x_train
    global x_test
    # Don't want strong correlations between the data (redundancy) so can combine sibsp and parch
    x_train['Relatives'] = x_train['SibSp'] + x_train['Parch']
    x_test['Relatives'] = x_test['SibSp'] + x_test['Parch']

    # Alone if 0 relatives
    x_train['IsAlone'] = x_train['Relatives'].apply(lambda x: 0 if x != 0 else 1)
    x_test['IsAlone'] = x_test['Relatives'].apply(lambda x: 0 if x != 0 else 1)

    # Because cabin has so few entries and because of a very clever notebook I read:
    # if cabin value is NaN then assume that guest did not have a cabin
    x_train['HasCabin'] = x_train['Cabin'].apply(lambda x: 0 if x != x else 1)  # x!= x if value is NaN
    x_test['HasCabin'] = x_test['Cabin'].apply(lambda x: 0 if x != x else 1)

    # Fill in embarked null values with most common value, because there are so few missing entries
    embarked_mode = pd_all_data['Embarked'].mode()[0]
    x_train['Embarked'] = x_train['Embarked'].fillna(embarked_mode)
    x_test['Embarked'] = x_test['Embarked'].fillna(embarked_mode)

    # filling in missing ages, since age should be normally distributed we can assume most values lie within
    # 1 std dev and add random ages in from that range
    mean_age = pd_all_data['Age'].mean()
    std_age = pd_all_data['Age'].std()

    x_train['Age'] = x_train['Age'].apply(
        lambda x: x if x == x else np.random.randint(mean_age - std_age, mean_age + std_age))
    x_test['Age'] = x_test['Age'].apply(
        lambda x: x if x == x else np.random.randint(mean_age - std_age, mean_age + std_age))

    # Mapping sex to value
    x_train['Sex'] = x_train['Sex'].map({'male': 0, 'female': 1})
    x_test['Sex'] = x_test['Sex'].map({'male': 0, 'female': 1})

    # Mapping embarked to value
    x_train['Embarked'] = x_train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    x_test['Embarked'] = x_test['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    # Single unknown value in of fare in test
    x_test['Fare'] = x_test['Fare'].fillna(pd_all_data['Fare'].mean())


def feature_selection():
    global x_train
    global x_test

    x_train = x_train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Parch', 'SibSp', 'Survived'], axis=1)
    x_test = x_test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Parch', 'SibSp'], axis=1)


def tune_knn():
    """
    Finding K value that gives the most consistent results in the K number of Neighbors algorithm
    :return: integer
    """
    scores = []
    neighbours = range(1, 50)
    for i in neighbours:
        knn = KNeighborsClassifier(n_neighbors=i)
        score = cross_val_score(knn, x_train, y_train, cv=10, scoring='accuracy')
        scores.append(score.mean())

    # plt.plot(neighbours, scores)
    # plt.show()
    return scores.index(max(scores)), max(scores)


# This takes really long to run best params:
def tune_rf():
    """
    Finding the parameter that give the best results for the random forest algorithm
    :return: dict of parameters
    """
    scores = {}

    # leaving all other params as default
    rf_params = {
        'n_estimators': 10,  # test from 100 -> 700 jump 100
        'max_features': 'sqrt',  # need to test sqrt, log2
        'min_samples_leaf': 20,  # test from  1->51 jump 2
        'verbose': 0
    }

    for n_estimators in range(100, 701, 100):
        for max_features in (['sqrt', 'log2']):
            for min_sample_leaf in (1, 51, 2):
                rf_params['n_estimators'] = n_estimators
                rf_params['max_features'] = max_features
                rf_params['min_samples_leaf'] = min_sample_leaf

                rf = RandomForestClassifier(**rf_params)

                score = cross_val_score(rf, x_train, y_train, cv=10).mean()
                scores[score] = rf_params
                print('done training')
    return scores[max(scores)], max(scores)  # returns params with max score


def compare(clf_one, clf_two):
    score_one = cross_val_score(clf_one, x_train, y_train, cv=10).mean()
    score_two = cross_val_score(clf_two, x_train, y_train, cv=10).mean()

    if score_one > score_two:
        return clf_one
    return clf_two


def main():
    global x_test
    global x_train
    global y_train

    clean()
    feature_selection()
    # print(pd_all_data.info())
    # print(pd_all_data.head(10))

    # finding the best params for both the models
    neighbours, knn_score = tune_knn()
    rf_params, rf_score = tune_rf()  # warning this method does take a couple minutes to run
    # These are the params it comes up with, if you don't want to run the method, just comment it out and uncomment
    # below and also comment out the if statement below (spoiler random forrest works better than knn)
    # rf_params = {'n_estimators': 700, 'max_features': 'log2', 'min_samples_leaf': 2, 'verbose': 0}

    # fitting the best model to all the training data
    model = RandomForestClassifier(**rf_params)
    if rf_score <= knn_score:
        model = KNeighborsClassifier(neighbours)

    # finally fit the best model to all the training data and submit the prediction
    model.fit(x_train, y_train)
    # Generate Submission File
    submission = pd.DataFrame({'PassengerId': y_test['PassengerId'],
                               'Survived': model.predict(x_test)})
    submission.to_csv("submission.csv", index=False)


if __name__ == '__main__':
    main()
