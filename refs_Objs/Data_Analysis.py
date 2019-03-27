import pandas
from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import linear_model
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import seaborn as sns
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV


loc = "E:\Hackathon\SPL_Data\\"


def transform_result(row):
    # if row.home_team_goal_count > row.away_team_goal_count:
    #     return 1
    # elif row.home_team_goal_count < row.away_team_goal_count:
    #     return -1
    # else:
    #     return 0
    return row.home_team_goal_count - row.away_team_goal_count


def data_processing():
    raw_data_1 = pandas.read_csv(loc + 'SPL-matches-2013-to-2014-stats.csv')
    raw_data_2 = pandas.read_csv(loc + 'SPL-matches-2014-to-2015-stats.csv')
    raw_data_3 = pandas.read_csv(loc + 'SPL-matches-2015-to-2016-stats.csv')
    raw_data_4 = pandas.read_csv(loc + 'SPL-matches-2016-to-2017-stats.csv')
    raw_data_5 = pandas.read_csv(loc + 'SPL-matches-2017-to-2018-stats.csv')
    raw_data_6 = pandas.read_csv(loc + 'SPL-matches-2018-to-2019-stats.csv')

    raw_data_1 = raw_data_1[['home_team_name', 'away_team_name', 'home_ppg', 'away_ppg', 'home_team_goal_count',
                             'away_team_goal_count', 'total_goal_count', 'total_goals_at_half_time',
                             'home_team_goal_count_half_time', 'home_team_goal_count_half_time',
                             'away_team_goal_count_half_time', 'home_team_corner_count', 'away_team_corner_count',
                             'home_team_red_cards', 'away_team_red_cards', 'home_team_shots_on_target',
                             'away_team_shots_on_target', 'home_team_fouls', 'away_team_fouls']]
    raw_data_2 = raw_data_2[['home_team_name', 'away_team_name', 'home_ppg', 'away_ppg', 'home_team_goal_count',
                             'away_team_goal_count', 'total_goal_count', 'total_goals_at_half_time',
                             'home_team_goal_count_half_time', 'home_team_goal_count_half_time',
                             'away_team_goal_count_half_time', 'home_team_corner_count', 'away_team_corner_count',
                             'home_team_red_cards', 'away_team_red_cards', 'home_team_shots_on_target',
                             'away_team_shots_on_target', 'home_team_fouls', 'away_team_fouls']]
    raw_data_3 = raw_data_3[['home_team_name', 'away_team_name', 'home_ppg', 'away_ppg', 'home_team_goal_count',
                             'away_team_goal_count', 'total_goal_count', 'total_goals_at_half_time',
                             'home_team_goal_count_half_time', 'home_team_goal_count_half_time',
                             'away_team_goal_count_half_time', 'home_team_corner_count', 'away_team_corner_count',
                             'home_team_red_cards', 'away_team_red_cards', 'home_team_shots_on_target',
                             'away_team_shots_on_target', 'home_team_fouls', 'away_team_fouls']]
    raw_data_4 = raw_data_4[['home_team_name', 'away_team_name', 'home_ppg', 'away_ppg', 'home_team_goal_count',
                             'away_team_goal_count', 'total_goal_count', 'total_goals_at_half_time',
                             'home_team_goal_count_half_time', 'home_team_goal_count_half_time',
                             'away_team_goal_count_half_time', 'home_team_corner_count', 'away_team_corner_count',
                             'home_team_red_cards', 'away_team_red_cards', 'home_team_shots_on_target',
                             'away_team_shots_on_target', 'home_team_fouls', 'away_team_fouls']]
    raw_data_5 = raw_data_5[['home_team_name', 'away_team_name', 'home_ppg', 'away_ppg', 'home_team_goal_count',
                             'away_team_goal_count', 'total_goal_count', 'total_goals_at_half_time',
                             'home_team_goal_count_half_time', 'home_team_goal_count_half_time',
                             'away_team_goal_count_half_time', 'home_team_corner_count', 'away_team_corner_count',
                             'home_team_red_cards', 'away_team_red_cards', 'home_team_shots_on_target',
                             'away_team_shots_on_target', 'home_team_fouls', 'away_team_fouls']]
    raw_data_6 = raw_data_6[['home_team_name', 'away_team_name', 'home_ppg', 'away_ppg', 'home_team_goal_count',
                             'away_team_goal_count', 'total_goal_count', 'total_goals_at_half_time',
                             'home_team_goal_count_half_time', 'home_team_goal_count_half_time',
                             'away_team_goal_count_half_time', 'home_team_corner_count', 'away_team_corner_count',
                             'home_team_red_cards', 'away_team_red_cards', 'home_team_shots_on_target',
                             'away_team_shots_on_target', 'home_team_fouls', 'away_team_fouls']]

    playing_stat = pandas.concat([raw_data_1, raw_data_2, raw_data_3, raw_data_4, raw_data_5, raw_data_6]
                                 , ignore_index=True)
    playing_stat["Result"] = playing_stat.apply(lambda row: transform_result(row), axis=1)
    print(playing_stat)
    playing_stat.to_csv(loc + "Merged_6_season_matches.csv")

    df = pandas.read_csv(loc + "Merged_6_season_matches.csv")
    table = pandas.DataFrame(columns=('Team', 'HGS', 'AGS', 'HAS', 'AAS', 'HGC', 'AGC', 'HDS', 'ADS'))
    print(len(df))
    avg_home_scored = df.home_team_goal_count.sum() / len(df)
    avg_away_scored = df.away_team_goal_count.sum() / len(df)
    avg_home_conceded = avg_away_scored
    avg_away_conceded = avg_home_scored
    print(avg_away_conceded, avg_home_conceded)

    res_home = df.groupby('home_team_name')
    res_away = df.groupby('away_team_name')
    print(res_home)

    table.Team = res_home.home_team_name.all().keys()
    table.HGS = res_home.home_team_goal_count.sum().values
    table.HGC = res_home.away_team_goal_count.sum().values
    table.AGS = res_away.away_team_goal_count.sum().values
    table.AGC = res_away.home_team_goal_count.sum().values
    # 15 Home matches for each team each season and 6 seasons therefore 90 home matches and 90 away matches
    table.HAS = (table.HGS / 90) / avg_home_scored
    table.AAS = (table.AGS / 90) / avg_away_scored
    table.HDS = (table.HGC / 90) / avg_home_conceded
    table.ADS = (table.AGC / 90) / avg_away_conceded

    print(table)

    feature_table = df.iloc[:, :40]
    feature_table = feature_table[['home_team_name', 'Result', 'away_team_name', 'home_ppg', 'away_ppg',
                                   'home_team_shots_on_target', 'away_team_shots_on_target',
                                   'home_team_corner_count', 'away_team_corner_count']]

    f_HAS = []
    f_HDS = []
    f_AAS = []
    f_ADS = []
    for index, row in feature_table.iterrows():
        f_HAS.append(table[table['Team'] == row['home_team_name']]['HAS'].values[0])
        f_HDS.append(table[table['Team'] == row['home_team_name']]['HDS'].values[0])
        f_AAS.append(table[table['Team'] == row['away_team_name']]['AAS'].values[0])
        f_ADS.append(table[table['Team'] == row['away_team_name']]['ADS'].values[0])

    feature_table['HAS'] = f_HAS
    feature_table['HDS'] = f_HDS
    feature_table['AAS'] = f_AAS
    feature_table['ADS'] = f_ADS
    feature_table.to_csv(loc + "Feature_table.csv")

    print(feature_table)

    train_Set_I = feature_table[['HAS', 'HDS', 'AAS', 'ADS', 'Result']]
    train_Set_II = feature_table[['HAS', 'HDS', 'AAS', 'ADS', 'home_team_shots_on_target', 'away_team_shots_on_target',
                                 'home_team_corner_count', 'away_team_corner_count', 'Result']]
    train_Set_III= feature_table[['HAS', 'HDS', 'AAS', 'ADS', 'home_team_shots_on_target', 'away_team_shots_on_target',
                                 'home_team_corner_count', 'away_team_corner_count', 'home_ppg', 'away_ppg', 'Result']]

    y = feature_table[['Result']]
    print('********************')
    print(y)

    print(train_Set_I.tail())
    print("Training Set 2 :", train_Set_II.tail())
    print(train_Set_III.tail())

    # x_train, x_test, y_train, y_test = train_test_split(train_Set_I, y_train, test_size=0.25)
    # #x2_train, x2_test, y2_train, y2_test = train_test_split(train_Set_II, y_train, test_size=0.20)
    # #x3_train, x3_test, y3_train, y3_test = train_test_split(train_Set_III, y_train, test_size=0.15)
    #
    # sv_classifier = SVC(kernel='linear')
    # sv_classifier.fit(x_train, y_train)
    #
    # y_pred = sv_classifier.predict(x_test)
    #
    # print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test, y_pred))

#     nbb_scores = []
#     nbb_scores_2 = []
#
#     for i in range(1, 1000, 50):
#         clf1 = MultinomialNB(alpha=i)
#         clf1.fit(train_Set_I, y_train)
# #        clf1.fit(train_Set_II, y_train)
#         scores = cross_val_score(clf1, train_Set_I, y_train, cv=10)
# #        scores_2 = cross_val_score(clf1, train_Set_II, y_train, cv=10)
#         print("NB alpha ", i, " ", scores.mean())#, " : ", scores_2.mean())
#
#         nbb_scores.append(scores.mean())
#   #      nbb_scores_2.append(scores_2.mean())

    # clf = [MultinomialNB(alpha=1), SVC(kernel='linear', C=1.5, probability=True), LogisticRegression()]
    #
    # labels = ['Naive Bayes', 'SVM', 'Log regres']
    #
    # mean_scores = []
    # mean_scores_2 = []
    # cms = []
    #
    # for i in range(0, 3):
    #     clf[i].fit(train_Set_I, y_train)
    #     #clf[i].fit(train_Set_II, y_train)
    #
    #     scores = cross_val_score(clf[i], train_Set_I, y_train, cv=10)
    #     #scores_2 = cross_val_score(clf[i], train_Set_II, y_train, cv=10)
    #     print(labels[i], " : ", scores.mean())#, " : ", scores_2.mean())
    #
    #     mean_scores.append(scores.mean())
    #     #mean_scores_2.append(scores_2.mean())

    # train_features, test_features, train_labels, test_labels = train_test_split(train_Set_I, y_train, test_size=0.25,
    #                                                                             random_state=42)
    # rf = RandomForestRegressor(n_estimators=1000, random_state=42)
    # rf.fit(train_features, train_labels)
    #
    # # Use the forest's predict method on the test data
    # predictions = rf.predict(test_features)
    # # Calculate the absolute errors
    # errors = abs(predictions - test_labels)
    # # Print out the mean absolute error (mae)
    # print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
    #
    # mape = 100 * (errors / test_labels)
    # # Calculate and display accuracy
    # accuracy = 100 - np.mean(mape)
    # print('Accuracy:', round(accuracy, 2), '%.')
    print("Length of Training Set is: ", len(train_Set_I))
    sns.pairplot(train_Set_I, kind="reg")

    train_Set_II = train_Set_II.drop('Result', axis=1)
    print('****************train after drop***********')
    print(train_Set_I)

    X_train, X_test, y_train, y_test = train_test_split(train_Set_III, y, test_size=0.001, random_state=0)
                                                        #random_state=20)
    print('below is y-train')
    print(y_train.astype(float))

    print('below is y-test')
    print(y_test.astype(float))

    scaler_X = preprocessing.StandardScaler().fit(X_train.astype(float))
    scaler_y = preprocessing.StandardScaler().fit(y_train.astype(float))
    X_train_scaled = scaler_X.transform(X_train.astype(float))
    y_train_scaled = scaler_y.transform(y_train.astype(float))

    clf = RandomForestRegressor()
    clf.fit(X_train_scaled, y_train_scaled.ravel())
    X_test_scaled = scaler_X.transform(X_test.astype(float))
    y_test_scaled = scaler_y.transform(y_test.astype(float))
    pred = clf.predict(X_test_scaled)
    r2_score(y_test_scaled, pred)
    print('First Score is: ', r2_score(y_test, pred))

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_scaled, pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test_scaled, pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_scaled, pred)))
    crossResult = cross_val_score(clf, X_train, y_train_scaled.ravel(), cv=5)
    print(crossResult.mean(), crossResult)

    # hyperparameters = {'randomforestregressor__max_features': ['auto', 'sqrt', 'log2'],
    #                    'randomforestregressor__max_depth': [None, 5, 3, 1]}
    #
    # pipeline = make_pipeline(preprocessing.StandardScaler(),
    #                          RandomForestRegressor(n_estimators=100))
    #
    # clf = GridSearchCV(pipeline, hyperparameters, cv=10)
    # clf.fit(X_train, y_train.ravel())
    # clf.best_params_
    # y_pred = clf.predict(X_test)
    # r2_score(y_test, y_pred)
    # print('Second Score is: ', r2_score(y_test, y_pred))




#    x_train, x_test, yaxis_train, y_test = train_test_split(train_Set_I, y_train.astype('float64'), test_size=0.001, random_state=0)

#    print("Length of test Set is: ", len(x_test))
#    print(x_test)


#     sns.heatmap(train_Set_I)
#
#     sc = StandardScaler()
#     x_train = sc.fit_transform(x_train, yaxis_train)
#     print("XTest=", x_test)
#     x_test = sc.transform(x_test)
#     y_test = sc.transform(y_test)
#
#     regressor = RandomForestRegressor(n_estimators=20, random_state=0)
#     regressor.fit(x_train, yaxis_train)
#     y_pred = regressor.predict(x_test)
#
#     print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
#     print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
#     print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#     crossResult = cross_val_score(regressor, x_train, yaxis_train)
#     print(crossResult.mean(), crossResult)


data_processing()
