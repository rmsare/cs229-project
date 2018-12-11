import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, \
                            median_absolute_error, r2_score


def aic(y_true, y_pred, nfeat):
    sse = np.sum((y_true - y_pred) ** 2)
    return 2 * nfeat - 2 * np.log(sse)


def bic(y_true, y_pred, nfeat):
    nobs = len(y_true)
    sse = np.sum((y_true - y_pred) ** 2)
    return np.log(nobs) * nfeat - 2 * np.log(sse)


def plot_learning_curve(train_sizes, train_scores, test_scores):
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', \
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', \
             label="Cross-validation score")
    
    plt.plot(train_sizes, train_scores_mean - train_scores_std, '--', color='r')
    plt.plot(train_sizes, train_scores_mean + train_scores_std, '--', color='r')
    
    plt.plot(train_sizes, test_scores_mean - test_scores_std, '--', color='g')
    plt.plot(train_sizes, test_scores_mean + test_scores_std, '--', color='g')
    
    plt.xlabel('# of training examples')
    plt.legend(loc="best")
    plt.show()


def print_report(y_true, y_pred, k, f):
    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - (1 - r2) * f
    a = aic(y_true, y_pred, k)
    b = bic(y_true, y_pred, k)
    print('{:.2f} & {:.2f} & {:.2f}'.format(rmse, adj_r2, a))


def score_model(y_true, y_pred, k):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    a = aic(y_true, y_pred, k)
    return rmse, r2, a
