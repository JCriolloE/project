import os
import sys

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import learning_curve
from utilities import custom_logging

logger = custom_logging.CustomLogging(__name__)

warnings.filterwarnings("ignore")

y_train = []
og_reg_pred = []
knears_pred = []
svc_pred = []
tree_pred = []
log_reg_pred = []

class MyFunctions:

    def __init__(self, model_path):
        self.model_path = model_path

    def graph_roc_curve_multiple(
            log_fpr,
            log_tpr,
            knear_fpr,
            knear_tpr,
            svc_fpr,
            svc_tpr,
            tree_fpr,
            tree_tpr):
        plt.figure(figsize=(16, 8))
        plt.title('ROC Curve \n Top 4 Classifiers', fontsize=18)
        plt.plot(
            log_fpr,
            log_tpr,
            label='Logistic Regression Classifier Score: {:.4f}'.format(
                roc_auc_score(
                    y_train,
                    log_reg_pred)))
        plt.plot(
            knear_fpr,
            knear_tpr,
            label='KNears Neighbors Classifier Score: {:.4f}'.format(
                roc_auc_score(
                    y_train,
                    knears_pred)))
        plt.plot(
            svc_fpr,
            svc_tpr,
            label='Support Vector Classifier Score: {:.4f}'.format(
                roc_auc_score(
                    y_train,
                    svc_pred)))
        plt.plot(
            tree_fpr,
            tree_tpr,
            label='Decision Tree Classifier Score: {:.4f}'.format(
                roc_auc_score(
                    y_train,
                    tree_pred)))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.axis([-0.01, 1, 0, 1])
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.annotate(
            'Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(
                0.5, 0.5), xytext=(
                0.6, 0.3), arrowprops=dict(
                    facecolor='#6E726D', shrink=0.05), )
        plt.legend()


    def plot_learning_curve(estimator1,
                            estimator2,
                            estimator3,
                            estimator4,
                            X,
                            y,
                            ylim=None,
                            cv=None,
                            n_jobs=1,
                            train_sizes=np.linspace(.1,
                                                    1.0,
                                                    5)):
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            2, 2, figsize=(20, 14), sharey=True)
        if ylim is not None:
            plt.ylim(*ylim)
        # First Estimator
        train_sizes, train_scores, test_scores = learning_curve(
            estimator1, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        ax1.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        color="#ff9124")
        ax1.fill_between(
            train_sizes,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
            color="#2492ff")
        ax1.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
                label="Training score")
        ax1.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
                label="Cross-validation score")
        ax1.set_title("Logistic Regression Learning Curve", fontsize=14)
        ax1.set_xlabel('Training size (m)')
        ax1.set_ylabel('Score')
        ax1.grid(True)
        ax1.legend(loc="best")

        # Second Estimator
        train_sizes, train_scores, test_scores = learning_curve(
            estimator2, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        ax2.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        color="#ff9124")
        ax2.fill_between(
            train_sizes,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
            color="#2492ff")
        ax2.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
                label="Training score")
        ax2.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
                label="Cross-validation score")
        ax2.set_title("Knears Neighbors Learning Curve", fontsize=14)
        ax2.set_xlabel('Training size (m)')
        ax2.set_ylabel('Score')
        ax2.grid(True)
        ax2.legend(loc="best")

        # Third Estimator
        train_sizes, train_scores, test_scores = learning_curve(
            estimator3, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        ax3.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        color="#ff9124")
        ax3.fill_between(
            train_sizes,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
            color="#2492ff")
        ax3.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
                label="Training score")
        ax3.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
                label="Cross-validation score")
        ax3.set_title("Support Vector Classifier \n Learning Curve", fontsize=14)
        ax3.set_xlabel('Training size (m)')
        ax3.set_ylabel('Score')
        ax3.grid(True)
        ax3.legend(loc="best")

        # Fourth Estimator
        train_sizes, train_scores, test_scores = learning_curve(
            estimator4, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        ax4.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        color="#ff9124")
        ax4.fill_between(
            train_sizes,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
            color="#2492ff")
        ax4.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
                label="Training score")
        ax4.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
                label="Cross-validation score")
        ax4.set_title("Decision Tree Classifier \n Learning Curve", fontsize=14)
        ax4.set_xlabel('Training size (m)')
        ax4.set_ylabel('Score')
        ax4.grid(True)
        ax4.legend(loc="best")

        logger.info("Plots executed")
        return plt
