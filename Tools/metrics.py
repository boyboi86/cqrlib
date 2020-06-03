import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import log_loss, accuracy_score
from sklearn.base import ClassifierMixin, ClusterMixin
from sklearn.model_selection import BaseCrossValidator

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from research.Tools.cross_validate import cv_score, PurgedKFold
from research.Util.multiprocess import mp_pandas_obj

def sample_weight_generator(X):
    sample_weight_ = np.ones(X.shape[0])
    sample_weight = pd.Series(sample_weight_, index = X.index).div(X.shape[0]) # if not weight assigned equal prob given
    return sample_weight

def mdi(fitted_model: ClassifierMixin,
        train_features: pd.DataFrame,
        clustered_subsets: ClusterMixin = None):
    """
    Advances in Financial Machine Learning, Snippet 8.2, page 115.
    MDI Feature importance
    Mean decrease impurity (MDI) is a fast, explanatory-importance (in-sample, IS) method specific to tree-based
    classifiers, like RF. At each node of each decision tree, the selected feature splits the subset it received in
    such a way that impurity is decreased. Therefore, we can derive for each decision tree how much of the overall
    impurity decrease can be assigned to each feature. And given that we have a forest of trees, we can average those
    values across all estimators and rank the features accordingly.
    Tip:
    Masking effects take place when some features are systematically ignored by tree-based classifiers in favor of
    others. In order to avoid them, set max_features=int(1) when using sklearn’s RF class. In this way, only one random
    feature is considered per level.
    Notes:
    * MDI cannot be generalized to other non-tree based classifiers
    * The procedure is obviously in-sample.
    * Every feature will have some importance, even if they have no predictive power whatsoever.
    * MDI has the nice property that feature importances add up to 1, and every feature importance is bounded between 0 and 1.
    * method does not address substitution effects in the presence of correlated features. MDI dilutes the importance of
      substitute features, because of their interchangeability: The importance of two identical features will be halved,
      as they are randomly chosen with equal probability.
    * Sklearn’s RandomForest class implements MDI as the default feature importance score. This choice is likely
      motivated by the ability to compute MDI on the fly, with minimum computational cost.
    Clustered Feature Importance( Machine Learning for Asset Manager snippet 6.4 page 86) :
    Clustered MDI  is the  modified version of MDI (Mean Decreased Impurity). It  is robust to substitution effect that
    takes place when two or more explanatory variables share a substantial amount of information (predictive power).CFI
    algorithm described by Dr Marcos Lopez de Prado  in Clustered Feature  Importance section of book Machine Learning
    for Asset Manager. Here  instead of  taking the importance  of  every feature, we consider the importance of every
    feature subsets, thus every feature receive the importance of subset it belongs to.
    :param model: (model object): Trained tree based classifier.
    :param feature_names: (list): Array of feature names.
    :param clustered_subsets: (list) Feature clusters for Clustered Feature Importance (CFI). Default None will not apply CFI.
                              Structure of the input must be a list of list/s i.e. a list containing the clusters/subsets of feature
                              name/s inside a list. E.g- [['I_0','I_1','R_0','R_1'],['N_1','N_2'],['R_3']]
    :return: (pd.DataFrame): Mean and standard deviation feature importance.
    """
    # Feature importance based on in-sample (IS) mean impurity reduction
    feature_imp_df = {i: tree.feature_importances_ for i, tree in enumerate(fitted_model.estimators_)}
    feature_imp_df = pd.DataFrame.from_dict(feature_imp_df, orient='index')
    feature_imp_df.columns = train_features

    # Make sure that features with zero importance are not averaged, since the only reason for a 0 is that the feature
    # was not randomly chosen. Replace those values with np.nan
    feature_imp_df = feature_imp_df.replace(0, np.nan)  # Because max_features = 1

    if clustered_subsets is not None:
        # Getting subset wise importance
        importance = pd.DataFrame(index = train_features.columns, columns=['mean', 'std'])
        for subset in clustered_subsets: # Iterating over each cluster
            subset_feat_imp = feature_imp_df[subset].sum(axis=1)
            # Importance of each feature within a subsets is equal to the importance of that subset
            importance.loc[subset, 'mean'] = subset_feat_imp.mean()
            importance.loc[subset, 'std'] = subset_feat_imp.std()*subset_feat_imp.shape[0]**-.5
    else:
        importance = pd.concat({'mean': feature_imp_df.mean(),
                                'std': feature_imp_df.std() * feature_imp_df.shape[0] ** -0.5},
                               axis=1)

    importance /= importance['mean'].sum()
    return importance


def mda(classifier: ClassifierMixin,
           X: pd.DataFrame,
           y: pd.Series,
           events: pd.DataFrame = None,
           cv_gen = None,
           n_splits: int = 10,
           pct_embargo: float = .0,
           sample_weight = None,
           scoring: str = "neg_log_loss",
           random_state=None):
    """
    Advances in Financial Machine Learning, Snippet 8.3, page 116-117.
    MDA Feature Importance
    Mean decrease accuracy (MDA) is a slow, predictive-importance (out-of-sample, OOS) method. First, it fits a
    classifier; second, it derives its performance OOS according to some performance score (accuracy, negative log-loss,
    etc.); third, it permutates each column of the features matrix (X), one column at a time, deriving the performance
    OOS after each column’s permutation. The importance of a feature is a function of the loss in performance caused by
    its column’s permutation. Some relevant considerations include:
    * This method can be applied to any classifier, not only tree-based classifiers.
    * MDA is not limited to accuracy as the sole performance score. For example, in the context of meta-labeling
      applications, we may prefer to score a classifier with F1 rather than accuracy. That is one reason a better
      descriptive name would have been “permutation importance.” When the scoring function does not correspond to a
      metric space, MDA results should be used as a ranking.
    * Like MDI, the procedure is also susceptible to substitution effects in the presence of correlated features.
      Given two identical features, MDA always considers one to be redundant to the other. Unfortunately, MDA will make
      both features appear to be outright irrelevant, even if they are critical.
    * Unlike MDI, it is possible that MDA concludes that all features are unimportant. That is because MDA is based on
      OOS performance.
    * The CV must be purged and embargoed.
    Clustered Feature Importance( Machine Learning for Asset Manager snippet 6.5 page 87) :
    Clustered MDA is the modified version of MDA (Mean Decreased Accuracy). It is robust to substitution effect that takes
    place when two or more explanatory variables share a substantial amount of information (predictive power).CFI algorithm
    described by Dr Marcos Lopez de Prado  in Clustered Feature  Importance (Presentation Slides)
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3517595. Instead of shuffling (permutating) all variables
    individually (like in MDA), we shuffle all variables in cluster together. Next, we follow all the  rest of the
    steps as in MDA. It can used by simply specifying the clustered_subsets argument.
    :param model: (sklearn.Classifier): Any sklearn classifier.
    :param X: (pd.DataFrame): Train set features.
    :param y: (pd.DataFrame, np.array): Train set labels.
    :param cv_gen: (cross_validation.PurgedKFold): Cross-validation object.
    :param clustered_subsets: (list) Feature clusters for Clustered Feature Importance (CFI). Default None will not apply CFI.
                              Structure of the input must be a list of list/s i.e. a list containing the clusters/subsets of feature
                              name/s inside a list. E.g- [['I_0','I_1','R_0','R_1'],['N_1','N_2'],['R_3']]
    :param sample_weight_train: (np.array) Sample weights used to train the model for each record in the dataset.
    :param sample_weight_score: (np.array) Sample weights used to evaluate the model quality.
    :param scoring: (function): Scoring function used to determine importance.
    :param random_state: (int) Random seed for shuffling the features.
    :return: (pd.DataFrame): Mean and standard deviation of feature importance.
    """
    if scoring not in ["neg_log_loss", "accuracy"]:
        scoring = "neg_log_loss" #replace raise ErrorValue
        
    if random_state is None:
        random_state = classifier.random_state

    if sample_weight is None:
        sample_weight = sample_weight_generator(X = X)
        
    if cv_gen is None:
        cv_gen = PurgedKFold(n_splits = n_splits, 
                             events = events,
                             pct_embargo = pct_embargo)

    fold_metrics_values, features_metrics_values = pd.Series(), pd.DataFrame(columns=X.columns)

    # Clustered feature subsets will be used for CFI if clustered_subsets exists else will operate on the single column as MDA
    
    for i, (train, test) in enumerate(cv_gen.split(X=X, y=y)):
        X0, y0, w0 = X.iloc[train, :], y.iloc[train], sample_weight.iloc[train]
        X1, y1, w1 = X.iloc[test, :], y.iloc[test], sample_weight.iloc[test]
        fit = classifier.fit(X = X0, y = y0, sample_weight = w0.values)

        # Get overall metrics value on out-of-sample fold
        if scoring == "neg_log_loss":
            prob = fit.predict_proba(X1)
            fold_metrics_values.loc[i] = -log_loss(y1,
                                                   prob,
                                                   sample_weight=w1.values,
                                                   labels=classifier.classes_)
        else:
            pred = fit.predict(X1)
            fold_metrics_values.loc[i] = accuracy_score(y1,
                                                        pred,
                                                        sample_weight=w1.values)

        # Get feature specific metric on out-of-sample fold
        for j in X.columns:
            X1_ = X1.copy(deep=True)
            np.random.RandomState(seed=random_state).shuffle(X1_[j].values)  # Permutation of a single column for MDA shuffle values within column
            if scoring == "neg_log_loss":
                prob = fit.predict_proba(X1_)
                features_metrics_values.loc[i, j] = -log_loss(y1,
                                                              prob,
                                                              sample_weight=w1.values,
                                                              labels=classifier.classes_)
            else:
                pred = fit.predict(X1_)
                features_metrics_values.loc[i, j] = accuracy_score(y1,
                                                                   pred,
                                                                   sample_weight=w1.values)

    importance = (-features_metrics_values).add(fold_metrics_values, axis=0)
    if scoring == "neg_log_loss":
        importance = importance / -features_metrics_values
    else:
        importance = importance / (1.0 - features_metrics_values).replace(0, np.nan)
    importance = pd.concat({'mean': importance.mean(), 'std': importance.std() * importance.shape[0] ** -.5}, axis=1)
    importance.replace([-np.inf, np.nan], 0, inplace=True)  # Replace infinite values

    return importance, fold_metrics_values.mean()


def sfi(classifier: ClassifierMixin,
        X: pd.DataFrame,
        y: pd.Series,
        cv_gen: BaseCrossValidator,
        scoring: str ="neg_log_loss",
        sample_weight=None):
    """
    Advances in Financial Machine Learning, Snippet 8.4, page 118.
    Implementation of SFI
    Substitution effects can lead us to discard important features that happen to be redundant. This is not generally a
    problem in the context of prediction, but it could lead us to wrong conclusions when we are trying to understand,
    improve, or simplify a model. For this reason, the following single feature importance method can be a good
    complement to MDI and MDA.
    Single feature importance (SFI) is a cross-section predictive-importance (out-of- sample) method. It computes the
    OOS performance score of each feature in isolation. A few considerations:
    * This method can be applied to any classifier, not only tree-based classifiers.
    * SFI is not limited to accuracy as the sole performance score.
    * Unlike MDI and MDA, no substitution effects take place, since only one feature is taken into consideration at a time.
    * Like MDA, it can conclude that all features are unimportant, because performance is evaluated via OOS CV.
    The main limitation of SFI is that a classifier with two features can perform better than the bagging of two
    single-feature classifiers. For example, (1) feature B may be useful only in combination with feature A;
    or (2) feature B may be useful in explaining the splits from feature A, even if feature B alone is inaccurate.
    In other words, joint effects and hierarchical importance are lost in SFI. One alternative would be to compute the
    OOS performance score from subsets of features, but that calculation will become intractable as more features are
    considered.
    :param clf: (sklearn.Classifier): Any sklearn classifier.
    :param X: (pd.DataFrame): Train set features.
    :param y: (pd.DataFrame, np.array): Train set labels.
    :param cv_gen: (cross_validation.PurgedKFold): Cross-validation object.
    :param sample_weight_train: (np.array) Sample weights used to train the model for each record in the dataset.
    :param sample_weight_score: (np.array) Sample weights used to evaluate the model quality.
    :param scoring: (function): Scoring function used to determine importance.
    :return: (pd.DataFrame): Mean and standard deviation of feature importance.
    """
    if isinstance(X, pd.Index):
        feature_names = X
    else:
        try:
            feature_names = X.columns
        except:
            raise ValueError("X params require features columns index input")
    
    if sample_weight is None:
        sample_weight = sample_weight_generator(X = X)

    importance = pd.DataFrame(columns=['mean', 'std'])
    for feat in feature_names:
        _cv_scores = cv_score(classifier = classifier,
                              X=X[[feat]],
                              y=y,
                              sample_weight = sample_weight,
                              scoring=scoring,
                              cv_gen=cv_gen)
        
        importance.loc[feat, 'mean'] = _cv_scores.mean()
        importance.loc[feat, 'std'] = _cv_scores.std() * _cv_scores.shape[0] ** -.5
    return importance

def mp_sfi(classifier: ClassifierMixin,
            feature_names: pd.Index,
            X: pd.DataFrame,
            y: pd.Series,
            cv_gen: BaseCrossValidator,
            scoring: str ="neg_log_loss",
            sample_weight=None):
    """
    Advances in Financial Machine Learning, Snippet 8.4, page 118.
    Implementation of SFI
    Substitution effects can lead us to discard important features that happen to be redundant. This is not generally a
    problem in the context of prediction, but it could lead us to wrong conclusions when we are trying to understand,
    improve, or simplify a model. For this reason, the following single feature importance method can be a good
    complement to MDI and MDA.
    Single feature importance (SFI) is a cross-section predictive-importance (out-of- sample) method. It computes the
    OOS performance score of each feature in isolation. A few considerations:
    * This method can be applied to any classifier, not only tree-based classifiers.
    * SFI is not limited to accuracy as the sole performance score.
    * Unlike MDI and MDA, no substitution effects take place, since only one feature is taken into consideration at a time.
    * Like MDA, it can conclude that all features are unimportant, because performance is evaluated via OOS CV.
    The main limitation of SFI is that a classifier with two features can perform better than the bagging of two
    single-feature classifiers. For example, (1) feature B may be useful only in combination with feature A;
    or (2) feature B may be useful in explaining the splits from feature A, even if feature B alone is inaccurate.
    In other words, joint effects and hierarchical importance are lost in SFI. One alternative would be to compute the
    OOS performance score from subsets of features, but that calculation will become intractable as more features are
    considered.
    :param clf: (sklearn.Classifier): Any sklearn classifier.
    :param X: (pd.DataFrame): Train set features.
    :param y: (pd.DataFrame, np.array): Train set labels.
    :param cv_gen: (cross_validation.PurgedKFold): Cross-validation object.
    :param sample_weight_train: (np.array) Sample weights used to train the model for each record in the dataset.
    :param sample_weight_score: (np.array) Sample weights used to evaluate the model quality.
    :param scoring: (function): Scoring function used to determine importance.
    :return: (pd.DataFrame): Mean and standard deviation of feature importance.
    """
    
    if sample_weight is None:
        sample_weight = sample_weight_generator(X = X)
    
    importance = pd.DataFrame(columns=['mean', 'std'])
    for feat in feature_names:
        _cv_scores = cv_score(classifier = classifier,
                              X=X[[feat]],
                              y=y,
                              sample_weight = sample_weight,
                              scoring=scoring,
                              cv_gen=cv_gen)
        
        importance.loc[feat, 'mean'] = _cv_scores.mean()
        importance.loc[feat, 'std'] = _cv_scores.std() * _cv_scores.shape[0] ** -.5
    return importance

def plot_feat_imp(feat_imp_df: pd.DataFrame,
                  oob_score: float,
                  oos_score: float,
                  method: str,
                  output_path = None,
                  save_fig: bool = False):
    """
    Advances in Financial Machine Learning, Snippet 8.10, page 124.
    Feature importance plotting function.
    Plot feature importance.
    :param importance_df: (pd.DataFrame): Mean and standard deviation feature importance.
    :param oob_score: (float): Out-of-bag score.
    :param oos_score: (float): Out-of-sample (or cross-validation) score.
    :param save_fig: (bool): Boolean flag to save figure to a file.
    :param output_path: (str): If save_fig is True, path where figure should be saved.
    """
    # Plot mean imp bars with std
    plt.figure(figsize=(10, feat_imp_df.shape[0] / 5))
    feat_imp_df.sort_values('mean', ascending=True, inplace=True)
    ax = feat_imp_df['mean'].plot(kind='barh', color='b', alpha=0.25, xerr=feat_imp_df['std'], error_kw={'ecolor': 'r'})
    
    if method == 'MDI':
        plt.axvline(1./feat_imp_df.shape[0], lw = 1, c = 'r', ls = '--')
    ax.get_yaxis().set_visible(False)
    for i,j in zip(ax.patches, feat_imp_df.index):
        ax.text(i.get_width()/2,
                i.get_y() + i.get_height()/2,
                j,
                ha="center",
                va="center",
                c = "black")
    plt.title('tag = {0} | OOB Score:{1:.6f}| OOS score:{2:.6f}'.format(method, oob_score, oos_score))

    if save_fig is True:
        plt.savefig(output_path)
    else:
        plt.show()
        
def _feat_imp_analysis(classifier,
                      X: pd.DataFrame,
                      y: pd.Series,
                      method: str = 'SFI',
                      events: pd.Series = None,
                      pct_embargo = 0.01,
                      sample_weight = None,
                      n_splits: int = 10,
                      min_weight_fraction_leaf = 0.0,
                      n_jobs: int = 1,
                      scoring: str = "accuracy"):
    
    if classifier is None:
        clf = DecisionTreeClassifier(criterion = 'entropy',
                                     max_features = int(1), # has to be 1 with dtype integer
                                     class_weight = 'balanced',
                                     min_weight_fraction_leaf = min_weight_fraction_leaf)
        
        classifier = BaggingClassifier(base_estimator=clf,
                                        n_estimators=1000,
                                        max_features=1.0, #if this is not a float, your max feature for all will only be 1
                                        max_samples=1.0,
                                        oob_score=True,
                                        n_jobs=n_jobs)
    
    if events is None:
        try:
            events = y.t1 #if you are using "yield" instead of "return", avoid try/ except/ finally
        except:
            raise ValueError("params events t1 required")
        
    fit_model = classifier.fit(X, y['bin'], sample_weight=sample_weight)
    oob_score = fit_model.oob_score_

    cv_gen = PurgedKFold(n_splits=n_splits, events=events, pct_embargo = pct_embargo)
    oos_score = cv_score(classifier, X, y['bin'], cv_gen=cv_gen, scoring=scoring).mean()

    if method == 'MDI':
        imp = mdi(fitted_model = fit_model,
                     train_features = X.columns)
        
    elif method == 'MDA':
        imp, oos_score = mda(classifier = classifier,
                                 X = X,
                                 y = y['bin'],
                                 events = events,
                                 cv_gen = cv_gen,
                                 pct_embargo = pct_embargo,
                                 scoring=scoring,
                                 sample_weight=sample_weight)
        
    elif method == 'SFI':
        n_jobs_ = classifier.n_jobs 
        classifier.n_jobs = 1 # reset to 1, use mp_pandas_obj instead
        imp = mp_pandas_obj(mp_sfi, ('feature_names', X.columns), 
                             num_threads = n_jobs_, 
                             classifier = classifier, 
                             X = X, 
                             y = y['bin'], 
                             cv_gen = cv_gen, 
                             scoring = scoring,
                             sample_weight=sample_weight)
        
    return imp, oob_score, oos_score


def feat_imp_analysis(classifier = None,
                      X = None,
                      y = None,
                      sample_weight = None,
                      methods=['MDI', 'MDA', 'SFI'],
                      output_path: str = None):
    for method in methods:
        feat_imp_score, oob_score, oos_score = _feat_imp_analysis(classifier = classifier,
                                                                   X=X,
                                                                   y=y,
                                                                   method = method,
                                                                   sample_weight = sample_weight)

        plot_feat_imp(feat_imp_df = feat_imp_score, 
                     oob_score=oob_score, 
                     oos_score=oos_score,
                     method = method,
                     save_fig=False, 
                     output_path=output_path)