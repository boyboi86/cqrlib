"""
Implements the book chapter 7 on Cross Validation for financial data.
"""

import pandas as pd
import numpy as np

from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import KFold, BaseCrossValidator, GridSearchCV, RandomizedSearchCV
from sklearn.base import ClassifierMixin

from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle


def sample_weight_generator(X):
    sample_weight_ = np.ones(X.shape[0])
    sample_weight = pd.Series(sample_weight_, index = X.index).div(X.shape[0]) # if not weight assigned equal weight given
    return sample_weight

def train_times(events: pd.DataFrame, test_times: pd.Series):
    # pylint: disable=invalid-name
    """
    Advances in Financial Machine Learning, Snippet 7.1, page 106.
    Purging observations in the training set
    This function find the training set indexes given the information on which each record is based
    and the range for the test set.
    Given test_times, find the times of the training observations.
    :param data: (pd.Series) The information range on which each record is constructed from
        *data.index*: Time when the information extraction started.
        *data.value*: Time when the information extraction ended.
    :param test_times: (pd.Series) Times for the test dataset.
    :return: (pd.Series) Training set
    """
    train = events.copy(deep=True)
    for start_ix, end_ix in test_times.items():
        df0 = train[(start_ix <= train.index) & (train.index <= end_ix)].index  # Train starts within test
        df1 = train[(start_ix <= train) & (train <= end_ix)].index  # Train ends within test
        df2 = train[(train.index <= start_ix) & (end_ix <= train)].index  # Train envelops test
        train = train.drop(df0.union(df1).union(df2))
    return train

def embargo_times(times, pct_embargo: float = .01):
    if pct_embargo == 0:
        _embargo = pd.Series(times, index = times.index)
    else:
        _embargo = pd.Series(times[pct_embargo:], index = times[:-pct_embargo].index)
    return _embargo

class PurgedKFold(KFold):
    """
    Extend KFold class to work with labels that span intervals
    The train is purged of observations overlapping test-label intervals
    Test set is assumed contiguous (shuffle=False), w/o training samples in between
    :param n_splits: (int) The number of splits. Default to 3
    :param data: (pd.Series) The information range on which each record is constructed from
        *data.index*: Time when the information extraction started.
        *data.value*: Time when the information extraction ended.
    :param pct_embargo: (float) Percent that determines the embargo size.
    """

    def __init__(self,
                 n_splits: int = 5,
                 events: pd.Series = None,
                 pct_embargo: float = .01):

        if not isinstance(events, pd.Series):
            try:
                events = events.t1 #replaced raise ErrorValue
            except:
                raise ValueError("events params does not contain t1, pls use tri_barriers func")
                
        super(PurgedKFold, self).__init__(n_splits, shuffle=False, random_state=None)

        self.events = events
        self.pct_embargo = pct_embargo

    # noinspection PyPep8Naming
    def split(self,
              X: pd.DataFrame,
              y: pd.Series = None,
              groups=None):
        """
        The main method to call for the PurgedKFold class
        :param X: (pd.DataFrame) Samples dataset that is to be split
        :param y: (pd.Series) Sample labels series
        :param groups: (array-like), with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        :return: (tuple) [train list of sample indices, and test list of sample indices]
        """
        if (X.index == self.events.index).sum() != len(self.events):
            raise ValueError("cannot match different shape {0} {1}".format(X.shape[0], self.events.shape[0]))

        _idx = np.arange(X.shape[0])
        embargo = int(X.shape[0] * self.pct_embargo) #similar to sklearn round

        
        test_ranges = [(ix[0], ix[-1] + 1) for ix in np.array_split(np.arange(X.shape[0]), self.n_splits)]
        for start_ix, end_ix in test_ranges:
            
            test_indices = _idx[start_ix:end_ix]

            _test = pd.Series(index=[self.events[start_ix]], data=[self.events[end_ix-1]])
            _train = train_times(self.events, _test)
            _train = embargo_times(_train, embargo)
            train_indices = []
            for train_ix in _train.index:
                train_indices.append(self.events.index.get_loc(train_ix))
            yield np.array(train_indices), test_indices
            """
            t0  = self.events.index[start_idx]
            print("t0",t0)
            test_idx = _idx[start_idx:end_idx]
            print("test",test_idx)
            print("before max",self.events[test_idx].max())
            max_event_idx = self.events.searchsorted(self.events[test_idx].max())
            print("max",max_event_idx)
            train_idx = self.events.searchsorted(self.events[self.events <= t0].index)
            print("first",train_idx)
            train_idx = np.concatenate((train_idx, _idx[max_event_idx + embargo:]))
            print("second",train_idx)
            yield train_idx, test_idx
            """

# noinspection PyPep8Naming
def cv_score(
        classifier: ClassifierMixin,
        X: pd.DataFrame,
        y: pd.Series,
        events: pd.Series = None,
        pct_embargo: float = .01,
        cv_gen: BaseCrossValidator = None,
        sample_weight: pd.Series = None,
        scoring: str = "neg_log_loss",
        shuffle_after_split: bool = False):

    """
    Advances in Financial Machine Learning, Snippet 7.4, page 110.
    Using the PurgedKFold Class.
    Function to run a cross-validation evaluation of the using sample weights and a custom CV generator.
    Note: This function is different to the book in that it requires the user to pass through a CV object. The book
    will accept a None value as a default and then resort to using PurgedCV, this also meant that extra arguments had to
    be passed to the function. To correct this we have removed the default and require the user to pass a CV object to
    the function.
    Example:
    .. code-block:: python
        cv_gen = PurgedKFold(n_splits=n_splits, data=data, pct_embargo=pct_embargo)
        scores_array = cross_val_score(classifier, X, y, cv_gen, sample_weight=None, scoring=accuracy_score)
    :param classifier: (ClassifierMixin) A sk-learn Classifier object instance.
    :param X: (pd.DataFrame) The dataset of records to evaluate.
    :param y: (pd.Series) The labels corresponding to the X dataset.
    :param cv_gen: (BaseCrossValidator) Cross Validation generator object instance.
    :param sample_weight: (np.array) Sample weights used to train the model for each record in the dataset.
    :return: (np.array) The computed score.
    """
    if cv_gen is None:
        cv_gen = PurgedKFold(n_splits = 5,
                             events = events,
                             pct_embargo = pct_embargo)
        
    if sample_weight is None:
        sample_weight = sample_weight_generator(X = X) # if not weight assigned equal weight given
        
    scores = []
    for train, test in cv_gen.split(X=X, y=y): #we set y = None as default so we can leave it as it is
#==============================================================
        
        if shuffle_after_split:        
            test = shuffle(test, 
                            random_state = classifier.random_state) #added for randomness
            
 #==============================================================       
        fit = classifier.fit(X = X.iloc[train, :],
                             y = y.iloc[train],
                             sample_weight = sample_weight.iloc[train].values)
        
        if scoring == "neg_log_loss":
            prob = fit.predict_proba(X.iloc[test,:])
            _score = -log_loss(y_true = y.iloc[test], #in sklearn the formula already update to negative
                               y_pred = prob,
                               sample_weight = sample_weight.iloc[test],
                               labels = classifier.classes_)
        else:
            pred = fit.predict(X.iloc[test,:])
            _score = accuracy_score(y_true = y.iloc[test],
                                    y_pred = pred,
                                    sample_weight = sample_weight.iloc[test].values)
        scores.append(_score)
    return np.array(scores)


class Pipe_line(Pipeline):
    def fit(self, X, y, sample_weight = None, **fit_params):
        if sample_weight is not None:
            fit_params[self.steps[-1][0] + '__sample_weight'] = sample_weight
        else:
            sample_weight = sample_weight_generator(X) #newly added I don't want to use sklearn sample weight
            
        return super(Pipe_line, self).fit(X, y, **fit_params)

def hyper_fit(feat: pd.DataFrame, # Train X
              label: pd.Series, #meta-labels [0,1] binary form, y['bin']
              events: pd.Series, # y['t1']
              pipe_clf: ClassifierMixin, #classifierMixin SVM()
              param_grid: list, 
              n_splits: int = 3,
              bagging: list = [0, None, 1.], #n_estimators, max_samples, max_features
              random_search: int = 0,
              n_jobs: int = 1,
              pct_embargo: float = 0.0,
              **fit_params):
    
    if set(label.values) == {0,1}:
        scoring = 'f1'
    else:
        scoring = 'neg_log_loss'
    inner_cv = PurgedKFold(n_splits = n_splits,
                           events = events,
                           pct_embargo = pct_embargo)
    if random_search == 0:
        gs = GridSearchCV(estimator = pipe_clf, 
                          param_grid = param_grid, 
                          scoring = scoring, 
                          cv = inner_cv, 
                          n_jobs = n_jobs) #iid = False depreciated version 0.22.0
    else:
        gs = RandomizedSearchCV(estimator = pipe_clf, 
                                param_grid = param_grid, 
                                scoring = scoring, 
                                cv = inner_cv, 
                                n_jobs = n_jobs,
                                n_iter = random_search)
    
    gs = gs.fit(feat, label, **fit_params).best_estimator_ #pipeline
    
    if bagging[1] > 0:
        gs = BaggingClassifier(base_estimator = Pipe_line(gs.steps), 
                               n_estimators = int(bagging[0]), 
                               max_samples = float(bagging[1]), 
                               max_features = float(bagging[2]), 
                               n_jobs = n_jobs)
        gs = gs.fit(feat, label, sample_weight = fit_params \
                    [gs.base_estimator.steps[-1][0] + '__sample_weight'])
        gs = Pipeline(['bag', gs])
    return gs