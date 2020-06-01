"""
Implements the book chapter 7 on Cross Validation for financial data.
"""

import pandas as pd
import numpy as np

from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import KFold
from sklearn.base import ClassifierMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import shuffle


def train_times(events: pd.DataFrame, test_times: pd.Series) -> pd.Series:
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
    train = events.t1.copy(deep=True)
    for start_ix, end_ix in test_times.items():
        df0 = train[(start_ix <= train.index) & (train.index <= end_ix)].index  # Train starts within test
        df1 = train[(start_ix <= train) & (train <= end_ix)].index  # Train ends within test
        df2 = train[(train.index <= start_ix) & (end_ix <= train)].index  # Train envelops test
        train = train.drop(df0.union(df1).union(df2))
    return train

def embargo_times(times, pct_embargo: float = .0):
    step = int(times.shape[0] * pct_embargo)
    if step == 0:
        _embargo = pd.Series(times, index = times)
    else:
        _embargo = pd.Series(times[step:], index = times[:-step])
        _embargo.append(pd.Series(times[-1:], index = times[-step:]))
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
                 pct_embargo: float = 0.):

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
              y: pd.Series,
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
        test_ranges = [(idx[0], idx[-1] + 1) for idx in np.array_split(np.arange(X.shape[0]), self.n_splits)]
        for start_idx, end_idx in test_ranges:
            t0  = self.events.index[start_idx]
            test_idx = _idx[start_idx:end_idx]
            
            max_event_idx = self.events.searchsorted(self.events[test_idx].max())
            train_idx = self.events.searchsorted(self.events[self.events <= t0].index)
            train_idx = np.concatenate((train_idx, _idx[max_event_idx + embargo:]))

            yield train_idx, test_idx


# noinspection PyPep8Naming
def cv_score(
        classifier: ClassifierMixin,
        X: pd.DataFrame,
        y: pd.Series,
        events: pd.DataFrame = None,
        pct_embargo: float = .0,
        cv_gen: BaseCrossValidator = None,
        sample_weight: np.ndarray = None,
        scoring: str = "neg_log_loss",
        shuffle_after_split: bool = True):

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
        sample_weight_ = np.ones(X.shape[0])
        sample_weight = pd.Series(sample_weight_, index = X.index)# if not weight assigned equal weight given
        
    scores = []
    for train, test in cv_gen.split(X=X, y=y): #we set y = None as default so we can leave it as it is
#==============================================================
        if shuffle_after_split is True:        
            train = shuffle(train, 
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