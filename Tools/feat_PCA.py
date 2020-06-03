import pandas as pd
import numpy as np
from scipy.stats import weightedtau, kendalltau, spearmanr, pearsonr


def _eigen_vector(dot_matrix: pd.DataFrame, var_threshold: float):
    """
    Advances in Financial Machine Learning, Snippet 8.5, page 119.
    Computation of Orthogonal Features
    Gets eigen values and eigen vector from matrix which explain % variance_thresh of total variance.
    :param dot_matrix: (np.array): Matrix for which eigen values/vectors should be computed.
    :param variance_thresh: (float): Percentage % of overall variance which compressed vectors should explain.
    :return: (pd.Series, pd.DataFrame): Eigenvalues, Eigenvectors.
    """
    # Compute eigen_vec from dot prod matrix, reduce dimension
    eigen_val, eigen_vec = np.linalg.eigh(dot_matrix) #Hermitian
    idx = eigen_val.argsort()[::-1]  # Arguments for sorting eigen_val desc
    eigen_val, eigen_vec = eigen_val[idx], eigen_vec[:, idx]

    # 2) Only positive eigen_vals
    eigen_val = pd.Series(eigen_val, index=['PC_' + str(i + 1) for i in range(eigen_val.shape[0])])
    eigen_vec = pd.DataFrame(eigen_vec, index=dot_matrix.index, columns=eigen_val.index)
    eigen_vec = eigen_vec.loc[:, eigen_val.index]

    # 3) Reduce dimension, form PCs
    cum_var = eigen_val.cumsum() / eigen_val.sum()
    dim = cum_var.values.searchsorted(var_threshold)
    eigen_val, eigen_vec = eigen_val.iloc[:dim + 1], eigen_vec.iloc[:, :dim + 1]
    return eigen_val, eigen_vec


def o_feat(feat_df: pd.DataFrame, var_threshold: float =.95):
    """
    Advances in Financial Machine Learning, Snippet 8.5, page 119.
    Computation of Orthogonal Features.
    Gets PCA orthogonal features.
    :param feature_df: (pd.DataFrame): Dataframe of features.
    :param variance_thresh: (float): Percentage % of overall variance which compressed vectors should explain.
    :return: (pd.DataFrame): Compressed PCA features which explain %variance_thresh of variance.
    """
    # Given a dataframe of features, compute orthogonal features
    feat_df = feat_df.sub(feat_df.mean(), axis=1).div(feat_df.std(), axis=1)  # Standardize
    dot_matrix = pd.DataFrame(np.dot(feat_df.T, feat_df),
                              index = feat_df.columns,
                              columns = feat_df.columns)
    _, eigen_vec = _eigen_vector(dot_matrix, var_threshold) #we only need vector for transformation
    _pca_feat = np.dot(feat_df, eigen_vec)
    return _pca_feat


def _pca_rank_weighted_kendall_tau(feature_imp, pca_rank):
    """
    Advances in Financial Machine Learning, Snippet 8.6, page 121.
    Computes Weighted Kendall's Tau Between Feature Importance and Inverse PCA Ranking.
    :param feature_imp: (np.array): Feature mean importance.
    :param pca_rank: (np.array): PCA based feature importance rank.
    :return: (float): Weighted Kendall Tau of feature importance and inverse PCA rank with p_value.
    """
    return weightedtau(feature_imp, pca_rank ** -1.0)


def feat_pca(feat_df: pd.DataFrame, feat_imp, var_threshold: float = 0.95):
    """
    Performs correlation analysis between feature importance (MDI for example, supervised) and PCA eigenvalues
    (unsupervised).
    High correlation means that probably the pattern identified by the ML algorithm is not entirely overfit.
    :param feature_df: (pd.DataFrame): Features dataframe.
    :param feature_importance: (pd.DataFrame): Individual MDI feature importance.
    :param variance_thresh: (float): Percentage % of overall variance which compressed vectors should explain in PCA compression.
    :return: (dict): Dictionary with kendall, spearman, pearson and weighted_kendall correlations and p_values.
    """
    feat_df = feat_df.sub(feat_df.mean(), axis=1).div(feat_df.std(), axis=1)# Standardize
    dot = pd.DataFrame(np.dot(feat_df.T, feat_df), index=feat_df.columns,
                       columns=feat_df.columns)
    eigen_val, eigen_vec = _eigen_vector(dot, var_threshold)

    # Compute correlations between eigen values for each eigen vector vs mdi importance
    all_eigen_values = []  # All eigen values in eigen vectors
    corr_dict = {'Pearson': [], 'Spearman': [], 'Kendall': []}  # Dictionary containing correlation metrics
    for vec in eigen_vec.columns:
        all_eigen_values.extend(abs(eigen_vec[vec].values * eigen_val[vec]))

    # We need to repeat importance array # of eigen vector times to generate correlation for all_eigen_values
    repeated_importance_array = np.tile(feat_imp['mean'].values, len(eigen_vec.columns))

    for corr_type, function in zip(corr_dict.keys(), [pearsonr, spearmanr, kendalltau]):
        corr_coef = function(repeated_importance_array, all_eigen_values)
        corr_dict[corr_type] = corr_coef

    # Get Rank based weighted Tau correlation
    feat_pca_rank = (eigen_val * eigen_vec).abs().sum(axis=1).rank(
        ascending=False)  # Sum of absolute values across all eigen vectors
    corr_dict['Weighted_Kendall_Rank'] = _pca_rank_weighted_kendall_tau(feat_imp['mean'].values,
                                                                           pca_rank = feat_pca_rank.values)
    return corr_dict