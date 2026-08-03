import numpy as np
import random
import mne
import scipy.io
import matplotlib.pyplot as plt
import copy
import math
import pickle
import os
import pandas as pd
import seaborn as sns
from numpy import linalg as LA
from scipy import signal
from scipy.linalg import toeplitz, eig, eigh, sqrtm, lstsq
from scipy.sparse.linalg import eigs
from scipy.stats import zscore, pearsonr, binomtest, binom, wilcoxon, mannwhitneyu, rankdata
from sklearn.covariance import LedoitWolf
from collections import Counter
from sklearn.cluster import DBSCAN


def eig_sorted(X, option='descending'):
    '''
    Eigenvalue decomposition, with ranked eigenvalues
    X = V @ np.diag(lam) @ LA.inv(V)
    Could be replaced by eig in scipy.linalg
    '''
    lam, V = LA.eig(X)
    # lam = np.real(lam)
    # V = np.real(V)
    if option == 'descending':
        idx = np.argsort(-lam)
    elif option =='ascending':
        idx = np.argsort(lam)
    else:
        idx = range(len(lam))
        print('Warning: Not sorted')
    lam = lam[idx] # rank eigenvalues
    V = V[:, idx] # rearrange eigenvectors accordingly
    return lam, V


def geig_sorted(A, B, n_comp, option='descending'):
    '''
    Solve the generalized eigenvalue problem A V = B V diag(lam)
    Assume that A and B are real symmetric matrices
    Return the first n_comp eigenvalues and eigenvectors
    '''
    lam, V = eig(A, B) 
    lam = np.real(lam)
    V = np.real(V)
    if option == 'descending':
        idx = np.argsort(-lam)
    elif option =='ascending':
        idx = np.argsort(lam)
    else:
        idx = range(len(lam))
        print('Warning: Not sorted')
    lam = lam[idx] # rank eigenvalues
    V = V[:, idx] # rearrange eigenvectors accordingly
    return lam[:n_comp], V[:,:n_comp]


def PCAreg_inv(X, rank):
    '''
    PCA Regularized inverse of a symmetric square matrix X
    rank could be a smaller number than rank(X)
    '''
    lam, V = eig_sorted(X)
    lam = lam[:rank]
    V = V[:, :rank]
    inv = V @ np.diag(1/lam) @ np.transpose(V)
    return inv


def schmidt_orthogonalization(vectors):
    vector_dim, num_vectors = vectors.shape
    orthogonal_vectors = np.zeros((vector_dim, num_vectors))
    for i in range(num_vectors):
        orthogonal_vector = vectors[:, i]
        for j in range(i):
            orthogonal_vector -= np.dot(vectors[:, i], orthogonal_vectors[:, j]) / np.dot(orthogonal_vectors[:, j], orthogonal_vectors[:, j]) * orthogonal_vectors[:, j]
        orthogonal_vectors[:, i] = orthogonal_vector / np.linalg.norm(orthogonal_vector)
    return orthogonal_vectors


def expand_data_to_3D(data):
    '''
    Expand 1D or 2D data to 3D data
    '''
    if np.ndim(data) == 1:
        data = np.expand_dims(np.expand_dims(data, axis=1), axis=2)
    elif np.ndim(data) == 2:
        data = np.expand_dims(data, axis=2)
    return data


def regress_out(X, Y):
    '''
    Regress out Y from X
    X: T x Dx or T x Dx x N
    Y: T x Dy or T,
    '''
    if np.ndim(Y) == 1:
        Y = np.expand_dims(Y, axis=1)
    if np.ndim(X) == 3:
        if np.ndim(Y) == 2:
            X_res = copy.deepcopy(X)
            for i in range(X.shape[2]):
                W = lstsq(Y, X[:,:,i])[0]
                X_res[:,:,i] = X[:,:,i] - Y @ W
        elif np.ndim(Y) == 3:
            X_res = copy.deepcopy(X)
            for i in range(X.shape[2]):
                W = lstsq(Y[:,:,i], X[:,:,i])[0]
                X_res[:,:,i] = X[:,:,i] - Y[:,:,i] @ W
        else:
            raise ValueError('Check the dimension of Y')
    elif np.ndim(X) == 2:
        assert np.ndim(Y) == 2, 'Check the dimension of Y'
        W = lstsq(Y, X)[0]
        X_res = X - Y @ W
    else:
        raise ValueError('Check the dimension of X')
    return X_res


def regress_out_4D(X, Y):
    '''
    Regress out Y from X
    X: T x Dx x N x nb_tasks
    Y: T x Dy x nb_tasks,
    '''
    X_res = copy.deepcopy(X)
    for i in range(X.shape[-1]):
        X_res[:,:,:,i] = regress_out(X[:,:,:,i], Y[:,:,i])
    return X_res


def regress_out_2D_pair(X, Y, C):
    '''
    Regress out C from both X and Y
    '''
    if np.ndim(X) == 1:
        X = np.expand_dims(X, axis=1)
    if np.ndim(Y) == 1:
        Y = np.expand_dims(Y, axis=1)
    X_res = regress_out(X, C)
    Y_res = regress_out(Y, C)
    return X_res, Y_res


def regress_out_2D_pair_trials(X_trials, Y_trials, C_trials):
    '''
    Regress out C from both X and Y
    '''
    X_res_trials = []
    Y_res_trials = []
    for X, Y, C in zip(X_trials, Y_trials, C_trials):
        X_res, Y_res = regress_out_2D_pair(X, Y, C)
        X_res_trials.append(X_res)
        Y_res_trials.append(Y_res)
    return X_res_trials, Y_res_trials


def further_regress_out(data_3D, confound_3D, L_d, L_c, offset_d, offset_c):
    N = data_3D.shape[2]
    data_clean_list = []
    for n in range(N):
        data = block_Hankel(data_3D[:,:,n], L_d, offset_d)
        confound = block_Hankel(confound_3D[:,:,n], L_c, offset_c)
        data_clean = regress_out(data, confound)
        data_clean_list.append(data_clean)
    data_clean_3D = np.stack(data_clean_list, axis=2)
    return data_clean_3D


def further_regress_out_list(X_list, confound_list, L_d, L_c, offset_d, offset_c):
    X_list = [expand_data_to_3D(X) for X in X_list]
    confound_list = [expand_data_to_3D(confound) for confound in confound_list]
    N = max(X_list[0].shape[2], confound_list[0].shape[2])
    X_list = [np.tile(X,(1,1,N)) if X.shape[2] == 1 else X for X in X_list]
    confound_list = [np.tile(confound,(1,1,N)) if confound.shape[2] == 1 else confound for confound in confound_list]
    # Do regression on the entire concatenated data
    # len_list = [X.shape[0] for X in X_list]
    # X_reg = further_regress_out(np.concatenate(tuple(X_list), axis=0), np.concatenate(tuple(confound_list), axis=0), L_d, L_c, offset_d, offset_c)
    # X_reg_list = np.split(X_reg, np.cumsum(len_list)[:-1], axis=0)
    # Do regression per video
    X_reg_list = [further_regress_out(X, confound, L_d, L_c, offset_d, offset_c) for X, confound in zip(X_list, confound_list)]
    return X_reg_list


def regress_out_confounds(data_list, feat_att_list, feat_unatt_list, confound_list, L_data, L_Stim, offset_data, offset_Stim):
    '''
    Regressing confound (modality to be controlled) out of the data and features
    '''
    data_reg = further_regress_out_list(data_list, confound_list, L_data, L_data, offset_data, offset_data)
    feat_att_reg = further_regress_out_list(feat_att_list, confound_list, L_Stim, L_data, offset_Stim, offset_data)
    feat_unatt_reg = further_regress_out_list(feat_unatt_list, confound_list, L_Stim, L_data, offset_Stim, offset_data)
    return (data_reg, feat_att_reg, feat_unatt_reg)


def stack_modal(modal_nested_list):
    nb_video = len(modal_nested_list[0])
    dim_list = [modal[0].shape[1] for modal in modal_nested_list]
    stacked_list = []
    for i in range(nb_video):
        modal_list = [modal[i] for modal in modal_nested_list]
        modal_stacked = np.concatenate(tuple(modal_list), axis=1)
        stacked_list.append(modal_stacked)
    return stacked_list, dim_list


def get_cov_mtx(X, dim_list, regularization=None):
    '''
    Get the covariance matrix of X (T x dimX) with or without regularization
    dim_ilst is a list of dimensions of each modality (data from different subjects can also be viewed as different modalities)
    sum(dim_list) = dimX
    '''
    Rxx = np.cov(X, rowvar=False)
    Dxx = np.zeros_like(Rxx)
    dim_accumu = 0
    for dim in dim_list:
        if regularization == 'lwcov':
            Rxx[dim_accumu:dim_accumu+dim, dim_accumu:dim_accumu+dim] = LedoitWolf().fit(X[:, dim_accumu:dim_accumu+dim]).covariance_
        Dxx[dim_accumu:dim_accumu+dim, dim_accumu:dim_accumu+dim] = Rxx[dim_accumu:dim_accumu+dim, dim_accumu:dim_accumu+dim]
        dim_accumu += dim
    return Rxx, Dxx


def bandpass(data, fs, band):
    '''
    Bandpass filter
    Inputs:
        data: T x D
        fs: sampling frequency
        band: frequency band
    Outputs:
        filtered data: T x D        
    '''
    b, a = scipy.signal.butter(5, np.array(band), btype='bandpass', fs=fs)
    filtered = scipy.signal.filtfilt(b, a, data, axis=0)
    return filtered


def extract_freq_band(eeg, fs, band, normalize=False):
    '''
    Extract frequency band from EEG data
    Inputs:
        eeg: EEG data
        fs: sampling frequency
        band: frequency band
        normalize: whether to normalize the bandpassed data
    Outputs:
        eeg_band: bandpassed EEG data
    '''
    if eeg.ndim < 3:
        eeg_band = bandpass(eeg, fs, band)
        eeg_band = eeg_band / np.linalg.norm(eeg_band, 'fro') if normalize else eeg_band
    else:
        N = eeg.shape[2]
        eeg_band =np.zeros_like(eeg)
        for n in range(N):
            eeg_band[:,:,n] = bandpass(eeg[:,:,n], fs, band)
            eeg_band[:,:,n] = eeg_band[:,:,n] / np.linalg.norm(eeg_band[:,:,n], 'fro') if normalize else eeg_band[:,:,n]
    return eeg_band


def Hankel_mtx(L_timefilter, x, offset=0):
    '''
    Calculate the Hankel matrix
    Convolution: y(t)=x(t)*h(t)
    In matrix form: y=Xh E.g. time lag = 3
    If offset=0,
    h = h(0); h(1); h(2)
    X = 
    x(0)   x(-1)  x(-2)
    x(1)   x(0)   x(-1)
            ...
    x(T-1) x(T-2) x(T-3)
    If offset !=0, e.g., offset=1,
    h = h(-1); h(0); h(1)
    X = 
    x(1)   x(0)   x(-1)
    x(2)   x(1)   x(0)
            ...
    x(T)   x(T-1) x(T-2)
    Unknown values are set as 0
    This is useful when we want to remove segments (e.g., blinks, saccades) in the signals.
    '''
    first_col = np.zeros(L_timefilter)
    first_col[0] = x[0]
    if offset != 0:
        x = np.append(x, [np.zeros((1,offset))])
    hankel_mtx = np.transpose(toeplitz(first_col, x))
    if offset != 0:
        hankel_mtx = hankel_mtx[offset:,:]
    return hankel_mtx


def block_Hankel(X, L, offset=0):
    '''
    For spatial-temporal filter, calculate the block Hankel matrix
    Inputs:
    X: T(#sample)xD(#channel)
    L: number of time lags; 
    offset: offset of time lags; from -(L-1) to 0 (offset=0) or offset-(L-1) to offset
    '''
    if np.ndim(X) == 1:
        X = np.expand_dims(X, axis=1)
    Hankel_list = [Hankel_mtx(L, X[:,i], offset) for i in range(X.shape[1])]
    blockHankel = np.concatenate(tuple(Hankel_list), axis=1)
    return blockHankel


def hankelize_data_multisub(data_multisub, L, offset):
    N = data_multisub.shape[2]
    X_list = [block_Hankel(data_multisub[:,:,n], L, offset) for n in range(N)]
    X = np.stack(X_list, axis=2)
    return X


def random_combination(iterable, r):
    "Random selection from itertools.combinations(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return [pool[i] for i in indices]


def transformed_GEVD(Dxx, Rxx, rho, dimStim, n_components):
    Rxx_hat = copy.deepcopy(Rxx)
    Rxx_hat[:, -dimStim:] = np.sqrt(rho) * Rxx_hat[:, -dimStim:]
    Rxx_hat[-dimStim:, :] = np.sqrt(rho) * Rxx_hat[-dimStim:, :]
    Rxx_hat = (Rxx_hat + Rxx_hat.T)/2
    lam, W = eigh(Dxx, Rxx_hat, subset_by_index=[0,n_components-1]) # automatically ascend
    W[-dimStim:, :] = np.sqrt(1/rho) * W[-dimStim:, :]
    # Alternatively:
    # Rxx_hat = copy.deepcopy(Rxx)
    # Rxx_hat[:,-D_stim*L_Stim:] = Rxx_hat[:,-D_stim*L_Stim:]*rho
    # Rxx_hat[-D_stim*L_Stim:,:] = Rxx_hat[-D_stim*L_Stim:,:]*rho
    # Dxx_hat = copy.deepcopy(Dxx)
    # Dxx_hat[:,-D_stim*L_Stim:] = Dxx_hat[:,-D_stim*L_Stim:]*rho
    # lam, W = eigh(Dxx_hat, Rxx_hat, subset_by_index=[0,self.n_components-1])
    return lam, W


def into_trials(data, fs, t=60, start_points=None):
    if np.ndim(data)==1:
        data = np.expand_dims(data, axis=1)
    T = data.shape[0]
    if start_points is not None:
        # has a target number of trials with specified start points, then randomly select nb_trials trials
        # select data from start_points along axis 0
        data_trials = [data[start:start+fs*t, ...] for start in start_points]
    else:
        # does not have a target number of trials, then divide data into t s trials without overlap
        # if T is not a multiple of t sec, then discard the last few samples
        T_trunc = T - T%(fs*t)
        data_intmin = data[:T_trunc, ...]
        # segment X_intmin into 1 min trials along axis 0
        data_trials = np.split(data_intmin, int(T/(fs*t)), axis=0)
    return data_trials


def shift_trials(data_trials, shift=None):
    '''
    Given a list of trials, move part of the trials to the end of the list
    '''
    nb_trials = len(data_trials)
    if shift is None:
        shift = nb_trials//3
    trials_shifted = [data_trials[(n+shift)%nb_trials] for n in range(nb_trials)]
    return trials_shifted


def select_mismatches(data_trials, nb_mismatch):
    '''
    Given a list of trials, select nb_mismatch mismatched trials for each trial
    '''
    nb_trials = len(data_trials)
    match_indices_list = []
    mismatch_indices_list = []
    for i in range(nb_trials):
        possible_indices = [j for j in range(nb_trials) if j != i] if nb_trials > 1 else [0]
        if nb_mismatch > len(possible_indices):
            nb_mismatch = len(possible_indices)
        mismatch_indices = random.sample(possible_indices, nb_mismatch)
        match_indices = [i]*nb_mismatch
        mismatch_indices_list.extend(mismatch_indices)
        match_indices_list.extend(match_indices)
    return mismatch_indices_list, match_indices_list


def match_mismatch_pairs(X_trials, Y_trials, nb_mismatch):
    mismatch_indices_list, match_indices_list = select_mismatches(Y_trials, nb_mismatch)
    matches_X = [X_trials[i] for i in match_indices_list]
    matches_Y = [Y_trials[i] for i in match_indices_list]
    mismatches_Y = [Y_trials[i] for i in mismatch_indices_list]
    return matches_X, matches_Y, mismatches_Y


def split_multi_mod_LVO(nested_datalist, leave_out=2):
    '''
    Datasets are organized in nested datalist: [[EEG_1, EEG_2, ... ], [Vis_1, Vis_2, ... ], [Sd_1, Sd_2, ... ]]
    Create training and test lists for leave-one-video-out cross-validation
    - A training list: [[EEG_train, Vis_train, Sd_train]_1, [EEG_train, Vis_train, Sd_train]_2, ...]
    - A test list: [[EEG_test, Vis_test, Sd_test]_1, [EEG_test, Vis_test, Sd_test]_2, ...]
    '''
    nb_videos = len(nested_datalist[0])
    train_list_folds = []
    test_list_folds = []
    for i in range(0, nb_videos, leave_out):
        indices_train = [j for j in range(nb_videos) if j not in range(i, i+leave_out)]
        indices_test = [j for j in range(i, i+leave_out)]
        train_list_folds.append([np.concatenate(tuple([mod[i] for i in indices_train]), axis=0) if mod is not None else None for mod in nested_datalist])
        test_list_folds.append([np.concatenate(tuple([mod[i] for i in indices_test]), axis=0) if mod is not None else None for mod in nested_datalist])
    return train_list_folds, test_list_folds


def wilcoxon_effect(a, b, zero_method="wilcox", alternative="greater"):
    """Paired Wilcoxon signed-rank: Z, p, and matched-pairs rank-biserial r.

    a, b : 1D arrays of paired measurements (e.g. per-subject accuracies
            in Task 3 and Task 2). 
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    d = d[d != 0]                       # drop zero differences (Wilcoxon default)
    n = d.size                          # number of non-zero pairs
    r_abs = rankdata(np.abs(d))
    W_plus = r_abs[d > 0].sum()
    W_minus = r_abs[d < 0].sum()

    # rank-biserial from the signed-rank sums (definitional form)
    rb = (W_plus - W_minus) / (W_plus + W_minus)

    # Z and two-sided p from the normal approximation (tie/continuity corrected)
    res = wilcoxon(a, b, method="approx", zero_method=zero_method, alternative=alternative)
    z, p = res.zstatistic, res.pvalue
    r_from_z = z / np.sqrt(n)           # should match rb up to sign/tie handling

    return dict(n=n, W_plus=W_plus, W_minus=W_minus,
                z=z, p=p, rank_biserial=rb, r_from_z=r_from_z)


def mwu_effect(x, y, continuity=True, alternative="two-sided"):
    """Independent Mann-Whitney U: U, Z, p, rank-biserial r, and CLES.

    x, y : 1D arrays for the two groups (e.g. x = free-viewing, y = fixation).
    Sign convention: positive r / CLES > 0.5 means x tends to exceed y.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    n1, n2 = x.size, y.size

    res = mannwhitneyu(x, y, alternative=alternative)
    U1, p = res.statistic, res.pvalue          # U for the first group (x)

    rb = 2 * U1 / (n1 * n2) - 1                 # independent rank-biserial
    cles = U1 / (n1 * n2)                       # P(x > y), common-language ES

    # Z from the normal approximation with tie correction
    N = n1 + n2
    ranks = rankdata(np.concatenate([x, y]))
    _, counts = np.unique(ranks, return_counts=True)
    tie = (counts**3 - counts).sum()
    mu = n1 * n2 / 2.0
    sigma = np.sqrt((n1 * n2 / 12.0) * ((N + 1) - tie / (N * (N - 1))))
    cc = 0.5 if continuity else 0.0
    z = (U1 - mu - np.sign(U1 - mu) * cc) / sigma

    return dict(n1=n1, n2=n2, U=U1, z=z, p=p,
                rank_biserial=rb, cles=cles, r_from_z=z / np.sqrt(N))


def sig_level_binomial_test(p_value, total_trials, p=0.5):
    critical_value = binom.ppf(1-p_value, total_trials, p=p)
    critical_value = critical_value+1 if (1 - binom.cdf(critical_value, total_trials, p=p) > p_value) else critical_value # in case ppf returns a value that leads to a closer but larger p-value
    sig_level = int(critical_value+1)/total_trials
    return sig_level


def eval_mm(corr_match_fold, corr_mismatch_fold, nb_comp_into_account=2):
    # Remove rows where any value is NaN (because the trial length is too long for some folds/subjects)
    idx_not_nan = ~np.isnan(corr_match_fold).any(axis=1)
    corr_match_fold = corr_match_fold[idx_not_nan,:]
    corr_mismatch_fold = corr_mismatch_fold[idx_not_nan,:]
    corr_match_cv = np.mean(corr_match_fold, axis=0)
    corr_mismatch_cv = np.mean(corr_mismatch_fold, axis=0)
    print('Mean corr with match features across trials and folds: ', corr_match_cv)
    print('Mean corr with mismatch features across trials and folds: ', corr_mismatch_cv)
    corr_match_fold = np.array([np.sort(row)[::-1] for row in corr_match_fold])
    corr_mismatch_fold = np.array([np.sort(row)[::-1] for row in corr_mismatch_fold])
    nb_correct = sum(corr_match_fold[:,:nb_comp_into_account].sum(axis=1)>corr_mismatch_fold[:,:nb_comp_into_account].sum(axis=1))
    nb_test = corr_match_fold.shape[0]
    acc = nb_correct/nb_test
    p_value = binomtest(nb_correct, nb_test, alternative='greater').pvalue
    acc_sig = sig_level_binomial_test(0.05, nb_test)
    print('Accuracy: ', acc, ' p-value: ', p_value, 'Number of tests: ', nb_test)
    return acc, p_value, acc_sig


def eval_compete(corr_att_fold, corr_unatt_fold, TRAIN_WITH_ATT, range_into_account=3, nb_comp_into_account=2, message=True):
    # Remove rows where any value is NaN (because the trial length is too long for some folds/subjects)
    idx_not_nan = ~np.isnan(corr_att_fold).any(axis=1)
    corr_att_fold = corr_att_fold[idx_not_nan,:]
    corr_unatt_fold = corr_unatt_fold[idx_not_nan,:]
    nb_test = corr_att_fold.shape[0]
    corr_att_cv = np.mean(corr_att_fold, axis=0)
    corr_unatt_cv = np.mean(corr_unatt_fold, axis=0)
    corr_att_fold = np.array([np.sort(row[:range_into_account])[::-1] for row in corr_att_fold])
    corr_unatt_fold = np.array([np.sort(row[:range_into_account])[::-1] for row in corr_unatt_fold])
    nb_correct = sum(corr_att_fold[:,:nb_comp_into_account].sum(axis=1)>corr_unatt_fold[:,:nb_comp_into_account].sum(axis=1))
    if not TRAIN_WITH_ATT:
        nb_correct = nb_test - nb_correct
    acc = nb_correct/nb_test 
    p_value = binomtest(nb_correct, nb_test, alternative='greater').pvalue
    acc_sig = sig_level_binomial_test(0.05, nb_test)
    if message:
        print('Mean corr with attended features across trials and folds: ', corr_att_cv)
        print('Mean corr with unattended features across trials and folds: ', corr_unatt_cv)
        print('Accuracy: ', acc, ' p-value: ', p_value, 'Number of tests: ', nb_test)
    return acc, p_value, acc_sig, corr_att_cv, corr_unatt_cv


def eval_compete_3D(corr_att_fold, corr_unatt_fold, TRAIN_WITH_ATT, range_into_account=3, nb_comp_into_account=2, message=True):
    acc_list = []
    p_value_list = []
    acc_sig_list = []
    corr_att_cv_list = []
    corr_unatt_cv_list = []
    for i in range(corr_att_fold.shape[2]):
        acc, p_value, acc_sig, corr_att_cv, corr_unatt_cv = eval_compete(corr_att_fold[:,:,i], corr_unatt_fold[:,:,i], TRAIN_WITH_ATT, range_into_account, nb_comp_into_account, message)
        acc_list.append(acc)
        p_value_list.append(p_value)
        acc_sig_list.append(acc_sig)
        corr_att_cv_list.append(corr_att_cv)
        corr_unatt_cv_list.append(corr_unatt_cv)
    acc = np.stack(acc_list, axis=0)
    p_value = np.stack(p_value_list, axis=0)
    acc_sig = np.stack(acc_sig_list, axis=0)
    corr_att_cv = np.stack(corr_att_cv_list, axis=0)
    corr_unatt_cv = np.stack(corr_unatt_cv_list, axis=0)
    return acc, p_value, acc_sig, corr_att_cv, corr_unatt_cv


def W_organize(W, datalist, Llist):
    '''
    Input: 
    W generated by GCCA_multi_modal
    Output:
    Organized W list containing W of each modality 
    '''
    W_list = []
    dim_start = 0
    for i in range(len(datalist)):
        rawdata = datalist[i]
        L = Llist[i]
        if np.ndim(rawdata) == 3:
            _, D, N = rawdata.shape
            dim_end = dim_start + D*L*N
            W_temp = W[dim_start:dim_end,:]
            W_stack = np.reshape(W_temp, (N,D*L,-1))
            W_list.append(np.transpose(W_stack, [1,0,2]))
        elif np.ndim(rawdata) == 2:
            _, D = rawdata.shape
            dim_end = dim_start + D*L
            W_list.append(W[dim_start:dim_end,:])
        else:
            print('Warning: Check the dim of data')
        dim_start = dim_end
    return W_list


def F_organize(F_redun, L, offset, avg=True):
    '''
    Extract the forward model corresponding to the correct time points from the redundant forward model F_redun
    Input: 
    F_redun: DLxNxK or DLxK
    Output:
    Forward model DxNxK or DxK
    '''
    if np.ndim(F_redun) == 3:
        DL, _, _ = F_redun.shape
    else:
        DL, _ = F_redun.shape
    D = int(DL/L)
    indices = [i*L+offset for i in range(D)]
    if np.ndim(F_redun) == 3:
        F = F_redun[indices,:,:]
        if avg:
            F = np.average(F, axis=1)
    else:
        F = F_redun[indices,:]
    return F


def F_avg_temporal(F_redun, L):
    DL, K = F_redun.shape
    D = int(DL/L)
    F_avg = np.zeros((D, K))
    F_blocks = np.split(F_redun, D, axis=0)
    for i in range(D):
        F_avg[i,:] = np.mean(F_blocks[i], axis=0)
    return F_avg


def forward_model(X, W_Hankel, L=1, offset=0):
    '''
    Reference: On the interpretation of weight vectors of linear models in multivariate neuroimaging https://www.sciencedirect.com/science/article/pii/S1053811913010914
    Backward models: Extract latent factors as functions of the observed data s(t) = W^T x(t)
    Forward models: Reconstruct observations from latent factors x(t) = As(t) + n(t)
    x(t): D-dimensional observations
    s(t): K-dimensional latent factors
    W: backward model
    A: forward model

    In our use case the backward model can be found using (G)CCA. Latent factors are the representations generated by different components.
    X_Hankel W_Hankel = S     X:TxDL W:DLxK S:TxK
    S F.T = X                 F: DxK
    F = X.T X_Hankel W_Hankel inv(W_Hankel.T X_Hankel.T X_Hankel W_Hankel)

    Inputs:
    X: observations (one subject) TxD
    W_Hankel: filters/backward models DLxK
    L: time lag (if temporal-spatial)

    Output:
    F: forward model
    '''
    if L == 1:
        Rxx = np.cov(X, rowvar=False)
        F = Rxx@W_Hankel@LA.inv(W_Hankel.T@Rxx@W_Hankel)
    else:
        X_block_Hankel = block_Hankel(X, L, offset)
        F = X.T@X_block_Hankel@W_Hankel@LA.inv(W_Hankel.T@X_block_Hankel.T@X_block_Hankel@W_Hankel)
    return F


def phase_scramble_2D(data):
    # Initialize an array to hold the scrambled data
    scrambled_data = np.zeros_like(data, dtype=complex)
    T, D = data.shape
    # Generate random phase shifts for half of the spectrum
    half_T = T // 2 if T % 2 == 0 else (T + 1) // 2
    random_phase_half = np.exp(1j * np.random.uniform(0, 2*np.pi, size=half_T))
    # Ensure conjugate symmetry
    random_phase_full = np.concatenate(([1], random_phase_half[1:half_T], [1], np.conj(random_phase_half[1:half_T][::-1]))) if T%2==0 else np.concatenate(([1], random_phase_half[1:half_T], np.conj(random_phase_half[1:half_T][::-1])))
    # Loop over channels
    for i in range(D):  # Assuming data.shape = (time, channel)
        # Perform FFT on each channel independently
        fft_result = np.fft.fft(data[:, i])
        amplitude = np.abs(fft_result)
        # Apply the random phase shifts
        scrambled_fft = amplitude * random_phase_full
        scrambled_data[:, i] = np.fft.ifft(scrambled_fft)
    return scrambled_data.real


def phase_scramble_3D(data):
    _, _, N = data.shape
    scrambled_data = np.zeros_like(data)
    for n in range(N):
        scrambled_data[:,:,n] = phase_scramble_2D(data[:,:,n])
    return scrambled_data


def shuffle_block(X, block_len):
    '''
    Shuffle the blocks of X along the time axis for each subject.
    '''
    T, D, N = X.shape
    if T%block_len != 0:
        append_arr = np.zeros((block_len-T%block_len, D, N))
        X = np.concatenate((X, append_arr), axis=0)
    T_appended = X.shape[0]
    X_shuffled = np.zeros_like(X)
    for n in range(N):
        blocks = [X[i:i+block_len, :, n] for i in range(0, T_appended, block_len)]
        random.shuffle(blocks)
        X_shuffled[:,:,n] = np.concatenate(tuple(blocks), axis=0)
    return X_shuffled


def shuffle_2D(X, block_len):
    T, D = X.shape
    if T%block_len != 0:
        append_arr = np.zeros((block_len-T%block_len, D))
        X = np.concatenate((X, append_arr), axis=0)
        T, _ = X.shape
    X_block = X.reshape((T//block_len, block_len, D))
    X_shuffle_block = np.random.permutation(X_block)
    X_shuffle = X_shuffle_block.reshape((T, D))
    return X_shuffle


def shuffle_3D(X, block_len):
    '''
    Same as shuffle_block(X, block_len)
    '''
    T, D, N = X.shape
    if T%block_len != 0:
        append_arr = np.zeros((block_len-T%block_len, D, N))
        X = np.concatenate((X, append_arr), axis=0)
    X_shuffled = np.zeros_like(X)
    for n in range(N):
        X_shuffled[:,:,n] = shuffle_2D(X[:,:,n], block_len)
    return X_shuffled


def shuffle_datalist(datalist, block_len):
    '''
    Shuffle the blocks of X along the time axis for each subject.
    '''
    datalist_shuffled = []
    for data in datalist:
        if np.ndim(data) == 2:
            datalist_shuffled.append(shuffle_2D(data, block_len))
        elif np.ndim(data) == 3:
            datalist_shuffled.append(shuffle_3D(data, block_len))
    return datalist_shuffled


def circular_permute_3D(X):
    '''
    Circularly permute X along the time axis for each subject.
    '''
    T, _, N = X.shape
    X_permuted = np.zeros_like(X)
    for n in range(N):
        shift = random.randint(0, T // 300) * 300  # Shift by a random multiple of 300 samples (10 seconds) 
        X_permuted[:,:,n] = np.roll(X[:,:,n], shift, axis=0)
    return X_permuted


def EEG_normalization(data, len_seg):
    '''
    Normalize the EEG data.
    Subtract data of each channel by the mean of it
    Divide data into several segments, and for each segment, divide the data matrix by its Frobenius norm.
    Inputs:
    data: EEG data D x T
    len_seg: length of the segments
    Output:
    normalized_data
    '''
    _, T = data.shape
    n_blocks = T // len_seg + 1
    data_blocks = np.array_split(data, n_blocks, axis=1)
    data_zeromean = [db - np.mean(db, axis=1, keepdims=True) for db in data_blocks]
    normalized_blocks = [db/LA.norm(db) for db in data_zeromean]
    normalized_data = np.concatenate(tuple(normalized_blocks), axis=1)
    return normalized_data


def extract_highfreq(EEG, resamp_freqs, band=[15,20], ch_eog=None, regression=False, normalize=True):
    '''
    EEG signals -> band-pass filter -> high-frequency signals -> Hilbert transform -> signal envelope -> low-pass filter -> down-sampled envelope -> noramalized envelope
    Inputs:
    EEG: EEG signals with original sampling rate
    resamp_freqs: resampling frequency
    band: the frequency band to be kept
    Outputs:
    envelope: the envelope of high-frequency signals
    '''
    # EOG channels are marked as 'eeg' now
    # Filter both eeg and eog channels with a band-pass filter
    EEG_band = EEG.filter(l_freq=band[0], h_freq=band[1], picks=['eeg'])
    # Extract the envelope of signals
    envelope = EEG_band.copy().apply_hilbert(picks=['eeg'], envelope=True)
    # Mark EOG channels as 'eog'
    if ch_eog is not None:
        type_true = ['eog']*len(ch_eog)
        change_type_dict = dict(zip(ch_eog, type_true))
        envelope.set_channel_types(change_type_dict)
        # Regress out the filtered EOG signals before extracting the envelope of high-frequency signals
        if regression:
            EOGweights = mne.preprocessing.EOGRegression(picks='eeg', proj=False).fit(envelope)
            envelope = EOGweights.apply(envelope, copy=False)
    envelope = envelope.resample(sfreq=resamp_freqs)
    if normalize:
        eeg_channel_indices = mne.pick_types(envelope.info, eeg=True)
        eegdata, _ = envelope[eeg_channel_indices]
        envelope._data[eeg_channel_indices, :] = EEG_normalization(eegdata, resamp_freqs*60)
    return envelope


def preprocessing(file_path, HP_cutoff = 0.5, AC_freqs=50, band=None, resamp_freqs=None, bads=[], eog=True, regression=False, normalize=False):
    '''
    Preprocessing of the raw signal
    Re-reference -> Highpass filter (-> downsample)
    No artifact removal technique has been applied yet
    Inputs:
    file_path: location of the eeg dataset
    HP_cutoff: cut off frequency of the high pass filter (for removing DC components and slow drifts)
    AC_freqs: AC power line frequency
    resamp_freqs: resampling frequency (if None then resampling is not needed)
    bads: list of bad channels
    eog: if contains 4 eog channels
    regression: whether regresses eog out
    Output:
    preprocessed: preprocessed eeg
    fs: the sample frequency of the EEG signal (original or down sampled)
    '''
    raw_lab = mne.io.read_raw_eeglab(file_path, preload=True)
    raw_lab.info['bads'] = bads
    fsEEG = raw_lab.info['sfreq']
    # Rename channels and set montages
    biosemi_layout = mne.channels.read_layout('biosemi')
    ch_names_map = dict(zip(raw_lab.info['ch_names'], biosemi_layout.names))
    raw_lab.rename_channels(ch_names_map)
    montage = mne.channels.make_standard_montage('biosemi64')
    raw_lab.set_montage(montage)
    if len(bads)>0:
        # Interpolate bad channels
        raw_lab.interpolate_bads()
    # Re-reference
    # raw_lab.set_eeg_reference(ref_channels=['Cz']) # Select the reference channel to be Cz
    raw_lab.set_eeg_reference(ref_channels='average')
    # If there are EOG channels, first treat them as EEG channels and do re-referencing, filtering and resampling.
    if eog:
        misc_names = [raw_lab.info.ch_names[i] for i in mne.pick_types(raw_lab.info, misc=True)]
        # eog_data, _ = raw_lab[misc_names]
        # eog_channel_indices = mne.pick_channels(raw_lab.info['ch_names'], include=misc_names)
        type_eeg = ['eeg']*len(misc_names)
        change_type_dict = dict(zip(misc_names, type_eeg))
        raw_lab.set_channel_types(change_type_dict)
        # Take the average of four EOG channels as the reference
        # raw_lab._data[eog_channel_indices, :] = eog_data - np.average(eog_data, axis=0)
    else:
        misc_names = None
    # Highpass filter - remove DC components and slow drifts
    raw_highpass = raw_lab.copy().filter(l_freq=HP_cutoff, h_freq=None)
    # raw_highpass.compute_psd().plot(average=True)
    # Remove power line noise
    raw_notch = raw_highpass.copy().notch_filter(freqs=AC_freqs)
    # raw_notch.compute_psd().plot(average=True)
    # Resampling:
    # Anti-aliasing has been implemented in mne.io.Raw.resample before decimation
    # https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.resample
    if resamp_freqs is not None:
        if band is not None:
            highfreq = extract_highfreq(raw_notch.copy(), resamp_freqs, band, misc_names, regression, normalize)
        else:
            highfreq = None
        raw_downsampled = raw_notch.copy().resample(sfreq=resamp_freqs)
        # raw_downsampled.compute_psd().plot(average=True)
        preprocessed = raw_downsampled
        fs = resamp_freqs
    else:
        highfreq = None
        preprocessed = raw_notch
        fs = fsEEG
    # Then set EOG channels to their true type
    if eog:
        type_true = ['eog']*len(misc_names)
        change_type_dict = dict(zip(misc_names, type_true))
        preprocessed.set_channel_types(change_type_dict)
    if regression:
        EOGweights = mne.preprocessing.EOGRegression(picks='eeg', proj=False).fit(preprocessed)
        preprocessed = EOGweights.apply(preprocessed, copy=False)
    if normalize:
        eeg_channel_indices = mne.pick_types(preprocessed.info, eeg=True)
        eegdata, _ = preprocessed[eeg_channel_indices]
        preprocessed._data[eeg_channel_indices, :] = EEG_normalization(eegdata, fs*60)
    return preprocessed, fs, highfreq


def clean_features(feats, smooth=True):
    y = copy.deepcopy(feats)
    T, nb_feature = y.shape
    for i in range(nb_feature):
        # interpolate NaN values (linearly)
        nans, x= np.isnan(y[:,i]), lambda z: z.nonzero()[0]
        if any(nans):
            f1 = scipy.interpolate.interp1d(x(~nans), y[:,i][~nans], fill_value='extrapolate')
            y[:,i][nans] = f1(x(nans))
        if smooth:
            # extract envelope by finding peaks and interpolating peaks with spline
            idx_peaks = scipy.signal.find_peaks(y[:,i])[0]
            idx_rest = np.setdiff1d(np.array(range(T)), idx_peaks)
            # consider use quadratic instead
            f2 = scipy.interpolate.interp1d(idx_peaks, y[:,i][idx_peaks], kind='cubic', fill_value='extrapolate')
            y[:,i][idx_rest] = f2(idx_rest)
    return y


def plot_spatial_resp(forward_model, file_name, fig_size=(10, 4)):
    _, n_components = forward_model.shape
    forward_model = np.abs(forward_model)
    biosemi_layout = mne.channels.read_layout('biosemi')
    create_info = mne.create_info(biosemi_layout.names, ch_types='eeg', sfreq=30)
    create_info.set_montage('biosemi64')
    vmax = np.max(forward_model)
    vmin = np.min(forward_model)
    if n_components < 5:
        n_row = 1
        n_column = n_components
    else:
        if n_components % 5 != 0:
            n_row = n_components//5 + 1
        else:
            n_row = n_components//5
        n_column = 5
    fig, axes = plt.subplots(nrows=n_row, ncols=n_column, figsize=fig_size)
    fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    comp = 0
    if n_components == 1:
        axes = np.array([axes])
    for ax in axes.flat:
        if comp < n_components:
            im, _ = mne.viz.plot_topomap(forward_model[:,comp], create_info, ch_type='eeg', axes=ax, show=False, vlim=(vmin, vmax))
            # if idx_sig is not None:
            #     color = 'b' if comp in idx_sig else 'black'
            # else:
            #     color = 'black'
            # if ifISC:
            #     ax.set_title("CC: {order}\n ISC: {corr:.3f}".format(order=comp+1, corr=np.mean(corr[:,comp])), color=color)
            # else:
            #     ax.set_title("CC: {order}\n corr: {corr:.3f}".format(order=comp+1, corr=np.mean(corr[:,comp])), color=color)
        else:
            ax.axis('off')
        comp += 1
    cbar_ax = fig.add_axes([0.85, 0.3, 0.02, 0.5])
    fig.colorbar(im, cax=cbar_ax, label='Weight')
    plt.savefig(file_name, dpi=600)
    plt.close()


def plot_three_tasks(task_models_dict, file_name, task_names=["Task 1", "Task 2", "Task 3"], fig_size=(12, 4)):
    """
    Plots 3 spatial topographies side-by-side sharing a single colorbar.
    
    Inputs:
    task_models_dict: A dictionary of 1D arrays, e.g., {0: array, 1: array, 2: array} 
                      (Output from the grand_average function)
    """
    # 1. Stack the dictionary arrays into a single 2D array
    forward_models = np.column_stack([task_models_dict[0], task_models_dict[1], task_models_dict[2]])
    
    # Force the maximum absolute peak of every task to be positive (red)
    for i in range(forward_models.shape[1]):
        task_map = forward_models[:, i]
        
        # Find the index of the largest weight (ignoring sign)
        peak_idx = np.argmax(np.abs(task_map))
        
        # If the actual value at that peak is negative, flip the whole map
        if task_map[peak_idx] < 0:
            forward_models[:, i] = -task_map
    
    # 2. Setup MNE Info (Biosemi 64)
    biosemi_layout = mne.channels.read_layout('biosemi')
    create_info = mne.create_info(biosemi_layout.names, ch_types='eeg', sfreq=30)
    create_info.set_montage('biosemi64')
    
    # 3. Calculate a symmetric global color limit for all subplots
    # This ensures that 0 is perfectly centered in your colormap
    max_val = np.max(np.abs(forward_models))
    vmin, vmax = -max_val, max_val
    
    # 4. Create the 1x3 Figure
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=fig_size)
    
    # 5. Loop through and plot each task
    for idx, ax in enumerate(axes):
        im, _ = mne.viz.plot_topomap(
            forward_models[:, idx], 
            create_info, 
            ch_type='eeg', 
            axes=ax, 
            show=False, 
            vlim=(vmin, vmax),
            cmap='RdBu_r' # Standard divergent colormap for EEG
        )
        ax.set_title(task_names[idx], fontsize=20, pad=10)
        
    # 6. Add a single, shared colorbar on the right
    plt.subplots_adjust(left=0.05, right=0.85) # Compress plots to the left
    cbar_ax = fig.add_axes([0.88, 0.15, 0.035, 0.7]) # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Weight (a.u.)', fontsize=20, labelpad=10)
    
    # Make the tick numbers larger
    cbar.ax.tick_params(labelsize=18)
    
    # 7. Save and close
    plt.savefig(file_name, dpi=600, bbox_inches='tight')
    plt.close()


def plot_spatial_resp_fold(forward_model_fold, corr_att_fold, corr_unatt_fold, sig_corr_fold, file_name, AVG=False, ISC=False):
    n_components = forward_model_fold[0].shape[1]
    Prefix = "ISC" if ISC else "Corr"
    if AVG:
        corr_att_fold = np.mean(corr_att_fold, axis=0, keepdims=1)
        if corr_unatt_fold is not None:
            corr_unatt_fold = np.mean(corr_unatt_fold, axis=0, keepdims=1)
        sig_corr_fold = [np.mean(sig_corr_fold)] if sig_corr_fold is not None else None
        # forward_model_fold = [np.abs(fm) for fm in forward_model_fold]
        forward_model_fold = [np.mean(forward_model_fold, axis=0)]
        file_name = file_name.replace('Folds', 'Avg')
    biosemi_layout = mne.channels.read_layout('biosemi')
    create_info = mne.create_info(biosemi_layout.names, ch_types='eeg', sfreq=30)
    create_info.set_montage('biosemi64')
    vmax = np.max([np.max(np.abs(fm)) for fm in forward_model_fold])
    vmin = np.min([np.min(np.abs(fm)) for fm in forward_model_fold])
    n_rows = len(forward_model_fold)
    n_cols = n_components
    fig_size = (n_cols*2, n_rows*2.5)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=fig_size)
    fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    # plot each component
    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i, j] if not AVG else axes[j]
            im, _ = mne.viz.plot_topomap(np.abs(forward_model_fold[i][:,j]), create_info, ch_type='eeg', axes=ax, show=False, vlim=(vmin, vmax))
            if corr_unatt_fold is not None:
                ax.set_title("Att: {corr_att:.3f}\n UnAtt: {corr_unatt:.3f}".format(corr_att=corr_att_fold[i,j], corr_unatt=corr_unatt_fold[i,j]), color='black')
            else:
                ax.set_title("{pref}: {corr:.3f}".format(pref=Prefix, corr=corr_att_fold[i,j]), color='black') 
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0 and sig_corr_fold is not None:  # add row title to the first subplot in each row
                ax.set_ylabel("Sig: {sig:.3f}".format(sig=sig_corr_fold[i]), fontsize=12)
    cbar_ax = fig.add_axes([0.85, 0.3, 0.02, 0.5])
    fig.colorbar(im, cax=cbar_ax, label='Weight')
    plt.savefig(file_name, dpi=600)
    plt.close()


def plot_accuracy_percentage(corr_att, corr_unatt, sim, figure_dir, Subj_ID, nb_comp_into_account=2, name_prepend='Test_'):
    # Remove rows where any value is NaN (because the trial length is too long for some folds/subjects)
    idx_not_nan = ~np.isnan(corr_att).any(axis=1)
    corr_att = corr_att[idx_not_nan,:]
    corr_unatt = corr_unatt[idx_not_nan,:]
    corr_att = np.array([np.sort(row)[::-1] for row in corr_att])
    corr_unatt = np.array([np.sort(row)[::-1] for row in corr_unatt])
    stat_att = np.sum(corr_att[:,:nb_comp_into_account], axis=1)
    stat_unatt = np.sum(corr_unatt[:,:nb_comp_into_account], axis=1)
    label = stat_att > stat_unatt
    rank_sim = np.argsort(sim)
    label = label[rank_sim]
    sim = sim[rank_sim]
    # Calculate cumulative sum of the label
    cum_label = np.sum(label) - np.cumsum(label)
    acc_step = cum_label[:-1] / (len(label) - np.arange(1, len(label)))
    
    plt.close('all')
    # Create two subplots with a shared x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    # Plot the violin plot of sim on the first subplot
    sns.violinplot(x=sim, hue=label, ax=ax1, fill=False, split=True)
    ax1.set_xlabel('Similarity')
    ax1.tick_params('y', colors='r')
    ax1.set_title('Similarity distributions when decoding correctly and incorrectly')
    # Plot the accuracy on the second subplot
    ax2.plot(sim[:-1], acc_step)
    ax2.set_ylabel('Acc')
    ax2.tick_params('y')
    ax2.set_xlabel('Similarity Threshold')
    ax2.set_title('Accuracy when considering the trials with similarity above a threshold')

    figure_name = figure_dir + f'{name_prepend}Acc_Percentage_Subj_{Subj_ID+1}.png'
    plt.savefig(figure_name)


def create_dir(path, CLEAR=False):
    if not os.path.exists(path):
        os.makedirs(path)
    if CLEAR:
        for file in os.listdir(path):
            os.remove(path + file)


def get_features(feats_path_folder, video_id, len_seg, smooth=True):
    with open(feats_path_folder + video_id + '_mask.pkl', 'rb') as f:
        feats = pickle.load(f)
    feats = np.concatenate(tuple(feats), axis=0)
    feats = clean_features(feats, smooth=smooth)
    feats = feats[:len_seg, ...]
    return feats


def get_gaze(gaze_path, len_seg, expdim=True):
    gaze = np.load(gaze_path, allow_pickle=True)
    # interpolate missing values
    gaze = np.array([np.nan if x is None else x for x in gaze])
    gaze_clean = clean_features(gaze.astype(np.float64), smooth=False)
    gaze_clean = gaze_clean[:len_seg, :]
    if expdim:
        gaze_clean = np.expand_dims(gaze_clean, axis=2)
    return gaze_clean


def get_eeg_eog(eeg_path, fsStim, bads, expdim=True, cut_off_sec=None):
    eeg_prepro, fs, _ = preprocessing(eeg_path, HP_cutoff = 0.5, AC_freqs=50, band=None, resamp_freqs=fsStim, bads=bads, eog=True, regression=False, normalize=True)
    eeg_channel_indices = mne.pick_types(eeg_prepro.info, eeg=True)
    eog_channel_indices = mne.pick_types(eeg_prepro.info, eog=True)
    eeg_downsampled, _ = eeg_prepro[eeg_channel_indices]
    eog_downsampled, _ = eeg_prepro[eog_channel_indices]
    if expdim:
        eeg_downsampled = np.expand_dims(eeg_downsampled.T, axis=2)
        eog_downsampled = np.expand_dims(eog_downsampled.T, axis=2)
    else:
        eeg_downsampled = eeg_downsampled.T
        eog_downsampled = eog_downsampled.T
    if cut_off_sec is not None:
        eeg_downsampled = eeg_downsampled[:cut_off_sec*fs, ...]
        eog_downsampled = eog_downsampled[:cut_off_sec*fs, ...]
    return eeg_downsampled, eog_downsampled, fs


def data_per_subj(eeg_folder, fsStim, bads, len_video, feats_path_folder=None, expdim=True, Task='1'):
    eeg_files_all = [file for file in os.listdir(eeg_folder) if file.endswith('.set')]
    files = [file for file in eeg_files_all if file[-5] == Task]
    files.sort()
    eeg_list = []
    eog_list = []
    gaze_list = []
    feat_list = [] if feats_path_folder is not None else None
    for file in files:
        eeg_downsampled, eog_downsampled, fs = get_eeg_eog(eeg_folder + file, fsStim, bads, expdim, cut_off_sec=len_video)
        eeg_list.append(eeg_downsampled)
        eog_list.append(eog_downsampled)
        gaze_file = file[:-4] + '_gaze.npy'
        # check if file exists
        if not os.path.exists(eeg_folder + gaze_file):
            print(f"Gaze file {gaze_file} not found in {eeg_folder}. Adding all zeros.")
            gaze = np.zeros_like(eog_downsampled)
        else:
            gaze = get_gaze(eeg_folder + gaze_file, len_video*fs)
        gaze_list.append(gaze)
        if feats_path_folder is not None:
            feats = get_features(feats_path_folder, file[:2], len_video*fs, smooth=True)
            feat_list.append(feats)
    return eeg_list, eog_list, feat_list, gaze_list, fs


def data_multi_subj(subj_path, fsStim, bads, feats_path_folder, len_video, SAVE=True, Task='1'):
    data_path = 'data/'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    nb_subj = len(subj_path)
    eeg_multisubj_list, eog_multisubj_list, feat_list, gaze_multisubj_list, fs = data_per_subj(subj_path[0], fsStim, bads[0], len_video, feats_path_folder, Task=Task)
    for n in range(1,nb_subj):
        eeg_list, eog_list, _, gaze_list, _ = data_per_subj(subj_path[n], fsStim, bads[n], len_video, Task=Task)
        eeg_multisubj_list = [np.concatenate((eeg_multi, eeg), axis=2) for eeg_multi, eeg in zip(eeg_multisubj_list, eeg_list)]
        eog_multisubj_list = [np.concatenate((eog_multi, eog), axis=2) for eog_multi, eog in zip(eog_multisubj_list, eog_list)]
        gaze_multisubj_list = [np.concatenate((gaze_multi, gaze), axis=2) for gaze_multi, gaze in zip(gaze_multisubj_list, gaze_list)]
    if SAVE:
        # save all data (eeg_multisubj_list, eog_multisubj_list, feat_att_list, feat_unatt_list, fs, nb_files) into a single file
        data = {'eeg_multisubj_list': eeg_multisubj_list, 'eog_multisubj_list': eog_multisubj_list, 'feat_list': feat_list, 'gaze_multisubj_list': gaze_multisubj_list, 'fs': fs}
        file_name = 'data_task_' + Task + '.pkl'
        with open(data_path + file_name, 'wb') as f:
            pickle.dump(data, f)
    return eeg_multisubj_list, eog_multisubj_list, feat_list, gaze_multisubj_list, fs


def add_new_data(subj_path, fsStim, bads, feats_path_folder, len_video, Task='1'):
    data_path = 'data/'
    file_name = 'data_task_' + Task + '.pkl'
    with open(data_path + file_name, 'rb') as f:
        data = pickle.load(f)
    nb_subj_old = data['eeg_multisubj_list'][0].shape[2]
    eeg_multisubj_add, eog_multisubj_add, _, gaze_multisubj_add, _ = data_multi_subj(subj_path[nb_subj_old:], fsStim, bads[nb_subj_old:], feats_path_folder, len_video, SAVE=False, Task=Task)
    eeg_multisubj_list = [np.concatenate((old, new), axis=2) for old, new in zip(data['eeg_multisubj_list'], eeg_multisubj_add)]
    eog_multisubj_list = [np.concatenate((old, new), axis=2) for old, new in zip(data['eog_multisubj_list'], eog_multisubj_add)]
    gaze_multisubj_list = [np.concatenate((old, new), axis=2) for old, new in zip(data['gaze_multisubj_list'], gaze_multisubj_add)]
    data['eeg_multisubj_list'] = eeg_multisubj_list
    data['eog_multisubj_list'] = eog_multisubj_list
    data['gaze_multisubj_list'] = gaze_multisubj_list
    with open(data_path + file_name, 'wb') as f:
        pickle.dump(data, f)
    return eeg_multisubj_list, eog_multisubj_list, data['feat_list'], gaze_multisubj_list, data['fs']


def load_subj(Subj_ID):
    eeg_list = []
    eog_list = []
    gaze_list = []
    feats_list = []
    data_path = 'data/data_task_1.pkl'
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
        np.expand_dims(data['eeg_multisubj_list'], axis=0)
    eeg_list = [np.expand_dims(d[:,:,Subj_ID], axis=2) for d in data['eeg_multisubj_list']]
    eog_list = [np.expand_dims(d[:,:,Subj_ID], axis=2) for d in data['eog_multisubj_list']]
    gaze_list = [np.expand_dims(d[:,:,Subj_ID], axis=2) for d in data['gaze_multisubj_list']]
    feats_list = [np.expand_dims(d, axis=2) for d in data['feat_list']]
    for TASK in ['2', '3']:
        data_path = 'data/data_task_' + TASK + '.pkl'
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        eeg_list = [np.concatenate((d_MT, np.expand_dims(d[:,:,Subj_ID], axis=2)), axis=2) for d_MT, d in zip(eeg_list, data['eeg_multisubj_list'])]
        eog_list = [np.concatenate((d_MT, np.expand_dims(d[:,:,Subj_ID], axis=2)), axis=2) for d_MT, d in zip(eog_list, data['eog_multisubj_list'])]
        gaze_list = [np.concatenate((d_MT, np.expand_dims(d[:,:,Subj_ID], axis=2)), axis=2) for d_MT, d in zip(gaze_list, data['gaze_multisubj_list'])]
        feats_list = [np.concatenate((d_MT, np.expand_dims(d, axis=2)), axis=2) for d_MT, d in zip(feats_list, data['feat_list'])]
    return eeg_list, eog_list, gaze_list, feats_list


def remove_shot_cuts_and_center(data, fs, time_points=None, remove_time=1, CENTER=True):
    T = data.shape[0]
    if time_points is None:
        time_points = [0, T]
    nearby_idx = []
    for p in time_points:
        len_points = int(remove_time*fs)
        nearby_idx = nearby_idx + list(range(max(0, p-len_points), min(p+len_points, T)))
    nearby_idx = list(set(nearby_idx))
    data_clean = np.delete(data, nearby_idx, axis=0)
    if CENTER:
        data_clean = data_clean - np.mean(data_clean, axis=0)
    return data_clean


def load_data(subj_path, fsStim, bads, feats_path_folder, LOAD_ONLY, ALL_NEW):
    file_name = 'data_twoobj.pkl'
    if LOAD_ONLY:
        data_path = 'data/'
        with open(data_path + file_name, 'rb') as f:
            data = pickle.load(f)
        eeg_multisubj_list = data['eeg_multisubj_list']
        eog_multisubj_list = data['eog_multisubj_list']
        feat_att_list = data['feat_att_list']
        feat_unatt_list = data['feat_unatt_list']
        gaze_multisubj_list = data['gaze_multisubj_list']
        fs = data['fs']
        len_seg_list = data['len_seg_list']
    else:
        if ALL_NEW:
            eeg_multisubj_list, eog_multisubj_list, feat_att_list, feat_unatt_list, gaze_multisubj_list, fs, len_seg_list = data_multi_subj(subj_path, fsStim, bads, feats_path_folder)
        else:
            eeg_multisubj_list, eog_multisubj_list, feat_att_list, feat_unatt_list, gaze_multisubj_list, fs, len_seg_list = add_new_data(subj_path, fsStim, bads, feats_path_folder)
    return eeg_multisubj_list, eog_multisubj_list, feat_att_list, feat_unatt_list, gaze_multisubj_list, fs, len_seg_list


# Check the alignment between eog and gaze. The synchronization is good if the peaks of two signals (eye blinks) are aligned.
def check_alignment(subj_ID, task_ID, eog_multitask_list, gaze_multitask_list, nb_points=500):
    eog_one_subj_list = [eog[:,:,task_ID-1] for eog in eog_multitask_list]
    gaze_one_subj_list = [gaze[:,:,task_ID-1] for gaze in gaze_multitask_list]
    eog_verti_list = [eog[:,0] - eog[:,1] for eog in eog_one_subj_list]
    gaze_y_list = [gaze[:,1] for gaze in gaze_one_subj_list]
    nb_videos = len(eog_verti_list)
    # make a subplot (3 x (nb_videos//3+1)) for each video
    # draw and save the plot
    nb_rows = 3
    nb_cols = nb_videos//3+1
    fig, ax = plt.subplots(nb_rows, nb_cols, figsize=(15, 10), sharey=True)
    for i in range(nb_videos):
        ax[i//nb_cols, i%nb_cols].plot(eog_verti_list[i][-nb_points:]/np.max(eog_verti_list[i][-nb_points:]), label='eog vertical')
        ax[i//nb_cols, i%nb_cols].plot(gaze_y_list[i][-nb_points:]/np.max(gaze_y_list[i][-nb_points:]), label='gaze y')
        ax[i//nb_cols, i%nb_cols].set_title(f"Video {i+1}, gaze y max {np.max(gaze_y_list[i][-nb_points:])}")
        ax[i//nb_cols, i%nb_cols].legend()
    plt.savefig(f"figures/alignment/Subj_{subj_ID}_Task_{task_ID}.png")


def calcu_gaze_velocity(gaze):
    if np.ndim(gaze) == 2:
        gaze = np.expand_dims(gaze, axis=2)
    _, D, _ = gaze.shape
    if D > 2:
        gaze = gaze[:,:2,:]
    pos_diff = np.diff(gaze, axis=0, prepend=np.expand_dims(gaze[0,:], axis=0))
    gaze_velocity = np.sqrt(np.sum(pos_diff**2, axis=1, keepdims=True))
    return gaze_velocity


def calcu_gaze_vel_from_EOG(eog):
    if np.ndim(eog) == 2:
        eog = np.expand_dims(eog, axis=2)
    eog_y = eog[:,0,:] - eog[:,1,:]
    eog_x = eog[:,2,:] - eog[:,3,:]
    eog_xy = np.stack((eog_x, eog_y), axis=1)
    gaze_velocity = calcu_gaze_velocity(eog_xy)
    return gaze_velocity
    

def refine_saccades(saccade_multisubj_list, blink_multisubj_list):
    saccade_multisubj_list = [saccade_multisubj.astype(bool) for saccade_multisubj in saccade_multisubj_list]
    blink_multisubj_list = [blink_multisubj.astype(bool) for blink_multisubj in blink_multisubj_list]
    saccade_multisubj_list = [np.logical_xor(np.logical_and(saccade_multisubj, blink_multisubj), saccade_multisubj) for saccade_multisubj, blink_multisubj in zip(saccade_multisubj_list, blink_multisubj_list)]
    saccade_multisubj_list = [saccade.astype(float) for saccade in saccade_multisubj_list]
    return saccade_multisubj_list


def remove_saccade(datalist, Sacc, remove_before=15, remove_after=30):
    T = Sacc.shape[0]
    # transform the saccade into a binary mask
    Sacc = Sacc > 0.5
    # find the indices of the points around the saccade
    idx_remove = np.where(Sacc)[0]
    idx_remove = np.concatenate([np.arange(i-remove_before, i+remove_after+1) for i in idx_remove])
    # remove the repeated indices and the indices out of the range
    idx_remove = np.unique(idx_remove)
    idx_remove = idx_remove[(idx_remove>=0) & (idx_remove<T)]
    # remove the time points around the saccade
    datalist = [np.delete(data, idx_remove, axis=0) for data in datalist]
    return datalist


def interpolate_blinks(ts, blinks):
    '''
    Interpolate the time series around the blinks indicated by the binary mask
    '''
    time_series = copy.deepcopy(ts)
    # If the time series is 3D, replicate the blinks along the second dimension
    if time_series.shape[1] != blinks.shape[1]:
        blinks = np.tile(blinks, (1, time_series.shape[1], 1))
    # Transform the blinks into a binary mask
    blinks = blinks > 0.5
    # Set the indices of the blinks to NaN
    time_series[blinks] = np.nan
    if np.ndim(time_series) == 2:
        time_series = clean_features(time_series, smooth=False)
    else:
        for i in range(time_series.shape[2]):
            time_series[:,:,i] = clean_features(time_series[:,:,i], smooth=False)
    return time_series
    

def pixels_to_dva(radius_in_pixels, distance_to_screen_cm=130, screen_width_cm=48.7, screen_width_px=1920):
    """
    Converts a radius in pixels to Degrees of Visual Angle (DVA).
    """
    # 1. Calculate how many centimeters one pixel represents
    cm_per_pixel = screen_width_cm / screen_width_px
    # 2. Convert the pixel radius into a physical radius (cm)
    radius_cm = radius_in_pixels * cm_per_pixel
    # 3. Use trigonometry to find the angle (in radians), then convert to degrees
    # math.atan2 is safe and returns radians
    dva_radians = math.atan2(radius_cm, distance_to_screen_cm)
    dva_degrees = math.degrees(dva_radians)
    return dva_degrees


def fixation_cluster(xy, eps=10, min_samples=5):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(xy)
    labels = clustering.labels_
    most_common_label = Counter(labels).most_common(1)[0][0]
    assert most_common_label != -1, "Most common label is -1, indicating noise."
    fixation_points = xy[labels == most_common_label]
    centroid = np.mean(fixation_points, axis=0)
    distances = np.linalg.norm(fixation_points - centroid, axis=1)
    radius_px = np.percentile(distances, 95)
    dva = pixels_to_dva(radius_px)
    return centroid, dva, labels, most_common_label


def get_mask_from_gaze(xy_multitask, saccade_multitask, blink_multitask, eps=10, nb_nearby_samples=None):
    nb_tasks = xy_multitask.shape[2]
    masks = []
    for i in range(nb_tasks):
        xy = xy_multitask[:,:,i]
        saccade = saccade_multitask[:,0,i].astype(bool)
        blink = blink_multitask[:,0,i].astype(bool)
        _, _, labels, most_common_label = fixation_cluster(xy, eps=eps)
        mask_to_discard = labels != most_common_label
        mask_to_discard = mask_to_discard & ~blink | saccade   
        if nb_nearby_samples is not None:
            original_mask = mask_to_discard.copy()
            for bf in range(1, nb_nearby_samples[0] + 1):
                mask_to_discard[:-bf] |= original_mask[bf:]
            for af in range(1, nb_nearby_samples[1] + 1):
                mask_to_discard[af:] |= original_mask[:-af]
        masks.append(np.expand_dims(mask_to_discard, axis=1))
    return np.stack(masks, axis=2)


def create_corr_df(Subj_ID, sig_corr_pool, corr_att_fold, corr_unatt_fold):
    corr_att = np.average(corr_att_fold, axis=0)
    corr_unatt = np.average(corr_unatt_fold, axis=0)
    n_component = len(corr_att)
    index = pd.MultiIndex.from_tuples([
        ('Subj '+str(Subj_ID+1), sig_corr_pool, 'CC {}'.format(i+1))
        for i in range(n_component)
    ], names=['Subject ID', 'Sig Level', 'Component'])
    data = {
        'Att': corr_att,
        'Unatt': corr_unatt
    }
    corr_df = pd.DataFrame(data, index=index)
    return corr_df

def save_corr_df(table_name, sig_corr_pool, corr_att_fold, corr_unatt_fold, Subj_ID, OVERWRITE=False):
    # check if the file exists
    if not os.path.isfile(table_name):
        res_df = create_corr_df(Subj_ID, sig_corr_pool, corr_att_fold, corr_unatt_fold)
    else:
        # read the dataframe
        res_df = pd.read_csv(table_name, header=0, index_col=[0,1,2])                
        if not 'Subj '+str(Subj_ID+1) in res_df.index.get_level_values('Subject ID'):
            res_add = create_corr_df(Subj_ID, sig_corr_pool, corr_att_fold, corr_unatt_fold)
            res_df = pd.concat([res_df, res_add], axis=0)
        elif OVERWRITE:
            res_df = res_df.drop('Subj '+str(Subj_ID+1), level='Subject ID')
            res_add = create_corr_df(Subj_ID, sig_corr_pool, corr_att_fold, corr_unatt_fold)
            res_df = pd.concat([res_df, res_add], axis=0)
        else:
            print(f"Results for Subj {Subj_ID+1} already exist in {table_name}")
    with open(table_name, 'w') as f:
        res_df.to_csv(f, header=True)

def create_acc_df(Subj_ID, trial_len_list, acc_list, dataloss=None):
    if dataloss is not None:
        columns = ['Subject ID'] + ['Trial_len='+str(tl) for tl in trial_len_list] + ['Data Loss']
        data = ['Subj '+str(Subj_ID+1)] + acc_list + [dataloss]
    else:
        columns = ['Subject ID'] + ['Trial_len='+str(tl) for tl in trial_len_list]
        data = ['Subj '+str(Subj_ID+1)] + acc_list
    acc_df = pd.DataFrame([data], columns=columns)
    return acc_df

def save_acc_df(table_name, Subj_ID, trial_len_list, res, OVERWRITE=False, data_loss=None):
    if not os.path.isfile(table_name):
        # create a pandas dataframe that contains Subj_ID, Corr_Att, Corr_Unatt, Sig_Corr
        res_df = create_acc_df(Subj_ID, trial_len_list, res, data_loss)
    else:
        # read the dataframe
        res_df = pd.read_csv(table_name, header=0)
        if ('Subj ' + str(Subj_ID + 1)) not in res_df['Subject ID'].values:
            res_add = create_acc_df(Subj_ID, trial_len_list, res, data_loss)
            res_df = pd.concat([res_df, res_add], axis=0)
        elif OVERWRITE:
            res_df = res_df[res_df['Subject ID'] != 'Subj ' + str(Subj_ID + 1)]
            res_add = create_acc_df(Subj_ID, trial_len_list, res, data_loss)
            res_df = pd.concat([res_df, res_add], axis=0)
        else:
            print(f"Results for Subj {Subj_ID+1} already exist in {table_name}")
    with open(table_name, 'w') as f:
        res_df.to_csv(f, header=True, index=False)


def create_ISC_df(ISC_fold, ISCov_fold, sig_ISC_pool, mod_name):
    n_folds = ISC_fold.shape[0]
    n_component = ISC_fold.shape[1]
    # Flatten the ISC and ISCov arrays to include fold information
    ISC_flat = ISC_fold.flatten()
    ISCov_flat = ISCov_fold.flatten()
    # Create a MultiIndex to include fold information
    index = pd.MultiIndex.from_tuples([
        (mod_name, sig_ISC_pool, f'CC {i+1}', f'Fold {j+1}')
        for j in range(n_folds)
        for i in range(n_component)
    ], names=['Modality', 'Sig Level (ISC)', 'Component', 'Fold'])
    data = {
        'ISC': ISC_flat,
        'ISCov': ISCov_flat
    }
    ISC_df = pd.DataFrame(data, index=index)
    return ISC_df


def read_res(table_dir, res, train_type):
    table_name = table_dir + f'{res}_Train_{train_type}.csv'
    res_df = pd.read_csv(table_name, header=0, index_col=[0,1,2])
    return res_df