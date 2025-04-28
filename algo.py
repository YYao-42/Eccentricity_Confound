'''
Inherited from the previous project.
Class CanonicalCorrelationAnalysis and Class GeneralizedCCA are heavily modified.
Class DiscriminativeCCA, GeneralizedCCA_MultiMod and GCCAPreprocessedCCA are newly added.
Other classes are not used and modified, and thus do not have all the new features.
'''

import numpy as np
import copy
import random
from sklearn.covariance import LedoitWolf
from tqdm import tqdm
from numpy import linalg as LA
from scipy.linalg import eig, eigh, sqrtm, lstsq
from scipy.stats import pearsonr
import utils


class CanonicalCorrelationAnalysis:
    def __init__(self, EEG_list, Stim_list, fs, L_EEG, L_Stim, offset_EEG=0, offset_Stim=0, dim_list_EEG=None, dim_list_Stim=None, leave_out=1, n_components=5, regularization='lwcov', message=True, signifi_level=True, p_value=0.05, EEG_masked=None, Stim_masked=None):
        '''
        EEG_list: list of EEG data, each element is a T(#sample)xDx(#channel)x(#task) array corresponding to a video 
        Stim_list: list of stimulus, each element is a T(#sample)xDy(#feature dim)x(#task) array corresponding to a video 
        fs: Sampling rate
        L_EEG/L_Stim: If use (spatial-) temporal filter, the number of taps
        offset_EEG/offset_Stim: If use (spatial-) temporal filter, the offset of time lags
        dim_list_EEG/dim_list_Stim: If 'EEG' is actually a stack of data from different modalities, dim_list_EEG should be a list of the dimensions of each modality. The same for 'Stim'.
        leave_out: Number of videos left out for each fold
        n_components: Number of components to be returned
        regularization: Regularization of the estimated covariance matrix
        message: Print out the results if True
        signifi_level: Calculate the significance level if True
        p_value: For calculating significance level
        '''
        self.EEG_list = EEG_list
        self.Stim_list = Stim_list
        self.fs = fs
        self.L_EEG = L_EEG
        self.L_Stim = L_Stim
        self.offset_EEG = offset_EEG
        self.offset_Stim = offset_Stim
        self.dim_list_EEG = dim_list_EEG
        self.dim_list_Stim = dim_list_Stim
        self.leave_out = leave_out
        self.n_components = n_components
        self.regularization = regularization
        self.message = message
        self.signifi_level = signifi_level
        self.p_value = p_value
        self.EEG_masked = EEG_masked
        self.Stim_masked = Stim_masked
        self.MASK = True if EEG_masked is not None and Stim_masked is not None else False

        self.nb_videos = len(self.Stim_list)
        assert self.nb_videos%self.leave_out == 0, "The number of videos should be a multiple of the leave_out parameter."
        self.nb_folds = self.nb_videos//self.leave_out

    def apply_mask(self, X, Y=None):
        idx_not_nan = ~np.isnan(X).any(axis=1)
        if Y is not None:
            idx_not_nan = idx_not_nan & ~np.isnan(Y).any(axis=1)
            Y = Y[idx_not_nan, :]
        X = X[idx_not_nan, :]
        return X, Y

    def fit(self, X, Y):
        if np.ndim(Y) == 1:
            Y = np.expand_dims(Y, axis=1)
        # X, Y = self.apply_mask(X, Y)
        T, Dx = X.shape
        _, Dy = Y.shape
        Lx = self.L_EEG
        Ly = self.L_Stim
        n_components = self.n_components
        mtx_X = utils.block_Hankel(X, Lx, self.offset_EEG)
        mtx_Y = utils.block_Hankel(Y, Ly, self.offset_Stim)
        mtx_X, mtx_Y = self.apply_mask(mtx_X, mtx_Y)
        dim_list_X = [d*Lx for d in self.dim_list_EEG] if self.dim_list_EEG is not None else [Dx*Lx]
        dim_list_Y = [d*Ly for d in self.dim_list_Stim] if self.dim_list_Stim is not None else [Dy*Ly]
        # compute covariance matrices
        Rx, _ = utils.get_cov_mtx(mtx_X, dim_list_X, self.regularization)
        Ry, _ = utils.get_cov_mtx(mtx_Y, dim_list_Y, self.regularization)
        covXY = np.cov(mtx_X, mtx_Y, rowvar=False)
        Rxy = covXY[:Dx*Lx,Dx*Lx:Dx*Lx+Dy*Ly]
        Ryx = covXY[Dx*Lx:Dx*Lx+Dy*Ly,:Dx*Lx]
        invRx = utils.PCAreg_inv(Rx, LA.matrix_rank(Rx))
        invRy = utils.PCAreg_inv(Ry, LA.matrix_rank(Ry))
        A = invRx@Rxy@invRy@Ryx
        B = invRy@Ryx@invRx@Rxy
        # lam of A and lam of B should be the same
        # can be used as a preliminary check for correctness
        # the correlation coefficients are already available by taking sqrt of the eigenvalues: corr_coe = np.sqrt(lam[:K_regu])
        # or we do the following to obtain transformed X and Y and calculate corr_coe from there
        Lam, V_A = utils.eig_sorted(A)
        _, V_B = utils.eig_sorted(B)
        Lam = np.real(Lam[:n_components])
        V_A = np.real(V_A[:,:n_components])
        V_B = np.real(V_B[:,:n_components])
        # mtx_X and mtx_Y should be centered according to the definition. But since we calculate the correlation coefficients, it does not matter.
        X_trans = mtx_X@V_A
        Y_trans = mtx_Y@V_B
        corr_coe = np.array([np.corrcoef(X_trans[:,k], Y_trans[:,k])[0,1] for k in range(n_components)])
        V_A[:,corr_coe<0] = -1*V_A[:,corr_coe<0]
        corr_coe[corr_coe<0] = -1*corr_coe[corr_coe<0]
        return corr_coe, V_A, V_B

    def get_transformed_data(self, X, Y, V_A, V_B):
        '''
        Get the transformed data
        X: EEG or other data modalities; V_A: filters for X
        Y: features; V_B: filters for Y
        C: competing features
        '''
        # X, Y = self.apply_mask(X, Y)
        mtx_X = utils.block_Hankel(X, self.L_EEG, self.offset_EEG)
        mtx_Y = utils.block_Hankel(Y, self.L_Stim, self.offset_Stim)
        mtx_X, mtx_Y = self.apply_mask(mtx_X, mtx_Y)
        mtx_X_centered = mtx_X - np.mean(mtx_X, axis=0, keepdims=True)
        mtx_Y_centered = mtx_Y - np.mean(mtx_Y, axis=0, keepdims=True)
        X_trans = mtx_X_centered@V_A
        Y_trans = mtx_Y_centered@V_B
        return X_trans, Y_trans
    
    def get_transformed_data_3D(self, X, Y, V_A, V_B):
        assert np.ndim(X) == np.ndim(Y) == 3, "The input data should be 3D."
        X_trans_all = []
        Y_trans_all = []
        for task in range(X.shape[2]):
            X_trans, Y_trans = self.get_transformed_data(X[:,:,task], Y[:,:,task], V_A, V_B)
            X_trans_all.append(X_trans)
            Y_trans_all.append(Y_trans)
        X_trans = np.stack(X_trans_all, axis=2)
        Y_trans = np.stack(Y_trans_all, axis=2)
        return X_trans, Y_trans

    def cal_corr_coe(self, X, Y, V_A=None, V_B=None):
        '''
        Same as get_corr_coe but with the input of the data and the filters
        '''
        if V_A is None:
            X_trans, Y_trans = X, Y
        else:
            X_trans, Y_trans = self.get_transformed_data(X, Y, V_A, V_B)
        corr_coe = np.array([np.corrcoef(X_trans[:,k], Y_trans[:,k])[0,1] for k in range(self.n_components)]) # (n_components,)
        return corr_coe

    def cal_corr_coe_3D(self, X, Y, V_A=None, V_B=None):
        '''
        Calculate the correlation coefficients for 3D data
        '''
        assert np.ndim(X) == np.ndim(Y) == 3, "The input data should be 3D."
        corr_all = []
        for task in range(X.shape[2]):
            corr_coe = self.cal_corr_coe(X[:,:,task], Y[:,:,task], V_A, V_B)
            corr_all.append(corr_coe)
        corr_coe = np.stack(corr_all, axis=1) # (n_components, nb_tasks)
        return corr_coe

    def cal_corr_coe_trials(self, X_trials, Y_trials, V_A=None, V_B=None, avg=True):
        THREE_D = np.ndim(X_trials[0]) == 3
        corr_coe_trials = [self.cal_corr_coe(X, Y, V_A, V_B) for X, Y in zip(X_trials, Y_trials)] if not THREE_D else [self.cal_corr_coe_3D(X, Y, V_A, V_B) for X, Y in zip(X_trials, Y_trials)]
        corr_coe = np.stack(corr_coe_trials, axis=0)
        if avg:
            corr_coe = np.mean(corr_coe, axis=0)
        return corr_coe

    def cal_corr_compete_trials(self, X, Y_att, V_X, V_Y, BOOTSTRAP, trial_len, given_start_points=None, BTfactor=2, overlap=0.9):
        X_trans, Y_att_trans = self.get_transformed_data(X, Y_att, V_X, V_Y) if np.ndim(X) == 2 else self.get_transformed_data_3D(X, Y_att, V_X, V_Y)
        T = X_trans.shape[0]
        assert T-trial_len*self.fs >= 0, "The trial length is too long."
        if given_start_points is None:
            if BOOTSTRAP:
                nb_trials = min(T//self.fs//BTfactor, 200)
                start_points = np.random.randint(0, T-trial_len*self.fs, size=nb_trials)
                start_points = np.sort(start_points)
            else:
                start_points = np.array(range(0, T - T%(self.fs*trial_len), round(self.fs*trial_len*(1-overlap))))
        else:
            start_points = given_start_points
        # Note: the data has been transformed
        X_trans_trials = utils.into_trials(X_trans, self.fs, trial_len, start_points=start_points)
        Y_att_trans_trials = utils.into_trials(Y_att_trans, self.fs, trial_len, start_points=start_points)
        Y_compete_trans_trials = [utils.select_distractors([Y_att_trans], self.fs, trial_len, start_point)[0] for start_point in start_points]
        # Y_compete_trans_trials = utils.shift_trials(Y_att_trans_trials)
        corr_att_trials = self.cal_corr_coe_trials(X_trans_trials, Y_att_trans_trials, avg=False)
        corr_compete_trials = self.cal_corr_coe_trials(X_trans_trials, Y_compete_trans_trials, avg=False)
        return corr_att_trials, corr_compete_trials, start_points, X_trans_trials, Y_att_trans_trials, Y_compete_trans_trials
    
    def cal_corr_compete_trials_mask(self, X, Y_att, V_X, V_Y, BOOTSTRAP, trial_len, given_start_points=None, BTfactor=2, overlap=0.9):
        T = X.shape[0]
        assert T-trial_len*self.fs >= 0, "The trial length is too long."
        if given_start_points is None:
            if BOOTSTRAP:
                nb_trials = min(T//self.fs//BTfactor, 200)
                start_points = np.random.randint(0, T-trial_len*self.fs, size=nb_trials)
                start_points = np.sort(start_points)
            else:
                start_points = np.array(range(0, T - T%(self.fs*trial_len), round(self.fs*trial_len*(1-overlap))))
        else:
            start_points = given_start_points
        # Note: the data has not been transformed
        X_trials = utils.into_trials(X, self.fs, trial_len, start_points=start_points)
        Y_att_trials = utils.into_trials(Y_att, self.fs, trial_len, start_points=start_points)
        Y_compete_trials = [utils.select_distractors([Y_att], self.fs, trial_len, start_point)[0] for start_point in start_points]
        corr_att_trials = self.cal_corr_coe_trials(X_trials, Y_att_trials, V_X, V_Y, avg=False)
        corr_compete_trials = self.cal_corr_coe_trials(X_trials, Y_compete_trials, V_X, V_Y, avg=False)
        return corr_att_trials, corr_compete_trials, start_points, X_trials, Y_att_trials, Y_compete_trials

    def permutation_test(self, X, Y, V_A, V_B, nb_permu=200, PHASE_SCRAMBLE=False, block_len=1, X_trans=None, Y_trans=None):
        '''
        Permutation test for the correlation coefficients. Use phase scrambling or block shuffling.
        '''
        corr_coe_topK = np.zeros((nb_permu, self.n_components))
        if X_trans is None and Y_trans is None:
            X_trans, Y_trans = self.get_transformed_data(X, Y, V_A, V_B) 
        for i in tqdm(range(nb_permu)):
            X_shuffled = utils.shuffle_2D(X_trans, block_len) if not PHASE_SCRAMBLE else utils.phase_scramble_2D(X_trans)
            Y_shuffled = utils.shuffle_2D(Y_trans, block_len) if not PHASE_SCRAMBLE else utils.phase_scramble_2D(Y_trans)
            corr_coe_topK[i,:] = np.array([np.corrcoef(X_shuffled[:,k], Y_shuffled[:,k])[0,1] for k in range(self.n_components)])
        return corr_coe_topK

    def calculate_sig_corr(self, corr_trials, nb_permu=200, nb_fold=1):
        '''
        Calculate the significance level based on the results of the permutation test
        '''
        assert self.n_components*nb_permu*nb_fold == corr_trials.shape[0]*corr_trials.shape[1]
        sig_idx = -int(nb_permu*self.p_value*self.n_components*nb_fold)
        corr_trials = np.sort(abs(corr_trials), axis=None)
        return corr_trials[sig_idx]
    
    def permutation_test_acc(self, X_trials, Y1_trials, Y2_trials, V_X=None, V_Y=None, nb_permu=100):
        acc_list = []
        for i in tqdm(range(nb_permu)):
            # randomly shuffle X_trials
            X_trials_shifted = copy.deepcopy(X_trials)
            shift_amount = random.randint(len(X_trials)//4, len(X_trials)//4*3)
            X_trials_shifted = X_trials_shifted[shift_amount:] + X_trials_shifted[:shift_amount]
            corr_X1_trials = self.cal_corr_coe_trials(X_trials_shifted, Y1_trials, V_X, V_Y, avg=False)
            corr_X2_trials = self.cal_corr_coe_trials(X_trials_shifted, Y2_trials, V_X, V_Y, avg=False)
            acc, _, _, _, _= utils.eval_compete_3D(corr_X1_trials, corr_X2_trials, TRAIN_WITH_ATT=True, message=False)
            acc_list.append(acc)
        return acc_list

    # def forward_model(self, X, V_A, X_trans=None):
    #     '''
    #     Inputs:
    #     X: observations (one subject) TxD
    #     V_A: filters/backward models DLxK
    #     X_trans: transformed TxK
    #     (Do not consider REGFEATS here)
    #     Output:
    #     F: forward model
    #     '''
    #     if X_trans is not None: 
    #         # Calculate the forward model based on the original data
    #         F = (lstsq(X_trans, X)[0]).T
    #     else: 
    #         # Calculate the forward model based on the Hankelized data and extract the forward model corresponding to the correct time points
    #         X_block_Hankel = utils.block_Hankel(X, self.L_EEG, self.offset_EEG)
    #         Rxx = np.cov(X_block_Hankel, rowvar=False)
    #         if np.ndim(Rxx) == 0:
    #             Rxx = np.array([[Rxx]])
    #         F_redun = Rxx@V_A@LA.inv(V_A.T@Rxx@V_A)
    #         F = utils.F_organize(F_redun, self.L_EEG, self.offset_EEG)
    #     return F

    def get_train_test_data(self):
        train_list_folds, test_list_folds = utils.split_multi_mod_LVO([self.EEG_list, self.Stim_list], self.leave_out)
        if self.MASK:
            _, test_list_folds = utils.split_multi_mod_LVO([self.EEG_masked, self.Stim_masked], self.leave_out)
        T, _, nb_tasks = train_list_folds[0][0].shape
        assert len(train_list_folds) == len(test_list_folds) == self.nb_folds, "The number of folds is not correct."
        train_list_folds = [[data[:,:,1:].transpose(2,0,1).reshape(T*2, -1) for data in EEGStim] for EEGStim in train_list_folds]
        # train_list_folds = [[data.transpose(2,0,1).reshape(T*nb_tasks, -1) for data in EEGStim] for EEGStim in train_list_folds]
        return train_list_folds, test_list_folds, nb_tasks

    def cross_val(self):
        '''
        Cross-validation with leave-one-pair-out; For single-object dataset only
        '''
        train_list_folds, test_list_folds, nb_tasks = self.get_train_test_data()
        n_components = self.n_components
        nb_folds = self.nb_folds
        corr_train_fold = np.zeros((nb_folds, n_components))
        corr_test_fold = np.zeros((nb_folds, n_components, nb_tasks))
        corr_permu_fold = []
        # forward_model_fold = []
        for idx in range(0, nb_folds):
            [EEG_train, Sti_train], [EEG_test, Sti_test] = train_list_folds[idx], test_list_folds[idx]
            corr_train_fold[idx,:], V_A_train, V_B_train = self.fit(EEG_train, Sti_train)
            corr_test_fold[idx,:,:] = self.cal_corr_coe_3D(EEG_test, Sti_test, V_A_train, V_B_train)
            # forward_model = self.forward_model(EEG_test, V_A_train)
            # forward_model_fold.append(forward_model)
            corr_permu_fold.append(self.permutation_test(EEG_test[:,:,0], Sti_test[:,:,0], V_A=V_A_train, V_B=V_B_train))
        corr_permu_all = np.concatenate(tuple(corr_permu_fold), axis=0)
        sig_corr_pool = self.calculate_sig_corr(corr_permu_all, nb_fold=nb_folds)
        if self.message:
            print('Average correlation coefficients of the top {} components on the training sets: {}'.format(n_components, np.average(corr_train_fold, axis=0)))
            print('Average correlation coefficients of the top {} components on the test sets: {}'.format(n_components, np.average(corr_test_fold, axis=0)))
            print('Significance level: {}'.format(sig_corr_pool))
        return corr_train_fold, corr_test_fold, sig_corr_pool

    def match_mismatch(self, trial_len, BOOTSTRAP=True, V_eeg=None, V_Stim=None, PERMU_TEST=True, overlap=0.9, given_start_points=None):
        '''
        Match-Mismatch task with leave-one-pair-out
        Always train on match and try to distinguish match from mismatch 
        Mismatch is a random segment that is not shown on the screen
        '''
        train_list_folds, test_list_folds, _ = self.get_train_test_data()
        corr_match_eeg = []
        corr_mismatch_eeg = []
        X_all_trials = []
        Y_att_all_trials = []
        Y_compete_all_trials = []
        start_points = None if given_start_points is None else given_start_points
        for idx in range(0, self.nb_folds):
            [EEG_train, Sti_train], [EEG_test, Sti_test] = train_list_folds[idx], test_list_folds[idx]
            if V_eeg is None:
                _, V_eeg_train, V_feat_train = self.fit(EEG_train, Sti_train)
            else:
                V_eeg_train, V_feat_train = V_eeg, V_Stim
            if not self.MASK:
                corr_match_eeg_i, corr_mismatch_eeg_i, start_points, X_trans_trials, Y_att_trans_trials, Y_compete_trans_trials = self.cal_corr_compete_trials(EEG_test, Sti_test, V_eeg_train, V_feat_train, BOOTSTRAP, trial_len, given_start_points=start_points, overlap=overlap) 
                X_all_trials = X_trans_trials + X_all_trials
                Y_att_all_trials = Y_att_trans_trials + Y_att_all_trials
                Y_compete_all_trials = Y_compete_trans_trials + Y_compete_all_trials
            else:
                corr_match_eeg_i, corr_mismatch_eeg_i, start_points, X_trials, Y_att_trials, Y_compete_trials = self.cal_corr_compete_trials_mask(EEG_test, Sti_test, V_eeg_train, V_feat_train, BOOTSTRAP, trial_len, given_start_points=start_points, overlap=overlap) 
                X_all_trials = X_trials + X_all_trials
                Y_att_all_trials = Y_att_trials + Y_att_all_trials
                Y_compete_all_trials = Y_compete_trials + Y_compete_all_trials
            corr_match_eeg.append(corr_match_eeg_i)
            corr_mismatch_eeg.append(corr_mismatch_eeg_i)
        corr_match_eeg = np.concatenate(tuple(corr_match_eeg), axis=0)
        corr_mismatch_eeg = np.concatenate(tuple(corr_mismatch_eeg), axis=0)
        if PERMU_TEST:
            acc_permu_list = self.permutation_test_acc(X_all_trials, Y_att_all_trials, Y_compete_all_trials) if not self.MASK else self.permutation_test_acc(X_all_trials, Y_att_all_trials, Y_compete_all_trials, V_X=V_eeg_train, V_Y=V_feat_train)
        else:
            acc_permu_list = None
        return corr_match_eeg, corr_mismatch_eeg, acc_permu_list, start_points
