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
from scipy.linalg import eig, eigh, sqrtm, lstsq, block_diag
from scipy.stats import pearsonr
import utils


class CanonicalCorrelationAnalysis:
    def __init__(self, EEG_list, Stim_list, fs, L_EEG, L_Stim, offset_EEG=0, offset_Stim=0, dim_list_EEG=None, dim_list_Stim=None, task_train=[2,3], leave_out=1, n_components=5, regularization='lwcov', message=True, signifi_level=True, p_value=0.05, EEG_masked=None, Stim_masked=None):
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
        self.task_train = [t-1 for t in task_train] # convert to 0-indexed
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
        rt_of_not_nan = np.sum(idx_not_nan)/X.shape[0]
        X = X[idx_not_nan, :]
        return X, Y, rt_of_not_nan

    def fit(self, X, Y):
        if np.ndim(Y) == 1:
            Y = np.expand_dims(Y, axis=1)
        T, Dx = X.shape
        _, Dy = Y.shape
        Lx = self.L_EEG
        Ly = self.L_Stim
        n_components = self.n_components
        mtx_X = utils.block_Hankel(X, Lx, self.offset_EEG)
        mtx_Y = utils.block_Hankel(Y, Ly, self.offset_Stim)
        mtx_X, mtx_Y, rt_of_not_nan = self.apply_mask(mtx_X, mtx_Y)
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
        return corr_coe, V_A, V_B, rt_of_not_nan

    def forward_model(self, X, V_A, Y):
        '''
        Calculate the forward model of the filters V_A on the data X
        Y is provided here just to get the correct mask
        '''
        nb_tasks = X.shape[2] if np.ndim(X) == 3 else 1
        forward_models = []
        for task in range(nb_tasks):
            mtx_X = utils.block_Hankel(X[:,:,task], self.L_EEG, self.offset_EEG)
            mtx_Y = utils.block_Hankel(Y[:,:,task], self.L_Stim, self.offset_Stim)
            mtx_X, _, _ = self.apply_mask(mtx_X, mtx_Y)
            mtx_X_centered = mtx_X - np.mean(mtx_X, axis=0, keepdims=True)
            forward_model = mtx_X_centered.T@mtx_X_centered@V_A@LA.inv(V_A.T@mtx_X_centered.T@mtx_X_centered@V_A)
            forward_model = utils.F_organize(forward_model, self.L_EEG, self.offset_EEG)
            # forward_model = utils.F_avg_temporal(forward_model, self.L_EEG)
            forward_models.append(forward_model)
        forward_models = np.stack(forward_models, axis=2) if nb_tasks > 1 else forward_models[0]
        return forward_models

    def get_transformed_data(self, X, Y, V_A, V_B):
        '''
        Get the transformed data
        X: EEG or other data modalities; V_A: filters for X
        Y: features; V_B: filters for Y
        C: competing features
        '''
        mtx_X = utils.block_Hankel(X, self.L_EEG, self.offset_EEG)
        mtx_Y = utils.block_Hankel(Y, self.L_Stim, self.offset_Stim)
        mtx_X, mtx_Y, rt_of_not_nan = self.apply_mask(mtx_X, mtx_Y)
        mtx_X_centered = mtx_X - np.mean(mtx_X, axis=0, keepdims=True)
        mtx_Y_centered = mtx_Y - np.mean(mtx_Y, axis=0, keepdims=True)
        X_trans = mtx_X_centered@V_A
        Y_trans = mtx_Y_centered@V_B
        return X_trans, Y_trans, rt_of_not_nan
    
    def get_transformed_data_3D(self, X, Y, V_A, V_B):
        assert self.MASK is False, "This function only works for non-masked data. Otherwise the dimensions of data of different tasks will not match."
        assert np.ndim(X) == np.ndim(Y) == 3, "The input data should be 3D."
        X_trans_all = []
        Y_trans_all = []
        for task in range(X.shape[2]):
            X_trans, Y_trans, _ = self.get_transformed_data(X[:,:,task], Y[:,:,task], V_A, V_B)
            X_trans_all.append(X_trans)
            Y_trans_all.append(Y_trans)
        X_trans = np.stack(X_trans_all, axis=2)
        Y_trans = np.stack(Y_trans_all, axis=2)
        rts_of_not_nan = None
        return X_trans, Y_trans, rts_of_not_nan

    def get_transformed_data_indexed(self, X, Y, V_A, V_B):
        '''
        Same as get_transformed_data, but also returns the indices, w.r.t. the original time axis,
        of the samples that survived the mask. They are needed to tell which part of the video a
        transformed sample comes from, once the masked samples have been discarded.
        '''
        mtx_X = utils.block_Hankel(X, self.L_EEG, self.offset_EEG)
        mtx_Y = utils.block_Hankel(Y, self.L_Stim, self.offset_Stim)
        idx_kept = np.where(~np.isnan(mtx_X).any(axis=1) & ~np.isnan(mtx_Y).any(axis=1))[0]
        mtx_X = mtx_X[idx_kept, :]
        mtx_Y = mtx_Y[idx_kept, :]
        rt_of_not_nan = len(idx_kept)/X.shape[0]
        mtx_X_centered = mtx_X - np.mean(mtx_X, axis=0, keepdims=True)
        mtx_Y_centered = mtx_Y - np.mean(mtx_Y, axis=0, keepdims=True)
        X_trans = mtx_X_centered@V_A
        Y_trans = mtx_Y_centered@V_B
        return X_trans, Y_trans, idx_kept, rt_of_not_nan

    def cal_corr_coe(self, X, Y, V_A=None, V_B=None):
        '''
        Same as get_corr_coe but with the input of the data and the filters
        '''
        if V_A is None:
            X_trans, Y_trans = X, Y
            rt_of_not_nan = None
        else:
            X_trans, Y_trans, rt_of_not_nan = self.get_transformed_data(X, Y, V_A, V_B)
        corr_coe = np.array([np.corrcoef(X_trans[:,k], Y_trans[:,k])[0,1] for k in range(self.n_components)]) # (n_components,)
        return corr_coe, rt_of_not_nan

    def cal_corr_coe_3D(self, X, Y, V_A=None, V_B=None):
        '''
        Calculate the correlation coefficients for 3D data
        '''
        assert np.ndim(X) == np.ndim(Y) == 3, "The input data should be 3D."
        corr_all = []
        rts_of_not_nan = []
        for task in range(X.shape[2]):
            corr_coe, rt_of_not_nan = self.cal_corr_coe(X[:,:,task], Y[:,:,task], V_A, V_B)
            corr_all.append(corr_coe)
            rts_of_not_nan.append(rt_of_not_nan)
        corr_coe = np.stack(corr_all, axis=1) # (n_components, nb_tasks)
        rts_of_not_nan = np.array(rts_of_not_nan).reshape(1, -1)
        return corr_coe, rts_of_not_nan

    def cal_corr_coe_trials(self, X_trials, Y_trials, V_A=None, V_B=None, avg=True):
        THREE_D = np.ndim(X_trials[0]) == 3
        corr_coe_rts = [self.cal_corr_coe(X, Y, V_A, V_B) for X, Y in zip(X_trials, Y_trials)] if not THREE_D else [self.cal_corr_coe_3D(X, Y, V_A, V_B) for X, Y in zip(X_trials, Y_trials)]
        corr_coe_trials = [stats[0] for stats in corr_coe_rts]
        rts_trials = [stats[1] for stats in corr_coe_rts]
        corr_coe = np.stack(corr_coe_trials, axis=0)
        rts_of_not_nan = np.stack(rts_trials, axis=0)
        if avg:
            corr_coe = np.mean(corr_coe, axis=0)
            rts_of_not_nan = np.mean(rts_of_not_nan, axis=0)
        return corr_coe, rts_of_not_nan

    def get_start_points(self, T, trial_len, BOOTSTRAP, BTfactor=2, overlap=0.9, given_start_points=None):
        assert T-trial_len*self.fs >= 0, f"The trial length is too long. T: {T}, trial_len*fs: {trial_len*self.fs}"
        if given_start_points is None:
            if BOOTSTRAP:
                nb_trials = min(T//self.fs//BTfactor, 200)
                start_points = np.random.randint(0, T-trial_len*self.fs, size=nb_trials)
                start_points = np.sort(start_points)
            else:
                start_points = np.array(range(0, T - T%(self.fs*trial_len), round(self.fs*trial_len*(1-overlap))))
                # if the last trial is too short, drop it
                if start_points[-1] + trial_len*self.fs > T and len(start_points) > 1:
                    start_points = start_points[:-1]
        else:
            start_points = given_start_points
        return start_points

    def cal_corr_compete_trials(self, X, Y_att, V_X, V_Y, BOOTSTRAP, trial_len, given_start_points=None, BTfactor=2, overlap=0.9, nb_compete=1):
        X_trans, Y_att_trans, _ = self.get_transformed_data(X, Y_att, V_X, V_Y) if np.ndim(X) == 2 else self.get_transformed_data_3D(X, Y_att, V_X, V_Y)
        T = X_trans.shape[0]
        assert T-trial_len*self.fs >= 0, "The trial length is too long."
        start_points = self.get_start_points(T, trial_len, BOOTSTRAP, BTfactor, overlap, given_start_points)
        # Note: the data has been transformed
        X_trans_trials = utils.into_trials(X_trans, self.fs, trial_len, start_points=start_points)
        Y_att_trans_trials = utils.into_trials(Y_att_trans, self.fs, trial_len, start_points=start_points)
        # Y_compete_trans_trials = utils.shift_trials(Y_att_trans_trials)
        X_trans_trials, Y_att_trans_trials, Y_compete_trans_trials = utils.match_mismatch_pairs(X_trans_trials, Y_att_trans_trials, nb_compete)
        corr_att_trials, _ = self.cal_corr_coe_trials(X_trans_trials, Y_att_trans_trials, avg=False)
        corr_compete_trials, _ = self.cal_corr_coe_trials(X_trans_trials, Y_compete_trans_trials, avg=False)
        return corr_att_trials, corr_compete_trials, start_points, X_trans_trials, Y_att_trans_trials, Y_compete_trans_trials
    
    def cal_corr_compete_trials_mask(self, X, Y_att, V_X, V_Y, BOOTSTRAP, trial_len, given_start_points=None, BTfactor=2, overlap=0.9, nb_compete=1):
        '''First divide data into trials and then apply the mask before calculating the correlation coefficients.'''
        T = X.shape[0]
        assert T-trial_len*self.fs >= 0, "The trial length is too long."
        start_points = self.get_start_points(T, trial_len, BOOTSTRAP, BTfactor, overlap, given_start_points)
        # Note: the data has not been transformed
        X_trials = utils.into_trials(X, self.fs, trial_len, start_points=start_points)
        Y_att_trials = utils.into_trials(Y_att, self.fs, trial_len, start_points=start_points)
        # Y_compete_trials = utils.shift_trials(Y_att_trials)
        X_trials, Y_att_trials, Y_compete_trials = utils.match_mismatch_pairs(X_trials, Y_att_trials, nb_compete)
        corr_att_trials, rts_kept_trials = self.cal_corr_coe_trials(X_trials, Y_att_trials, V_X, V_Y, avg=False)
        corr_compete_trials, _ = self.cal_corr_coe_trials(X_trials, Y_compete_trials, V_X, V_Y, avg=False)
        return corr_att_trials, corr_compete_trials, start_points, X_trials, Y_att_trials, Y_compete_trials, rts_kept_trials

    def cal_corr_compete_mask_trials(self, X, Y_att, V_X, V_Y, BOOTSTRAP, trial_len, BTfactor=2, overlap=0.9, nb_compete=1):
        '''First apply the mask and then divide data into trials.'''
        rts_kept = []
        corr_att_trials_tasks = []
        corr_compete_trials_tasks = []
        X_trans_trials_tasks = []
        Y_att_trans_trials_tasks = []
        Y_compete_trans_trials_tasks = []
        min_nb_trials = np.inf
        for task in range(X.shape[2]):
            X_trans, Y_trans, rt_of_not_nan = self.get_transformed_data(X[:,:,task], Y_att[:,:,task], V_X, V_Y)
            assert X_trans.shape[0] == Y_trans.shape[0], "The number of samples in X and Y should be the same after transformation."
            rts_kept.append(rt_of_not_nan)
            T = X_trans.shape[0]
            start_points = self.get_start_points(T, trial_len, BOOTSTRAP, BTfactor, overlap)
            X_trans_trials = utils.into_trials(X_trans, self.fs, trial_len, start_points=start_points)
            Y_att_trans_trials = utils.into_trials(Y_trans, self.fs, trial_len, start_points=start_points)
            # Y_compete_trans_trials = utils.shift_trials(Y_att_trans_trials)
            X_trans_trials, Y_att_trans_trials, Y_compete_trans_trials = utils.match_mismatch_pairs(X_trans_trials, Y_att_trans_trials, nb_compete)
            corr_att_trials, _ = self.cal_corr_coe_trials(X_trans_trials, Y_att_trans_trials, avg=False)
            corr_compete_trials, _ = self.cal_corr_coe_trials(X_trans_trials, Y_compete_trans_trials, avg=False)
            corr_att_trials_tasks.append(corr_att_trials)
            corr_compete_trials_tasks.append(corr_compete_trials)
            X_trans_trials_tasks.append(X_trans_trials)
            Y_att_trans_trials_tasks.append(Y_att_trans_trials)
            Y_compete_trans_trials_tasks.append(Y_compete_trans_trials)
            min_nb_trials = min(min_nb_trials, len(corr_att_trials))
        # randomly select the same number of trials from each task
        indices = [np.sort(random.sample(range(len(trials)), min_nb_trials)) for trials in corr_att_trials_tasks]
        corr_att_trials_tasks = [trials[ind] for trials, ind in zip(corr_att_trials_tasks, indices)]
        corr_compete_trials_tasks = [trials[ind] for trials, ind in zip(corr_compete_trials_tasks, indices)]
        X_trans_trials_tasks = [[trials[i] for i in ind] for trials, ind in zip(X_trans_trials_tasks, indices)]
        Y_att_trans_trials_tasks = [[trials[i] for i in ind] for trials, ind in zip(Y_att_trans_trials_tasks, indices)]
        Y_compete_trans_trials_tasks = [[trials[i] for i in ind] for trials, ind in zip(Y_compete_trans_trials_tasks, indices)]
        corr_att_trials = np.stack(corr_att_trials_tasks, axis=2)
        corr_compete_trials = np.stack(corr_compete_trials_tasks, axis=2)
        X_trans_trials = [np.stack([task_trials[i] for task_trials in X_trans_trials_tasks], axis=2) for i in range(min_nb_trials)]
        Y_att_trans_trials = [np.stack([task_trials[i] for task_trials in Y_att_trans_trials_tasks], axis=2) for i in range(min_nb_trials)]
        Y_compete_trans_trials = [np.stack([task_trials[i] for task_trials in Y_compete_trans_trials_tasks], axis=2) for i in range(min_nb_trials)]
        rts_kept = np.array(rts_kept).reshape(1, -1)
        return corr_att_trials, corr_compete_trials, X_trans_trials, Y_att_trans_trials, Y_compete_trans_trials, rts_kept

    def permutation_test(self, X, Y, V_A, V_B, nb_permu=200, PHASE_SCRAMBLE=False, block_len=1, X_trans=None, Y_trans=None):
        '''
        Permutation test for the correlation coefficients. Use phase scrambling or block shuffling.
        '''
        corr_coe_topK = np.zeros((nb_permu, self.n_components))
        if X_trans is None and Y_trans is None:
            X_trans, Y_trans, _ = self.get_transformed_data(X, Y, V_A, V_B) 
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
            corr_X1_trials, _ = self.cal_corr_coe_trials(X_trials_shifted, Y1_trials, V_X, V_Y, avg=False)
            corr_X2_trials, _ = self.cal_corr_coe_trials(X_trials_shifted, Y2_trials, V_X, V_Y, avg=False)
            acc, _, _, _, _= utils.eval_compete_3D(corr_X1_trials, corr_X2_trials, TRAIN_WITH_ATT=True, message=False)
            acc_list.append(acc)
        return acc_list

    def get_train_test_data(self):
        train_list_folds, test_list_folds = utils.split_multi_mod_LVO([self.EEG_list, self.Stim_list], self.leave_out)
        if self.MASK:
            _, test_list_folds = utils.split_multi_mod_LVO([self.EEG_masked, self.Stim_masked], self.leave_out)
        T, _, nb_tasks = train_list_folds[0][0].shape
        assert len(train_list_folds) == len(test_list_folds) == self.nb_folds, "The number of folds is not correct."
        train_list_folds = [[data[:,:,self.task_train].transpose(2,0,1).reshape(T*len(self.task_train), -1) for data in EEGStim] for EEGStim in train_list_folds]
        # train_list_folds = [[data.transpose(2,0,1).reshape(T*nb_tasks, -1) for data in EEGStim] for EEGStim in train_list_folds]
        return train_list_folds, test_list_folds, nb_tasks

    def cross_val(self, PERMU_TEST=False):
        '''
        Cross-validation with leave-one-pair-out; For single-object dataset only
        '''
        train_list_folds, test_list_folds, nb_tasks = self.get_train_test_data()
        n_components = self.n_components
        nb_folds = self.nb_folds
        corr_train_fold = np.zeros((nb_folds, n_components))
        corr_test_fold = np.zeros((nb_folds, n_components, nb_tasks))
        rt_fold = np.zeros((nb_folds, 1, nb_tasks))
        corr_permu_fold = []
        forward_model_fold = []
        for idx in range(0, nb_folds):
            [EEG_train, Sti_train], [EEG_test, Sti_test] = train_list_folds[idx], test_list_folds[idx]
            corr_train_fold[idx,:], V_A_train, V_B_train, rt_of_not_nan = self.fit(EEG_train, Sti_train)
            print('Training, Fold {}: {}% of the data is not NaN'.format(idx, rt_of_not_nan*100))
            corr_test_fold[idx,:,:], rt_fold[idx,:,:] = self.cal_corr_coe_3D(EEG_test, Sti_test, V_A_train, V_B_train)
            forward_models = self.forward_model(EEG_test, V_A_train, Sti_test)
            forward_model_fold.append(forward_models)
            if PERMU_TEST:
                corr_permu_fold.append(self.permutation_test(EEG_test[:,:,0], Sti_test[:,:,0], V_A=V_A_train, V_B=V_B_train))
        if PERMU_TEST:
            corr_permu_all = np.concatenate(tuple(corr_permu_fold), axis=0)
            sig_corr_pool = self.calculate_sig_corr(corr_permu_all, nb_fold=nb_folds)
        else:
            sig_corr_pool = None
        if self.message:
            print('Average correlation coefficients of the top {} components on the training sets: {}'.format(n_components, np.average(corr_train_fold, axis=0)))
            print('Average correlation coefficients of the top {} components on the test sets: {}'.format(n_components, np.average(corr_test_fold, axis=0)))
            print('Significance level: {}'.format(sig_corr_pool))
        return corr_train_fold, corr_test_fold, sig_corr_pool, forward_model_fold

    def match_mismatch(self, trial_len, BOOTSTRAP=True, V_eeg=None, V_Stim=None, PERMU_TEST=True, overlap=0.5, given_start_points=None, nb_compete=1):
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
        rts_kept = []
        start_points = None if given_start_points is None else given_start_points
        for idx in range(0, self.nb_folds):
            [EEG_train, Sti_train], [EEG_test, Sti_test] = train_list_folds[idx], test_list_folds[idx]
            if V_eeg is None:
                _, V_eeg_train, V_feat_train, _ = self.fit(EEG_train, Sti_train)
            else:
                V_eeg_train, V_feat_train = V_eeg, V_Stim
            if not self.MASK:
                corr_match_eeg_i, corr_mismatch_eeg_i, start_points, X_trans_trials, Y_att_trans_trials, Y_compete_trans_trials = self.cal_corr_compete_trials(EEG_test, Sti_test, V_eeg_train, V_feat_train, BOOTSTRAP, trial_len, given_start_points=start_points, overlap=overlap, nb_compete=nb_compete) 
            else:
                corr_match_eeg_i, corr_mismatch_eeg_i, X_trans_trials, Y_att_trans_trials, Y_compete_trans_trials, rts = self.cal_corr_compete_mask_trials(EEG_test, Sti_test, V_eeg_train, V_feat_train, BOOTSTRAP, trial_len, overlap=overlap, nb_compete=nb_compete) 
            X_all_trials = X_trans_trials + X_all_trials
            Y_att_all_trials = Y_att_trans_trials + Y_att_all_trials
            Y_compete_all_trials = Y_compete_trans_trials + Y_compete_all_trials
            corr_match_eeg.append(corr_match_eeg_i)
            corr_mismatch_eeg.append(corr_mismatch_eeg_i)
            if self.MASK:
                rts_kept.append(rts)
        corr_match_eeg = np.concatenate(tuple(corr_match_eeg), axis=0)
        corr_mismatch_eeg = np.concatenate(tuple(corr_mismatch_eeg), axis=0)
        if self.MASK:
            rts_kept = np.concatenate(tuple(rts_kept), axis=0)
        if PERMU_TEST:
            # randomly choose nb_trials/nb_compete trials from the original trials to calculate the accuracy
            idx_trials = random.sample(range(len(X_all_trials)), len(X_all_trials)//nb_compete)
            X_reduced = [X_all_trials[i] for i in idx_trials]
            Y_att_reduced = [Y_att_all_trials[i] for i in idx_trials]
            Y_compete_reduced = [Y_compete_all_trials[i] for i in idx_trials]
            acc_permu_list = self.permutation_test_acc(X_reduced, Y_att_reduced, Y_compete_reduced)
        else:
            acc_permu_list = None
        return corr_match_eeg, corr_mismatch_eeg, acc_permu_list, start_points, rts_kept

    def mm_blocks(self, trial_len, BOOTSTRAP=True, overlap=0.9, block_len=90, block_ol=0.8, nb_compete=1):
        '''
        Match-Mismatch task with leave-one-pair-out
        Always train on match and try to distinguish match from mismatch 
        Mismatch is a random segment that is not shown on the screen
        '''
        train_list_folds, test_list_folds, _ = self.get_train_test_data()
        corr_match_dict = {}
        corr_mismatch_dict = {}
        rts_kept_dict = {}
        for idx in range(0, self.nb_folds):
            [EEG_train, Sti_train], [EEG_test, Sti_test] = train_list_folds[idx], test_list_folds[idx]
            _, V_eeg_train, V_feat_train, _ = self.fit(EEG_train, Sti_train)
            block_start_points = self.get_start_points(EEG_test.shape[0], block_len, BOOTSTRAP=False, overlap=block_ol)
            EEG_blocks = utils.into_trials(EEG_test, self.fs, block_len, start_points=block_start_points)
            Sti_blocks = utils.into_trials(Sti_test, self.fs, block_len, start_points=block_start_points)
            for i, (eeg, sti) in enumerate(zip(EEG_blocks, Sti_blocks)):
                if not self.MASK:
                    try:
                        corr_match_eeg_i, corr_mismatch_eeg_i, _, _, _, _ = self.cal_corr_compete_trials(eeg, sti, V_eeg_train, V_feat_train, BOOTSTRAP, trial_len, overlap=overlap, nb_compete=nb_compete) 
                    except:
                        continue
                else:
                    try:
                        corr_match_eeg_i, corr_mismatch_eeg_i, _, _, _, rts_i = self.cal_corr_compete_mask_trials(eeg, sti, V_eeg_train, V_feat_train, BOOTSTRAP, trial_len, overlap=overlap, nb_compete=nb_compete) 
                    except:
                        continue
                if i not in corr_match_dict:
                    corr_match_dict[i] = corr_match_eeg_i
                    corr_mismatch_dict[i] = corr_mismatch_eeg_i
                    rts_kept_dict[i] = rts_i if self.MASK else None
                else:
                    corr_match_dict[i] = np.concatenate((corr_match_dict[i], corr_match_eeg_i), axis=0)
                    corr_mismatch_dict[i] = np.concatenate((corr_mismatch_dict[i], corr_mismatch_eeg_i), axis=0)
                    if self.MASK:
                        rts_kept_dict[i] = np.concatenate((rts_kept_dict[i], rts_i), axis=0)
        return corr_match_dict, corr_mismatch_dict, rts_kept_dict

    def get_mismatch_pool(self, test_indices, V_X, V_Y, nb_tasks):
        '''
        Transform the stimulus of the videos that are not in the test set, to be used as a pool of
        competing segments. A segment taken from another video was, by construction, not on the
        screen during the match trial, and the size of the pool no longer depends on the length of
        the block, nor on the length of the test video.
        Note that the videos of the pool are the ones the filters were trained on. Only their
        stimulus is used here, which is not related in time to the EEG of the test video.
        Inputs:
        test_indices: indices of the videos that are left out for testing in the current fold
        Output:
        A list (one element per task) of lists of transformed stimuli, one per video of the pool
        '''
        EEG_list = self.EEG_masked if self.MASK else self.EEG_list
        Stim_list = self.Stim_masked if self.MASK else self.Stim_list
        pool_tasks = [[] for _ in range(nb_tasks)]
        for idx_video in range(self.nb_videos):
            if idx_video in test_indices:
                continue
            for task in range(nb_tasks):
                _, Y_trans, _, _ = self.get_transformed_data_indexed(EEG_list[idx_video][:,:,task], Stim_list[idx_video][:,:,task], V_X, V_Y)
                pool_tasks[task].append(Y_trans)
        return pool_tasks

    def cal_corr_compete_blocks(self, X, Y_att, V_X, V_Y, block_start_points, block_len, trial_len, Y_pool_tasks=None, BOOTSTRAP=True, BTfactor=2, overlap=0.9, nb_compete=1, guard_len=None):
        '''
        Same purpose as cal_corr_compete_mask_trials, but the competing segments are not drawn from
        the block the match trial belongs to. The mask is applied to the whole recording first, and
        each block is then located on the retained time axis, such that a match trial still comes
        from its own block, while the diversity of the competing segments no longer shrinks when
        the block gets short after masking.
        Inputs:
        block_start_points: start points of the blocks, on the original time axis
        block_len/trial_len: length of the blocks/trials in seconds
        Y_pool_tasks: if given (see get_mismatch_pool), the competing segments are drawn from this
        pool of other videos; if None, they are drawn from the test video itself, at least
        guard_samples away from the matched segment
        guard_len: only used when Y_pool_tasks is None; minimum distance (s) between a match trial
        and its competitors, defaults to trial_len, i.e., no overlap at all. Match trials for which
        the test video is too short to provide such a competitor are discarded
        Outputs:
        Dictionaries, indexed by the block index, of the correlation coefficients with the matched
        and the competing segments, and of the ratio of data kept in each block
        '''
        nb_tasks = X.shape[2]
        T = X.shape[0]
        block_len_samples = int(block_len*self.fs)
        trial_len_samples = int(trial_len*self.fs)
        guard_samples = trial_len_samples if guard_len is None else int(guard_len*self.fs)
        # transform the whole recording once per task and keep track of the samples that survived the mask
        X_trans_tasks = []
        Y_trans_tasks = []
        idx_kept_tasks = []
        for task in range(nb_tasks):
            X_trans, Y_trans, idx_kept, _ = self.get_transformed_data_indexed(X[:,:,task], Y_att[:,:,task], V_X, V_Y)
            X_trans_tasks.append(X_trans)
            Y_trans_tasks.append(Y_trans)
            idx_kept_tasks.append(idx_kept)
        corr_match_blocks = {}
        corr_mismatch_blocks = {}
        rts_blocks = {}
        for idx_block, block_start in enumerate(block_start_points):
            bounds = [utils.retained_block_bounds(idx_kept, block_start, block_len_samples) for idx_kept in idx_kept_tasks]
            # skip the block if any task does not have enough data left in it after masking
            if any(hi - lo <= trial_len_samples for lo, hi in bounds):
                continue
            match_starts_tasks = []
            for lo, hi in bounds:
                starts = self.get_start_points(hi-lo, trial_len, BOOTSTRAP=BOOTSTRAP, BTfactor=BTfactor, overlap=overlap)
                starts = lo + starts[starts + trial_len_samples <= hi - lo]
                match_starts_tasks.append(starts)
            # keep the same number of match trials for every task, since the results are stacked along the task axis
            min_nb_trials = min([len(starts) for starts in match_starts_tasks])
            if min_nb_trials == 0:
                continue
            indices = [np.sort(random.sample(range(len(starts)), min_nb_trials)) for starts in match_starts_tasks]
            match_starts_tasks = [starts[ind] for starts, ind in zip(match_starts_tasks, indices)]
            corr_match_tasks = []
            corr_mismatch_tasks = []
            rts_tasks = []
            for task in range(nb_tasks):
                X_trans, Y_trans = X_trans_tasks[task], Y_trans_tasks[task]
                if Y_pool_tasks is None:
                    match_starts, mismatch_starts = utils.sample_mismatch_starts(match_starts_tasks[task], X_trans.shape[0], trial_len_samples, nb_compete, guard_samples)
                    Y_mismatch_trials = utils.into_trials(Y_trans, self.fs, trial_len, start_points=mismatch_starts)
                else:
                    Y_pool = Y_pool_tasks[task]
                    match_starts, picks = utils.sample_mismatch_from_pool(match_starts_tasks[task], [Y.shape[0] for Y in Y_pool], trial_len_samples, nb_compete)
                    Y_mismatch_trials = [Y_pool[idx_pool][start:start+trial_len_samples, :] for idx_pool, start in picks]
                if len(match_starts) == 0:
                    continue
                X_trials = utils.into_trials(X_trans, self.fs, trial_len, start_points=match_starts)
                Y_match_trials = utils.into_trials(Y_trans, self.fs, trial_len, start_points=match_starts)
                corr_match, _ = self.cal_corr_coe_trials(X_trials, Y_match_trials, avg=False)
                corr_mismatch, _ = self.cal_corr_coe_trials(X_trials, Y_mismatch_trials, avg=False)
                corr_match_tasks.append(corr_match)
                corr_mismatch_tasks.append(corr_mismatch)
                lo, hi = bounds[task]
                rts_tasks.append((hi-lo)/(min(T, block_start+block_len_samples)-block_start))
            if len(corr_match_tasks) < nb_tasks:
                continue
            nb_pairs = min([corr.shape[0] for corr in corr_match_tasks])
            corr_match_blocks[idx_block] = np.stack([corr[:nb_pairs] for corr in corr_match_tasks], axis=2)
            corr_mismatch_blocks[idx_block] = np.stack([corr[:nb_pairs] for corr in corr_mismatch_tasks], axis=2)
            rts_blocks[idx_block] = np.array(rts_tasks).reshape(1, -1)
        return corr_match_blocks, corr_mismatch_blocks, rts_blocks

    def mm_blocks_global_mismatch(self, trial_len, BOOTSTRAP=True, overlap=0.9, block_len=90, block_ol=0.8, nb_compete=1, mismatch_scope='other_videos', guard_len=None):
        '''
        Same as mm_blocks, except that the competing segments are not drawn from the block the
        match trial belongs to (see cal_corr_compete_blocks)
        mismatch_scope: 'other_videos' to draw them from the videos that are not being tested in
        the current fold, 'test_video' to draw them from the test video itself, at least guard_len
        seconds away from the matched segment
        '''
        assert mismatch_scope in ['other_videos', 'test_video'], "mismatch_scope should be 'other_videos' or 'test_video'."
        train_list_folds, test_list_folds, nb_tasks = self.get_train_test_data()
        corr_match_dict = {}
        corr_mismatch_dict = {}
        rts_kept_dict = {}
        for idx in range(0, self.nb_folds):
            [EEG_train, Sti_train], [EEG_test, Sti_test] = train_list_folds[idx], test_list_folds[idx]
            _, V_eeg_train, V_feat_train, _ = self.fit(EEG_train, Sti_train)
            test_indices = range(idx*self.leave_out, (idx+1)*self.leave_out)
            Y_pool_tasks = self.get_mismatch_pool(test_indices, V_eeg_train, V_feat_train, nb_tasks) if mismatch_scope == 'other_videos' else None
            block_start_points = self.get_start_points(EEG_test.shape[0], block_len, BOOTSTRAP=False, overlap=block_ol)
            corr_match_blocks, corr_mismatch_blocks, rts_blocks = self.cal_corr_compete_blocks(EEG_test, Sti_test, V_eeg_train, V_feat_train, block_start_points, block_len, trial_len, Y_pool_tasks=Y_pool_tasks, BOOTSTRAP=BOOTSTRAP, overlap=overlap, nb_compete=nb_compete, guard_len=guard_len)
            for i in corr_match_blocks:
                if i not in corr_match_dict:
                    corr_match_dict[i] = corr_match_blocks[i]
                    corr_mismatch_dict[i] = corr_mismatch_blocks[i]
                    rts_kept_dict[i] = rts_blocks[i]
                else:
                    corr_match_dict[i] = np.concatenate((corr_match_dict[i], corr_match_blocks[i]), axis=0)
                    corr_mismatch_dict[i] = np.concatenate((corr_mismatch_dict[i], corr_mismatch_blocks[i]), axis=0)
                    rts_kept_dict[i] = np.concatenate((rts_kept_dict[i], rts_blocks[i]), axis=0)
        return corr_match_dict, corr_mismatch_dict, rts_kept_dict


class GeneralizedCCA:
    '''
    Perform GCCA on data of different subjects. If subjects have multi-modal data, then the data are concatenated along the channel axis.
    '''
    def __init__(self, EEG_list, fs, L, offset, hankelized=False, dim_list=None, task_train=[2,3], leave_out=1, n_components=5, regularization='lwcov', message=True, signifi_level=True, n_permu=500, p_value=0.05, save_W_perfold=True, EEG_list_masked=None):
        '''
        EEG_list: list of EEG data, each element is a T(#sample)xDx(#channel)xN(#subj)x(#task) array corresponding to a video 
        fs: Sampling rate
        L: If use (spatial-) temporal filter, the number of taps
        offset: If use (spatial-) temporal filter, the offset of the time lags
        hankelized: If the data is already hankelized because of regression
        dim_list: If 'EEG' contains data from multiple sources that have significantly different scales, then specify the dimensions of each source
        leave_out: Number of pairs to leave out for leave-one-pair-out cross-validation
        n_components: Number of components to be returned
        regularization: Regularization of the estimated covariance matrix
        message: If print message
        signifi_level: If calculate significance level
        n_permu: Number of permutations for significance level calculation
        p_value: P-value for significance level calculation
        save_W_perfold: If save the weights per fold
        '''
        self.EEG_list = EEG_list
        self.fs = fs
        self.L = L
        self.offset = offset
        self.hankelized = hankelized
        self.dim_list = dim_list
        self.task_train = [t-1 for t in task_train] # convert to 0-indexed
        self.leave_out = leave_out
        self.n_components = n_components
        self.regularization = regularization
        self.message = message
        self.signifi_level = signifi_level
        self.n_permu = n_permu
        self.p_value = p_value
        self.save_W_perfold = save_W_perfold
        if self.save_W_perfold:
            self.test_list = []
            self.W_train_list = []

        self.nb_videos = len(self.EEG_list)
        assert self.nb_videos%self.leave_out == 0, "The number of videos should be a multiple of the leave_out parameter."
        self.nb_folds = self.nb_videos//self.leave_out
        self.EEG_list_masked = EEG_list_masked

    def apply_mask(self, X):
        idx_not_nan = ~np.isnan(X).any(axis=1)
        X = X[idx_not_nan, :]
        return X

    def get_train_test_data(self, MASKED=False):
        train_list_folds, test_list_folds = utils.split_multi_mod_LVO([self.EEG_list], self.leave_out) if not MASKED else utils.split_multi_mod_LVO([self.EEG_list_masked], self.leave_out)
        T, _, N, nb_tasks = train_list_folds[0][0].shape
        assert len(train_list_folds) == len(test_list_folds) == self.nb_folds, "The number of folds is not correct."
        train_list_folds = [[np.concatenate(tuple([data[0][:,:,:,t] for t in self.task_train]), axis=0)] for data in train_list_folds]
        return train_list_folds, test_list_folds, nb_tasks

    def fit(self, X_stack):
        '''
        Inputs:
        X_stack: stacked (along axis 2) data of different subjects
        Outputs:
        W_stack: weights with shape DLxNxn_components
        S: shared subspace with shape Txn_components
        F_stack: forward model with shape Dxn_components calculated from the shared subspace in the training set
        lam: eigenvalues, related to mean squared error (not used)
        '''
        T, D, N = X_stack.shape
        L = self.L
        dim_list_extended = [d*L for d in self.dim_list]*N if self.dim_list is not None else [D*L]*N
        # From [X1; X2; ... XN] to [X1 X2 ... XN]
        # each column represents a variable, while the rows contain observations
        X_list = [utils.block_Hankel(X_stack[:,:,n], L, self.offset) for n in range(N)]
        X = np.concatenate(tuple(X_list), axis=1)
        X = self.apply_mask(X) # remove rows with NaN
        X_center = X - np.mean(X, axis=0, keepdims=True)
        Rxx, _ = utils.get_cov_mtx(X, dim_list_extended, self.regularization)
        Dxx = np.zeros_like(Rxx)
        for n in range(N):
            Dxx[n*D*L:(n+1)*D*L, n*D*L:(n+1)*D*L] = Rxx[n*D*L:(n+1)*D*L, n*D*L:(n+1)*D*L]
        lam, W = eigh(Dxx, Rxx, subset_by_index=[0,self.n_components-1]) # automatically ascend
        Lam = np.diag(lam)
        # Right scaling
        W = W @ sqrtm(LA.inv(Lam.T @ W.T @ Rxx * T @ W @ Lam))
        # Shared subspace
        S = X_center@W@Lam
        # Forward models
        F_redun = T * Dxx @ W
        # Reshape W as (DL*n_components*N)
        W_stack = np.reshape(W, (N,D*L,-1))
        W_stack = np.transpose(W_stack, [1,0,2])
        F_redun_stack = np.reshape(F_redun, (N,D*L,-1))
        F_redun_stack = np.transpose(F_redun_stack, [1,0,2])
        F_stack = utils.F_organize(F_redun_stack, L, self.offset, avg=True)
        return W_stack, S, F_stack, lam

    def fit_corrca(self, X_stack):
        T, _, N = X_stack.shape
        X_list = [utils.block_Hankel(X_stack[:,:,n], self.L, self.offset) for n in range(N)]
        X_list = [self.apply_mask(X) for X in X_list] # remove rows with NaN
        X = np.stack(X_list, axis=2)
        X_center = X - np.mean(X, axis=0, keepdims=True)
        _, D, _ = X.shape
        Rw = np.zeros([D,D])
        for n in range(N):
            if self.regularization == 'lwcov':
                Rw += LedoitWolf().fit(X[:,:,n]).covariance_
            else:
                Rw += np.cov(np.transpose(X[:,:,n])) # Inside np.cov: observations in the columns
        if self.regularization == 'lwcov':
            Rt = N**2*LedoitWolf().fit(np.average(X, axis=2)).covariance_
        else:
            Rt = N**2*np.cov(np.transpose(np.average(X, axis=2)))
        Rb = (Rt - Rw)/(N-1)
        ISC, W = eigh(Rb, Rw, subset_by_index=[D-self.n_components,D-1])
        ISC = np.squeeze(np.fliplr(np.expand_dims(ISC, axis=0)))
        W = np.fliplr(W)
        # right scaling
        Lam = np.diag(1/(ISC*(N-1)+1))
        W = W @ sqrtm(LA.inv(Lam.T @ W.T @ Rt * T @ W @ Lam))
        # shared subspace
        S = np.sum(X_center, axis=2) @ W @ Lam
        # Forward models
        F_redun = T * Rw @ W / N
        F = utils.F_organize(F_redun, self.L, self.offset)
        return W, S, F, Lam

    def get_transformed_data(self, X_stack, W_stack):
        '''
        Get the transformed data
        '''
        _, _, N = X_stack.shape
        Hankellist = [np.expand_dims(self.apply_mask(utils.block_Hankel(X_stack[:,:,n], self.L, self.offset)), axis=2) for n in range(N)]
        Hankel_center = [hankel - np.mean(hankel, axis=0, keepdims=True) for hankel in Hankellist]
        X_center = np.concatenate(tuple(Hankel_center), axis=2)
        if W_stack is not None:
            if np.ndim (W_stack) == 2: # for correlated component analysis
                W_stack = np.expand_dims(W_stack, axis=1)
                W_stack = np.repeat(W_stack, N, axis=1)
            X_trans = np.einsum('tdn,dkn->tkn', X_center, np.transpose(W_stack, (0,2,1)))
        else:
            X_trans = X_center
        return X_trans
    
    def get_transformed_data_4D(self, X_stack, W_stack):
        assert np.ndim(X_stack) == 4, "The input data should be 4D."
        nb_tasks = X_stack.shape[3]
        X_trans_all = []
        for task in range(nb_tasks):
            X_trans = self.get_transformed_data(X_stack[:,:,:,task], W_stack)
            X_trans_all.append(X_trans)
        X_trans = np.stack(X_trans_all, axis=3) # (T, DL, N, nb_tasks)
        return X_trans

    def cal_avg_corr_coe(self, X_stack, W_stack=None):
        '''
        Calculate the inter-subject correlation (average pairwise correlation)
        '''
        if W_stack is None:
            X_trans = X_stack
        else:
            X_trans = self.get_transformed_data(X_stack, W_stack)
        _, _, N = X_trans.shape
        n_components = self.n_components
        corr_mtx_stack = np.zeros((N,N,n_components))
        avg_corr = np.zeros(n_components)
        for component in range(n_components):
            corr_mtx_stack[:,:,component] = np.corrcoef(X_trans[:,component,:], rowvar=False)
            avg_corr[component] = np.sum(corr_mtx_stack[:,:,component]-np.eye(N))/N/(N-1)
        return avg_corr
    
    def cal_avg_cov(self, X_stack, W_stack=None):
        '''
        Calculate the inter-subject covariance (average pairwise covariance)
        '''
        if W_stack is None:
            X_trans = X_stack
        else:
            X_trans = self.get_transformed_data(X_stack, W_stack)
        _, _, N = X_trans.shape
        n_components = self.n_components
        cov_mtx_stack = np.zeros((N,N,n_components))
        avg_cov = np.zeros(n_components)
        for component in range(n_components):
            cov_mtx_stack[:,:,component] = np.cov(X_trans[:,component,:], rowvar=False)
            cov_diag = np.diag(np.diag(cov_mtx_stack[:,:,component]))
            avg_cov[component] = np.sum(cov_mtx_stack[:,:,component]-cov_diag)/N/(N-1)
        return avg_cov

    def cal_avg_corr_coe_4D(self, X_stack, W_stack=None):
        assert np.ndim(X_stack) == 4, "The input data should be 4D."
        corr_all = []
        for task in range(X_stack.shape[3]):
            avg_corr = self.cal_avg_corr_coe(X_stack[:,:,:,task], W_stack)
            corr_all.append(avg_corr)
        avg_corr = np.stack(corr_all, axis=1) # (n_components, nb_tasks)
        return avg_corr

    def cal_avg_cov_4D(self, X_stack, W_stack=None):
        assert np.ndim(X_stack) == 4, "The input data should be 4D."
        cov_all = []
        for task in range(X_stack.shape[3]):
            avg_cov = self.cal_avg_cov(X_stack[:,:,:,task], W_stack)
            cov_all.append(avg_cov)
        avg_cov = np.stack(cov_all, axis=1) # (n_components, nb_tasks)
        return avg_cov

    def cal_avg_corr_coe_trials(self, X_trials, W_stack=None, avg=True):
        Four_D = np.ndim(X_trials[0]) == 4
        corr_coe_trials = [self.cal_avg_corr_coe(X, W_stack) for X in X_trials] if not Four_D else [self.cal_avg_corr_coe_4D(X, W_stack) for X in X_trials]
        corr_coe = np.stack(corr_coe_trials, axis=0)
        if avg:
            corr_coe = np.mean(corr_coe, axis=0)
        return corr_coe

    def permutation_test(self, X_stack, W_stack, PHASE_SCRAMBLE=True):
        corr_coe_topK = np.empty((0, self.n_components))
        cov_topK = np.empty((0, self.n_components))
        X_trans = self.get_transformed_data(X_stack, W_stack)
        for i in tqdm(range(self.n_permu)):
            X_shuffled = utils.circular_permute_3D(X_trans) if not PHASE_SCRAMBLE else utils.phase_scramble_3D(X_trans)
            corr_coe = self.cal_avg_corr_coe(X_shuffled)
            cov = self.cal_avg_cov(X_shuffled)
            corr_coe_topK = np.concatenate((corr_coe_topK, np.expand_dims(corr_coe, axis=0)), axis=0)
            cov_topK = np.concatenate((cov_topK, np.expand_dims(cov, axis=0)), axis=0)
        return corr_coe_topK, cov_topK

    def calculate_sig_corr(self, corr_trials, nb_fold=1):
        assert self.n_components*self.n_permu*nb_fold == corr_trials.shape[0]*corr_trials.shape[1]
        sig_idx = -int(self.n_permu*self.p_value*self.n_components*nb_fold)
        corr_trials = np.sort(abs(corr_trials), axis=None)
        return corr_trials[sig_idx]
    
    def calculate_sig_multi_comp(self, corr_trials, nb_fold=1, nb_components=2):
        corr_agg = np.sum(corr_trials[:,:nb_components], axis=1)
        corr_agg = np.sort(abs(corr_agg), axis=None)
        assert self.n_permu*nb_fold == corr_agg.shape[0]
        sig_idx = -int(self.n_permu*self.p_value*nb_fold)
        return corr_agg[sig_idx]

    def cross_val(self, CORRCA=True, sig_components=2):
        train_list_folds, test_list_folds, nb_tasks = self.get_train_test_data()
        if self.EEG_list_masked is not None:
            _, test_list_folds, _ = self.get_train_test_data(MASKED=True)
        n_components = self.n_components
        nb_folds = self.nb_folds
        corr_train_fold = np.zeros((nb_folds, n_components))
        corr_test_fold = np.zeros((nb_folds, n_components, nb_tasks))
        cov_train_fold = np.zeros((nb_folds, n_components))
        cov_test_fold = np.zeros((nb_folds, n_components, nb_tasks))
        corr_permu_fold = []
        cov_permu_fold = []
        for idx in range(0, nb_folds):
            [EEG_train], [EEG_test] = train_list_folds[idx], test_list_folds[idx]
            W_train, _, F, _ = self.fit(EEG_train) if not CORRCA else self.fit_corrca(EEG_train)
            corr_train_fold[idx,:] = self.cal_avg_corr_coe(EEG_train, W_train)
            corr_test_fold[idx,:,:] = self.cal_avg_corr_coe_4D(EEG_test, W_train)
            cov_train_fold[idx,:] = self.cal_avg_cov(EEG_train, W_train)
            cov_test_fold[idx,:,:] = self.cal_avg_cov_4D(EEG_test, W_train)
            if self.save_W_perfold:
                self.test_list.append(EEG_test)
                self.W_train_list.append(W_train)
            if self.signifi_level:
                corr_permu, cov_permu = self.permutation_test(EEG_test[:,:,:,0], W_train, PHASE_SCRAMBLE=False)
                corr_permu_fold.append(corr_permu)
                cov_permu_fold.append(cov_permu)
        if self.signifi_level:
            sig_corr_fold = [self.calculate_sig_multi_comp(corr_permu, nb_components=sig_components) for corr_permu in corr_permu_fold]
            corr_permu_all = np.concatenate(tuple(corr_permu_fold), axis=0)
            sig_corr_pool = self.calculate_sig_multi_comp(corr_permu_all, nb_fold=nb_folds, nb_components=sig_components)
            cov_permu_all = np.concatenate(tuple(cov_permu_fold), axis=0)
            sig_cov_pool = self.calculate_sig_multi_comp(cov_permu_all, nb_fold=nb_folds, nb_components=1)
        else:
            sig_corr_fold = None
            sig_corr_pool = None
            sig_cov_pool = None
        if self.message:
            # print('Average ISC of the top {} components on the training sets: {}'.format(n_components, np.average(corr_train_fold, axis=0)))
            print('Average ISC of the top {} components on the test sets: {}'.format(n_components, np.average(corr_test_fold, axis=0)))
            # print('Average ISCov of the top {} components on the training sets: {}'.format(n_components, np.average(cov_train_fold, axis=0)))
            print('Average ISCov of the top {} components on the test sets: {}'.format(n_components, np.average(cov_test_fold, axis=0)))
            print('Significance level (Corr, CC1+CC2): {}'.format(sig_corr_pool))
            print('Significance level (Cov): {}'.format(sig_cov_pool))
        return corr_train_fold, corr_test_fold, sig_corr_fold, sig_corr_pool, F, cov_train_fold, cov_test_fold
    
    def cross_val_trials(self, BOOTSTRAP, trial_len, given_start_points=None, BTfactor=2, overlap=0.5, CORRCA=True):
        train_list_folds, test_list_folds, nb_tasks = self.get_train_test_data()
        if self.EEG_list_masked is not None:
            _, test_list_folds, _ = self.get_train_test_data(MASKED=True)
        n_components = self.n_components
        nb_folds = self.nb_folds
        corr_test_folds = []
        for idx in range(0, nb_folds):
            [EEG_train], [EEG_test] = train_list_folds[idx], test_list_folds[idx]
            W_train, _, F, _ = self.fit(EEG_train) if not CORRCA else self.fit_corrca(EEG_train)
            if self.save_W_perfold:
                self.test_list.append(EEG_test)
                self.W_train_list.append(W_train)
            T = EEG_test.shape[0]
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
            EEG_trials = utils.into_trials(EEG_test, self.fs, trial_len, start_points=start_points)
            corr_trials = self.cal_avg_corr_coe_trials(EEG_trials, W_train, avg=False)
            corr_test_folds.append(corr_trials)
        if self.message:
            print('Average ISC of the top {} components on the test sets: {}'.format(n_components, np.average(np.concatenate(corr_test_folds, axis=0), axis=0)))
        return corr_test_folds, start_points, F
    
    def get_enhanced_data(self, CORRCA=True):
        assert self.leave_out == 1, "This function only works for leave-one-pair-out cross-validation."
        train_list_folds, test_list_folds, nb_tasks = self.get_train_test_data()
        n_components = self.n_components
        nb_folds = self.nb_folds
        enhanced_list = []
        for idx in range(0, nb_folds):
            [EEG_train], [EEG_test] = train_list_folds[idx], test_list_folds[idx]
            W_train, _, _, _ = self.fit(EEG_train) if not CORRCA else self.fit_corrca(EEG_train)
            EEG_test_trans = self.get_transformed_data_4D(EEG_test, W_train)
            EEG_enhanced = np.mean(EEG_test_trans, axis=2)  # Average across subjects
            enhanced_list.append(EEG_enhanced)
        return enhanced_list

    def forward_model(self):
        assert self.save_W_perfold, "Please set save_W_perfold=True when initializing the class."
        forward_model_list = []
        for EEG_test, W_train in zip(self.test_list, self.W_train_list):
            fm_tasks = []
            nb_tasks = EEG_test.shape[-1]
            for task in range(nb_tasks):
                EEG_trans = self.get_transformed_data(EEG_test[..., task], W_train)
                EEG_task = self.get_transformed_data(EEG_test[..., task], None)
                EEG_flat = np.concatenate([EEG_task[:, :, n] for n in range(nb_tasks)], axis=0)
                EEG_trans_flat = np.concatenate([EEG_trans[:, :, n] for n in range(nb_tasks)], axis=0)
                fm_redun = EEG_flat.T @ EEG_trans_flat @ LA.inv(EEG_trans_flat.T @ EEG_trans_flat)
                fm_tasks.append(utils.F_organize(fm_redun, self.L, self.offset))
            F = np.stack(fm_tasks, axis=2)
            forward_model_list.append(F)
        return forward_model_list


class BlockCCA:
    def __init__(self, list_X, list_Y, fs, L, offset, sharefilter='set', hankelized=False, leave_out=1, n_components=5, regularization='lwcov', message=True, signifi_level=True, n_permu=500, p_value=0.05, save_W_perfold=True):
        '''
        list_X, list_Y: lists of data, each element is a T(#sample)xDx(#channel)xN(#subj) array corresponding to a video 
        fs: Sampling rate
        L: If use (spatial-) temporal filter, the number of taps
        offset: If use (spatial-) temporal filter, the offset of the time lags
        sharefilter: Share filters for the views in each set if 'set'; Share filters for all views if 'all'; Do not share filters if 'none'
        hankelized: If the data is already hankelized because of regression
        leave_out: Number of pairs to leave out for leave-one-pair-out cross-validation
        n_components: Number of components to be returned
        regularization: Regularization of the estimated covariance matrix
        message: If print message
        signifi_level: If calculate significance level
        n_permu: Number of permutations for significance level calculation
        p_value: P-value for significance level calculation
        save_W_perfold: If save the weights per fold
        '''
        self.list_X = list_X
        self.list_Y = list_Y
        self.fs = fs
        self.L = L
        self.offset = offset
        self.sharefilter = sharefilter
        self.hankelized = hankelized
        self.leave_out = leave_out
        self.n_components = n_components
        self.regularization = regularization
        self.message = message
        self.signifi_level = signifi_level
        self.n_permu = n_permu
        self.p_value = p_value
        self.save_W_perfold = save_W_perfold
        if self.save_W_perfold:
            self.test_list = []
            self.W_train_list = []

    def hankelize_data(self, X_stack, concat=True, center=True):
        _, _, N = X_stack.shape
        if not self.hankelized:
            X_stack = utils.hankelize_data_multisub(X_stack, self.L, self.offset)
        dim_list = [X_stack.shape[1]]*N
        if concat:
            X_stack = np.concatenate(tuple([X_stack[:,:,n] for n in range(N)]), axis=1)
        if center:
            X_stack = X_stack - np.mean(X_stack, axis=0, keepdims=True)
        return X_stack, dim_list

    def fit(self, X_stack, Y_stack):
        Nx = X_stack.shape[2]
        Ny = Y_stack.shape[2]
        X_center, dim_list_X = self.hankelize_data(X_stack)
        Y_center, dim_list_Y = self.hankelize_data(Y_stack)
        agg = np.concatenate((X_center, Y_center), axis=1)
        dim_list = dim_list_X + dim_list_Y
        R, D = utils.get_cov_mtx(agg, dim_list, self.regularization)
        dimX = sum(dim_list_X)
        # R[:dimX, :dimX] = np.diag(np.diag(R[:dimX, :dimX]))
        R[:dimX, :dimX] = 0
        R[dimX:, dimX:] = 0
        R = R + D
        if self.sharefilter == 'set':
            Ix = [np.eye(d) for d in dim_list_X]
            Ix = np.concatenate(tuple(Ix), axis=0)
            Iy = [np.eye(d) for d in dim_list_Y]
            Iy = np.concatenate(tuple(Iy), axis=0)
            I = block_diag(Ix, Iy)
            R = I.T @ R @ I
            D = I.T @ D @ I
            dim_list = [Ix.shape[1], Iy.shape[1]]
        elif self.sharefilter == 'all':
            I = [np.eye(d) for d in dim_list]
            I = np.concatenate(tuple(I), axis=0)
            R = I.T @ R @ I
            D = I.T @ D @ I
            dim_list = [I.shape[1]]
        elif self.sharefilter == 'none':
            pass
        else:
            raise ValueError("Invalid value for sharefilter. Choose from 'set', 'all', or 'none'.")
        # _, W = eigh(D, R, subset_by_index=[0,self.n_components-1]) # automatically ascending
        _, W = eigh(R, D, subset_by_index=[D.shape[0]-self.n_components, D.shape[0]-1])
        W = np.fliplr(W) # descending order
        # divide W based on dim_list
        W_list = np.vsplit(W, np.cumsum(dim_list)[:-1])
        if len(W_list) == 2:
            assert self.sharefilter == 'set', "If there are only two views, then sharefilter should be 'set'."
            W_list = [W_list[0]] * Nx + [W_list[1]] * Ny
        elif len(W_list) == 1:
            assert self.sharefilter == 'all', "If there is only one view, then sharefilter should be 'all'."
            W_list = [W_list[0]] * (Nx + Ny)
        else:
            assert len(W_list) == Nx + Ny, "The number of views does not match for mode 'none'."
        W = np.stack(W_list, axis=2) # (D_total, n_components, N_total)
        return W
    
    def get_transformed_data(self, X_stack, Y_stack, W):
        X_stack_center, _ = self.hankelize_data(X_stack, concat=False)
        Y_stack_center, _ = self.hankelize_data(Y_stack, concat=False)
        Nx = X_stack_center.shape[2]
        Ny = Y_stack_center.shape[2]
        assert W.shape[2] == Nx + Ny, "The number of views does not match."
        Wx = W[:,:,:Nx]
        Wy = W[:,:,Nx:]
        X_trans = np.einsum('tdn,dkn->tkn', X_stack_center, Wx)
        Y_trans = np.einsum('tdn,dkn->tkn', Y_stack_center, Wy)
        return X_trans, Y_trans

    def cal_avg_corr_coe(self, X_stack, Y_stack, W_stack=None):
        '''
        Calculate the inter-subject correlation (average pairwise correlation)
        '''
        if W_stack is None:
            X_trans, Y_trans = X_stack, Y_stack
        else:
            X_trans, Y_trans = self.get_transformed_data(X_stack, Y_stack, W_stack)
        Nx = X_trans.shape[2]
        Ny = Y_trans.shape[2]
        n_components = self.n_components
        corr_tensor = np.zeros((Nx + Ny, Nx + Ny, n_components))
        for component in range(n_components):
            agg_trans = np.hstack((X_trans[:,component,:], Y_trans[:,component,:]))
            corr_tensor[:,:,component] = np.corrcoef(agg_trans, rowvar=False)
        # cross_corr_tensor = corr_tensor[Nx:, :Nx, :]
        # avg_corr = np.mean(cross_corr_tensor, axis=(0,1))
        return corr_tensor
    
    def cross_val(self):
        train_list_folds, test_list_folds = utils.split_multi_mod_LVO([self.list_X, self.list_Y], self.leave_out)
        n_components = self.n_components
        nb_folds = len(train_list_folds)
        corr_train_folds = []
        corr_test_folds = []
        for idx in range(0, nb_folds):
            [setA_train, setB_train], [setA_test, setB_test] = train_list_folds[idx], test_list_folds[idx]
            W_train = self.fit(setA_train, setB_train)
            corr_train_folds.append(self.cal_avg_corr_coe(setA_train, setB_train, W_train))
            corr_test_folds.append(self.cal_avg_corr_coe(setA_test, setB_test, W_train))
            if self.save_W_perfold:
                self.test_list.append(test_list_folds[idx])
                self.W_train_list.append(W_train)
        return corr_train_folds, corr_test_folds
    
    def forward_model(self):
        assert self.save_W_perfold, "Please set save_W_perfold=True when initializing the class."
        fm_A_list = []
        fm_B_list = []
        for set_AB_test, W_train in zip(self.test_list, self.W_train_list):
            setA_test, setB_test = set_AB_test
            setA_trans, setB_trans = self.get_transformed_data(setA_test, setB_test, W_train)
            shared_A = np.mean(setA_trans, axis=2)
            shared_B = np.mean(setB_trans, axis=2)
            F_A = np.transpose(setA_test, (1, 2, 0)) @ shared_A @ LA.inv(shared_A.T @ shared_A)
            F_B = np.transpose(setB_test, (1, 2, 0)) @ shared_B @ LA.inv(shared_B.T @ shared_B)
            fm_A_list.append(F_A.mean(axis=1)) # average across subjects
            fm_B_list.append(F_B.mean(axis=1)) # average across subjects
        return fm_A_list, fm_B_list