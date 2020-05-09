#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 11:32:51 2018

@author: ravi
"""

import numpy as np
import h5py
import itertools
import sys
import scipy.signal as scisig

from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from scipy import interpolate
from scipy.signal import butter, filtfilt

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange', 'm', 'g', 'k'])

def pre_emp(x, coeff=0.95):

    assert len(x.shape)==1
    
    start_val = 0
    pre_emped = np.zeros((len(x),), dtype=np.float64)

    for i in range(len(x)):
        pre_emped[i] = x[i] - coeff*start_val
        start_val = x[i]

    return pre_emped

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5*fs
    normal_cutoff = cutoff / nyq
    b,a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def smooth(x,window_len=7,window='hanning'):
    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = eval('np.'+window+'(window_len)')
    y = np.convolve(w/w.sum(),s,mode='same')
    return y[window_len-1:-1*(window_len-1)]

def smooth_contour(data, window=3):
    for i in range(data.shape[0]):
        x = smooth(data[i], window)
        data[i] = x[window-1:-1*(window-1)]
    return data

def generate_context(features, axis=0, context=1):
    """
    Axis specifies the dimension along which the features expand
    """
    
    backward = features.copy()
    forward = features.copy()
    if axis==0:
        for c in range(context):
            backward = np.roll(backward, 1, axis=1)
            forward = np.roll(forward, -1, axis=1)
            backward[:,0] = 0
            forward[:,-1] = 0
            features = np.concatenate((backward, features, forward), axis=axis)
            
    else:
        for c in range(context):
            backward = np.roll(backward, 1, axis=0)
            forward = np.roll(forward, -1, axis=0)
            backward[0,:] = 0
            forward[-1,:] = 0
            features = np.concatenate((backward, features, forward), axis=axis)

    return features

def generate_interpolation(f0):
#    f0 = scisig.medfilt(f0, kernel_size=3)
    nz_idx = np.where(f0>0.0)[0]
    mnz = []
    fnz = []
    
    if 0 not in nz_idx:
        mnz = [0]
        fnz = [f0[nz_idx[0]]]
    
    mnz.extend(nz_idx.tolist())
    fnz.extend(f0[nz_idx].tolist())
    
    if len(f0) - 1 not in nz_idx:
        mnz.extend([len(f0)-1])
        fnz.extend([f0[nz_idx[-1]]])
    
    interp = interpolate.interp1d(np.asarray(mnz), np.asarray(fnz))
    
    x = np.arange(0, len(f0))
    y = interp(x)
    return y

def concat_features_f0_mom(data, keep_norm=False, shuffle=False, keep_tar=False, energy=False):
    features_src = data['src_straight_cep_feat']
    f0_src = data['src_f0_feat']
    if energy:
        ec_src = data['src_ec_feat']

    if keep_tar:
        features_tar = data['tar_straight_cep_feat']
        f0_tar = data['tar_f0_feat']
        if energy:
            ec_tar = data['tar_ec_feat']
    
#    if not keep_norm:
#        features_src = features_src[:,:-1]
#        if keep_tar:
#            features_tar = features_tar[:,:-1]
    
    if energy:
        feat_src = np.concatenate((features_src, f0_src, ec_src), 1)
    else:
        feat_src = np.concatenate((features_src, f0_src), 1)
        
    if keep_tar:
        if energy:
            feat_tar = np.concatenate((features_tar, f0_tar, ec_tar), 1)
        else:
            feat_tar = np.concatenate((features_tar, f0_tar), 1)

    dim_feat = feat_src.shape[1]
    momentum_pitch = data['momentum_pitch']
    dim_pitch = momentum_pitch.shape[1]
    if energy:
        momentum_energy = data['momentum_energy']
        dim_energy = momentum_energy.shape[1]
    
    if shuffle:
        if keep_tar:
            if energy:
                joint_data = np.concatenate((feat_src, feat_tar, momentum_pitch, momentum_energy), 1)
            else:
                joint_data = np.concatenate((feat_src, feat_tar, momentum_pitch), 1)
        else:
            if energy:
                joint_data = np.concatenate((feat_src, momentum_pitch, momentum_energy), 1)
            else:
                joint_data = np.concatenate((feat_src, momentum_pitch), 1)
        np.random.shuffle(joint_data)
        
        feat_src = joint_data[:, :dim_feat]
        if keep_tar:
            feat_tar = joint_data[:,dim_feat:2*dim_feat]
            momentum_pitch = joint_data[:, 2*dim_feat:2*dim_feat+dim_pitch]
            if energy:
                momentum_energy = joint_data[:, 2*dim_feat+dim_pitch:]
        else:
            momentum_pitch = joint_data[:, dim_feat:dim_feat+dim_pitch]
            if energy:
                momentum_energy = joint_data[:, dim_feat+dim_pitch:]
    
    if keep_tar:
        if energy:
            return np.asarray(feat_src, np.float32), \
                    np.asarray(feat_tar, np.float32), \
                    np.asarray(momentum_pitch, np.float32), \
                    np.asarray(momentum_energy, np.float32)
        else:
            return np.asarray(feat_src, np.float32), \
                    np.asarray(feat_tar, np.float32), \
                    np.asarray(momentum_pitch, np.float32)
    else:
        if energy:
            return np.asarray(feat_src, np.float32), \
                    np.asarray(momentum_pitch, np.float32), \
                    np.asarray(momentum_energy, np.float32)
        else:
            return np.asarray(feat_src, np.float32), \
                    np.asarray(momentum_pitch, np.float32)


def cdf_transform(Y_train, Y_valid, Y_test, bins=256):
    
    Y_train_cdf = np.zeros(Y_train.shape)
    Y_valid_cdf = np.zeros(Y_valid.shape)
    Y_test_cdf = np.zeros(Y_test.shape)
    
    for dim in range(Y_train.shape[1]):
        histo = np.histogram(Y_train[:,dim], bins=bins)
        
        bin_count = histo[0]
        bin_ends = histo[1]
        
        cummu_count = np.cumsum(bin_count)
        for i in range(1, bin_ends.shape[0]):
            idx_train = np.where((Y_train[:,dim]<=bin_ends[i]) & (Y_train[:,dim]>bin_ends[i-1]))[0]
            idx_valid = np.where((Y_valid[:,dim]<=bin_ends[i]) & (Y_valid[:,dim]>bin_ends[i-1]))[0]
            idx_test = np.where((Y_test[:,dim]<=bin_ends[i]) & (Y_test[:,dim]>bin_ends[i-1]))[0]
            Y_train_cdf[idx_train,dim] = cummu_count[i-1] / Y_train.shape[0]
            Y_valid_cdf[idx_valid,dim] = cummu_count[i-1] / Y_train.shape[0]
            Y_test_cdf[idx_test,dim] = cummu_count[i-1] / Y_train.shape[0]
        
        idx_train = np.where(Y_train[:,dim]<=bin_ends[0])[0]
        Y_train_cdf[idx_train,dim] = cummu_count[0] / Y_train.shape[0]
        
        idx_valid = np.where(Y_valid[:,dim]<=bin_ends[0])[0]
        Y_valid_cdf[idx_valid,dim] = cummu_count[0] / Y_train.shape[0]
        idx_valid = np.where(Y_valid[:,dim]>=bin_ends[-1])[0]
        Y_valid_cdf[idx_valid,dim] = cummu_count[-1] / Y_train.shape[0]
        
        idx_test = np.where(Y_test[:,dim]<=bin_ends[0])[0]
        Y_test_cdf[idx_test,dim] = cummu_count[0] / Y_train.shape[0]
        idx_test = np.where(Y_test[:,dim]>=bin_ends[-1])[0]
        Y_test_cdf[idx_test,dim] = cummu_count[-1] / Y_train.shape[0]

    return Y_train_cdf, Y_valid_cdf, Y_test_cdf

def create_train_valid_fold(data, fold, speaker_dict, keep_norm=False, shuffle=False, \
                            keep_tar=False, energy=False):
    file_idx = data['file_idx']
    features_src = data['src_cep']
    if keep_tar:
        features_tar = data['tar_cep']
    
    if not keep_norm:
        features_src = features_src[:,:-1]
        if keep_tar:
            features_tar = features_tar[:,:-1]
    
    f0_src = data['src_f0']
    if keep_tar:
        f0_tar = data['tar_f0']

    if energy:
        ec_src = data['src_ec']
        if keep_tar:
            ec_tar = data['tar_ec']

    if energy:
        feat_src = np.concatenate((features_src, f0_src, ec_src), 1)
        if keep_tar:
            feat_tar = np.concatenate((features_tar, f0_tar, ec_tar), 1)
    else:
        feat_src = np.concatenate((features_src, f0_src), 1)
        if keep_tar:
            feat_tar = np.concatenate((features_tar, f0_tar), 1)

    mom_f0 = data['mom_f0']
    if energy:
        mom_ec = data['mom_ec']
    
    dim_feats = feat_src.shape[1]
    dim_mom = mom_f0.shape[1]

    if shuffle:
        if keep_tar:
            if energy:
                joint_data = np.concatenate((feat_src, feat_tar, mom_f0, mom_ec), 1)
            else:
                joint_data = np.concatenate((feat_src, feat_tar, mom_f0), 1)
        else:
            if energy:
                joint_data = np.concatenate((feat_src, mom_f0, mom_ec), 1)
            else:
                joint_data = np.concatenate((feat_src, mom_f0), 1)
        joint_data = np.concatenate((joint_data, file_idx), axis=1)
        np.random.shuffle(joint_data)
        file_idx = joint_data[:,-1]
        joint_data = joint_data[:,:-1]
        z = np.where((file_idx>=speaker_dict[fold-1][0]) & (file_idx<=speaker_dict[fold-1][1]))[0]
        valid_data = joint_data[z]
        train_data = np.delete(joint_data, z, axis=0)
        
    if keep_tar:
        train_feats_src = train_data[:,:dim_feats]
        train_feats_tar = train_data[:,dim_feats:2*dim_feats]
        train_mom = train_data[:,2*dim_feats:]
        valid_feats_src = valid_data[:,:dim_feats]
        valid_feats_tar = valid_data[:,dim_feats:2*dim_feats]
        valid_mom = valid_data[:,2*dim_feats:]
        return train_feats_src, train_feats_tar, train_mom, valid_feats_src, valid_feats_tar, valid_mom
    else:
        train_feats = train_data[:,:dim_feats]
        train_mom = train_data[:,dim_feats:]
        valid_feats = valid_data[:,:dim_feats]
        valid_mom = valid_data[:,dim_feats:]
        return train_feats, train_mom, valid_feats, valid_mom

def speaker_normalization(train, valid, files_train, files_valid):
    speaker_id = joblib.load('./speaker_file_info.pkl')
    speaker_id = speaker_id['neutral_angry']
    scaler_array = []
    gender_train = np.zeros((train.shape[0],1))
    gender_valid = np.zeros((valid.shape[0],1))
    for i in range(len(speaker_id)):
        scaler = StandardScaler()
        speaker_info = speaker_id[i]
        try:
            idx_train = np.where((files_train>=speaker_info[0]) \
                                  & (files_train<=speaker_info[1]))[0]
            scaler.fit(train[idx_train,:])
            train[idx_train,:] = scaler.transform(train[idx_train,:])
            gender_train[idx_train,0] = 1 if speaker_info[2] == 'M' else 0
        except Exception as e:
            print(e)
        
        try:
            idx_valid = np.where((files_valid>=speaker_info[0]) \
                                  & (files_valid<=speaker_info[1]))[0]
            valid[idx_valid,:] = scaler.transform(valid[idx_valid,:])
            gender_valid[idx_valid,0] = 1 if speaker_info[2] == 'M' else 0
        except Exception as e:
            print(e)
        scaler_array.append(scaler)
    train = np.concatenate((train, gender_train), axis=1)
    valid = np.concatenate((valid, gender_valid), axis=1)
    return (train, valid, scaler_array)

def load_arrays_h5py(file_name):
    f = h5py.File(file_name, 'r+')
    arrays = {}
    for k,v in f.items():
        arrays[k] = np.transpose(np.asarray(v))
    return arrays

def kl_div(p_1, p_2):
    idx = np.where(p_1<=0)[0]
    p_1[idx] = 1e-15
    p_1 = np.divide(p_1, np.sum(p_1))
    idx = np.where(p_2<=0)[0]
    p_2[idx] = 1e-15
    p_2 = np.divide(p_2, np.sum(p_2))
    return np.sum(np.multiply(p_1, np.log(np.divide(p_1, p_2))))

def make_train_valid_test(data, files, fold, speaker_list):
    if speaker_list is None:
        speaker_list = joblib.load('./speaker_file_info.pkl')
    idx = np.where((files>=speaker_list[fold-1][0]) \
                   & (files<=speaker_list[fold-1][1]))[0]
    final_test = data[idx, :]
    data = np.delete(data, idx, axis=0)
    files = np.delete(files, idx, axis=0)
    hist_dist = 1e10
    for rand_set in range(2):
        train = np.empty((0, data.shape[1]))
        valid = np.empty((0, data.shape[1]))
        unique_files = np.unique(files)
        np.random.shuffle(unique_files)
        utt_train = int(0.85*unique_files.shape[0])
        for utt in range(0, utt_train):
            idx = np.where(files==unique_files[utt])[0]
            train= np.asarray(np.concatenate((train, data[idx,:]), \
                                                   axis=0), np.float32)
        
        for utt in range(utt_train, unique_files.shape[0]):
            idx = np.where(files==unique_files[utt])[0]
            valid = np.asarray(np.concatenate((valid, data[idx,:]), \
                                        axis=0), np.float32)
        
        trb = np.histogram(train[:,-1], bins=100, density=True)
        vab = np.histogram(valid[:,-1], trb[1], density=True)
        dist = kl_div(trb[0], vab[0])
        if dist < hist_dist:
            hist_dist = dist
            final_train = train
            final_valid = valid
        print('Running {}th set having distance- {}'.format(rand_set, dist))
        sys.stdout.flush()
    return final_train, final_valid, final_test









































 
