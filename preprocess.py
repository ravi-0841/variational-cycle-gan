import librosa
import numpy as np
import os
import pyworld
import scipy.io.wavfile as scwav
import scipy.ndimage.filters as scifilt

from joblib import Parallel, delayed

def load_wavs(wav_dir, sr):

    wavs = list()
    for file in sorted(os.listdir(wav_dir)):
        file_path = os.path.join(wav_dir, file)
#        wav, _ = librosa.load(file_path, sr = sr, mono = True)
        wav = scwav.read(file_path)
        wav = wav[1].astype(np.float64)
        #wav = wav.astype(np.float64)
        wavs.append(wav)

    return wavs

def world_decompose(wav, fs, frame_period=5.0):

    # Decompose speech signal into f0, spectral envelope and aperiodicity using WORLD
    wav = wav.astype(np.float64)
    f0, timeaxis = pyworld.harvest(wav, fs, \
            frame_period=frame_period, f0_floor=50.0, f0_ceil=800.0)
    sp = pyworld.cheaptrick(wav, f0, timeaxis, fs)
    ap = pyworld.d4c(wav, f0, timeaxis, fs)

    return (f0, sp, ap)

def world_encode_spectral_envelope(sp, fs, dim = 24):

    # Get Mel-cepstral coefficients (MCEPs)

    #sp = sp.astype(np.float64)
    coded_sp = pyworld.code_spectral_envelope(sp, fs, dim)

    return coded_sp

def world_decode_spectral_envelope(coded_sp, fs):

    fftlen = pyworld.get_cheaptrick_fft_size(fs)
    #coded_sp = coded_sp.astype(np.float32)
    #coded_sp = np.ascontiguousarray(coded_sp)
    decoded_sp = pyworld.decode_spectral_envelope(coded_sp, fs, fftlen)

    return decoded_sp

def world_encode_data(wavs, fs, frame_period = 5.0, coded_dim = 24):

    f0s = list()
    timeaxes = list()
    sps = list()
    aps = list()
    coded_sps = list()
    
    world_params = Parallel(n_jobs=6)(delayed(world_decompose)(w,fs,frame_period) for w in wavs)
    f0s = [z[0] for z in world_params]
    timeaxes = [z[1] for z in world_params]
    sps = [z[2] for z in world_params]
    aps = [z[3] for z in world_params]
    coded_sps = [world_encode_spectral_envelop(z[2],fs,coded_dim) for z in world_params]

    return f0s, timeaxes, sps, aps, coded_sps

def transpose_in_list(lst):

    transposed_lst = list()
    for array in lst:
        transposed_lst.append(array.T)
    return transposed_lst

def world_decode_data(coded_sps, fs):

    decoded_sps =  list()

    for coded_sp in coded_sps:
        decoded_sp = world_decode_spectral_envelop(coded_sp, fs)
        decoded_sps.append(decoded_sp)

    return decoded_sps

def world_speech_synthesis(f0, decoded_sp, ap, fs, frame_period):

    #decoded_sp = decoded_sp.astype(np.float64)
    wav = pyworld.synthesize(f0, decoded_sp, ap, fs, frame_period)
    # Librosa could not save wav if not doing so
    wav = wav.astype(np.float32)

    return wav

def world_synthesis_data(f0s, decoded_sps, aps, fs, frame_period):

    wavs = list()

    for f0, decoded_sp, ap in zip(f0s, decoded_sps, aps):
        wav = world_speech_synthesis(f0, decoded_sp, ap, fs, frame_period)
        wavs.append(wav)

    return wavs

def coded_sps_normalization_fit_transform(coded_sps):

    coded_sps_concatenated = np.concatenate(coded_sps, axis = 1)
    coded_sps_mean = np.mean(coded_sps_concatenated, axis = 1, keepdims = True)
    coded_sps_std = np.std(coded_sps_concatenated, axis = 1, keepdims = True)

    coded_sps_normalized = list()
    for coded_sp in coded_sps:
        coded_sps_normalized.append((coded_sp - coded_sps_mean) / coded_sps_std)
    
    return coded_sps_normalized, coded_sps_mean, coded_sps_std

def coded_sps_normalization_transform(coded_sps, coded_sps_mean, coded_sps_std):

    coded_sps_normalized = list()
    for coded_sp in coded_sps:
        coded_sps_normalized.append((coded_sp - coded_sps_mean) / coded_sps_std)
    
    return coded_sps_normalized

def coded_sps_normalization_inverse_transform(normalized_coded_sps, coded_sps_mean, coded_sps_std):

    coded_sps = list()
    for normalized_coded_sp in normalized_coded_sps:
        coded_sps.append(normalized_coded_sp * coded_sps_std + coded_sps_mean)

    return coded_sps

def coded_sp_padding(coded_sp, multiple = 4):

    num_features = coded_sp.shape[0]
    num_frames = coded_sp.shape[1]
    num_frames_padded = int(np.ceil(num_frames / multiple)) * multiple
    num_frames_diff = num_frames_padded - num_frames
    num_pad_left = num_frames_diff // 2
    num_pad_right = num_frames_diff - num_pad_left
    coded_sp_padded = np.pad(coded_sp, ((0, 0), (num_pad_left, num_pad_right)), 'constant', constant_values = 0)

    return coded_sp_padded

def wav_padding(wav, sr, frame_period, multiple = 4):

    assert wav.ndim == 1 
    num_frames = len(wav)
    num_frames_padded = int((np.ceil((np.floor(num_frames / (sr * frame_period / 1000)) + 1) / multiple + 1) * multiple - 1) * (sr * frame_period / 1000))
    num_frames_diff = num_frames_padded - num_frames
    num_pad_left = num_frames_diff // 2
    num_pad_right = num_frames_diff - num_pad_left
    wav_padded = np.pad(wav, (num_pad_left, num_pad_right), 'constant', constant_values = 0)

    return wav_padded

def logf0_statistics(f0s):

    log_f0s_concatenated = np.ma.log(np.concatenate(f0s))
    log_f0s_mean = log_f0s_concatenated.mean()
    log_f0s_std = log_f0s_concatenated.std()

    return log_f0s_mean, log_f0s_std

def pitch_conversion(f0, mean_log_src, std_log_src, mean_log_target, std_log_target):

    # Logarithm Gaussian normalization for Pitch Conversions
    f0_converted = np.exp((np.log(f0) - mean_log_src) / std_log_src * std_log_target + mean_log_target)

    return f0_converted

def wavs_to_specs(wavs, n_fft = 1024, hop_length = None):

    stfts = list()
    for wav in wavs:
        stft = librosa.stft(wav, n_fft = n_fft, hop_length = hop_length)
        stfts.append(stft)

    return stfts

def wavs_to_mfccs(wavs, sr, n_fft = 1024, hop_length = None, n_mels = 128, n_mfcc = 24):

    mfccs = list()
    for wav in wavs:
        mfcc = librosa.feature.mfcc(y = wav, sr = sr, n_fft = n_fft, hop_length = hop_length, n_mels = n_mels, n_mfcc = n_mfcc)
        mfccs.append(mfcc)

    return mfccs

def mfccs_normalization(mfccs):

    mfccs_concatenated = np.concatenate(mfccs, axis = 1)
    mfccs_mean = np.mean(mfccs_concatenated, axis = 1, keepdims = True)
    mfccs_std = np.std(mfccs_concatenated, axis = 1, keepdims = True)

    mfccs_normalized = list()
    for mfcc in mfccs:
        mfccs_normalized.append((mfcc - mfccs_mean) / mfccs_std)
    
    return mfccs_normalized, mfccs_mean, mfccs_std

def sample_train_data(dataset_A, dataset_B, n_frames = 128, parallel=False):

    num_samples = min(len(dataset_A), len(dataset_B))
    train_data_A_idx = np.arange(len(dataset_A))
    train_data_B_idx = np.arange(len(dataset_B))
    np.random.shuffle(train_data_A_idx)
    if parallel:
        train_data_B_idx = np.copy(train_data_A_idx)
    else:
        np.random.shuffle(train_data_B_idx)
    train_data_A_idx_subset = train_data_A_idx[:num_samples]
    train_data_B_idx_subset = train_data_B_idx[:num_samples]

    train_data_A = list()
    train_data_B = list()

    for idx_A, idx_B in zip(train_data_A_idx_subset, train_data_B_idx_subset):
        data_A = dataset_A[idx_A]
        data_B = dataset_B[idx_B]
        frames_A_total = data_A.shape[1]
        frames_B_total = data_B.shape[1]

        if frames_A_total >= n_frames and frames_B_total >= n_frames:
            if parallel:
                start = np.random.randint(np.min([frames_B_total, frames_A_total]) - n_frames + 1)
                end = start + n_frames
                train_data_A.append(data_A[0:1,start:end])            
                train_data_B.append(data_B[0:1,start:end])
            else:    
                start_A = np.random.randint(frames_A_total - n_frames + 1)
                end_A = start_A + n_frames
                train_data_A.append(data_A[0:1,start_A:end_A])
                
                start_B = np.random.randint(frames_B_total - n_frames + 1)
                end_B = start_B + n_frames
                train_data_B.append(data_B[0:1,start_B:end_B])

    train_data_A = np.array(train_data_A)
    train_data_B = np.array(train_data_B)

    return train_data_A, train_data_B

def sample_data(mfc_A, pitch_A, mfc_B, pitch_B):
    mfc_data_A = list()
    mfc_data_B = list()
    pitch_data_A = list()
    pitch_data_B = list()

    for i in range(mfc_A.shape[0]):
        q = np.random.randint(0, mfc_A.shape[1])
        mfc_data_A.append(np.squeeze(mfc_A[i,q,:,:]))
        mfc_data_B.append(np.squeeze(mfc_B[i,q,:,:]))
        pitch_data_A.append(np.squeeze(pitch_A[i,q,:,:]))
        pitch_data_B.append(np.squeeze(pitch_B[i,q,:,:]))
    
    mfc_data_A = np.transpose(np.asarray(mfc_data_A), axes=(0,2,1))
    mfc_data_B = np.transpose(np.asarray(mfc_data_B), axes=(0,2,1))
    pitch_data_A = np.transpose(np.expand_dims(np.asarray(pitch_data_A), \
                                               axis=-1), axes=(0,2,1))
    pitch_data_B = np.transpose(np.expand_dims(np.asarray(pitch_data_B), \
                                               axis=-1), axes=(0,2,1))

    return mfc_data_A, pitch_data_A, mfc_data_B, pitch_data_B

def smooth_mfc(mfc):
    
    assert len(mfc.shape) > 1

    if len(mfc.shape)==4:
        for utt in range(mfc.shape[0]):
            for randsamp in range(mfc.shape[1]):
                mfc[utt,randsamp,:,:] \
                        = scifilt.gaussian_filter1d(np.squeeze(mfc[utt,randsamp,:,:]), \
                            sigma=1.0, axis=0)
    elif len(mfc.shape)==3:
        for utt in range(mfc.shape[0]):
            mfc[utt,:,:]  = scifilt.gaussian_filter1d(np.squeeze(mfc[utt,:,:]), \
                        sigma=1.0, axis=0)
    else:
        mfc = scifilt.gaussian_filter1d(mfc, sigma=1.0, axis=0)

    return mfc

def sample_data_smoothed(mfc_A, pitch_A, mfc_B, pitch_B):
    mfc_data_A = list()
    mfc_data_B = list()
    pitch_data_A = list()
    pitch_data_B = list()

    for i in range(mfc_A.shape[0]):
        q = np.random.randint(0, mfc_A.shape[1])
        mfc_data_A.append(np.squeeze(mfc_A[i,q,:,:]))
        mfc_data_B.append(np.squeeze(mfc_B[i,q,:,:]))
        pitch_data_A.append(np.squeeze(pitch_A[i,q,:,:]))
        pitch_data_B.append(np.squeeze(pitch_B[i,q,:,:]))
    
    mfc_data_A = np.transpose(np.asarray(mfc_data_A), axes=(0,2,1))
    mfc_data_B = np.transpose(np.asarray(mfc_data_B), axes=(0,2,1))
    pitch_data_A = np.transpose(np.expand_dims(np.asarray(pitch_data_A), \
                                               axis=-1), axes=(0,2,1))
    pitch_data_B = np.transpose(np.expand_dims(np.asarray(pitch_data_B), \
                                               axis=-1), axes=(0,2,1))
    
    mfc_A = smooth_mfc(mfc_A)
    mfc_B = smooth_mfc(mfc_B)

    return mfc_data_A, pitch_data_A, mfc_data_B, pitch_data_B

def normalize_f0s(f0_A, f0_B):

    if len(f0_A.shape)==3:
        utt = f0_A.shape[0]
        rand_samp = f0_A.shape[1]
        means_A = np.zeros((utt*rand_samp,1))
        stds_A = np.zeros((utt*rand_samp, 1))
        means_B = np.zeros((utt*rand_samp,1))
        stds_B = np.zeros((utt*rand_samp, 1))

        f0_A = np.vstack(f0_A)
        f0_B = np.vstack(f0_B)

        for i in range(utt*rand_samp):
            means_A[i,0] = np.mean(f0_A[i,:])
            stds_A[i,0] = np.std(f0_A[i,:])

            means_B[i,0] = np.mean(f0_B[i,:])
            stds_B[i,0] = np.std(f0_B[i,:])
            
            f0_A[i,:] = (f0_A[i,:] - means_A[i,0]) / stds_A[i,0]
            f0_B[i,:] = (f0_B[i,:] - means_B[i,0]) / stds_B[i,0]
        
        f0_A = np.reshape(f0_A, (utt, rand_samp, -1))
        f0_B = np.reshape(f0_B, (utt, rand_samp, -1))

        means_A = np.reshape(means_A, (utt, rand_samp, 1))
        means_B = np.reshape(means_B, (utt, rand_samp, 1))

        stds_A = np.reshape(stds_A, (utt, rand_samp, 1))
        stds_B = np.reshape(stds_B, (utt, rand_samp, 1))

    elif len(f0_A.shape)==2:
        utt = f0_A.shape[0]
        means_A = np.zeros((utt,1))
        stds_A = np.zeros((utt, 1))
        means_B = np.zeros((utt,1))
        stds_B = np.zeros((utt, 1))

        for i in range(utt):
            means_A[i,0] = np.mean(f0_A[i,:])
            stds_A[i,0] = np.std(f0_A[i,:])

            means_B[i,0] = np.mean(f0_B[i,:])
            stds_B[i,0] = np.std(f0_B[i,:])
            
            f0_A[i,:] = (f0_A[i,:] - means_A[i,0]) / stds_A[i,0]
            f0_B[i,:] = (f0_B[i,:] - means_B[i,0]) / stds_B[i,0]
        
    return f0_A, f0_B, means_A, means_B, stds_A, stds_B



