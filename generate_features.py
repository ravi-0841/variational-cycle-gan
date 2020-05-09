from glob import glob
import os
import scipy.io.wavfile as scwav
import numpy as np
import librosa
import scipy.io as scio
import scipy.signal as scisig
import pyworld as pw
import argparse

import warnings
warnings.filterwarnings('ignore')

from dtw import constrained_dtw
from feat_utils import smooth, \
    generate_interpolation, highpass_filter

def get_feats(files_list, sample_rate, window_len, 
              window_stride, modified_dtw, n_feats, n_mfc):
    """ 
    FILE_LIST: A list containing the source and target utterances location
    sample_rate: Sampling frequency of the speech
    window_len: Length of the analysis window for getting features (in ms)
    """
    files_list_src = files_list['src']
    files_list_tar = files_list['tar']

    f0_feat_src = []
    f0_feat_tar = []
    
    mfc_feat_src = []
    mfc_feat_tar = []

    file_list   = []

    for s,t in zip(files_list_src, files_list_tar):
        
        print(t)
        
        """
        Utterance level features for context expansion
        """
        utt_f0_src          = list()
        utt_f0_tar          = list()
        utt_mfc_src         = list()
        utt_mfc_tar         = list()
        
        file_id = int(s.split('/')[-1][:-4])
        try:
            src_wav = scwav.read(s)
            src = np.asarray(src_wav[1], np.float64)
            src = np.copy(highpass_filter(src, 20, 16000), order='C')

            tar_wav = scwav.read(t)
            tar = np.asarray(tar_wav[1], np.float64)
            tar = np.copy(highpass_filter(tar, 20, 16000), order='C')
            
            f0_src, t_src   = pw.harvest(src, sample_rate, frame_period=int(1000*window_len))
            straight_src    = pw.cheaptrick(src, f0_src, t_src, sample_rate)

            f0_tar, t_tar   = pw.harvest(tar, sample_rate,frame_period=int(1000*window_len))
            straight_tar    = pw.cheaptrick(tar, f0_tar, t_tar, sample_rate)

            f0_src = scisig.medfilt(f0_src, kernel_size=3)
            f0_tar = scisig.medfilt(f0_tar, kernel_size=3)
            f0_src = np.asarray(f0_src, np.float32)
            f0_tar = np.asarray(f0_tar, np.float32)

            f0_src = np.asarray(generate_interpolation(f0_src), np.float32)
            f0_tar = np.asarray(generate_interpolation(f0_tar), np.float32)
            
            f0_src = smooth(f0_src, window_len=13)
            f0_tar = smooth(f0_tar, window_len=13)

            src_mfc = pw.code_spectral_envelope(straight_src, sample_rate, n_mfc)
            tar_mfc = pw.code_spectral_envelope(straight_tar, sample_rate, n_mfc)

            src_mfcc = librosa.feature.mfcc(y=src, sr=sample_rate, \
                                            hop_length=int(sample_rate*window_len), \
                                            win_length=int(sample_rate*window_len), \
                                            n_fft=1024, n_mels=128)
            
            tar_mfcc = librosa.feature.mfcc(y=tar, sr=sample_rate, \
                                            hop_length=int(sample_rate*window_len), \
                                            win_length=int(sample_rate*window_len), \
                                            n_fft=1024, n_mels=128)
            if not modified_dtw:
                _, cords = librosa.sequence.dtw(X=src_mfcc, Y=tar_mfcc, metric='cosine')
            else:
                _, _, cords = constrained_dtw(x=src_mfcc.T, y=tar_mfcc.T)

            del src_mfcc, tar_mfcc
            
            f0_src = f0_src.reshape(-1,1)
            f0_tar = f0_tar.reshape(-1,1)
            
            ext_src_f0 = []
            ext_tar_f0 = []
            ext_src_mfc = []
            ext_tar_mfc = []
            
            for i in range(len(cords)-1, -1, -1):
                ext_src_f0.append(f0_src[cords[i,0],0])
                ext_tar_f0.append(f0_tar[cords[i,1],0])
                ext_src_mfc.append(src_mfc[cords[i,0],:])
                ext_tar_mfc.append(tar_mfc[cords[i,1],:])
            
            ext_src_f0 = np.reshape(np.asarray(ext_src_f0), (-1,1))
            ext_tar_f0 = np.reshape(np.asarray(ext_tar_f0), (-1,1))
            ext_src_mfc = np.asarray(ext_src_mfc)
            ext_tar_mfc = np.asarray(ext_tar_mfc)

            if cords.shape[0]<n_feats:
                continue
            else:
                for sample in range(20):
                    start = np.random.randint(0, cords.shape[0]-n_feats+1)
                    end = start + n_feats
                    
                    utt_f0_src.append(ext_src_f0[start:end,:])
                    utt_f0_tar.append(ext_tar_f0[start:end,:])
                    
                    utt_mfc_src.append(ext_src_mfc[start:end,:])
                    utt_mfc_tar.append(ext_tar_mfc[start:end,:])
                
                f0_feat_src.append(utt_f0_src)
                f0_feat_tar.append(utt_f0_tar)
                
                mfc_feat_src.append(utt_mfc_src)
                mfc_feat_tar.append(utt_mfc_tar)
                    
                file_list.append(file_id)

        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    file_list = np.asarray(file_list).reshape(-1,1)
    return file_list, (f0_feat_src, mfc_feat_src, 
            f0_feat_tar, mfc_feat_tar)


##----------------------------------generate all features---------------------------------
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, 
            help='Directory for source emotion wav files')
    parser.add_argument('--target_dir', type=str, 
            help='Directory for target emotion wav files')
    parser.add_argument('--save_dir', type=str, 
            help='Directory for saving the features')
    parser.add_argument('--modified_dtw', type=bool, 
            help='Use modified version of DTW', default=0)
    parser.add_argument('--n_frames', type=int, 
            help='Number of frames for training', default=128)
    parser.add_argument('--n_mfc', type=int, 
            help='Dimensions of mfcc features', default=23)
    parser.add_argument('--emo_pair', type=str, 
            help='Emotion pair', default='neu-ang')
    parser.add_argument('--fraction', type=str, 
                        help='which fraction of data is being generated', 
                        choices=['train', 'valid', 'test'], default='train')

    args = parser.parse_args()
    
    """
    For testing
    """
    args.source_dir = '/home/ravi/Downloads/Emo-Conv/neutral-happy/test/neutral/'
    args.target_dir = '/home/ravi/Downloads/Emo-Conv/neutral-happy/test/happy/'
    args.fraction = 'test'
    args.save_dir = '/home/ravi/Desktop/'

    emo_dict = {'neutral-angry':'neu-ang', 'neutral-happy':'neu-hap', 
                'neutral-sad':'neu-sad'}

    try:
        files_src = sorted(glob(os.path.join(args.source_dir, '*.wav')))
        files_tar = sorted(glob(os.path.join(args.target_dir, '*.wav')))
       
        sample_rate = 16000.0
        window_len = 0.005
        window_stride = 0.005
       
        files_list = {'src':files_src, 'tar':files_tar}
       
        file_names, (src_f0_feat, src_mfc_feat, tar_f0_feat, 
                tar_mfc_feat) = get_feats(files_list, sample_rate, 
                                window_len, window_stride, 
                                n_feats=args.n_frames, n_mfc=args.n_mfc, 
                                modified_dtw=args.modified_dtw)

        scio.savemat(os.path.join(args.save_dir, args.fraction+'.mat'), \
                     { \
                         'src_mfc_feat':           src_mfc_feat, \
                         'tar_mfc_feat':           tar_mfc_feat, \
                         'src_f0_feat':            src_f0_feat, \
                         'tar_f0_feat':            tar_f0_feat, \
                         'file_names':             file_names
                     })

        del file_names, src_mfc_feat, src_f0_feat, \
            tar_mfc_feat, tar_f0_feat
    except Exception as ex:
        print(ex)

