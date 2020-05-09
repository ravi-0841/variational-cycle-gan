import argparse
import os
import numpy as np
import tensorflow as tf
import librosa
import scipy.signal as scisig

import preprocess as preproc

from model_mceps import CycleGAN as CycleGAN_mceps
from model import CycleGAN as CycleGAN_f0s
from helper import smooth, generate_interpolation

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def f0_conversion(model_f0_path, input_mfc, input_pitch, direction):
    tf.reset_default_graph()
    model_f0 = CycleGAN_f0s(mode='test')
    model_f0.load(filepath=model_f0_path)
    conv_f0 = model_f0.test(input_pitch=input_pitch, input_mfc=input_mfc, \
                        direction=conversion_direction)[0]
    return conv_f0

def mcep_conversion(model_mcep_path, features, direction):
    tf.reset_default_graph()
    model_mceps = CycleGAN_mceps(num_features=24, mode='test')
    model_mceps.load(filepath=model_mcep_path)
    coded_sp_converted_norm = model_mceps.test(inputs=features, \
                    direction=conversion_direction)[0]
    return coded_sp_converted_norm


def conversion(model_f0_path, model_mcep_path, mcep_nmz_path, data_dir, 
        conversion_direction, output_dir, emo_pair):

    num_mceps = 24
    sampling_rate = 16000
    frame_period = 5.0

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    mcep_normalization_params = np.load(mcep_nmz_path)
    mcep_mean_A = mcep_normalization_params['mean_A']
    mcep_std_A = mcep_normalization_params['std_A']
    mcep_mean_B = mcep_normalization_params['mean_B']
    mcep_std_B = mcep_normalization_params['std_B']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.listdir(data_dir):
        
        try:

            filepath = os.path.join(data_dir, file)
            wav, _ = librosa.load(filepath, sr = sampling_rate, mono = True)
            wav = preproc.wav_padding(wav = wav, sr = sampling_rate, \
                    frame_period = frame_period, multiple = 4)
            f0, sp, ap = preproc.world_decompose(wav = wav, \
                    fs = sampling_rate, frame_period = frame_period)

            coded_sp = preproc.world_encode_spectral_envelope(sp = sp, \
                    fs = sampling_rate, dim = num_mceps)

            coded_sp_f0 = preproc.world_encode_spectral_envelope(sp=sp, \
                    fs=sampling_rate, dim=23)

            coded_sp_transposed = coded_sp.T
    
            if conversion_direction == 'A2B':

                coded_sp_norm = (coded_sp_transposed - mcep_mean_A) / mcep_std_A
    
                # test mceps
                coded_sp_converted_norm = mcep_conversion(model_mcep_path=model_mcep_path, \
                                                features=np.array([coded_sp_norm]), \
                                                direction=conversion_direction)
                # test f0:
                f0 = scisig.medfilt(f0, kernel_size=3)
                z_idx = np.where(f0<10.0)[0]
                
                f0 = generate_interpolation(f0)
                f0 = smooth(f0, window_len=13)
                f0 = np.reshape(f0, (1,1,-1))

                coded_sp_f0 = np.expand_dims(coded_sp_f0, axis=0)
                coded_sp_f0 = np.transpose(coded_sp_f0, (0,2,1))

                f0_converted = f0_conversion(model_f0_path=model_f0_path, 
                        input_mfc=coded_sp_f0, input_pitch=f0, direction='A2B')

                f0_converted = np.asarray(np.reshape(f0_converted, (-1,)), np.float64)
                f0_converted[z_idx] = 0.0
                f0_converted = np.ascontiguousarray(f0_converted)
    
            else:
                raise Exception("Please specify A2B as conversion direction")
    
            coded_sp_converted = coded_sp_converted_norm * mcep_std_B + mcep_mean_B
            coded_sp_converted = coded_sp_converted.T
            coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
            decoded_sp_converted = preproc.world_decode_spectral_envelope(coded_sp=coded_sp_converted, \
                    fs=sampling_rate)
            wav_transformed = preproc.world_speech_synthesis(f0=f0_converted, \
                    decoded_sp=decoded_sp_converted, ap=ap, fs=sampling_rate, \
                    frame_period=frame_period)
            librosa.output.write_wav(os.path.join(output_dir, \
                    os.path.basename(file)), wav_transformed, sampling_rate)

            print("Processed "+filepath)
        except Exception as ex:
            print(ex)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Convert voices using pre-trained CycleGAN model.')
    
    # NA - lc_0.0001_lm_1e-05_all_spk_lvi, 400
    # NH - lc_1e-05_lm_0.0001_all_spk_lvi, 500
    # NS - lc_1e-05_lm_1e-06, 400

    parser.add_argument('--emo_pair', type=str, help='Pair of emotions', \
                        default='neu-ang', \
                        choices=['neu-ang', 'neu-hap', \
                                 'neu-sad'])
    parser.add_argument('--data_dir', type=str, 
            help='Folder containing wav files for conversion')
    parser.add_argument('--model_f0_path', type=str, 
            help='Path to the trained F0 model', 
            default='./model/model_f0/neu-ang.ckpt')
    parser.add_argument('--model_mcep_path', type=str, 
            help='Path to the trained mcep_model', 
            default='./model/model_mcep/neu-ang.ckpt')
    parser.add_argument('--mcep_nmz_path', type=str, 
            help='Path to mcep normalization parameter', 
            default='./model/model_mcep/neu-ang_mcep_nmz.npz')
    parser.add_argument('--output_dir', type=str, 
            help='Directory to store converted files', 
            default='./converted')
    argv = parser.parse_args()
    
    emo_pair = argv.emo_pair
    model_f0_path = argv.model_f0_path
    model_mcep_path = argv.model_mcep_path
    mcep_nmz_path = argv.mcep_nmz_path
    data_dir = argv.data_dir
    output_dir = argv.output_dir
    conversion_direction = 'A2B'
    
    data_dir = '/home/ravi/Desktop/check'
    output_dir = '/home/ravi/Desktop/check_out'

    conversion(model_f0_path=model_f0_path, model_mcep_path=model_mcep_path, 
        mcep_nmz_path=mcep_nmz_path, data_dir=data_dir, 
        conversion_direction=conversion_direction, 
        output_dir=output_dir, emo_pair=emo_pair)


