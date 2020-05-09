import os
import numpy as np
import argparse
import time
import scipy.io as scio
import pylab
import logging

import preprocess as preproc

from glob import glob
from model import CycleGAN

from helper import smooth, generate_interpolation

def train(train_dir, model_dir, model_name, random_seed, \
            validation_dir, output_dir, pre_train=None, \
            lambda_cycle=0, lambda_momenta=0):

    np.random.seed(random_seed)

    num_epochs = 500
    mini_batch_size = 1

    generator_learning_rate = 0.0001
    discriminator_learning_rate = 0.0000001

    num_mcep = 23
    n_frames = 128

    lambda_cycle = lambda_cycle
    lambda_momenta = lambda_momenta

    lc_lm = "lc_"+str(lambda_cycle) \
            +"_lm_"+str(lambda_momenta)

    logger_file = './log/'+lc_lm+'.log'
    
    if not os.path.exists('./log'):
        os.mkdir('./log')
    
    if os.path.exists(logger_file):
        os.remove(logger_file)

    logging.basicConfig(filename=logger_file, \
                            level=logging.DEBUG)


    logging.info("lambda_cycle - {}".format(lambda_cycle))
    logging.info("lambda_momenta - {}".format(lambda_momenta))

    if not os.path.isdir("./generated_pitch/"+lc_lm):
        os.mkdir("./generated_pitch/" + lc_lm)
    else:
        for f in glob(os.path.join("./generated_pitch/", "*.png")):
            os.remove(f)


    start_time = time.time()

    data_train = scio.loadmat(os.path.join(train_dir, 'train.mat'))
    data_valid = scio.loadmat(os.path.join(train_dir, 'valid.mat'))

    pitch_A_train = np.expand_dims(data_train['src_f0_feat'], axis=-1)
    pitch_B_train = np.expand_dims(data_train['tar_f0_feat'], axis=-1)
    mfc_A_train = data_train['src_mfc_feat']
    mfc_B_train = data_train['tar_mfc_feat']
    
    pitch_A_valid = np.expand_dims(data_valid['src_f0_feat'], axis=-1)
    pitch_B_valid = np.expand_dims(data_valid['tar_f0_feat'], axis=-1)
    mfc_A_valid = data_valid['src_mfc_feat']
    mfc_B_valid = data_valid['tar_mfc_feat']

    # Shuffle to get non-parallel training data
    indices_train = np.arange(0, pitch_A_train.shape[0])
    np.random.shuffle(indices_train)
    pitch_A_train = pitch_A_train[indices_train]
    mfc_A_train = mfc_A_train[indices_train]
    np.random.shuffle(indices_train)
    pitch_B_train = pitch_B_train[indices_train]
    mfc_B_train = mfc_B_train[indices_train]

    mfc_A_valid, pitch_A_valid, \
        mfc_B_valid, pitch_B_valid = preproc.sample_data(mfc_A=mfc_A_valid, \
                                    mfc_B=mfc_B_valid, pitch_A=pitch_A_valid, \
                                    pitch_B=pitch_B_valid)

    if validation_dir is not None:
        validation_output_dir = os.path.join(output_dir, lc_lm)
        if not os.path.exists(validation_output_dir):
            os.makedirs(validation_output_dir)

    end_time = time.time()
    time_elapsed = end_time - start_time

    print('Time Elapsed for Data Preprocessing: %02d:%02d:%02d' % (time_elapsed // 3600, \
                                                                   (time_elapsed % 3600 // 60), \
                                                                   (time_elapsed % 60 // 1)))
    
    #use pre_train arg to provide trained model
    model = CycleGAN(dim_pitch=1, dim_mfc=num_mcep, \
                n_frames=n_frames, pre_train=pre_train)
    
    for epoch in range(1,num_epochs+1):

        print('Epoch: %d' % epoch)
        logging.info('Epoch: %d' % epoch)

        start_time_epoch = time.time()

        mfc_A, pitch_A, \
            mfc_B, pitch_B = preproc.sample_data(mfc_A=mfc_A_train, \
                            mfc_B=mfc_B_train, pitch_A=pitch_A_train, \
                            pitch_B=pitch_B_train)
        
        n_samples = mfc_A.shape[0]
        
        train_gen_loss = []
        train_disc_loss = []

        for i in range(n_samples // mini_batch_size):

            start = i * mini_batch_size
            end = (i + 1) * mini_batch_size

            generator_loss, discriminator_loss, \
            gen_A, gen_B, \
            mom_A, mom_B = model.train(mfc_A=mfc_A[start:end], \
                                        mfc_B=mfc_B[start:end], \
                                        pitch_A=pitch_A[start:end], \
                                        pitch_B=pitch_B[start:end], \
                                        lambda_cycle=lambda_cycle, \
                                        lambda_momenta=lambda_momenta, \
                                        generator_learning_rate=generator_learning_rate, \
                                        discriminator_learning_rate=discriminator_learning_rate)
            
            train_gen_loss.append(generator_loss)
            train_disc_loss.append(discriminator_loss)

        
        logging.info("Train Generator Loss- {}".format(np.mean(train_gen_loss)))
        logging.info("Train Discriminator Loss- {}".format(np.mean(train_disc_loss)))

        if epoch%100 == 0:

            for i in range(mfc_A_valid.shape[0]):

                gen_A, gen_B, mom_A, mom_B \
                        = model.test_gen(mfc_A=mfc_A_valid[i:i+1], \
                                mfc_B=mfc_B_valid[i:i+1], \
                                pitch_A=pitch_A_valid[i:i+1], \
                                pitch_B=pitch_B_valid[i:i+1])

                pylab.figure(figsize=(12,12))
                pylab.subplot(121)
                pylab.plot(pitch_A_valid[i].reshape(-1,), label='Input A')
                pylab.plot(gen_B.reshape(-1,), label='Generated B')
                pylab.plot(mom_B.reshape(-1,), label='Generated momenta')
                pylab.legend(loc=2)

                pylab.subplot(122)
                pylab.plot(pitch_B_valid[i].reshape(-1,), label='Input B')
                pylab.plot(gen_A.reshape(-1,), label='Generated A')
                pylab.plot(mom_A.reshape(-1,), label='Generated momenta')
                pylab.legend(loc=2)

                pylab.title('Epoch '+str(epoch)+' example '+str(i+1))
                pylab.savefig('./generated_pitch/'+str(epoch)+'_'+str(i+1)+'.png')
                pylab.close()
        
        end_time_epoch = time.time()
        time_elapsed_epoch = end_time_epoch - start_time_epoch

        logging.info('Time Elapsed for This Epoch: %02d:%02d:%02d' % (time_elapsed_epoch // 3600, \
                (time_elapsed_epoch % 3600 // 60), (time_elapsed_epoch % 60 // 1))) 

        if epoch % 100 == 0:
            cur_model_name = model_name+"_"+str(epoch)+".ckpt"
            model.save(directory=model_dir, filename=cur_model_name)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Train CycleGAN model for datasets.')

    emo_dict = {'neu-ang':['neutral', 'angry'], \
                        'neu-sad':['neutral', 'sad'], \
                        'neu-hap':['neutral', 'happy']}

    emo_pair = 'neu-hap'
    model_dir_default = './model/'
    model_name_default = 'model'
    output_dir_default = './validation_output/'
    random_seed_default = 0

    parser.add_argument('--train_dir', type=str, help='Directory for A.', \
                        default=train_dir_default)
    parser.add_argument('--model_dir', type=str, help='Directory for saving models.', \
                        default=model_dir_default)
    parser.add_argument('--model_name', type=str, help='File name for saving model.', \
                        default=model_name_default)
    parser.add_argument('--random_seed', type=int, help='Random seed for model training.', \
                        default=random_seed_default)
    parser.add_argument('--validation_dir', type=str, \
                        help='Convert validation after each training epoch. Set None for no conversion', \
                        default=validation_dir_default)
    parser.add_argument('--output_dir', type=str, \
                        help='Output directory for converted validation voices.', default=output_dir_default)
    parser.add_argument("--lambda_cycle", type=float, help="hyperparam for cycle loss", \
                        default=0.0001)#0.0001
    parser.add_argument("--lambda_momenta", type=float, help="hyperparam for momenta magnitude", \
                        default=1e-5)#0.1

    argv = parser.parse_args()

    train_dir = argv.train_dir
    model_dir = argv.model_dir
    model_name = argv.model_name
    random_seed = argv.random_seed
    validation_dir = None if argv.validation_dir == 'None' or argv.validation_dir == 'none' \
                        else argv.validation_dir
    output_dir = argv.output_dir

    lambda_cycle = argv.lambda_cycle
    lambda_momenta = argv.lambda_momenta

    train(train_dir=train_dir, model_dir=model_dir, model_name=model_name, 
            random_seed=random_seed, validation_dir=validation_dir, 
            output_dir=output_dir, pre_train=None, lambda_cycle=lambda_cycle, 
            lambda_momenta=lambda_momenta)
