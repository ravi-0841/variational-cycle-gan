import os
import tensorflow as tf
import numpy as np

from nn_models import discriminator, generator
from utils import l1_loss
from datetime import datetime
from tf_forward_tan import forward_tan

class CycleGAN(object):

    def __init__(self, dim_pitch=1, dim_mfc=23, n_frames=128, \
                 discriminator=discriminator, \
                 generator=generator, mode='train', \
                 log_dir='./log', pre_train=None):
        
        self.n_frames = n_frames
        self.pitch_shape = [None, dim_pitch, None] #[batch_size, num_features, num_frames]
        self.mfc_shape = [None, dim_mfc, None]

        self.center_difference_mat = np.zeros((n_frames, n_frames), np.float32)
        for i in range(self.n_frames-1):
            self.center_difference_mat[i,i+1] = 0.5
        for i in range(1, self.n_frames):
            self.center_difference_mat[i,i-1] = -0.5
            
        self.first_order_diff_mat = np.eye(self.n_frames, dtype=np.float32)
        for i in range(1, self.n_frames):
            self.first_order_diff_mat[i-1,i] = -1

        self.discriminator = discriminator
        self.generator = generator
        self.mode = mode

        self.build_model()
        self.optimizer_initializer()

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        if pre_train is not None:
            self.saver.restore(self.sess, pre_train)
        else:
            self.sess.run(tf.global_variables_initializer())

        if self.mode == 'train':
            self.train_step = 0

    def build_model(self):

        # Placeholders for real training samples
        self.pitch_A_real = tf.placeholder(tf.float32, \
                            shape=self.pitch_shape, name='pitch_A_real')
        self.pitch_B_real = tf.placeholder(tf.float32, \
                            shape=self.pitch_shape, name='pitch_B_real')
        
        self.mfc_A = tf.placeholder(tf.float32, \
                            shape=self.mfc_shape, name='mfc_A_real')
        self.mfc_B = tf.placeholder(tf.float32, \
                            shape=self.mfc_shape, name='mfc_B_real')

        # Placeholders for fake generated samples
        self.pitch_A_fake = tf.placeholder(tf.float32, \
                            shape=self.pitch_shape, name='pitch_A_fake')
        self.pitch_B_fake = tf.placeholder(tf.float32, \
                            shape=self.pitch_shape, name='pitch_B_fake')
        
        # Placeholder for test samples
        self.pitch_A_test = tf.placeholder(tf.float32, \
                            shape=self.pitch_shape, name='pitch_A_test')
        self.mfc_A_test = tf.placeholder(tf.float32, \
                            shape=self.mfc_shape, name='mfc_A_test')

        self.pitch_B_test = tf.placeholder(tf.float32, \
                            shape=self.pitch_shape, name='pitch_B_test')
        self.mfc_B_test = tf.placeholder(tf.float32, \
                            shape=self.mfc_shape, name='mfc_B_test')

        # Place holder for lambda_cycle and lambda_identity
        self.lambda_cycle = tf.placeholder(tf.float32, None, \
                                name='lambda_cycle')
        self.lambda_momenta = tf.placeholder(tf.float32, \
                                None, name='lambda_momenta')

        # Create the kernel for lddmm
        self.kernel = tf.expand_dims(tf.constant([6,50], \
                            dtype=tf.float32), axis=0)
        
        # Generate pitch from A to B
        self.momentum_A2B = self.generator(input_pitch=self.pitch_A_real, \
                            input_mfc=self.mfc_A, \
                            reuse=False, scope_name='generator_A2B')
        self.generation_A2B = forward_tan(x=self.pitch_A_real, \
                            p=self.momentum_A2B, kernel=self.kernel)
        self.momentum_cycle_A2A = self.generator(input_pitch=self.generation_A2B, \
                            input_mfc=self.mfc_B, \
                            reuse=False, scope_name='generator_B2A')
        self.cycle_A2A = forward_tan(x=self.generation_A2B, \
                            p=self.momentum_cycle_A2A, kernel=self.kernel)
        
        # Generate pitch from B to A
        self.momentum_B2A = self.generator(input_pitch=self.pitch_B_real, \
                            input_mfc=self.mfc_B, \
                            reuse=True, scope_name='generator_B2A')
        self.generation_B2A = forward_tan(x=self.pitch_B_real, \
                            p=self.momentum_B2A, kernel=self.kernel)
        self.momentum_cycle_B2B = self.generator(input_pitch=self.generation_B2A, \
                            input_mfc=self.mfc_A, \
                            reuse=True, scope_name='generator_A2B')
        self.cycle_B2B = forward_tan(x=self.generation_B2A, \
                            p=self.momentum_cycle_B2B, kernel=self.kernel)

        # Generator Discriminator Loss
        self.discrimination_B_fake \
                = self.discriminator(input1=self.pitch_A_real, \
                    input2=self.generation_A2B, reuse=False, \
                    scope_name='discriminator_A')
        self.discrimination_A_fake \
                = self.discriminator(input1=self.pitch_B_real, \
                    input2=self.generation_B2A, reuse=False, \
                    scope_name='discriminator_B')

        # Cycle loss
        self.cycle_loss = (l1_loss(y=self.pitch_A_real, y_hat=self.cycle_A2A) \
                        + l1_loss(y=self.pitch_B_real, y_hat=self.cycle_B2B)) / 2.0


        # Generator loss
        # Generator wants to fool discriminator
        self.generator_loss_A2B \
            = l1_loss(y=tf.ones_like(self.discrimination_B_fake), \
                y_hat=self.discrimination_B_fake)
        self.generator_loss_B2A \
            = l1_loss(y=tf.ones_like(self.discrimination_A_fake), \
                y_hat=self.discrimination_A_fake)
        self.gen_disc_loss = (self.generator_loss_A2B \
                                + self.generator_loss_B2A) / 2.0

        self.momentum_loss_A2B \
            = tf.reduce_sum(tf.square(tf.matmul(self.first_order_diff_mat, \
                tf.reshape(self.momentum_A2B, [-1,1])))) \
                + tf.reduce_sum(tf.square(tf.matmul(self.first_order_diff_mat, \
                tf.reshape(self.momentum_cycle_A2A, [-1,1]))))

        self.momentum_loss_B2A \
            = tf.reduce_sum(tf.square(tf.matmul(self.first_order_diff_mat, \
                tf.reshape(self.momentum_B2A, [-1,1])))) \
                + tf.reduce_sum(tf.square(tf.matmul(self.first_order_diff_mat, \
                tf.reshape(self.momentum_cycle_B2B, [-1,1]))))

        self.momenta_loss = (self.momentum_loss_A2B + self.momentum_loss_B2A) / 2.0

        # Merge the two generators, the cycle loss and vector field regularization
        self.generator_loss = (1 - self.lambda_cycle - self.lambda_momenta) \
                                * self.gen_disc_loss \
                                + self.lambda_cycle * self.cycle_loss \
                                + self.lambda_momenta * self.momenta_loss

        # Compute the discriminator probability for pair of inputs
        self.discrimination_input_A_real_B_fake \
            = self.discriminator(input1=self.pitch_A_real, \
                input2=self.pitch_B_fake, reuse=True, \
                scope_name='discriminator_A')
        self.discrimination_input_A_fake_B_real \
            = self.discriminator(input1=self.pitch_A_fake, \
                input2=self.pitch_B_real, reuse=True, \
                scope_name='discriminator_A')

        self.discrimination_input_B_real_A_fake \
            = self.discriminator(input1=self.pitch_B_real, \
                input2=self.pitch_A_fake, reuse=True, \
                scope_name='discriminator_B')
        self.discrimination_input_B_fake_A_real \
            = self.discriminator(input1=self.pitch_B_fake, \
                input2=self.pitch_A_real, reuse = True, \
                scope_name='discriminator_B')

        # Compute discriminator loss for backprop
        self.discriminator_loss_input_A_real \
            = l1_loss(y=tf.zeros_like(self.discrimination_input_A_real_B_fake), \
                            y_hat=self.discrimination_input_A_real_B_fake)
        self.discriminator_loss_input_A_fake \
            = l1_loss(y=tf.ones_like(self.discrimination_input_A_fake_B_real), \
                            y_hat = self.discrimination_input_A_fake_B_real)
        self.discriminator_loss_A = (self.discriminator_loss_input_A_real \
                                     + self.discriminator_loss_input_A_fake) / 2.0

        self.discriminator_loss_input_B_real \
            = l1_loss(y=tf.zeros_like(self.discrimination_input_B_real_A_fake), \
                            y_hat=self.discrimination_input_B_real_A_fake)
        self.discriminator_loss_input_B_fake \
            = l1_loss(y=tf.ones_like(self.discrimination_input_B_fake_A_real), \
                            y_hat = self.discrimination_input_B_fake_A_real)
        self.discriminator_loss_B = (self.discriminator_loss_input_B_real \
                                     + self.discriminator_loss_input_B_fake) / 2.0

        # Merge the two discriminators into one
        self.discriminator_loss = (self.discriminator_loss_A \
                                    + self.discriminator_loss_B) / 2.0

        # Categorize variables to optimize the two sets separately
        trainable_variables = tf.trainable_variables()
        self.discriminator_vars = [var for var in trainable_variables \
                                   if 'discriminator' in var.name]
        self.generator_vars = [var for var in trainable_variables \
                               if 'generator' in var.name]

        # Reserved for test
        self.momentum_A2B_test = self.generator(input_pitch=self.pitch_A_test, \
                                    input_mfc=self.mfc_A_test, \
                                    reuse=True, scope_name='generator_A2B')
        self.generation_A2B_test = forward_tan(x=self.pitch_A_test, \
                                    p=self.momentum_A2B_test, kernel=self.kernel)

        self.momentum_B2A_test = self.generator(input_pitch=self.pitch_B_test, \
                                    input_mfc=self.mfc_B_test, \
                                    reuse=True, scope_name='generator_B2A')
        self.generation_B2A_test = forward_tan(x=self.pitch_B_test, \
                                    p=self.momentum_B2A_test, kernel=self.kernel)


    def optimizer_initializer(self):

        self.generator_learning_rate = tf.placeholder(tf.float32, \
                                    None, name = 'generator_learning_rate')
        self.discriminator_learning_rate = tf.placeholder(tf.float32, \
                                    None, name = 'discriminator_learning_rate')
        self.discriminator_optimizer \
            = tf.train.AdamOptimizer(learning_rate=self.discriminator_learning_rate, \
                beta1 = 0.5).minimize(self.discriminator_loss, \
                var_list = self.discriminator_vars)
        self.generator_optimizer \
            = tf.train.AdamOptimizer(learning_rate=self.generator_learning_rate, \
                beta1 = 0.5).minimize(self.generator_loss, \
                var_list=self.generator_vars) 

    def train(self, pitch_A, mfc_A, pitch_B, \
              mfc_B, lambda_cycle, \
              lambda_momenta, generator_learning_rate, \
              discriminator_learning_rate):

        generation_A, generation_B, momentum_A, momentum_B, \
        generator_loss, _ \
            = self.sess.run([self.generation_B2A, self.generation_A2B, \
                             self.momentum_B2A, self.momentum_A2B, \
                             self.gen_disc_loss, self.generator_optimizer], \
                                feed_dict = {self.lambda_cycle:lambda_cycle, \
                                             self.lambda_momenta:lambda_momenta, \
                                             self.pitch_A_real:pitch_A, \
                                             self.pitch_B_real:pitch_B, \
                                             self.mfc_A:mfc_A, \
                                             self.mfc_B:mfc_B, \
                                             self.generator_learning_rate:generator_learning_rate})

        discriminator_loss, _ \
            = self.sess.run([self.discriminator_loss, \
                             self.discriminator_optimizer], \
                            feed_dict = {self.pitch_A_real:pitch_A, \
                                         self.pitch_B_real:pitch_B, \
                                         self.mfc_A:mfc_A, \
                                         self.mfc_B:mfc_B, \
                                         self.discriminator_learning_rate:discriminator_learning_rate, \
                                         self.pitch_A_fake:generation_A, \
                                         self.pitch_B_fake:generation_B})

        self.train_step += 1

        return generator_loss, discriminator_loss, \
                generation_A, generation_B, momentum_A, momentum_B

    def train_generators(self, pitch_A, mfc_A, pitch_B, mfc_B, \
                lambda_cycle, lambda_momenta, generator_learning_rate):
        
        generation_A, generation_B, momenta_A, momenta_B, \
                generator_loss, _ = self.sess.run([self.generation_B2A, \
                                    self.generation_A2B, self.momentum_B2A, \
                                    self.momentum_A2B, self.gen_disc_loss, \
                                    self.generator_optimizer], \
                                    feed_dict={self.pitch_A_real:pitch_A, \
                                        self.pitch_B_real:pitch_B, \
                                        self.mfc_A:mfc_A, \
                                        self.mfc_B:mfc_B, \
                                        self.lambda_cycle:lambda_cycle, \
                                        self.lambda_momenta:lambda_momenta, \
                                        self.generator_learning_rate:generator_learning_rate})
                                    
        return generator_loss, generation_A, generation_B, momenta_A, momenta_B

    def train_discriminators(self, pitch_A, mfc_A, pitch_B, mfc_B, \
                            discriminator_learning_rate):

        gen_A, gen_B = self.sess.run([self.generation_B2A, self.generation_A2B], \
                                    feed_dict={self.pitch_A_real:pitch_A, \
                                        self.pitch_B_real:pitch_B, \
                                        self.mfc_A:mfc_A, \
                                        self.mfc_B:mfc_B})

        discriminator_loss, _  = self.sess.run([self.discriminator_loss, \
                                    self.discriminator_optimizer], \
                                    feed_dict = {self.pitch_A_real:pitch_A, \
                                        self.pitch_B_real:pitch_B, \
                                        self.mfc_A:mfc_A, \
                                        self.mfc_B:mfc_B, \
                                        self.discriminator_learning_rate:discriminator_learning_rate, \
                                        self.pitch_A_fake:gen_A, \
                                        self.pitch_B_fake:gen_B})
        
        return discriminator_loss

    def test_gen(self, mfc_A, pitch_A, mfc_B, pitch_B):
        gen_mom_B, gen_pitch_B = self.sess.run([self.momentum_A2B_test, \
                                    self.generation_A2B_test], \
                                    feed_dict={self.pitch_A_test:pitch_A, \
                                        self.mfc_A_test:mfc_A})


        gen_mom_A, gen_pitch_A = self.sess.run([self.momentum_B2A_test, \
                                    self.generation_B2A_test], \
                                    feed_dict={self.pitch_B_test:pitch_B, \
                                        self.mfc_B_test:mfc_B})
        
        return gen_pitch_A, gen_pitch_B, gen_mom_A, gen_mom_B

    def test(self, input_pitch, input_mfc, direction):

        if direction == 'A2B':
            generation = self.sess.run(self.generation_A2B_test, \
                                feed_dict = {self.pitch_A_test:input_pitch, \
                                    self.mfc_A_test:input_mfc})
        elif direction == 'B2A':
            generation = self.sess.run(self.generation_B2A_test, \
                                feed_dict = {self.pitch_B_test:input_pitch, \
                                    self.mfc_B_test:input_mfc})
        else:
            raise Exception('Conversion direction must be specified.')

        return generation


    def save(self, directory, filename):

        if not os.path.exists(directory):
            os.makedirs(directory)
        self.saver.save(self.sess, \
                        os.path.join(directory, filename))
        

    def load(self, filepath):

        self.saver.restore(self.sess, filepath)


    def summary(self):

        with tf.name_scope('generator_summaries'):
            cycle_loss_summary = tf.summary.scalar('cycle_loss', \
                                    self.cycle_loss)
            identity_loss_summary = tf.summary.scalar('identity_loss', \
                                    self.identity_loss)
            generator_loss_A2B_summary = tf.summary.scalar('generator_loss_A2B', \
                                    self.generator_loss_A2B)
            generator_loss_B2A_summary = tf.summary.scalar('generator_loss_B2A', \
                                    self.generator_loss_B2A)
            generator_loss_summary = tf.summary.scalar('generator_loss', \
                                    self.generator_loss)
            generator_summaries = tf.summary.merge([cycle_loss_summary, \
                                    identity_loss_summary, \
                                    generator_loss_A2B_summary, \
                                    generator_loss_B2A_summary, \
                                    generator_loss_summary])

        with tf.name_scope('discriminator_summaries'):
            discriminator_loss_A_summary \
                = tf.summary.scalar('discriminator_loss_A', \
                        self.discriminator_loss_A)
            discriminator_loss_B_summary \
                = tf.summary.scalar('discriminator_loss_B', \
                        self.discriminator_loss_B)
            discriminator_loss_summary \
                = tf.summary.scalar('discriminator_loss', \
                        self.discriminator_loss)
            discriminator_summaries \
                = tf.summary.merge([discriminator_loss_A_summary, \
                        discriminator_loss_B_summary, \
                        discriminator_loss_summary])

        return generator_summaries, discriminator_summaries


