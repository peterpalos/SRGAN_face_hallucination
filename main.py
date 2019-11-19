from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras import layers as ls
from keras.applications import VGG19
from keras.optimizers import Adam
import numpy as np
import os
import matplotlib.pyplot as plt

### Deep Learning project 2. station
# This is only a prototype with small training dataset and few epochs. Thy hyperparameters aren't improved. Some generated sample pictures are exported to the git folder but these aren't test image predictions just from the training set to show how it works.
# There is some problem probably with the GPU and if we run the model it freeze the computer therefore we attach a colab friendly code too if the problem also appears.
# It's important to set the train_number because of the training for loop

class MPSRGAN:
    def __init__(self):
        self.hr_size = (256, 256, 3)
        self.lr_size = (64, 64, 3)
        self.channels = 3
        self.batch_size = 20
        self.train_number = 500
        self.train_hr = 'train_hr'
        self.train_lr = 'train_lr'
        self.valid_hr = ''
        self.valid_lr = ''
        self.test_dir = ''
        self.df = 64
        self.optimizer = Adam(0.0002, 0.5)
        self.optimizer_d = Adam(0.00005, 0.5)

    def build_datagen(self):
        d = ImageDataGenerator(
            rescale=1./255)

        self.train_generator_lr = d.flow_from_directory(
            self.train_lr,
            target_size=(64, 64),
            batch_size=self.batch_size,
            class_mode=None,
            shuffle=False)

        self.train_generator_hr = d.flow_from_directory(
            self.train_hr,
            target_size=(256, 256),
            batch_size=self.batch_size,
            class_mode=None,
            shuffle=False)
        
        '''
        self.valid_generator_lr = d.flow_from_directory(
        self.valid_lr,
        target_size = self.lr_size,
        batch_size = self.batch_size,
        class_mode=None,
        shuffle=True)

        self.valid_generator_hr = d.flow_from_directory(
        self.valid_hr,
        target_size = self.hr_size,
        batch_size = self.batch_size,
        class_mode=None,
        shuffle=True)

        self.test_generator_lr = d.flow_from_directory(
        self.test_dir,
        target_size = self.hr_size,
        batch_size = self.batch_size,
        class_mode=None,
        shuffle=True)
        '''

    def build_generator(self, ):
        def residual_block(layer_input, filters):
            d = ls.Conv2D(filters, kernel_size=3, strides=1,
                          padding='same')(layer_input)
            d = ls.Activation('relu')(d)
            d = ls.BatchNormalization(momentum=0.8)(d)
            d = ls.Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
            d = ls.BatchNormalization(momentum=0.8)(d)
            d = ls.Add()([d, layer_input])
            return d

        def deconv2d(layer_input):
            u = ls.UpSampling2D(size=2)(layer_input)
            u = ls.Conv2D(256, kernel_size=3, strides=1, padding='same')(u)
            u = ls.Activation('relu')(u)
            return u

        img_lr = ls.Input(shape=self.lr_size)

        c1 = ls.Conv2D(64, kernel_size=9, strides=1, padding='same')(img_lr)
        c1 = ls.Activation('relu')(c1)

        r = residual_block(c1, 64)

        for _ in range(16 - 1):
            r = residual_block(r, 64)

        c2 = ls.Conv2D(64, kernel_size=3, strides=1, padding='same')(r)
        c2 = ls.BatchNormalization(momentum=0.8)(c2)
        c2 = ls.Add()([c2, c1])

        u1 = deconv2d(c2)
        u2 = deconv2d(u1)

        gen_hr = ls.Conv2D(self.channels, kernel_size=9,
                           strides=1, padding='same', activation='tanh')(u2)

        return Model(img_lr, gen_hr)

    def build_discriminator(self):
        def d_block(layer_input, filters, strides=1, bn=True):
            d = ls.Conv2D(filters, kernel_size=3, strides=strides,
                          padding='same')(layer_input)
            d = ls.LeakyReLU(alpha=0.2)(d)
            if bn:
                d = ls.BatchNormalization(momentum=0.8)(d)
            return d

        d0 = ls.Input(shape=self.hr_size)

        d1 = d_block(d0, self.df, bn=False)
        d2 = d_block(d1, self.df, strides=2)
        d3 = d_block(d2, self.df*2)
        d4 = d_block(d3, self.df*2, strides=2)
        d5 = d_block(d4, self.df*4)
        d6 = d_block(d5, self.df*4, strides=2)
        d7 = d_block(d6, self.df*8)
        d8 = d_block(d7, self.df*8, strides=2)

        d9 = ls.Dense(self.df*16)(d8)
        d10 = ls.LeakyReLU(alpha=0.2)(d9)
        validity = ls.Dense(1, activation='sigmoid')(d10)

        return Model(d0, validity)

    def build_VGG19(self):
        vgg = VGG19(weights="imagenet")
        vgg.outputs = [vgg.layers[9].output]
        img = ls.Input(shape=self.hr_size)
        img_features = vgg(img)

        return Model(img, img_features)

    def build_combined(self):
        self.discriminator.trainable = False
        img_lr = ls.Input(shape=self.lr_size)
        fake_hr = self.generator(img_lr)
        validity = self.discriminator(fake_hr)
        fake_features = self.VGG19(fake_hr)

        return Model([img_lr], [validity, fake_features])

    def train(self, epochs=5, show_images=False):
        self.build_datagen()

        self.VGG19 = self.build_VGG19()
        self.VGG19.trainable = False
        self.VGG19.compile(loss='mse',
                           optimizer=self.optimizer)

        self.generator = self.build_generator()

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
                                   optimizer=self.optimizer_d)

        self.combined = self.build_combined()
        self.combined.compile(loss=['binary_crossentropy', 'mse'],
                              loss_weights=[1e-3, 1],
                              optimizer=self.optimizer)
        
        # The reason why we use the 'next()' method instead of putting the iterator in the for loop because the original keras code has a bug and doesn't stop. 
        cnt = 0
        for _ in range(epochs):
            cnt += 1
            for _ in range( self.train_number // self.batch_size):

                imgs_lr = self.train_generator_lr.next()
                imgs_hr = self.train_generator_hr.next()
           
                valid = np.ones((self.batch_size,) + (16, 16, 1))
                fake = np.zeros((self.batch_size,) + (16, 16, 1))
            
                
                # Train generator
                self.discriminator.trainable = False
                image_features = self.VGG19.predict(imgs_hr)

                self.g_loss = self.combined.train_on_batch([imgs_lr], [valid, image_features])

                # Train discriminator
                self.discriminator.trainable = True
                fake_hr = self.generator.predict(imgs_lr)
                
                d_loss_fake = self.discriminator.train_on_batch(fake_hr, fake)
                d_loss_real = self.discriminator.train_on_batch(imgs_hr, valid)
                self.d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            if show_images:
                plt.subplot(1,3,1)
                plt.title('original')
                plt.imshow(( imgs_hr[0]*255).astype('uint8'))

                plt.subplot(1,3,2)
                plt.title('generated: %d. epoch' % cnt)
                plt.imshow(( fake_hr[0]*255).astype('uint8'))

                plt.subplot(1,3,3)
                plt.title('input')
                plt.imshow(( imgs_lr[0]*255).astype('uint8'))

                plt.show()

            print ('epoch: %d: [Discriminator loss: %f], [ Generator loss: %f]' % (cnt, self.d_loss, self.g_loss[0]))



trainer = MPSRGAN()
trainer.train( show_images=True)