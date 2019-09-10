from keras.layers import Layer
from keras_contrib.layers.normalization.instancenormalization import InputSpec

import numpy as np
import matplotlib.image as mpimg
from progress.bar import Bar
import datetime
import time
import json
import csv
import os

import keras.backend as K
import tensorflow as tf


class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            if len(image.shape) == 3:
                image = image[np.newaxis, :, :, :]

            if self.num_imgs < self.pool_size:  # fill up the image pool
                self.num_imgs = self.num_imgs + 1
                if len(self.images) == 0:
                    self.images = image
                else:
                    self.images = np.vstack((self.images, image))

                if len(return_images) == 0:
                    return_images = image
                else:
                    return_images = np.vstack((return_images, image))

            else:  # 50% chance that we replace an old synthetic image
                p = np.random.rand()
                if p > 0.5:
                    random_id = np.random.randint(0, self.pool_size)
                    tmp = self.images[random_id, :, :, :]
                    tmp = tmp[np.newaxis, :, :, :]
                    self.images[random_id, :, :, :] = image[0, :, :, :]
                    if len(return_images) == 0:
                        return_images = tmp
                    else:
                        return_images = np.vstack((return_images, tmp))
                else:
                    if len(return_images) == 0:
                        return_images = image
                    else:
                        return_images = np.vstack((return_images, image))

        return return_images


def load_data(subfolder='', generator=False):
    def create_image_array(image_list, image_path, image_size, nr_of_channels):
        bar = Bar('Loading...', max=len(image_list))

        # Define image array
        image_array = np.empty((len(image_list),) + (image_size) + (nr_of_channels,))
        i = 0
        for image_name in image_list:
            # If file is image...
            if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
                # Load image and convert into np.array
                image = mpimg.imread(os.path.join(image_path, image_name))  # Normalized to [0,1]
                # image = np.array(Image.open(os.path.join(image_path, image_name)))

                # Add third dimension if image is 2D
                if nr_of_channels == 1:  # Gray scale image -> MR image
                    image = image[:, :, np.newaxis]

                # Normalize image with (max 8 bit value - 1)
                image = image * 2 - 1
                # image = image / 127.5 - 1

                # Add image to array
                image_array[i, :, :, :] = image
                i += 1
                bar.next()
        bar.finish()

        return image_array

    # Image paths
    trainA_path = os.path.join('data', subfolder, 'trainA')
    trainB_path = os.path.join('data', subfolder, 'trainB')
    testA_path = os.path.join('data', subfolder, 'testA')
    testB_path = os.path.join('data', subfolder, 'testB')

    # Image file names
    trainA_image_names = sorted(os.listdir(trainA_path))
    trainB_image_names = sorted(os.listdir(trainB_path))
    testA_image_names = sorted(os.listdir(testA_path))
    testB_image_names = sorted(os.listdir(testB_path))

    # Examine one image to get size and number of channels
    im_test = mpimg.imread(os.path.join(trainA_path, trainA_image_names[0]))
    # im_test = np.array(Image.open(os.path.join(trainA_path, trainA_image_names[0])))

    if len(im_test.shape) == 2:
        image_size = im_test.shape
        nr_of_channels = 1
    else:
        image_size = im_test.shape[0:-1]
        nr_of_channels = im_test.shape[-1]

    trainA_images = create_image_array(trainA_image_names, trainA_path, image_size, nr_of_channels)
    trainB_images = create_image_array(trainB_image_names, trainB_path, image_size, nr_of_channels)
    testA_images = create_image_array(testA_image_names, testA_path, image_size, nr_of_channels)
    testB_images = create_image_array(testB_image_names, testB_path, image_size, nr_of_channels)

    return {"image_size": image_size, "nr_of_channels": nr_of_channels,
            "trainA_images": trainA_images, "trainB_images": trainB_images,
            "testA_images": testA_images, "testB_images": testB_images,
            "trainA_image_names": trainA_image_names,
            "trainB_image_names": trainB_image_names,
            "testA_image_names": testA_image_names,
            "testB_image_names": testB_image_names}


def write_metadata_to_JSON(model, opt):
    # Save meta_data
    data = {}
    data['meta_data'] = []
    data['meta_data'].append({
        'img shape: height,width,channels': opt['img_shape'],
        'batch size': opt['batch_size'],
        'save training img interval': opt['save_training_img_interval'],
        'normalization function': str(model['normalization']),
        'lambda_ABA': opt['lambda_ABA'],
        'lambda_BAB': opt['lambda_BAB'],
        'lambda_adversarial': opt['lambda_adversarial'],
        'learning_rate_D': opt['learning_rate_D'],
        'learning rate G': opt['learning_rate_G'],
        'epochs': opt['epochs'],
        'use linear decay on learning rates': opt['use_linear_decay'],
        'epoch where learning rate linear decay is initialized (if use_linear_decay)': opt['decay_epoch'],
        'generator iterations': opt['generator_iterations'],
        'discriminator iterations': opt['discriminator_iterations'],
        'use patchGan in discriminator': opt['use_patchgan'],
        'beta 1': opt['beta_1'],
        'beta 2': opt['beta_2'],
        'REAL_LABEL': opt['REAL_LABEL'],
        'number of A train examples': len(opt['A_train']),
        'number of B train examples': len(opt['B_train']),
        'number of A test examples': len(opt['A_test']),
        'number of B test examples': len(opt['B_test']),
        'discriminator sigmoid': opt['discriminator_sigmoid'],
        'resize convolution': opt['use_resize_convolution'],
    })

    with open('{}/meta_data.json'.format(opt['out_dir']), 'w') as outfile:
        json.dump(data, outfile, sort_keys=True)


def write_loss_data_to_file(opt, history):
    keys = sorted(history.keys())
    with open('images/{}/loss_output.csv'.format(opt['date_time']), 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(keys)
        writer.writerows(zip(*[history[key] for key in keys]))


def join_and_save(opt, images, save_path):
    # Join images
    image = np.hstack(images)

    # Save images
    if opt['channels'] == 1:
        image = image[:, :, 0]

    mpimg.imsave(save_path, image, vmin=-1, vmax=1, cmap='gray')


def save_epoch_images(model, opt, epoch, num_saved_images=1):
    # Save training images
    nr_train_im_A = opt['A_train'].shape[0]
    nr_train_im_B = opt['B_train'].shape[0]

    rand_ind_A = np.random.randint(nr_train_im_A)
    rand_ind_B = np.random.randint(nr_train_im_B)

    real_image_A = opt['A_train'][rand_ind_A]
    real_image_B = opt['B_train'][rand_ind_B]
    synthetic_image_B = model['G_A2B'].predict(real_image_A[np.newaxis])[0]
    synthetic_image_A = model['G_B2A'].predict(real_image_B[np.newaxis])[0]
    reconstructed_image_A = model['G_B2A'].predict(synthetic_image_B[np.newaxis])[0]
    reconstructed_image_B = model['G_A2B'].predict(synthetic_image_A[np.newaxis])[0]

    save_path_A = '{}/train_A/epoch{}.png'.format(opt['out_dir'], epoch)
    save_path_B = '{}/train_B/epoch{}.png'.format(opt['out_dir'], epoch)
    if opt['paired_data']:
        real_image_Ab = opt['B_train'][rand_ind_A]
        real_image_Ba = opt['A_train'][rand_ind_B]
        join_and_save(opt, (real_image_Ab, real_image_A, synthetic_image_B, reconstructed_image_A), save_path_A)
        join_and_save(opt, (real_image_Ba, real_image_B, synthetic_image_A, reconstructed_image_B), save_path_B)
    else:
        join_and_save(opt, (real_image_A, synthetic_image_B, reconstructed_image_A), save_path_A)
        join_and_save(opt, (real_image_B, synthetic_image_A, reconstructed_image_B), save_path_B)

    # Save test images
    real_image_A = opt['A_test'][0]
    real_image_B = opt['B_test'][0]
    synthetic_image_B = model['G_A2B'].predict(real_image_A[np.newaxis])[0]
    synthetic_image_A = model['G_B2A'].predict(real_image_B[np.newaxis])[0]
    reconstructed_image_A = model['G_B2A'].predict(synthetic_image_B[np.newaxis])[0]
    reconstructed_image_B = model['G_A2B'].predict(synthetic_image_A[np.newaxis])[0]

    save_path_A = '{}/test_A/epoch{}.png'.format(opt['out_dir'], epoch)
    save_path_B = '{}/test_B/epoch{}.png'.format(opt['out_dir'], epoch)
    if opt['paired_data']:
        real_image_Ab = opt['B_test'][0]
        real_image_Ba = opt['A_test'][0]
        join_and_save(opt, (real_image_Ab, real_image_A, synthetic_image_B, reconstructed_image_A), save_path_A)
        join_and_save(opt, (real_image_Ba, real_image_B, synthetic_image_A, reconstructed_image_B), save_path_B)
    else:
        join_and_save(opt, (real_image_A, synthetic_image_B, reconstructed_image_A), save_path_A)
        join_and_save(opt, (real_image_B, synthetic_image_A, reconstructed_image_B), save_path_B)


def save_tmp_images(model, opt, real_image_A, real_image_B, synthetic_image_A, synthetic_image_B):
    try:
        reconstructed_image_A = model['G_B2A'].predict(synthetic_image_B[np.newaxis])[0]
        reconstructed_image_B = model['G_A2B'].predict(synthetic_image_A[np.newaxis])[0]

        real_images = np.vstack((real_image_A, real_image_B))
        synthetic_images = np.vstack((synthetic_image_B, synthetic_image_A))
        reconstructed_images = np.vstack((reconstructed_image_A, reconstructed_image_B))

        save_path = '{}/tmp.png'.format(opt['out_dir'])
        join_and_save(opt, (real_images, synthetic_images, reconstructed_images), save_path)
    except: # Ignore if file is open
        pass


def get_lr_linear_decay_rate(opt):
    # Calculate decay rates
    # max_nr_images = max(len(opt['A_train']), len(opt['B_train']))

    nr_train_im_A = opt['A_train'].shape[0]
    nr_train_im_B = opt['B_train'].shape[0]
    nr_batches_per_epoch = int(np.ceil(np.max((nr_train_im_A, nr_train_im_B)) / opt['batch_size']))

    updates_per_epoch_D = 2 * nr_batches_per_epoch
    updates_per_epoch_G = nr_batches_per_epoch
    nr_decay_updates_D = (opt['epochs'] - opt['decay_epoch'] + 1) * updates_per_epoch_D
    nr_decay_updates_G = (opt['epochs'] - opt['decay_epoch'] + 1) * updates_per_epoch_G
    decay_D = opt['learning_rate_D'] / nr_decay_updates_D
    decay_G = opt['learning_rate_G'] / nr_decay_updates_G

    return decay_D, decay_G


def update_lr(model, decay):
    new_lr = K.get_value(model.optimizer.lr) - decay
    if new_lr < 0:
        new_lr = 0
    # print(K.get_value(model.optimizer.lr))
    K.set_value(model.optimizer.lr, new_lr)


def print_ETA(opt, start_time, epoch, nr_im_per_epoch, loop_index):
    passed_time = time.time() - start_time

    iterations_so_far = ((epoch - 1) * nr_im_per_epoch + loop_index) / opt['batch_size']
    iterations_total = opt['epochs'] * nr_im_per_epoch / opt['batch_size']
    iterations_left = iterations_total - iterations_so_far
    eta = round(passed_time / (iterations_so_far + 1e-5) * iterations_left)

    passed_time_string = str(datetime.timedelta(seconds=round(passed_time)))
    eta_string = str(datetime.timedelta(seconds=eta))
    print('Elapsed time', passed_time_string, ': ETA in', eta_string)


def save_model(opt, model, epoch):
    # Create folder to save model architecture and weights
    directory = os.path.join('saved_models', opt['date_time'])
    if not os.path.exists(directory):
        os.makedirs(directory)

    weights_path = '{}/{}_weights_epoch_{}.hdf5'.format(directory, model.name, epoch)
    model.save_weights(weights_path)
    model_path = '{}/{}_model_epoch_{}.json'.format(directory, model.name, epoch)
    model.save_weights(model_path)
    json_string = model.to_json()
    with open(model_path, 'w') as outfile:
        json.dump(json_string, outfile)
    print('{} has been saved in saved_models/{}/'.format(model.name, opt['date_time']))

