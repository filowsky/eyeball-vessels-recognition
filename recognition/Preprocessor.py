import numpy as np
import pandas as pd
from skimage import io
from skimage.util import img_as_float
import os
import glob
import time


class Preprocessor:
    def __init__(self, total_patches, positive_proportion, patch_dim):
        self.total_patches = total_patches
        self.positive_proportion = positive_proportion
        self.patch_dim = patch_dim
        self.num_training_images = None
        self.patches_per_image = None
        self.current_img_index = -1
        self.current_img = None
        self.current_mask = None
        self.current_gt = None
        self.df = None

    def get_path(self, directory):
        imgs = glob.glob(directory + '/images/*.tif')
        imgs.sort()

        mask = glob.glob(directory + '/mask/*.gif')
        mask.sort()

        gt = glob.glob(directory + '/1st_manual/*.gif')
        gt.sort()

        return map(os.path.abspath, imgs), map(os.path.abspath, mask), map(os.path.abspath, gt)

    def load_next_img(self, data, mask_data, gt_data):
        if self.current_img_index < len(data) - 1:
            self.current_img_index += 1
            print("Working on image %d" % (self.current_img_index + 1))
            self.current_img = io.imread(data[self.current_img_index])
            self.current_mask = img_as_float(io.imread(mask_data[self.current_img_index]))
            self.current_gt = img_as_float(io.imread(gt_data[self.current_img_index]))

            return True
        else:
            print('No more images left in set')
            return False

    def save_img_data(self, proportion):
        pos_count = 0
        neg_count = 0

        while pos_count + neg_count < self.patches_per_image:
            i = np.random.randint(int(self.patch_dim / 2), self.current_img.shape[0] - int(self.patch_dim / 2))
            j = np.random.randint(int(self.patch_dim / 2), self.current_img.shape[1] - int(self.patch_dim / 2))
            h = int((self.patch_dim - 1) / 2)

            if int(np.sum(self.current_mask[i - h:i + h + 1, j - h:j + h + 1]) / self.patch_dim ** 2) == 1:
                ind = self.current_img_index * self.patches_per_image + pos_count + neg_count
                if int(self.current_gt[i, j]) == 1 and pos_count < proportion * self.patches_per_image:
                    self.df.loc[ind][0:-1] = np.reshape(self.current_img[i - h:i + h + 1, j - h:j + h + 1], -1)
                    self.df.loc[ind][self.patch_dim ** 2 * 3] = int(self.current_gt[i, j])
                    pos_count += 1
                elif int(self.current_gt[i, j]) == 0 and neg_count < (1 - proportion) * self.patches_per_image:
                    self.df.loc[ind][0:-1] = np.reshape(self.current_img[i - h:i + h + 1, j - h:j + h + 1], -1)
                    self.df.loc[ind][self.patch_dim ** 2 * 3] = int(self.current_gt[i, j])
                    neg_count += 1

    def preprocess(self):
        train, mask_train, gt_train = self.get_path('DRIVE/training')

        train_list = list(train)
        mask_train_list = list(mask_train)
        gt_train_list = list(gt_train)

        num_training_images = len(train_list)
        self.patches_per_image = self.total_patches / num_training_images
        self.current_img = io.imread(train_list[0])
        self.current_mask = img_as_float(io.imread(mask_train_list[0]))
        self.current_gt = img_as_float(io.imread(gt_train_list[0]))

        begin = time.time()
        print("Creating DataFrame")

        self.df = pd.DataFrame(index=np.arange(self.total_patches), columns=np.arange(self.patch_dim ** 2 * 3 + 1))

        print("Dataframe ready")

        while self.load_next_img(train_list, mask_train_list, gt_train_list):
            start = time.time()
            self.save_img_data(self.positive_proportion)
            print("Time taken for this image = %f secs" % (time.time() - start))

        last = len(self.df.columns) - 1
        labels = self.df[last]

        labels_matrix = labels.as_matrix()

        patches_dataset = self.df
        patches_dataset[last] = labels

        print("Writing to pickle\n")

        patches_dataset.to_pickle('results/patches_data_set.pkl')

        print("Total time taken = %f mins\n" % ((time.time() - begin) / 60.0))

        patches = self.df.iloc[:, :-1].as_matrix() / 255
        patches_formatted = patches.reshape(len(patches), int(len(patches[0]) / (self.patch_dim * 3)),
                                            int(len(patches[0]) / (self.patch_dim * 3)), 3)

        return patches_formatted, labels_matrix
