import numpy as np
import pandas as pd
from skimage import io
from skimage.util import img_as_float
import os
import glob
import time


class Preprocessor:
    total_patches = 80000
    num_training_images = None
    patches_per_image = None
    patch_dim = None
    current_img_index = -1
    current_img = None
    current_mask = None
    current_gt = None
    positive_proportion = 0.8
    df = None

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

        global df

        while pos_count + neg_count < self.patches_per_image:
            # Choose a random point
            i = np.random.randint(int(self.patch_dim / 2), self.current_img.shape[0] - int(self.patch_dim / 2))
            j = np.random.randint(int(self.patch_dim / 2), self.current_img.shape[1] - int(self.patch_dim / 2))
            h = int((self.patch_dim - 1) / 2)
            if int(np.sum(self.current_mask[i - h:i + h + 1, j - h:j + h + 1]) / self.patch_dim ** 2) == 1:
                ind = self.current_img_index * self.patches_per_image + pos_count + neg_count
                # If a positive sample is found and positive count hasn't reached its limit
                if int(self.current_gt[i, j]) == 1 and pos_count < proportion * self.patches_per_image:
                    df.loc[ind][0:-1] = np.reshape(self.current_img[i - h:i + h + 1, j - h:j + h + 1], -1)
                    df.loc[ind][self.patch_dim ** 2 * 3] = int(self.current_gt[i, j])
                    pos_count += 1
                # If a negative sample is found and negative count hasn't reached its limit
                elif int(self.current_gt[i, j]) == 0 and neg_count < (1 - proportion) * self.patches_per_image:
                    df.loc[ind][0:-1] = np.reshape(self.current_img[i - h:i + h + 1, j - h:j + h + 1], -1)
                    df.loc[ind][self.patch_dim ** 2 * 3] = int(self.current_gt[i, j])
                    neg_count += 1

    def preprocess(self, dataset_size, proportion, patch_size):
        self.patch_dim = patch_size
        train, mask_train, gt_train = self.get_path('DRIVE/training')

        train_list = list(train)
        mask_train_list = list(mask_train)
        gt_train_list = list(gt_train)

        num_training_images = len(train_list)
        self.patches_per_image = dataset_size / num_training_images
        self.current_img = io.imread(train_list[0])
        self.current_mask = img_as_float(io.imread(mask_train_list[0]))
        self.current_gt = img_as_float(io.imread(gt_train_list[0]))

        begin = time.time()
        print("Creating DataFrame")

        self.df = pd.DataFrame(index=np.arange(dataset_size), columns=np.arange(patch_size ** 2 * 3 + 1))

        print("Dataframe ready")

        while self.load_next_img(train_list, mask_train_list, gt_train_list):
            start = time.time()
            self.save_img_data(proportion)
            print("Time taken for this image = %f secs" % (time.time() - start))

        print("\nMean Normalising\n")
        last = len(df.columns) - 1
        labels = df[last]

        labels_matrix = labels.as_matrix()

        mean_normalised_df = df
        mean_normalised_df[last] = labels

        # print("Randomly shuffling the datasets\n")
        # mean_normalised_df = mean_normalised_df.iloc[np.random.permutation(len(df))]
        # mean_normalised_df = mean_normalised_df.reset_index(drop=True)

        print("Writing to pickle\n")

        mean_normalised_df.to_pickle('mean_normalised_df_no_class_bias.pkl')

        print("Total time taken = %f mins\n" % ((time.time() - begin) / 60.0))

        patches = df.iloc[:, :-1].as_matrix() / 255
        patches_formatted = patches.reshape(len(patches), int(len(patches[0]) / (self.patch_dim * 3)), int(len(patches[0]) / (self.patch_dim * 3)), 3)

        return patches_formatted, labels_matrix
