from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from matplotlib import pyplot as plt


def main():
    # train(patch_dim=15)

    model = load_model('model_result.h5')
    img_rows, img_cols = 565, 584

    img = image.load_img('DRIVE/test/images/01_test.tif', target_size=(img_rows, img_cols))
    x = image.img_to_array(img) / 255

    first_idx_row, first_idx_col = 7, 7

    patches = []
    for i in range(first_idx_row, img_rows-7):
        for j in range(first_idx_col, img_cols-7):
            element = x[i-7:i+7+1, j-7:j+7+1]
            patches.append(element)

    to_predict = np.asarray(patches)

    res = model.predict(to_predict)

    mask = []
    for k in range(0, len(res)):
        if res[k][1] >= 0.8:
            mask.append([1.0, 1.0, 1.0])
        else:
            mask.append([0.0, 0.0, 0.0])

    mask_img = np.asarray(mask).reshape(551, 570, 3)

    plt.imshow(mask_img, interpolation=None)
    plt.show()


if __name__ == '__main__':
    main()
