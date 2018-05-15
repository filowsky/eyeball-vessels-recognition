from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from matplotlib import pyplot as plt
from recognition.train import train


def main(new_learning=False):
    if new_learning:
        train(patch_dim=15,
              tuple_size=3,
              num_classes=2,
              epochs=5)

    model = load_model('results/model_result.h5')
    img_rows, img_cols = 565, 584
    first_idx_row, first_idx_col = 7, 7
    is_vessel_prediction_threshold = 0.9
    res_rows, res_cols = 551, 570
    tuple_size = 3

    img = image.load_img('DRIVE/test/images/01_test.tif', target_size=(img_rows, img_cols))
    x = image.img_to_array(img) / 255

    patches = []
    for i in range(first_idx_row, img_rows - first_idx_row):
        for j in range(first_idx_col, img_cols - first_idx_col):
            element = x[i - first_idx_row:i + first_idx_row + 1, j - first_idx_col:j + first_idx_col + 1]
            patches.append(element)

    to_predict = np.asarray(patches)

    res = model.predict(to_predict)

    mask = []
    for k in range(0, len(res)):
        if res[k][1] >= is_vessel_prediction_threshold:
            mask.append([1.0, 1.0, 1.0])
        else:
            mask.append([0.0, 0.0, 0.0])

    mask_img = np.asarray(mask).reshape(res_rows, res_cols, tuple_size)

    plt.imshow(mask_img, interpolation=None)
    plt.show()


if __name__ == '__main__':
    main()
