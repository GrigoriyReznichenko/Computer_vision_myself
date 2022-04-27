import numpy as np
import cv2
from skimage.transform import rescale


def generate(threhs_count=0.99, threhs_size=0.1, init_size=100):
    return np.uint8(cv2.GaussianBlur(rescale(np.uint8(np.random.rand(init_size, init_size) > threhs_count) * 255, 6),
                                     (21, 21), 3) > threhs_size) * 255


def draw_random_line(img, point_count=20):
    b0 = img.shape[0] * (np.random.rand() + np.random.rand()) / 5
    b1 = (np.random.rand() - 0.5) * 3
    x1 = np.random.randint(int(img.shape[1] / 2) - 1)
    x2 = np.random.randint(x1 + 1, img.shape[1])
    x = np.arange(x1, x2)
    y = np.int0(b0 + b1 * x)
    x = x[(y >= 0) & (y < img.shape[0])]
    y = y[(y >= 0) & (y < img.shape[0])]
    coords = np.dstack((x, y))
    coords = coords.squeeze()
    img_copy = img.copy()
    point_count = min(point_count, x.shape[0])
    for i in np.random.choice(range(x.shape[0]), point_count, replace=False):
        cv2.circle(img_copy, coords[i] + np.random.randint(0, max(int(point_count / 2), 1), 2), 7, 255, -1)
    return img_copy


def super_generaton(count=1, init_size=280, rand_line_count=5):
    images = []
    for i in range(count):
        gen = generate(np.random.uniform(0.999, 1), 0.1, init_size=init_size)
        if rand_line_count:
            for j in range(np.random.randint(0, rand_line_count) + 1):
                gen = draw_random_line(gen, np.random.randint(10, 50))
        images.append(gen)
    return np.array(images)[0, :, :]

