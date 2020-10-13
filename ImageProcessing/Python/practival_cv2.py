import cv2
import numpy as np

def image_transf(img_path: str, t_x: int, t_y: int):
    """     This function transfer images
    Args:
        img_path (str): the image's path
        t_x: the number of pixel to shift x-axis
        t_y: the number of pixel to shift y-axis
    Return:
        translated (np.array): the transfer image
    """
    img = cv2.imread(img_path)
    mtx = np.float32([
        [1, 0, t_x],
        [0, 1, t_y]
    ])
    translated = cv2.warpAffine(img, mtx, (img.shape[1], img.shape[0]))

    return translated


def image_rotation(img_path: str, angle: int, scale: float):
    """         This function rotation image
    Args:
        img_path (str): the image's path
        angle: the number of angle (degree) rotation
        scale: the ratio of image
    Return:
        rotator (np.array): the rotation image
    """
    

