import numpy as np
import skimage
import utils
import pathlib


def otsu_thresholding(im: np.ndarray) -> int:
    """
        Otsu's thresholding algorithm that segments an image into 1 or 0 (True or False)
        The function takes in a grayscale image and outputs a boolean image

        args:
            im: np.ndarray of shape (H, W) in the range [0, 255] (dtype=np.uint8)
        return:
            (int) the computed thresholding value
    """
    assert im.dtype == np.uint8
    ### START YOUR CODE HERE ### (You can change anything inside this block) 
    # You can also define other helper functions

    # 1) Compute normalized histogram
    L = 256 	# = Number of intensities
    (hist, _) = np.histogram(im, L, (0, L - 1))
    p = hist / np.sum(hist)

    # 2) Compute cumulative sums
    P_1 = np.cumsum(p)

    # 3) Comput cumulative means
    m = np.cumsum(np.multiply(p, np.arange(0,L)))

    # 4) Compute global mean
    m_g = np.sum(np.multiply(p, np.arange(0,L)))

    # 5) Compute between-class variance term
    sigma_b = np.zeros(L)
    for k in range(L):
    	denominator = P_1[k]*(1 - P_1[k])
    	if np.abs(denominator) <= 1e-8:
    		sigma_b[k] = 0
    	else:
    		sigma_b[k] = ((m_g * P_1[k] - m[k]) ** 2) / denominator

    # 6) Obtain Otsu threshold
    k_maxes = [j for j, i in enumerate(sigma_b) if i == np.amax(sigma_b)]
    threshold = int(sum(k_maxes) / len(k_maxes))

    return threshold
    ### END YOUR CODE HERE ### 


if __name__ == "__main__":
    # DO NOT CHANGE
    impaths_to_segment = [
        pathlib.Path("thumbprint.png"),
        pathlib.Path("polymercell.png")
    ]
    for impath in impaths_to_segment:
        im = utils.read_image(impath)
        threshold = otsu_thresholding(im)
        print("Found optimal threshold:", threshold)

        # Segment the image by threshold
        segmented_image = (im >= threshold)
        assert im.shape == segmented_image.shape, \
            "Expected image shape ({}) to be same as thresholded image shape ({})".format(
                im.shape, segmented_image.shape)
        assert segmented_image.dtype == np.bool, \
            "Expected thresholded image dtype to be np.bool. Was: {}".format(
                segmented_image.dtype)

        segmented_image = utils.to_uint8(segmented_image)

        save_path = "{}-segmented.png".format(impath.stem)
        utils.save_im(save_path, segmented_image)


