import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from torchvision.transforms import ToTensor, ToPILImage


def pil_to_opencv(pil_im):
    """Converts a PIL image into an OpenCV image."""
    return np.array(pil_im)[..., ::-1]


def opencv_to_pil(open_cv_arr):
    """Converts an OpenCV image into a PIL image."""
    return Image.fromarray(open_cv_arr[..., ::-1])


def torch_to_pil(torch_tensor):
    """Converts a torch tensor into a PIL image."""
    return ToPILImage()(torch_tensor)


def pil_to_torch(pil_im):
    """Converts a PIL image into a torch tensor."""
    return ToTensor()(pil_im)


def show_ims(ims, columns=5, figsize=(20, 10), titles=None, suptitle=None):
    """
    Show the given PIL images using matplotlib.
    
    Args:
        ims: A list of the PIL images
        columns: The number of columns
        figsize: The size of the matplotlib figure
        titles: A list with the same length as `imgs`, giving the
            title to display above each respective image.
        suptitle: The main title on top of the figure
    """
    if titles is not None:
        assert len(titles) == len(ims)

    plt.figure(figsize=figsize)
    for i, im in enumerate(ims):
        plt.subplot(len(ims) // columns + 1, columns, i + 1)
        plt.imshow(np.asarray(im))
        plt.grid(b=False)
        plt.axis('off')
        
        if titles is not None:
            plt.title(titles[i])
    if suptitle is not None:
        plt.suptitle(suptitle)


def show_imgs(imgs, columns=5, figsize=(20, 10), titles=None, suptitle=None):
    """
    Show the given images using matplotlib.
    
    Args:
        imgs: A list of the image paths
        columns: The number of columns
        figsize: The size of the matplotlib figure
        titles: A list with the same length as `imgs`, giving the
            title to display above each respective image.
        suptitle: The main title on top of the figure
    """
    return  show_ims([Image.open(img) for img in imgs],
                     columns=columns, figsize=figsize,
                     titles=titles, suptitle=suptitle)


def scale_im(im, scale=0.3):
    """
    Return a rescaled version of the given PIL Image.
    """
    im = im.copy()
    im.thumbnail([int(s*scale) for s in im.size])
    return im