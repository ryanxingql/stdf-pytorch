import skimage.metrics as skm


def calculate_psnr(img0, img1, data_range=None):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).
    
    Args:
        img0 (ndarray)
        img1 (ndarray)
        data_range (int, optional): Distance between minimum and maximum 
            possible values). By default, this is estimated from the image 
            data-type.
    
    Return:
        psnr (float)
    """
    psnr = skm.peak_signal_noise_ratio(img0, img1, data_range=data_range) 
    return psnr


def calculate_ssim(img0, img1, data_range=None):
    """Calculate SSIM (Structural SIMilarity).

    Args:
        img0 (ndarray)
        img1 (ndarray)
        data_range (int, optional): Distance between minimum and maximum 
            possible values). By default, this is estimated from the image 
            data-type.
    
    Return:
        ssim (float)
    """
    ssim = skm.structural_similarity(img0, img1, data_range=data_range)
    return ssim


def calculate_mse(img0, img1):
    """Calculate MSE (Mean Square Error).

    Args:
        img0 (ndarray)
        img1 (ndarray)

    Return:
        mse (float)
    """
    mse = skm.mean_squared_error(img0, img1)
    return mse
