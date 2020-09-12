import numpy as np


def import_yuv_v1(video_path, startfrm, nfs, height_frame=0,
        width_frame=0, opt_bar=False, opt_clear=False):
    """Import Y U V channels from a yuv video.

    nfs: num of frames that you need.
    startfrm: start from 0.

    return: Y, U, V, each with (nfs, height, width), [0, 255], uint8
        if startfrm excesses the file len, return [] & no error.
    """
    fp = open(video_path, 'rb')  # 0101...bytes

    # retrieve resolution info from video path
    if height_frame == 0:
        res = video_path.split("-")[2].split("_")[0]
        width_frame = int(res.split("x")[0])
        height_frame = int(res.split("x")[1])

    d0 = height_frame // 2
    d1 = width_frame // 2
    y_size = height_frame * width_frame
    u_size = d0 * d1
    v_size = d0 * d1

    # target at startfrm
    blk_size = y_size + u_size + v_size
    fp.seek(blk_size * startfrm, 0)

    # init
    y_batch = []
    u_batch = []
    v_batch = []

    # extract
    for ite_frame in range(nfs):

        if ite_frame == 0:
            tmp_c = fp.read(1)
            if tmp_c == b'':  # startfrm > the last frame
                return [], [], []
            fp.seek(-1, 1)  # offset=-1, start from the present position

        y_frame = [ord(fp.read(1)) for i in range(y_size)]  # bytes -> ascii
        y_frame = np.array(y_frame, dtype=np.uint8).reshape((height_frame, \
            width_frame))
        y_batch.append(y_frame)

        u_frame = [ord(fp.read(1)) for i in range(u_size)]
        u_frame = np.array(u_frame, dtype=np.uint8).reshape((d0, d1))
        u_batch.append(u_frame)

        v_frame = [ord(fp.read(1)) for i in range(v_size)]
        v_frame = np.array(v_frame, dtype=np.uint8).reshape((d0, d1))
        v_batch.append(v_frame)

        if opt_bar:
            print("\r<%d, %d>" % (ite_frame, nfs - 1), end="", flush=True)
    if opt_clear:
        print("\r" + 20 * " ", end="\r", flush=True)
        
    fp.close()

    y_batch = np.array(y_batch)
    u_batch = np.array(u_batch)
    v_batch = np.array(v_batch)
    return y_batch, u_batch, v_batch


def import_y_v1(video_path, height_frame, width_frame, nfs,
        startfrm, opt_bar=False, opt_clear=False):
    """Import Y channel from a yuv 420p video.
    startfrm: start from 0
    return: y_batch, (nfs * height * width), dtype=uint8
    """
    fp_data = open(video_path, 'rb')

    y_size = height_frame * width_frame
    u_size = height_frame // 2 * (width_frame // 2)
    v_size = u_size

    # target at startfrm
    blk_size = y_size + u_size + v_size
    fp_data.seek(blk_size * startfrm, 0)

    # extract
    y_batch = []
    for ite_frame in range(nfs):
        
        y_frame = [ord(fp_data.read(1)) for k in range(y_size)]
        y_frame = np.array(y_frame, dtype=np.uint8).reshape((height_frame, \
            width_frame))
        fp_data.read(u_size + v_size)  # skip u and v
        y_batch.append(y_frame)

        if opt_bar:
            print("\r<%d, %d>" % (ite_frame, nfs - 1), end="", flush=True)
    if opt_clear:
        print("\r" + 20 * " ", end="\r", flush=True)

    fp_data.close()
    
    y_batch = np.array(y_batch)
    return y_batch


def calculate_psnr_v1(img1, img2, data_range=1.):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Args:
        img1 (ndarray): Input image 1/2 with type of np.float32 and range of 
            [0, data_range]. No matter HWC or CHW.
        img2 (ndarray): Input image 2/2 with type of np.float32 and range of 
            [0, data_range]
    
    Return:
        float: The PSNR result (ave over all channels).

    Hint:
        If calculate PSNR between two uint8 images, first .astype(np.uint8),
            and set data_range=255..
        10 * log_10 (A / mse_ave) = 10 * [log_10 (A) - log_10 (mse_ave)]
            = 10 * log_10 (A) - 10 * log_10 [(mse_c1 + mse_c2 + mse_c3) / 3]
            = C - 10 * log_10 (mse_c1 + mse_c2 + mse_c3)
            != PSNR_ave
    """
    assert img1.shape == img2.shape, (
        f"Image shapes are different: {img1.shape} vs. {img2.shape}.")
    assert img1.dtype == np.float32, (
        f"Image 1's type {img1.dtype} != np.float32.")
    assert img2.dtype == np.float32, (
        f"Image 2's type {img2.dtype} != np.float32.")

    mse = np.mean((img1 - img2)**2, dtype=np.float32)
    if mse == 0:
        return float('inf')
    psnr = 10 * np.log10(float(data_range**2) / mse)
    return psnr


def calculate_mse_v1(img1, img2):
    """Calculate MSE (Mean Square Error).
    
    Args:
        img1 (ndarray): Input image 1/2 with type of np.float32.
        img2 (ndarray): Input image 2/2 with type of np.float32.
    
    Return:
        (float): The MSE result.
    """
    assert img1.shape == img2.shape, (
        f"Image shapes are different: {img1.shape} vs. {img2.shape}.")
    assert img1.dtype == np.float32, (
        f"Image 1's type {img1.dtype} != np.float32.")
    assert img2.dtype == np.float32, (
        f"Image 2's type {img2.dtype} != np.float32.")

    # default to average flattened array. no need to first reshape into 1D array
    return np.mean((img1 - img2)**2, dtype=np.float32)
