"""Toolbox for python code.

Content:
    File IO:
        import_yuv: 读取8bit yuv420p video.
        FileClient
        dict2str
    Conversion: 
        img2float32 / ndarray2img: float32或uint8输入, 自动判断类型
        rgb2ycbcr / ycbcr2rgb: 输入(..., 3)图像.
        rgb2gray / gray2rgb
        bgr2rgb / rgb2bgr
        paired_random_crop
        augment
        totensor
    Metrics:
        calculate_psnr
        calculate_ssim
        calculate_mse
    Deep learning:
        set_random_seed
        run_mp_train: 运行torch.multiprocessing
        get_dist_info
        DistSampler
    System:
        mkdir
        get_timestr
        Timer
    LMDB:
        make_lmdb_from_imgs
    Wasted

To-do:
    yuv2rgb: 8 bit yuv 420p
    yuv2ycbcr: 8 bit yuv 420p
    ycbcr2yuv: 8 bit yuv 420p
    imread

Ref:
    scikit-image: https://scikit-image.org/docs/
    mmcv: https://mmcv.readthedocs.io/en/latest/
    opencv-python
    BasicSR: https://github.com/xinntao/BasicSR

Principle:
    集成常用函数, 统一调用所需包, 统一命名格式.
    不重复造轮子! 了解原理即可.

Contact: xingql@buaa.edu.cn
"""
from .file_io import (import_yuv, FileClient, dict2str, CPUPrefetcher)
from .conversion import (img2float32, ndarray2img, rgb2ycbcr, ycbcr2rgb, 
rgb2gray, gray2rgb, bgr2rgb, rgb2bgr, paired_random_crop, augment, 
totensor)
from .metrics import (calculate_psnr, calculate_ssim, calculate_mse)
from .deep_learning import (set_random_seed, init_dist, get_dist_info, 
DistSampler, create_dataloader, CharbonnierLoss, PSNR, CosineAnnealingRestartLR)
from .system import (mkdir, get_timestr, Timer, Counter)
from .lmdb import make_lmdb_from_imgs, make_y_lmdb_from_yuv


__all__ = [
    'import_yuv', 'FileClient', 'dict2str', 'CPUPrefetcher', 
    'img2float32', 'ndarray2img', 'rgb2ycbcr', 'ycbcr2rgb', 'rgb2gray', 
    'gray2rgb', 'bgr2rgb', 'rgb2bgr', 'paired_random_crop', 'augment', 
    'totensor', 
    'calculate_psnr', 'calculate_ssim', 'calculate_mse', 
    'set_random_seed', 'init_dist', 'get_dist_info', 'DistSampler', 
    'create_dataloader', 'CharbonnierLoss', 'PSNR', 'CosineAnnealingRestartLR', 
    'mkdir', 'get_timestr', 'Timer', 'Counter', 
    'make_lmdb_from_imgs', 'make_y_lmdb_from_yuv', 
    ]
