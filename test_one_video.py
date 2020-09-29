import torch
import numpy as np
from collections import OrderedDict
from net_stdf import MFVQE
import utils
from tqdm import tqdm


ckp_path = 'exp/MFQEv2_R3_enlarge300x/ckp_290000.pt'  # trained at QP37, LDP, HM16.5

raw_yuv_path = '/media/x/Database/MFQEv2_dataset/test_18/raw/BasketballPass_416x240_500.yuv'
lq_yuv_path = '/media/x/Database/MFQEv2_dataset/test_18/HM16.5_LDP/QP37/BasketballPass_416x240_500.yuv'
h, w, nfs = 240, 416, 500


def main():
    # ==========
    # Load pre-trained model
    # ==========
    opts_dict = {
        'radius': 3, 
        'stdf': {
            'in_nc': 1, 
            'out_nc': 64, 
            'nf': 32, 
            'nb': 3, 
            'base_ks': 3, 
            'deform_ks': 3, 
            },
        'qenet': {
            'in_nc': 64, 
            'out_nc': 1, 
            'nf': 48, 
            'nb': 8, 
            'base_ks': 3, 
            },
        }   
    model = MFVQE(opts_dict=opts_dict)
    msg = f'loading model {ckp_path}...'
    print(msg)
    checkpoint = torch.load(ckp_path)
    if 'module.' in list(checkpoint['state_dict'].keys())[0]:  # multi-gpu training
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:]  # remove module
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:  # single-gpu training
        model.load_state_dict(checkpoint['state_dict'])

    msg = f'> model {ckp_path} loaded.'
    print(msg)
    model = model.cuda()
    model.eval()

    # ==========
    # Load entire video
    # ==========
    msg = f'loading raw and low-quality yuv...'
    print(msg)
    raw_y = utils.import_yuv(
        seq_path=raw_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=True
        )
    lq_y = utils.import_yuv(
        seq_path=lq_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=True
        )
    raw_y = raw_y.astype(np.float32) / 255.
    lq_y = lq_y.astype(np.float32) / 255.
    msg = '> yuv loaded.'
    print(msg)

    # ==========
    # Define criterion
    # ==========
    criterion = utils.PSNR()
    unit = 'dB'

    # ==========
    # Test
    # ==========
    pbar = tqdm(total=nfs, ncols=80)
    ori_psnr_counter = utils.Counter()
    enh_psnr_counter = utils.Counter()
    for idx in range(nfs):
        # load lq
        idx_list = list(range(idx-3,idx+4))
        idx_list = np.clip(idx_list, 0, nfs-1)
        input_data = []
        for idx_ in idx_list:
            input_data.append(lq_y[idx_])
        input_data = torch.from_numpy(np.array(input_data))
        input_data = torch.unsqueeze(input_data, 0).cuda()

        # enhance
        enhanced_frm = model(input_data)

        # eval
        gt_frm = torch.from_numpy(raw_y[idx]).cuda()
        batch_ori = criterion(input_data[0, 3, ...], gt_frm)
        batch_perf = criterion(enhanced_frm[0, 0, ...], gt_frm)
        ori_psnr_counter.accum(volume=batch_ori)
        enh_psnr_counter.accum(volume=batch_perf)

        # display
        pbar.set_description(
            "[{:.3f}] {:s} -> [{:.3f}] {:s}"
            .format(batch_ori, unit, batch_perf, unit)
            )
        pbar.update()
    
    pbar.close()
    ori_ = ori_psnr_counter.get_ave()
    enh_ = enh_psnr_counter.get_ave()
    print('ave ori [{:.3f}] {:s}, enh [{:.3f}] {:s}, delta [{:.3f}] {:s}'.format(
        ori_, unit, enh_, unit, (enh_ - ori_) , unit
        ))
    print('> done.')


if __name__ == '__main__':
    main()
