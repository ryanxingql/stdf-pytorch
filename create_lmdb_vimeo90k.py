"""
Create LMDB for the training set of Vimeo-90K.

GT: 64,612 training sequences out of 91701 7-frame sequences.
LQ: HM16.5-intra-compressed sequences.
key: assigned from 00000 to 99999.

Sym-link Vimeo-90K dataset root to ./data/vimeo90k folder.
"""
import argparse
import os
import glob
import yaml
import os.path as op
from utils import make_y_lmdb_from_yuv

parser = argparse.ArgumentParser()
parser.add_argument(
    '--opt_path', type=str, default='option_R3_vimeo90k_4G.yml', 
    help='Path to option YAML file.'
    )
args = parser.parse_args()

yml_path = args.opt_path
radius = 3  # must be 3!!! otherwise, you should change dataset.py


def create_lmdb_for_vimeo90k():
    # video info
    with open(yml_path, 'r') as fp:
        fp = yaml.load(fp, Loader=yaml.FullLoader)
        root_dir = fp['dataset']['root']
        gt_folder = fp['dataset']['train']['gt_folder']
        lq_folder = fp['dataset']['train']['lq_folder']
        gt_path = fp['dataset']['train']['gt_path']
        lq_path = fp['dataset']['train']['lq_path']
        meta_path = fp['dataset']['train']['meta_path']
    gt_dir = op.join(root_dir, gt_folder)
    lq_dir = op.join(root_dir, lq_folder)
    lmdb_gt_path = op.join(root_dir, gt_path)
    lmdb_lq_path = op.join(root_dir, lq_path)
    meta_path = op.join(root_dir, meta_path)

    # scan all videos
    print('Scaning meta list...')
    gt_video_list = []
    lq_video_list = []
    meta_fp = open(meta_path, 'r')
    while True:
        new_line = meta_fp.readline().split('\n')[0]
        if new_line == '':
            break
        vid_name = new_line.split('/')[0] + '_' + new_line.split('/')[1]
        qt_path = op.join(
            gt_dir, vid_name + '.yuv'
            )
        gt_video_list.append(qt_path)
        lq_path = op.join(
            lq_dir, vid_name + '.yuv'
            )
        lq_video_list.append(lq_path)        

    msg = f'> {len(gt_video_list)} videos found.'
    print(msg)

    # generate LMDB for GT
    print("Scaning GT frames (only center frames of each sequence)...")
    frm_list = []
    for gt_video_path in gt_video_list:
        nfs = 7
        num_seq = nfs // (2 * radius + 1)
        frm_list.append([radius + iter_seq * (2 * radius + 1) for iter_seq in range(num_seq)])
    num_frm_total = sum([len(frms) for frms in frm_list])
    msg = f'> {num_frm_total} frames found.'
    print(msg)
    key_list = []
    video_path_list = []
    index_frame_list = []
    for iter_vid in range(len(gt_video_list)):
        frms = frm_list[iter_vid]
        for iter_frm in range(len(frms)):
            key_list.append('{:03d}/{:03d}/im4.png'.format(iter_vid+1, iter_frm+1))
            video_path_list.append(gt_video_list[iter_vid])
            index_frame_list.append(frms[iter_frm])
    print("Writing LMDB for GT data...")
    make_y_lmdb_from_yuv(
        video_path_list=video_path_list, 
        yuv_type='444p', 
        h=256, 
        w=448, 
        index_frame_list=index_frame_list, 
        key_list=key_list, 
        lmdb_path=lmdb_gt_path, 
        multiprocessing_read=True, 
        )
    print("> Finish.")

    # generate LMDB for LQ
    print("Scaning LQ frames...")
    len_input = 2 * radius + 1
    frm_list = []
    for lq_video_path in lq_video_list:
        nfs = 7
        num_seq = nfs // len_input
        frm_list.append([list(range(iter_seq * len_input, (iter_seq + 1) \
            * len_input)) for iter_seq in range(num_seq)])
    num_frm_total = sum([len(frms) * len_input for frms in frm_list])
    msg = f'> {num_frm_total} frames found.'
    print(msg)
    key_list = []
    video_path_list = []
    index_frame_list = []
    for iter_vid in range(len(lq_video_list)):
        frm_seq = frm_list[iter_vid]
        for iter_seq in range(len(frm_seq)):
            key_list.extend(['{:03d}/{:03d}/im{:d}.png'.format(iter_vid+1, \
                iter_seq+1, i) for i in range(1, len_input+1)])
            video_path_list.extend([lq_video_list[iter_vid]] * len_input)
            index_frame_list.extend(frm_seq[iter_seq])
    print("Writing LMDB for LQ data...")
    make_y_lmdb_from_yuv(
        video_path_list=video_path_list, 
        yuv_type='444p', 
        h=256, 
        w=448, 
        index_frame_list=index_frame_list, 
        key_list=key_list, 
        lmdb_path=lmdb_lq_path, 
        multiprocessing_read=True, 
        )
    print("> Finish.")

    # sym-link
    if not op.exists('data/vimeo90k'):
        if not op.exists('data/'):
            os.system("mkdir data/")
        os.system(f"ln -s {root_dir} ./data/vimeo90k")
        print("Sym-linking done.")
    else:
        print("data/vimeo90k already exists.")
    

if __name__ == '__main__':
    create_lmdb_for_vimeo90k()
