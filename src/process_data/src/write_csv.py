# modified from https://github.com/TengdaHan/CoCLR
import os
import argparse
import csv
import glob

def write_list(data_list, path, ):
    with open(path, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for row in data_list:
            if row: writer.writerow(row)
    print('split saved to %s' % path)

def main_UCF101(f_root, splits_root, csv_root='../data/ucf101/'):
    '''generate training/testing split, count number of available frames, save in csv'''
    if not os.path.exists(csv_root): os.makedirs(csv_root)
    for which_split in [1,2,3]:
        train_set = []
        test_set = []
        train_split_file = os.path.join(splits_root, 'trainlist%02d.txt' % which_split)
        with open(train_split_file, 'r') as f:
            for line in f:
                vpath = os.path.join(f_root, line.split(' ')[0][0:-4]) + '/'
                train_set.append([vpath, len(glob.glob(os.path.join(vpath, '*.jpg')))])

        test_split_file = os.path.join(splits_root, 'testlist%02d.txt' % which_split)
        with open(test_split_file, 'r') as f:
            for line in f:
                vpath = os.path.join(f_root, line.rstrip()[0:-4]) + '/'
                test_set.append([vpath, len(glob.glob(os.path.join(vpath, '*.jpg')))])

        write_list(train_set, os.path.join(csv_root, 'train_split%02d.csv' % which_split))
        write_list(test_set, os.path.join(csv_root, 'test_split%02d.csv' % which_split))


def main_HMDB51(f_root, splits_root, csv_root='../data/hmdb51/'):
    '''generate training/testing split, count number of available frames, save in csv'''
    if not os.path.exists(csv_root): os.makedirs(csv_root)
    for which_split in [1,2,3]:
        train_set = []
        test_set = []
        split_files = sorted(glob.glob(os.path.join(splits_root, '*_test_split%d.txt' % which_split)))
        assert len(split_files) == 51
        for split_file in split_files:
            action_name = os.path.basename(split_file)[0:-16]
            with open(split_file, 'r') as f:
                for line in f:
                    video_name = line.split(' ')[0]
                    _type = line.split(' ')[1]
                    vpath = os.path.join(f_root, action_name, video_name[0:-4]) + '/'
                    if _type == '1':
                        train_set.append([vpath, len(glob.glob(os.path.join(vpath, '*.jpg')))])
                    elif _type == '2':
                        test_set.append([vpath, len(glob.glob(os.path.join(vpath, '*.jpg')))])

        write_list(train_set, os.path.join(csv_root, 'train_split%02d.csv' % which_split))
        write_list(test_set, os.path.join(csv_root, 'test_split%02d.csv' % which_split))

### For Kinetics ###
def get_split(root, split_path, mode):
    print('processing %s split ...' % mode)
    print('checking %s' % root)
    split_list = []
    split_content = pd.read_csv(split_path).iloc[:,0:4]
    split_list = Parallel(n_jobs=64)\
                 (delayed(check_exists)(row, root) \
                 for i, row in tqdm(split_content.iterrows(), total=len(split_content)))
    return split_list

def check_exists(row, root):
    dirname = '_'.join([row['youtube_id'], '%06d' % row['time_start'], '%06d' % row['time_end']])
    full_dirname = os.path.join(root, row['label'], dirname)
    if os.path.exists(full_dirname):
        n_frames = len(glob.glob(os.path.join(full_dirname, '*.jpg')))
        return [full_dirname, n_frames]
    else:
        return None

def main_Kinetics400(mode, k400_path, f_root, csv_root='../data/kinetics400'):
    train_split_path = os.path.join(k400_path, 'kinetics_train/kinetics_train.csv')
    val_split_path = os.path.join(k400_path, 'kinetics_val/kinetics_val.csv')
    test_split_path = os.path.join(k400_path, 'kinetics_test/kinetics_test.csv')
    if not os.path.exists(csv_root): os.makedirs(csv_root)
    if mode == 'train':
        train_split = get_split(os.path.join(f_root, 'train_split'), train_split_path, 'train')
        write_list(train_split, os.path.join(csv_root, 'train_split.csv'))
    elif mode == 'val':
        val_split = get_split(os.path.join(f_root, 'val_split'), val_split_path, 'val')
        write_list(val_split, os.path.join(csv_root, 'val_split.csv'))
    elif mode == 'test':
        test_split = get_split(f_root, test_split_path, 'test')
        write_list(test_split, os.path.join(csv_root, 'test_split.csv'))
    else:
        raise IOError('wrong mode')

if __name__ == '__main__':
    # f_root is the frame path
    # edit 'your_path' here: 

    parser = argparse.ArgumentParser()
    parser.add_argument("--f_root", type=str, default='/home/mtoering/data_dpc/UCF101/frame')
    parser.add_argument("--splits_root", type=str, default='/home/mtoering/data_dpc/UCF101/splits_classification')
    parser.add_argument("--dataset", type=str, default='ucf101')

    args = parser.parse_args()

    if args.dataset == 'ucf101':
        main_UCF101(f_root=args.f_root, splits_root=args.splits_root)

    if args.dataset == 'hmdb51':
        main_HMDB51(f_root=args.f_root, splits_root=args.splits_root)

    if args.dataset == 'k-400':
        main_kinetics400(f_root=args.f_root, splits_root=args.splits_root)