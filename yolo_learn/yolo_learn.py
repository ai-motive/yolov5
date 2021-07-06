import os
import sys
import argparse
import json
import yaml
import subprocess
from pprint import pprint
from enum import Enum
from sklearn.model_selection import train_test_split
from python_utils.common import general as cg, logger as cl, string as cs
from python_utils.image import general as ig
from python_utils.json import general as jg

import train


_this_folder_ = os.path.dirname(os.path.abspath(__file__))
_this_basename_ = os.path.splitext(os.path.basename(__file__))[0]
_project_folder_ = os.path.abspath(os.path.join(_this_folder_, os.pardir))

MARGIN = '\t' * 20

# OPERATION MODE
class OpMode(Enum):
    PROCESS_ALL = 0
    CROP        = 1
    GENERATE    = 2
    SPLIT       = 3
    MERGE       = 4
    TRAIN_TEST  = 5

# TARGET MODE
class TgtMode(Enum):
    TRAIN      = 0
    TEST       = 1

# OBJECT TYPES (problem / graph / graph_table / total)
class ObjType(Enum):
    PROBLEM       = 0
    GRAPH         = 1
    GRAPH_TABLE   = 2
    TOTAL         = 3


# OBJECT NAMES (problem_intro / problem_whole / problem_text / graph / table / ko / math)
class ObjName(Enum):
    PROBLEM_INTRO = 0
    PROBLEM_WHOLE = 1
    PROBLEM_TEXT  = 2
    GRAPH         = 3
    TABLE         = 4
    KO            = 5
    MATH          = 6


class ObjInfo:
    def __init__(self, obj_type, obj_name):
        self.obj_type = obj_type
        self.obj_name = obj_name

    def get_class_number(self):
        obj_class_number = {
            ObjType.PROBLEM.name : {
                ObjName.PROBLEM_INTRO.name : 0,
                ObjName.PROBLEM_WHOLE.name : 1,
                ObjName.PROBLEM_TEXT.name  : 2,
            },
            ObjType.GRAPH.name : {
                ObjName.GRAPH.name : 0
            },
            ObjType.GRAPH_TABLE.name : {
                ObjName.GRAPH.name : 0,
                ObjName.TABLE.name : 1
            },
            ObjType.TOTAL.name : {
                ObjName.GRAPH.name : 0,
                ObjName.TABLE.name : 1,
                ObjName.KO.name    : 2,
                ObjName.MATH.name  : 3,
            },
        }

        return obj_class_number[self.obj_type.upper()].get(self.obj_name.upper())


def main_crop(ini, common_info, logger=None):
    # Init. local variables
    vars = {}
    for key, val in ini.items():
        vars[key] = cs.replace_string_from_dict(val, common_info)

    cg.folder_exists(ini['img_path'], create_=True)

    raw_path = os.path.join(_project_folder_, ini['raw_path'])
    ann_path = os.path.join(_project_folder_, ini['ann_path'])
    raw_fnames = sorted(cg.get_filenames(raw_path, extensions=ig.IMG_EXTENSIONS))
    ann_fnames = sorted(cg.get_filenames(ann_path, extensions=jg.META_EXTENSION))
    logger.info(" [CROP] # Total file number to be processed: {:d}.".format(len(raw_fnames)))

    for idx, raw_fname in enumerate(raw_fnames):
        logger.info(" [CROP] # Processing {} ({:d}/{:d})".format(raw_fname, (idx + 1), len(raw_fnames)))

        _, raw_core_name, raw_ext = cg.split_fname(raw_fname)
        img = ig.imread(raw_fname, color_fmt='RGB')

        # Load json
        ann_fname = ann_fnames[idx]
        _, ann_core_name, _ = cg.split_fname(ann_fname)
        if ann_core_name == raw_core_name + raw_ext:
            with open(ann_fname) as json_file:
                json_data = json.load(json_file)
                objects = json_data['objects']
                # pprint.pprint(objects)

        # Extract crop position
        object_cnt = 0
        for obj in objects:
            class_name = obj['classTitle']
            if class_name != ini['object_class']:
                continue

            [x1, y1], [x2, y2] = obj['points']['exterior']
            x_min, y_min, x_max, y_max = int(min(x1, x2)), int(min(y1, y2)), int(max(x1, x2)), int(max(y1, y2))
            if x_max - x_min <= 0 or y_max - y_min <= 0:
                continue

            try:
                crop_img = img[y_min:y_max, x_min:x_max]
            except TypeError:
                logger.error(" [CROP] # Crop error : {}".format(raw_fname))
                logger.error(" [CROP] # Error pos : {}, {}, {}, {}".format(x_min, x_max, y_min, y_max))
                pass

            # Save cropped image
            rst_fpath = os.path.join(_project_folder_, ini['img_path'] + raw_core_name + '_' + str(object_cnt) + raw_ext)
            ig.imwrite(crop_img, rst_fpath)
            object_cnt += 1

    logger.info(" # {} in {} mode finished.".format(_this_basename_, OP_MODE))
    return True


def main_generate(ini, common_info, logger=None):
    # Init. local variables
    vars = {}
    for key, val in ini.items():
        vars[key] = cs.replace_string_from_dict(val, common_info)

    label_path = os.path.join(_project_folder_, vars['label_path'])
    cg.folder_exists(label_path, create_=True)

    raw_path = os.path.join(_project_folder_, vars['raw_path'])
    ann_path = os.path.join(_project_folder_, vars['ann_path'])
    raw_fnames = sorted(cg.get_filenames(raw_path, extensions=ig.IMG_EXTENSIONS))
    ann_fnames = sorted(cg.get_filenames(ann_path, extensions=jg.META_EXTENSION))
    logger.info(" [GENERATE] # Total file number to be processed: {:d}.".format(len(raw_fnames)))

    for idx, raw_fname in enumerate(raw_fnames):
        _, raw_core_name, raw_ext = cg.split_fname(raw_fname)

        img = ig.imread(raw_fname, color_fmt='RGB')

        h, w, c = img.shape

        # Load json
        ann_fname = ann_fnames[idx]

        _, ann_core_name, _ = cg.split_fname(ann_fname)
        if ann_core_name == raw_core_name + raw_ext:
            with open(ann_fname) as json_file:
                json_data = json.load(json_file)
                objects = json_data['objects']
                # pprint.pprint(objects)

        # Extract crop position
        obj_names = common_info['obj_names'].replace(' ', '').split(',')
        obj_type = common_info['obj_type']

        labels = []
        for obj in objects:
            obj_name = obj['classTitle']

            if obj_name not in obj_names:
                continue

            class_num = ObjInfo(obj_type, obj_name).get_class_number()

            [x1, y1], [x2, y2] = obj['points']['exterior']
            x_min, y_min, x_max, y_max = int(min(x1, x2)), int(min(y1, y2)), int(max(x1, x2)), int(max(y1, y2))
            if x_max - x_min <= 0 or y_max - y_min <= 0:
                continue

            class_no, x_center, y_center, width, height = \
                str(class_num), str(((x_max + x_min) / 2) / w), str(((y_max + y_min) / 2) / h), str((x_max - x_min) / w), str((y_max - y_min) / h)

            label = "{} {} {} {} {}\r\n".format(class_no, x_center, y_center, width, height)
            labels.append(label)

        # Save object info to COCO format
        rst_fpath = os.path.join(_project_folder_, vars['label_path'] + raw_core_name + '.txt')

        if cg.file_exists(rst_fpath):
            logger.info(" [GENERATE] # File already exist {} ({:d}/{:d})".format(rst_fpath, (idx + 1), len(raw_fnames)))
        else:
            with open(rst_fpath, 'w', encoding='utf8') as f:
                for label in labels:
                    f.write("{}\n".format(label))

            logger.info(" [GENERATE] # File is saved {} ({:d}/{:d})".format(rst_fpath, (idx + 1), len(raw_fnames)))

    logger.info(" # {} in {} mode finished.".format(_this_basename_, OP_MODE))
    return True


def main_split(ini, common_info, logger=None):
    # Init. local variables
    vars = {}
    for key, val in ini.items():
        vars[key] = cs.replace_string_from_dict(val, common_info)

    base_dir_name = common_info['base_dir_name']

    cg.folder_exists(vars['img_path'], create_=False)
    cg.folder_exists(vars['gt_path'], create_=False)

    cg.file_exists(vars['train_path'])
    cg.file_exists(vars['test_path'])

    train_ratio = float(vars['train_ratio'])
    test_ratio = round(1 - train_ratio, 2)

    img_fpaths = sorted(cg.get_filenames(vars['img_path'], extensions=ig.IMG_EXTENSIONS))
    src_train_img_fpaths, src_test_img_fpaths = train_test_split(img_fpaths, train_size=train_ratio, random_state=2000)
    src_train_gt_fpaths, src_test_gt_fpaths = [img_fpath.replace('img', base_dir_name).replace('.jpg', '.txt') for img_fpath in src_train_img_fpaths], \
                                              [img_fpath.replace('img', base_dir_name).replace('.jpg', '.txt') for img_fpath in src_test_img_fpaths]

    dst_train_img_dirpath, dst_test_img_dirpath = os.path.join(vars['train_path'], 'img/'), os.path.join(vars['test_path'], 'img/')
    dst_train_gt_dirpath, dst_test_gt_dirpath = os.path.join(vars['train_path'], base_dir_name + '/'), os.path.join(vars['test_path'], base_dir_name + '/')
    cg.folder_exists(dst_train_img_dirpath, create_=True), cg.folder_exists(dst_test_img_dirpath, create_=True)
    cg.folder_exists(dst_train_gt_dirpath, create_=True), cg.folder_exists(dst_test_gt_dirpath, create_=True)

    # Apply symbolic link for gt & img path
    dst_train_img_fpaths, dst_test_img_fpaths = cg.get_filenames(dst_train_img_dirpath, extensions=ig.IMG_EXTENSIONS), cg.get_filenames(dst_test_img_dirpath, extensions=ig.IMG_EXTENSIONS)
    dst_train_gt_fpaths, dst_test_gt_fpaths = cg.get_filenames(dst_train_gt_dirpath, extensions=cg.TEXT_EXTENSIONS), cg.get_filenames(dst_test_gt_dirpath, extensions=cg.TEXT_EXTENSIONS)
    link_img_, link_gt_ = (len(dst_train_img_fpaths) == 0 and len(dst_test_img_fpaths) == 0), (len(dst_train_gt_fpaths) == 0 and len(dst_test_gt_fpaths) == 0)

    for tgt_mode in list(TgtMode.__members__):
        if tgt_mode == TgtMode.TRAIN.name:
            src_img_fpaths = src_train_img_fpaths
            src_gt_fpaths = src_train_gt_fpaths

            dst_img_dirpath = dst_train_img_dirpath
            dst_gt_dirpath = dst_train_gt_dirpath

        elif tgt_mode == TgtMode.TEST.name:
            src_img_fpaths = src_test_img_fpaths
            src_gt_fpaths = src_test_gt_fpaths

            dst_img_dirpath = dst_test_img_dirpath
            dst_gt_dirpath = dst_test_gt_dirpath

        # Link img files
        if link_img_:
            for src_img_fpath in src_img_fpaths:
                sym_cmd = f'ln "{src_img_fpath}" "{dst_img_dirpath}"'
                subprocess.call(sym_cmd, shell=True)

            logger.info(" # Link img files {}\n{}->{}.".format(src_img_fpath, MARGIN, dst_img_dirpath))
        else:
            logger.info(" # Link img already processed !!!")

        # Link gt files
        if link_gt_:
            for src_gt_fpath in src_gt_fpaths:
                sym_cmd = f'ln "{src_gt_fpath}" "{dst_gt_dirpath}"'
                subprocess.call(sym_cmd, shell=True)

            logger.info(" # Link gt files {}\n{}->{}.".format(src_gt_fpath, MARGIN, dst_gt_dirpath))
        else:
            logger.info(" # Link gt already processed !!!")

    # Save train & test.txt file
    train_base_dirpath = os.path.join(vars['train_path'], base_dir_name)
    train_img_list_fpath = os.path.join(train_base_dirpath, 'img_list.txt')

    with open(train_img_list_fpath, 'w') as f:
        f.write('\n'.join(dst_train_img_fpaths) + '\n')

    test_base_dirpath = os.path.join(vars['test_path'], base_dir_name)
    test_img_list_fpath = os.path.join(test_base_dirpath, 'img_list.txt')

    with open(test_img_list_fpath, 'w') as f:
        f.write('\n'.join(dst_test_img_fpaths) + '\n')

    logger.info(f" [SPLIT] # Train : Test ratio -> {train_ratio * 100} % : {test_ratio * 100} %")
    logger.info(f" [SPLIT] # Train : Test size  -> {len(dst_train_img_fpaths)} : {len(dst_test_img_fpaths)}")

    return True

    # # Modify yaml file
    # ref_yaml_path = os.path.join(_project_folder_, vars['ref_yaml_path'])
    # with open(ref_yaml_path, 'r') as f:
    #     data = yaml.safe_load(f)
    #
    # data['train'] = os.path.join(_project_folder_, vars['train_path'])
    # data['val'] = os.path.join(_project_folder_, vars['val_path'])
    # data['names'] = common_info['obj_names'].replace(' ', '').split(',')
    # data['nc'] = len(data['names'])
    #
    # # Save yaml file
    # rst_yaml_path = os.path.join(_project_folder_, vars['rst_yaml_path'])
    # with open(rst_yaml_path, 'w') as f:
    #     yaml.dump(data, f)
    #     pprint(data)
    #
    # logger.info(" # {} in {} mode finished.".format(_this_basename_, OP_MODE))
    # return True


def main_merge(ini, common_info, logger=None):
    # Init. path variables
    vars = {}
    for key, val in ini.items():
        vars[key] = cs.replace_string_from_dict(val, common_info)

    cg.folder_exists(vars['total_dataset_path'], create_=True)

    datasets = [dataset for dataset in os.listdir(vars['dataset_path']) if (dataset != 'total') and ('meta.json' not in dataset)]
    sort_datasets = sorted(datasets, key=lambda x: (int(x.split('_')[0])))

    base_dir_name = common_info['base_dir_name']
    TRAIN, TEST = TgtMode.TRAIN.name, TgtMode.TEST.name
    train, test = TRAIN.lower(), TEST.lower()

    # Process total files
    train_img_list_fpaths, test_img_list_fpaths = [], []
    if len(sort_datasets) != 0:
        for dir_name in sort_datasets:
            src_train_path, src_test_path               = os.path.join(vars['dataset_path'], dir_name, train), os.path.join(vars['dataset_path'], dir_name, test)
            src_train_img_dirpath, src_test_img_dirpath = os.path.join(src_train_path, 'img/'), os.path.join(src_test_path, 'img/')
            src_train_gt_dirpath,  src_test_gt_dirpath  = os.path.join(src_train_path, base_dir_name + '/'), os.path.join(src_test_path, base_dir_name + '/')

            dst_train_path,        dst_test_path        = os.path.join(vars['total_dataset_path'], train), os.path.join(vars['total_dataset_path'], test)
            dst_train_img_dirpath, dst_test_img_dirpath = os.path.join(dst_train_path, 'img/'), os.path.join(dst_test_path, 'img/')
            dst_train_gt_dirpath,  dst_test_gt_dirpath  = os.path.join(dst_train_path, base_dir_name + '/'), os.path.join(dst_test_path, base_dir_name + '/')

            exist_train_img_dir_, exist_test_img_dir_ = cg.folder_exists(dst_train_img_dirpath), cg.folder_exists(dst_test_img_dirpath)
            exist_train_gt_dir_,  exist_test_gt_dir_  = cg.folder_exists(dst_train_gt_dirpath), cg.folder_exists(dst_test_gt_dirpath)

            if (exist_train_img_dir_ and exist_test_img_dir_) and (exist_train_gt_dir_ and exist_test_gt_dir_):
                logger.info(" # Already {} is exist".format(vars['total_dataset_path']))
            else:
                cg.folder_exists(dst_train_img_dirpath, create_=True), cg.folder_exists(dst_test_img_dirpath, create_=True)
                cg.folder_exists(dst_train_gt_dirpath, create_=True), cg.folder_exists(dst_test_gt_dirpath, create_=True)

            # Apply symbolic link for img & gt path
            for tgt_mode in [TRAIN, TEST]:
                if tgt_mode == TRAIN:
                    src_img_dirpath, dst_img_dirpath = src_train_img_dirpath, dst_train_img_dirpath
                    src_gt_dirpath,  dst_gt_dirpath  = src_train_gt_dirpath, dst_train_gt_dirpath
                elif tgt_mode == TEST:
                    src_img_dirpath, dst_img_dirpath = src_test_img_dirpath, dst_test_img_dirpath,
                    src_gt_dirpath,  dst_gt_dirpath  = src_test_gt_dirpath, dst_test_gt_dirpath

                # check file exists
                src_imgs, dst_imgs = sorted(cg.get_filenames(src_img_dirpath, extensions=ig.IMG_EXTENSIONS)), sorted(cg.get_filenames(dst_img_dirpath, extensions=ig.IMG_EXTENSIONS))
                src_gts,  dst_gts  = sorted(cg.get_filenames(src_gt_dirpath, extensions=ig.IMG_EXTENSIONS)), sorted(cg.get_filenames(dst_gt_dirpath, extensions=cg.TEXT_EXTENSIONS))

                src_img_fnames, dst_img_fnames = [cg.split_fname(src_img)[1] for src_img in src_imgs], [cg.split_fname(dst_img)[1] for dst_img in dst_imgs]
                src_gt_fnames,  dst_gt_fnames  = [cg.split_fname(src_gt)[1] for src_gt in src_gts], [cg.split_fname(dst_gt)[1] for dst_gt in dst_gts]

                exist_img_ = any(src_fname in dst_img_fnames for src_fname in src_img_fnames)
                exist_gt_  = any(src_fname in dst_gt_fnames for src_fname in src_gt_fnames)

                # link img_path
                if not(exist_img_):
                    cmd = 'ln "{}"* "{}"'.format(src_img_dirpath, dst_img_dirpath)  # to all files
                    subprocess.call(cmd, shell=True)
                    logger.info(" # Link img files {}\n{}->{}.".format(src_img_dirpath, MARGIN, dst_img_dirpath))

                # link gt_path
                if not (exist_gt_):
                    cmd = 'ln "{}"* "{}"'.format(src_gt_dirpath, dst_gt_dirpath)  # to all files
                    subprocess.call(cmd, shell=True)
                    logger.info(" # Link gt files {}\n{}->{}.".format(src_gt_dirpath, MARGIN, dst_gt_dirpath))


            # Add to list all label files
            for tar_mode in [TRAIN, TEST]:
                if tar_mode == TRAIN:
                    src_train_img_list_fpath = os.path.join(src_train_path, f'{base_dir_name}', 'img_list.txt')
                    train_img_list_fpaths.append(src_train_img_list_fpath)

                elif tar_mode == TEST:
                    src_test_img_list_fpath = os.path.join(src_test_path, f'{base_dir_name}', 'img_list.txt')
                    test_img_list_fpaths.append(src_test_img_list_fpath)

        dst_train_img_list_fpath = os.path.join(dst_train_path, f'{base_dir_name}', 'img_list.txt')
        dst_test_img_list_fpath  = os.path.join(dst_test_path, f'{base_dir_name}', 'img_list.txt')

        logger.info(" # Train img_list file paths : {}".format(train_img_list_fpaths))
        logger.info(" # Test img_list file paths : {}".format(test_img_list_fpaths))

        # Merge all label files
        with open(dst_train_img_list_fpath, 'w') as outfile:
            for fpath in train_img_list_fpaths:
                with open(fpath) as infile:
                    for line in infile:
                        outfile.write(line)

        with open(dst_test_img_list_fpath, 'w') as outfile:
            for fpath in test_img_list_fpaths:
                with open(fpath) as infile:
                    for line in infile:
                        outfile.write(line)

        logger.info(" # Train & Test gt files are merged !!!")

    logger.info(" # {} in {} mode finished.".format(_this_basename_, OP_MODE))
    return True


def main_train_test(ini, common_info, logger=None):
    # Init. local variables
    vars = {}
    for key, val in ini.items():
        vars[key] = cs.replace_string_from_dict(val, common_info)

    img_size, batch, epoch = ini['img_size'], ini['batch'], ini['epoch']
    device = ini['device'].replace(' ', '')
    data_yaml_path, data_hyp_path, model_yaml_path, model_weight_path, rst_dir_name = \
        ini['data_yaml_path'], ini['data_hyp_path'], ini['model_yaml_path'], ini['model_weight_path'], ini['rst_dir_name']

    args = [
        '--img', img_size, '--batch', batch, '--epoch', epoch,
        '--device', device,
        '--data', os.path.abspath(data_yaml_path), '--hyp', os.path.abspath(data_hyp_path),
        '--cfg', os.path.abspath(model_yaml_path), '--weight',  model_weight_path,
        '--name', rst_dir_name
    ]

    train.main(train.parse_arguments(args))

    return True


def main(args):
    ini = cg.get_ini_parameters(os.path.join(_this_folder_, args.ini_fname))
    common_info = {}
    for key, val in ini['COMMON'].items():
        common_info[key] = val

    logger = cl.setup_logger_with_ini(ini['LOGGER'],
                                      logging_=args.logging_, console_=args.console_logging_)

    if args.op_mode == OpMode.PROCESS_ALL.name:
        # Init. local variables
        vars = {}
        for key, val in ini[OpMode.PROCESS_ALL.name].items():
            vars[key] = cs.replace_string_from_dict(val, common_info)

        # Run generate & split
        tgt_dir_names = vars['tgt_dir_names'].replace(' ', '').split(',')
        for tgt_dir_name in tgt_dir_names:
            common_info['tgt_dir_name'] = tgt_dir_name
            main_generate(ini[OpMode.GENERATE.name], common_info, logger=logger)
            main_split(ini[OpMode.SPLIT.name], common_info, logger=logger)

        # Run merge
        main_merge(ini[OpMode.MERGE.name], common_info, logger=logger)

    elif args.op_mode == OpMode.CROP.name:
        main_crop(ini[OpMode.CROP.name], common_info, logger=logger)
    elif args.op_mode == OpMode.GENERATE.name:
        main_generate(ini[OpMode.GENERATE.name], common_info, logger=logger)
    elif args.op_mode == OpMode.SPLIT.name:
        main_split(ini[OpMode.SPLIT.name], common_info, logger=logger)
    elif args.op_mode == OpMode.MERGE.name:
        main_merge(ini[OpMode.MERGE.name], common_info, logger=logger)
    elif args.op_mode == OpMode.TRAIN_TEST.name:
        main_train_test(ini[OpMode.TRAIN_TEST.name], common_info, logger=logger)
    else:
        print(" @ Error: op_mode, {}, is incorrect.".format(args.op_mode))

    return True


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--obj_type", required=True, choices=[type.lower() for type in list(ObjType.__members__)], help="Object types")
    parser.add_argument("--op_mode", required=True, choices=[mode for mode in list(OpMode.__members__)], help="Operation mode")
    parser.add_argument("--ini_fname", required=True, help="System code ini filename")
    parser.add_argument("--model_dir", default="", help="Model directory")

    parser.add_argument("--logging_", default=False, action='store_true', help="Activate logging")
    parser.add_argument("--console_logging_", default=False, action='store_true', help="Activate logging")

    args = parser.parse_args(argv)

    return args


SELF_TEST_ = True
OBJ_TYPE = ObjType.TOTAL.name.lower()  # problem / graph / graph_table / total
OP_MODE = OpMode.TRAIN_TEST.name
# PROCESS_ALL
# (GENERATE / SPLIT / MERGE) / TRAIN_TEST (CROP은 별도의 기능)

if OBJ_TYPE == ObjType.PROBLEM.name.lower():
    INI_FNAME = _this_basename_ + f'_{ObjType.PROBLEM.name.lower()}.ini'
elif OBJ_TYPE == ObjType.GRAPH.name.lower():
    INI_FNAME = _this_basename_ + f'_{ObjType.GRAPH.name.lower()}.ini'
elif OBJ_TYPE == ObjType.GRAPH_TABLE.name.lower():
    INI_FNAME = _this_basename_ + f'_{ObjType.GRAPH_TABLE.name.lower()}.ini'
elif OBJ_TYPE == ObjType.TOTAL.name.lower():
    INI_FNAME = _this_basename_ + f'_{ObjType.TOTAL.name.lower()}.ini'


if __name__ == "__main__":
    if len(sys.argv) == 1:
        if SELF_TEST_:
            sys.argv.extend(["--obj_type", OBJ_TYPE])
            sys.argv.extend(["--op_mode", OP_MODE])
            sys.argv.extend(["--ini_fname", INI_FNAME])
            sys.argv.extend(["--logging_"])
            sys.argv.extend(["--console_logging_"])
        else:
            sys.argv.extend(["--help"])

    main(parse_arguments(sys.argv[1:]))