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


_this_folder_ = os.path.dirname(os.path.abspath(__file__))
_this_basename_ = os.path.splitext(os.path.basename(__file__))[0]
_project_folder_ = os.path.abspath(os.path.join(_this_folder_, os.pardir))


# OP_MODE
class OpMode(Enum):
    CROP       = 0
    GENERATE   = 1
    SPLIT      = 2
    TRAIN_TEST = 3


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
        for obj in objects:
            obj_name = obj['classTitle']

            if obj_name not in obj_names:
                continue

            class_num = ObjInfo(obj_type, obj_name).get_class_number()

            [x1, y1], [x2, y2] = obj['points']['exterior']
            x_min, y_min, x_max, y_max = int(min(x1, x2)), int(min(y1, y2)), int(max(x1, x2)), int(max(y1, y2))
            if x_max - x_min <= 0 or y_max - y_min <= 0:
                continue

            # Save object info to COCO format
            rst_fpath = os.path.join(_project_folder_, vars['label_path'] + raw_core_name + '.txt')
            class_no, x_center, y_center, width, height = \
                str(class_num), str(((x_max+x_min)/2) / w), str(((y_max+y_min)/2) / h), str((x_max-x_min)/w), str((y_max-y_min)/h)

            if cg.file_exists(rst_fpath):
                logger.info(" [GENERATE] # File already exist {} ({:d}/{:d})".format(rst_fpath, (idx + 1), len(raw_fnames)))
            else:
                with open(rst_fpath, 'a') as f:
                    strResult = "{} {} {} {} {}\r\n".format(class_no, x_center, y_center, width, height)
                    f.write(strResult)

                logger.info(" [GENERATE] # File is saved {} ({:d}/{:d})".format(rst_fpath, (idx + 1), len(raw_fnames)))

    logger.info(" # {} in {} mode finished.".format(_this_basename_, OP_MODE))
    return True


def main_split(ini, common_info, logger=None):
    # Init. local variables
    vars = {}
    for key, val in ini.items():
        vars[key] = cs.replace_string_from_dict(val, common_info)

    cg.folder_exists(vars['img_path'], create_=False)

    if cg.file_exists(vars['train_path']):
        print(" @ Warning: train text file path, {}, already exists".format(vars["train_path"]))
        ans = input(" % Proceed (y/n) ? ")
        if ans.lower() != 'y':
            sys.exit()
    if cg.file_exists(vars['val_path']):
        print(" @ Warning: test text file path, {}, already exists".format(vars["val_path"]))
        ans = input(" % Proceed (y/n) ? ")
        if ans.lower() != 'y':
            sys.exit()

    # Apply symbolic link for img path
    raw_path = os.path.join(_project_folder_, vars['raw_path'])
    img_path = os.path.join(_project_folder_, vars['img_path'])
    cg.folder_exists(img_path, create_=True)

    img_fnames = sorted(cg.get_filenames(img_path, extensions=ig.IMG_EXTENSIONS))
    if len(img_fnames) == 0:
        sym_cmd = "ln -s {} {}".format(raw_path + '*', img_path) # to all files
        subprocess.call(sym_cmd, shell=True)

    img_fnames = sorted(cg.get_filenames(img_path, extensions=ig.IMG_EXTENSIONS))

    train_ratio = float(vars['train_ratio'])
    test_ratio = (1.0 - train_ratio)

    train_img_list, test_img_list = train_test_split(img_fnames,
                                                     test_size=test_ratio, random_state=2000)
    # Save train.txt file
    train_path = os.path.join(_project_folder_, vars['train_path'])
    with open(train_path, 'w') as f:
        f.write('\n'.join(train_img_list) + '\n')

    val_path = os.path.join(_project_folder_, vars['val_path'])
    with open(val_path, 'w') as f:
        f.write('\n'.join(test_img_list) + '\n')

    logger.info(" [SPLIT] # Train : Test ratio -> {} : {}".format(train_ratio, test_ratio))
    logger.info(" [SPLIT] # Train : Test size  -> {} : {}".format(len(train_img_list), len(test_img_list)))

    # Modify yaml file
    ref_yaml_path = os.path.join(_project_folder_, vars['ref_yaml_path'])
    with open(ref_yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    data['train'] = os.path.join(_project_folder_, vars['train_path'])
    data['val'] = os.path.join(_project_folder_, vars['val_path'])
    data['names'] = common_info['obj_names'].replace(' ', '').split(',')
    data['nc'] = len(data['names'])

    # Save yaml file
    rst_yaml_path = os.path.join(_project_folder_, vars['rst_yaml_path'])
    with open(rst_yaml_path, 'w') as f:
        yaml.dump(data, f)
        pprint(data)

    logger.info(" # {} in {} mode finished.".format(_this_basename_, OP_MODE))
    return True


def main_train_test(ini, common_info, logger=None):
    # Init. local variables
    vars = {}
    for key, val in ini.items():
        vars[key] = cs.replace_string_from_dict(val, common_info)

    img_size, batch, epoch, data_yaml_path, model_yaml_path, model_weight_path, rst_dir_name = \
        ini['img_size'], ini['batch'], ini['epoch'], ini['data_yaml_path'], ini['model_yaml_path'], ini['model_weight_path'], ini['rst_dir_name']

    train_cmd = ini['train_py_cmd']
    train_args = ['--img', img_size, '--batch', batch,
                  '--epoch', epoch, '--data', data_yaml_path,
                  '--cfg', model_yaml_path, '--weight',  model_weight_path,
                  '--name', rst_dir_name]

    for arg in train_args:
        train_cmd += ''.join([' ', arg])

    logger.info(" [TRAIN] # Train shell cmd : {}".format(train_cmd))
    subprocess.call(train_cmd, shell=True, cwd=ini['train_root_path'])      # train by python
    # subprocess.call(train_cmd, shell=True, cwd=ini['train_root_path'])    # train by shell

    return True


def main(args):
    ini = cg.get_ini_parameters(os.path.join(_this_folder_, args.ini_fname))
    common_info = {}
    for key, val in ini['COMMON'].items():
        common_info[key] = val

    logger = cl.setup_logger_with_ini(ini['LOGGER'],
                                      logging_=args.logging_, console_=args.console_logging_)

    if args.op_mode == OpMode.CROP.name:
        main_crop(ini[OpMode.CROP.name], common_info, logger=logger)
    elif args.op_mode == OpMode.GENERATE.name:
        main_generate(ini[OpMode.GENERATE.name], common_info, logger=logger)
    elif args.op_mode == OpMode.SPLIT.name:
        main_split(ini[OpMode.SPLIT.name], common_info, logger=logger)
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
OP_MODE = OpMode.GENERATE.name  # GENERATE / SPLIT / TRAIN_TEST (CROP은 별도의 기능)

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