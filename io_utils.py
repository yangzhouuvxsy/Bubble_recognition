# io_utils.py
import json
import shutil
import pandas as pd
from nms_utils import nms
import numpy as np
from tqdm import tqdm
import os


def read_pkl_dict(pkl_path):
    """
    读取结构为 dict[filename] = [np.ndarray(N, 5)] 的检测结果pkl文件，
    应用NMS，返回结构化DataFrame，含帧编号 frame_idx。

    :param pkl_path: str，pkl文件路径
    :return: result_df (DataFrame)，字段：
             ['filename', 'frame_idx', 'box_id', 'x1', 'y1', 'x2', 'y2', 'score']
    """
    data_dict = pd.read_pickle(pkl_path)
    all_records = []

    # 按文件名排序，确保 frame_idx 从 1 递增
    sorted_items = sorted(data_dict.items(), key=lambda x: x[0])

    for frame_idx, (fname, box_list) in enumerate(tqdm(sorted_items, desc='📦 正在解析识别结果'), start=1):
        if not isinstance(box_list, list) or len(box_list) == 0:
            continue

        box_array = np.array(box_list[0])
        if box_array.shape[0] == 0:
            continue

        filtered = nms(box_array)

        for box_id, box in enumerate(filtered):
            x1, y1, x2, y2, score = box
            all_records.append([fname, frame_idx, box_id, x1, y1, x2, y2, score])

    result_df = pd.DataFrame(all_records, columns=[
        'filename', 'frame_idx', 'box_id', 'x1', 'y1', 'x2', 'y2', 'score'
    ])
    return result_df

def read_labelme(json_path):
    """
    读取 labelme json 文件（COCO格式）
    :param json_path: str
    :return: segmentation 数据与图像元信息统计
    """
    with open(json_path, "r") as f:
        ann = json.load(f)

    img_ids, img_names, img_counts = [], [], []
    segmentations, current = [], []
    count_tracker = -1

    for img in ann["images"]:
        img_ids.append(img['id'])
        img_names.append(img['file_name'])

    for ann_item in ann["annotations"]:
        img_id = ann_item['image_id']
        img_counts.append(img_id)
        if img_id == count_tracker:
            current.append(ann_item['segmentation'])
        else:
            if current:
                segmentations.append(current)
            current = [ann_item['segmentation']]
            count_tracker = img_id
    segmentations.append(current)

    total_label = pd.DataFrame([ann_item['id'] + 1], columns=['标记总数'])
    img_counts_df = pd.DataFrame(img_counts, columns=['img_count'])
    img_ids_df = pd.DataFrame(img_ids, columns=['img_id'])
    img_names_df = pd.DataFrame(img_names, columns=['img_name'])

    count_summary = img_counts_df.groupby('img_count').size().reset_index(name='label_count')
    img_info = pd.concat([img_ids_df, img_names_df, count_summary, total_label], axis=1)

    # 提取每张图的标注框
    new_data = []
    for items in segmentations:
        box_list = [seg[0] for seg in items]
        new_data.append(box_list)

    return new_data, img_info


def file_extraction(folder_path, file_type=None, out_path=None, frequency=1):
    """
    提取指定类型和频率的文件至输出目录
    :param folder_path: str
    :param file_type: str or None
    :param out_path: str or None
    :param frequency: int
    """
    if not os.path.exists(folder_path):
        print('文件夹不存在:', folder_path)
        return

    if not out_path:
        out_path = os.path.join(folder_path, 'out')
    os.makedirs(out_path, exist_ok=True)

    files = os.listdir(folder_path)
    selected = []

    if file_type:
        selected = [f for f in files if f.endswith(file_type)]
    else:
        selected = files

    for fname in selected[::frequency]:
        src = os.path.join(folder_path, fname)
        dst = os.path.join(out_path, fname)
        shutil.copy(src, dst)


if __name__ == "__main__":
    # 示例调用
    pkl_path = 'all_results.pkl'
    df = read_pkl_dict(pkl_path)
    print(df.head())
    print(df.tail())
    print(len(df))
