import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from io_utils import read_pkl_dict  # 假设你已封装了读取函数：返回 DataFrame
import re
import os
import cv2
def get_box_center(box: dict):
    """
    根据 box = {'x1':..., 'y1':..., 'x2':..., 'y2':...} 返回中心点 (cx, cy)
    """
    cx = 0.5 * (box['x1'] + box['x2'])
    cy = 0.5 * (box['y1'] + box['y2'])
    return cx, cy
def get_box_radius(box: dict) -> float:
    """
    根据 (x1,y1,x2,y2) 返回内接圆的半径，即 min(width, height)/2
    """
    w = box['x2'] - box['x1']
    h = box['y2'] - box['y1']
    r = min(w, h) / 2.0
    # 避免出现负值
    return max(r, 0.0)
def get_box_center_width_height(box):
    """
    给定 {x1,y1,x2,y2}，返回 (cx, cy, w, h)
    """
    x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
    cx = 0.5*(x1 + x2)
    cy = 0.5*(y1 + y2)
    w = x2 - x1
    h = y2 - y1
    return cx, cy, w, h

def build_box_from_center(cx, cy, w, h):
    """
    给定中心(cx,cy) 和宽高(w,h)，返回 dict(x1=..., y1=..., x2=..., y2=...)
    """
    x1 = cx - w/2
    x2 = cx + w/2
    y1 = cy - h/2
    y2 = cy + h/2
    return dict(x1=x1, y1=y1, x2=x2, y2=y2)
def compute_iou_batch(box, boxes):
    """
    向量化计算一个 box 与多个 boxes 之间的 IOU
    :param box: list[x1, y1, x2, y2]
    :param boxes: np.ndarray, shape (N, 4)
    :return: np.ndarray, shape (N,) 的 IOU 数组
    """
    x1, y1, x2, y2 = box
    box_area = (x2 - x1) * (y2 - y1)

    # 计算交集
    xx1 = np.maximum(x1, boxes[:, 0])
    yy1 = np.maximum(y1, boxes[:, 1])
    xx2 = np.minimum(x2, boxes[:, 2])
    yy2 = np.minimum(y2, boxes[:, 3])

    inter_w = np.maximum(0, xx2 - xx1)
    inter_h = np.maximum(0, yy2 - yy1)
    inter_area = inter_w * inter_h

    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - inter_area

    iou = inter_area / (union_area + 1e-6)
    return iou
def ios_ratio(box1: dict, box2: dict) -> float:
    """计算两个框的交集面积与较小框面积之比"""
    inter_x1 = max(box1['x1'], box2['x1'])
    inter_y1 = max(box1['y1'], box2['y1'])
    inter_x2 = min(box1['x2'], box2['x2'])
    inter_y2 = min(box1['y2'], box2['y2'])

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
    area2 = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
    small_area = min(area1, area2)

    if small_area <= 0:
        return 0.0
    return inter_area / small_area
def is_same_box(box1, box2):
    return (
        box1['x1'] == box2['x1'] and
        box1['y1'] == box2['y1'] and
        box1['x2'] == box2['x2'] and
        box1['y2'] == box2['y2']
    )
def merge_disconnected_tracks(result_df, max_frame_gap=5, iou_thresh_merge=0.4, center_dist_thresh=40):
    # 按轨迹ID分组并过滤短轨迹
    id_to_group = result_df.groupby('bubble_id')
    id_start_end = {
        bid: {
            'start_frame': group['frame_idx'].iloc[0],
            'end_frame': group['frame_idx'].iloc[-1],
            'start_box': group.iloc[0],
            'end_box': group.iloc[-1],
            'length': len(group)  # 记录轨迹长度
        }
        for bid, group in id_to_group if len(group) >= 2  # 过滤短轨迹
    }


    # for bid, group in id_to_group:
    #     group = group.sort_values('frame_idx')
    #     id_start_end[bid] = {
    #         'start_frame': group['frame_idx'].iloc[0],
    #         'end_frame': group['frame_idx'].iloc[-1],
    #         'start_box': group.iloc[0],
    #         'end_box': group.iloc[-1]
    #     }

    merge_map = {}  # id_b → id_a
    protected_ids = {}
    for id_a, info_a in id_start_end.items():
        for id_b, info_b in id_start_end.items():
            if id_a == id_b or id_b in protected_ids or id_b in merge_map:
                continue
            frame_gap = info_b['start_frame'] - info_a['end_frame']
            if 1 <= frame_gap <= max_frame_gap:
                box_a = info_a['end_box']
                box_b = info_b['start_box']
                # 新增：检查是否接近图片顶端
                skip_merge = False
                # 检查轨迹B的起始框是否接近顶端
                if box_b['y1'] <= 20:
                    skip_merge = True
                    print(f"[轨迹合并跳过] 轨迹 {id_b} 起始框接近顶端({box_b['y1']:.1f}px)，不合并到轨迹 {id_a}")

                if skip_merge:
                    continue
                cx1, cy1 = get_box_center(box_a)
                cx2, cy2 = get_box_center(box_b)
                dist = np.hypot(cx1 - cx2, cy1 - cy2)

                iou_val = compute_iou_batch(
                    [box_a['x1'], box_a['y1'], box_a['x2'], box_a['y2']],
                    np.array([[box_b['x1'], box_b['y1'], box_b['x2'], box_b['y2']]])
                )[0]

                if dist < center_dist_thresh and iou_val > iou_thresh_merge:
                    # print(f"[轨迹合并] {id_b} → {id_a} | Δframe={frame_gap}, dist={dist:.1f}, IoU={iou_val:.2f}")
                    merge_map[id_b] = id_a

    # 避免循环合并
    def get_merged_id(bid):
        while bid in merge_map:
            bid = merge_map[bid]
        return bid

    result_df['bubble_id'] = result_df['bubble_id'].apply(get_merged_id)

    merged_ids = set(merge_map.keys())
    return result_df, merged_ids


def track_bubbles_as_dataframe_fast(
    result_df: pd.DataFrame,
    iou_thresh=0.2,
    max_backtrack=1,
    radius_thresh = 60,
) -> pd.DataFrame:
    """
    以逐框处理的简化逻辑，支持：单轨迹匹配、前一帧多轨迹匹配(融合: 选最近中心)、拆分(spilt) 以及多帧回溯(只单轨迹)。
    :param result_df: 输入 DataFrame，包含 ['frame_idx', 'x1', 'y1', 'x2', 'y2', 'filename']
    :param iou_thresh: 单轨迹 / 多轨迹匹配的 IoU 阈值
    :param max_backtrack: 最多向前回溯几帧（只做单轨迹匹配）
    :return: 带 'bubble_id' 的 DataFrame
    """
    # 初始化多轨迹融合计数器
    merged_track_ids = set()
    # --------- 新增：按文件名数字排序 ---------
    # # 提取文件名中的数字
    # result_df['frame_number'] = result_df['filename'].apply(
    #     lambda x: int(re.search(r'(\d+)', x).group(1))
    # )
    # # 按数字排序
    # result_df = result_df.sort_values(by='frame_number').reset_index(drop=True)
    # # 重新生成 frame_idx 从 1 开始
    # result_df['frame_idx'] = result_df.groupby('frame_number').ngroup() + 1
    # result_df.drop(columns=['frame_number'], inplace=True)
    # 1) 按帧排序
    result_df = result_df.sort_values(by='frame_idx').reset_index(drop=True)
    # 2) 预备数据结构
    bubble_ids = [-1] * len(result_df)         # 存放最终的 bubble_id
    bubble_tracks = []                         # 每个 track_id 对应的一组 box
    track_ends_by_frame = defaultdict(list)    # {frame_idx: [(track_id, last_box)]}
    # 方便边遍历边赋值
    index_map = {idx: row for idx, row in result_df.iterrows()}
    track_latest_id = -1                       # 轨迹 ID 递增计数
    # 3) 遍历每个检测框(按帧顺序)
    for idx, cur_box in tqdm(index_map.items(), desc="🧬 Tracking", total=len(result_df)):

        frame_idx = cur_box['frame_idx']
        matched_track_ids = []
        current_max_backtrack = max_backtrack
        # ---------- (A) 首先只看“前一帧” ----------
        # 优先匹配上一帧
        # ① 获取候选轨迹并记录是来自哪一帧
        frame_used = None
        candidates = track_ends_by_frame.get(frame_idx - 1, [])
        if candidates:
            frame_used = frame_idx - 1
        # else:
        #     candidates = track_ends_by_frame.get(frame_idx - 2, [])
        #     if candidates:
        #         frame_used = frame_idx - 2
        # # ② 初始化
        matched_track_ids = []
        best_ios = 0
        best_tid = None
        if candidates:
            track_ids, last_boxes = zip(*candidates)
            last_boxes_array = np.array([[b['x1'], b['y1'], b['x2'], b['y2']] for b in last_boxes])
            cur_box_array = [cur_box['x1'], cur_box['y1'], cur_box['x2'], cur_box['y2']]
            ious = compute_iou_batch(cur_box_array, last_boxes_array)
            for i, iou_val in enumerate(ious):
                if iou_val > iou_thresh:
                    matched_track_ids.append(i)
            # 如果 IoU 不足，尝试 IoS 匹配
            if not matched_track_ids:
                for i, last_box in enumerate(last_boxes):
                    ios = ios_ratio(cur_box, last_box)
                    if 0.5 < ios < 1:
                        matched_box = last_boxes[i]
                        center_x, center_y = get_box_center(matched_box)
                        matched_tid = track_ids[i]  # ✅ 拿到真正的轨迹 ID
                        print(
                            f"[Frame {frame_idx}] IoU 不足但 IoS={ios:.2f}，添加轨迹 {matched_tid} 进入融合候选 "
                            f"（来源帧: {frame_used}，中心: ({center_x:.1f}, {center_y:.1f})）"
                        )
                        matched_track_ids.append(i)
        # 先根据上一帧匹配情况决定怎么做：
        if len(matched_track_ids) == 1:
                # ========== (1) 单轨迹匹配 ==========
            i_match = matched_track_ids[0]
            base_id = track_ids[i_match]
            # 检查“拆分(split)”：
            #   如果本帧有多个 box 都匹配到同一个 base_id，就说明拆分。
            #   简化处理：若本帧同一个track_id已经分配过，就给它新开一条。
            already_matched = any(
                (tid == base_id) for (tid, _) in track_ends_by_frame.get(frame_idx, [])
            )
            if not already_matched:
                # 直接追加到 base_id
                bubble_tracks[base_id].append(cur_box)
                bubble_ids[idx] = base_id
                track_ends_by_frame[frame_idx].append((base_id, cur_box))
                # 获取中心坐标并打印
                cx, cy = get_box_center(cur_box)
                # print(
                #     f"[Frame {frame_idx}] Bubble idx {idx} ✅ 单轨迹匹配成功，继承轨迹: {base_id}，中心坐标: ({cx:.1f}, {cy:.1f})")
            else:
                i_match = matched_track_ids[0]
                base_id = track_ids[i_match]
                last_box = last_boxes[i_match]

                cur_box_array = [cur_box['x1'], cur_box['y1'], cur_box['x2'], cur_box['y2']]
                last_box_array = np.array([[last_box['x1'], last_box['y1'], last_box['x2'], last_box['y2']]])
                iou = compute_iou_batch(cur_box_array, last_box_array)
                iou = float(iou)

                existing_matches = [box for tid, box in track_ends_by_frame.get(frame_idx, []) if tid == base_id]

                if existing_matches:
                    existing_box = existing_matches[0]
                    existing_box_array = [existing_box['x1'], existing_box['y1'], existing_box['x2'],
                                          existing_box['y2']]
                    existing_iou = compute_iou_batch(existing_box_array, last_box_array)
                    existing_iou = float(existing_iou)

                    if iou > existing_iou:
                        for i, (tid, box) in enumerate(track_ends_by_frame[frame_idx]):
                            if tid == base_id:
                                track_latest_id += 1
                                new_id = track_latest_id
                                bubble_tracks.append([box])
                                prev_idx = next(
                                    (i for i, bid in enumerate(bubble_ids)
                                     if bid == base_id and result_df.loc[i, 'frame_idx'] == frame_idx and is_same_box(
                                        result_df.loc[i], box)),
                                    None
                                )
                                if prev_idx is not None:
                                    bubble_ids[prev_idx] = new_id
                                track_ends_by_frame[frame_idx][i] = (new_id, box)
                                break

                        bubble_tracks[base_id].append(cur_box)
                        bubble_ids[idx] = base_id
                        track_ends_by_frame[frame_idx].append((base_id, cur_box))
                        print(f"[Frame {frame_idx}] Bubble idx {idx} ⚠️ 抢占轨迹 {base_id}（原框回退）")
                    else:
                        track_latest_id += 1
                        new_id = track_latest_id
                        bubble_tracks.append([cur_box])
                        bubble_ids[idx] = new_id
                        track_ends_by_frame[frame_idx].append((new_id, cur_box))
                        print(f"[Frame {frame_idx}] Bubble idx {idx} 🚨新建轨迹2 {new_id}（无继承）")

        elif len(matched_track_ids) > 1:
            # ===== Step 1: 当前帧中所有其他框（排除自己） =====
            cur_cx, cur_cy = get_box_center(cur_box)
            print(
                f"\n[Frame {frame_idx}] Bubble idx {idx} 进入 Step 1：多轨迹融合逻辑，中心坐标 = ({cur_cx:.1f}, {cur_cy:.1f})")

            cur_frame_all = result_df[result_df['frame_idx'] == frame_idx]
            other_boxes = [
                {
                    'x1': row.x1, 'y1': row.y1, 'x2': row.x2, 'y2': row.y2,
                    'index': row.Index,
                    'center': get_box_center(row._asdict())
                }
                for row in cur_frame_all.itertuples()
                if row.Index != idx
            ]
            nearest_arr = np.array([[b['x1'], b['y1'], b['x2'], b['y2']] for b in other_boxes])
            print(f"Step 1 完成：当前帧共有 {len(other_boxes)} 个其他框作为邻近参考")
            nearest_indices = [b['index'] for b in other_boxes]
            # ===== Step 2: 判断是否是“未融合”疑似情况 =====
            print("进入 Step 2：判断是否疑似未融合")
            iou_thresh_target = 0.7
            suspected_unmerged = False
            for i in matched_track_ids:
                tid = track_ids[i]
                last_box = last_boxes[i]
                last_arr = [last_box['x1'], last_box['y1'], last_box['x2'], last_box['y2']]
                ious = compute_iou_batch(last_arr, nearest_arr)
                for j, iou_val in enumerate(ious):
                    if iou_val > iou_thresh_target:
                        suspected_unmerged = True
                        print(f"  ⛔ 疑似未融合：轨迹 {tid} 与框 {nearest_indices[j]} 的 IoU = {iou_val:.2f} > 阈值")
                        break
                if suspected_unmerged:
                    break
            print(f"Step 2 完成：suspected_unmerged = {suspected_unmerged}")

            # ===== Step 3: 如果是疑似未融合 → 尝试独立继承最近轨迹 =====
            if suspected_unmerged:
                print("进入 Step 3：尝试从匹配轨迹中选最近中心")
                cur_center = (cur_cx, cur_cy)
                min_dist = float('inf')
                best_tid = None
                for i in matched_track_ids:
                    tid = track_ids[i]
                    last_center = get_box_center(last_boxes[i])
                    dist = np.hypot(cur_center[0] - last_center[0], cur_center[1] - last_center[1])
                    print(f"  轨迹 {tid} 与当前框中心距离 = {dist:.1f}")
                    if dist < min_dist:
                        min_dist = dist
                        best_tid = tid
                print(f"Step 3 完成：选择最近轨迹 {best_tid}，距离 = {min_dist:.1f}")

                # ===== Step 4: 若轨迹已被继承，判断是否抢占 =====
                print("进入 Step 4：检查是否抢占当前帧已继承轨迹")
                existing = [
                    i for i, bid in enumerate(bubble_ids)
                    if bid == best_tid and result_df.loc[i, 'frame_idx'] == frame_idx and i != idx
                ]
                if existing:
                    prev_idx = existing[0]
                    prev_box = result_df.loc[prev_idx]
                    last_box = next(b for (tid, b) in track_ends_by_frame[frame_used] if tid == best_tid)
                    iou_prev = compute_iou_batch(
                        [prev_box['x1'], prev_box['y1'], prev_box['x2'], prev_box['y2']],
                        np.array([[last_box['x1'], last_box['y1'], last_box['x2'], last_box['y2']]])
                    )[0]
                    iou_cur = compute_iou_batch(
                        [cur_box['x1'], cur_box['y1'], cur_box['x2'], cur_box['y2']],
                        np.array([[last_box['x1'], last_box['y1'], last_box['x2'], last_box['y2']]])
                    )[0]
                    print(f"  ⚔️ 冲突检测：cur_iou = {iou_cur:.2f}, prev_iou = {iou_prev:.2f}")
                    if iou_cur > iou_prev:
                        print(f"  → 当前框抢占轨迹 {best_tid}")
                        bubble_ids[prev_idx] = -1
                        bubble_tracks[best_tid].pop()
                        track_ends_by_frame[frame_idx] = [
                            pair for pair in track_ends_by_frame[frame_idx]
                            if not is_same_box(pair[1], prev_box)
                        ]
                        track_latest_id += 1
                        new_id = track_latest_id
                        bubble_tracks.append([prev_box])
                        bubble_ids[prev_idx] = new_id
                        track_ends_by_frame[frame_idx].append((new_id, prev_box))
                        print(f"    → 撤销框 {prev_idx} 的继承，重新分配为轨迹 {new_id}")
                        bubble_ids[idx] = best_tid
                        bubble_tracks[best_tid].append(cur_box)
                        track_ends_by_frame[frame_idx].append((best_tid, cur_box))
                        merged_track_ids.add(best_tid)
                        print(f"    → 当前框分配轨迹 {best_tid}")
                        #bubble_tracks[best_tid].pop()
                    else:
                        print(f"  → 当前框 IoU 较低（{iou_cur:.2f}），不抢占轨迹")
                        track_latest_id += 1
                        new_id = track_latest_id
                        bubble_ids[idx] = new_id
                        bubble_tracks.append([cur_box])
                        track_ends_by_frame[frame_idx].append((new_id, cur_box))
                        print(f"    → 分配新轨迹 {new_id}")
                else:
                    print(f"  → 当前轨迹 {best_tid} 未被继承，直接分配")
                    bubble_ids[idx] = best_tid
                    bubble_tracks[best_tid].append(cur_box)
                    track_ends_by_frame[frame_idx].append((best_tid, cur_box))
                    merged_track_ids.add(best_tid)
                    print(f"    → 当前框继承轨迹 {best_tid}")
                continue
            else:
                # ------- Step 4: 确认真正融合，进入继承逻辑 -------
                # === Step 1: 获取有历史轨迹的 matched 轨迹 ===
                historical_infos = []
                for i in matched_track_ids:
                    tid = track_ids[i]
                    if len(bubble_tracks[tid]) > 0:
                        radii_history = [get_box_radius(box) for box in bubble_tracks[tid]]
                        has_small = any(r < radius_thresh for r in radii_history)
                        max_radius = max(radii_history)
                        historical_infos.append((tid, has_small, max_radius))

                if len(historical_infos) == 1:
                    # 仅一个有历史信息 → 继承它
                    chosen_tid = historical_infos[0][0]
                    print(f"[Frame {frame_idx}] Bubble idx {idx} 仅一个有历史轨迹 {chosen_tid} → 直接继承")
                    bubble_ids[idx] = chosen_tid
                    bubble_tracks[chosen_tid].append(cur_box)
                    track_ends_by_frame[frame_idx].append((chosen_tid, cur_box))
                    merged_track_ids.add(chosen_tid)

                elif len(historical_infos) > 1:
                    # 多个有历史信息 → 选历史最大半径的轨迹（可选加小半径优先规则）
                    filtered = [info for info in historical_infos if info[1]]  # 有小于60历史的优先
                    if filtered:
                        target_list = filtered
                    else:
                        target_list = historical_infos

                    chosen_tid = max(target_list, key=lambda x: x[2])[0]
                    print(f"[Frame {frame_idx}] Bubble idx {idx} 多个历史轨迹中继承 max_radius 的轨迹 {chosen_tid}")
                    bubble_ids[idx] = chosen_tid
                    bubble_tracks[chosen_tid].append(cur_box)
                    track_ends_by_frame[frame_idx].append((chosen_tid, cur_box))
                    merged_track_ids.add(chosen_tid)

                else:
                    # 所有轨迹都没有历史信息 → 不继承
                    track_latest_id += 1
                    new_id = track_latest_id
                    bubble_ids[idx] = new_id
                    bubble_tracks.append([cur_box])
                    track_ends_by_frame[frame_idx].append((new_id, cur_box))
                    print(f"[Frame {frame_idx}] Bubble idx {idx} 所有轨迹都无历史记录 → 分配新轨迹 {new_id}")
        else:
            # ========== (3) 没有和上一帧匹配到 ==========
            def try_backtrack_match(cur_box, frame_idx, max_bt):
                """
                简化版：对 [frame_idx-2, ..., frame_idx-max_backtrack] 这几帧上的所有候选框
                一次性计算 IOU，选最大的那条轨迹（只要超过阈值即可）
                """
                # 1. 收集所有候选 (track_id, box)
                all_cands = []
                for back in range(2, max_bt + 1):
                    tgt = frame_idx - back
                    for tid, last_box in track_ends_by_frame.get(tgt, []):
                        all_cands.append((tid, last_box))
                if not all_cands:
                    return None
                # 2. 准备数组批量计算 IOU
                cur_arr = [cur_box['x1'], cur_box['y1'], cur_box['x2'], cur_box['y2']]
                boxes_arr = np.array([[b['x1'], b['y1'], b['x2'], b['y2']] for _, b in all_cands])
                ious = compute_iou_batch(cur_arr, boxes_arr)
                # 计算中心坐标
                # 3. 找到最大 IOU
                max_idx = int(np.argmax(ious))
                max_iou = ious[max_idx]

                chosen_tid = all_cands[max_idx][0]
                # 4. 判断阈值——这里可以用原始 iou_thresh，也可以再动态放宽
                if max_iou >= 0.3:
                    return chosen_tid
                return None
            # 遍历主循环里 “没有上一帧匹配” 的分支
            # ========== (3) 没有和上一帧匹配到 ==========
            #    尝试多帧回溯（支持拆分）
            # 新增：检查是否接近图片顶端，如果是则不进行多帧回溯
            skip_backtrack = False
            if not skip_backtrack:
                single_tid = try_backtrack_match(cur_box, frame_idx, current_max_backtrack)
                if single_tid is not None:
                    bubble_tracks[single_tid].append(cur_box)
                    bubble_ids[idx] = single_tid
                    track_ends_by_frame[frame_idx].append((single_tid, cur_box))
                    print(f"[Frame {frame_idx}] Bubble idx {idx} 🕰️ 多帧回溯成功，继承轨迹: {single_tid}")
                else:
                    track_latest_id += 1
                    new_id = track_latest_id
                    bubble_tracks.append([cur_box])
                    bubble_ids[idx] = new_id
                    track_ends_by_frame[frame_idx].append((new_id, cur_box))
                    cx = (cur_box['x1'] + cur_box['x2']) / 2
                    cy = (cur_box['y1'] + cur_box['y2']) / 2
                    # ✅ 打印新建框信息
                    print(f"[Frame {frame_idx}] Bubble idx {idx} ❌中心坐标: ({cx:.1f}, {cy:.1f}) 未匹配成功，创建新轨迹 {new_id}")



    # 2. 将更新后的 bubble_ids 写回 result_df
    result_df['bubble_id'] = bubble_ids
    # # ✅ 过滤掉未分配轨迹的框
    result_df = result_df[result_df['bubble_id'] != -1].reset_index(drop=True)
    frame_to_filename = (
        result_df.sort_values(['frame_idx'])[['frame_idx', 'filename']]
        .drop_duplicates('frame_idx')
        .set_index('frame_idx')['filename']
        .to_dict()
    )
    # ---------- (B) 补帧处理：分段插值中心和尺寸 ----------
    filled_rows = []
    # 新增：检查是否接近图片顶端，如果是则不进行多帧回溯

    result_df, merged_ids = merge_disconnected_tracks(
        result_df,
        max_frame_gap=20,  # 你可以改成你希望的间隔帧数
        iou_thresh_merge=0.4,  # IoU 合并阈值
        center_dist_thresh=40  # 中心点距离阈值
    )
    grouped = result_df.groupby('bubble_id')
    print(f"总共合并了 {len(merged_ids)} 条轨迹")
    print(f"被合并的轨迹ID: {merged_ids}")
    for bubble_id, group in grouped:
        group = group.sort_values('frame_idx')
        frames = group['frame_idx'].tolist()
        if len(frames) <= 1:
            # 只有一个点的轨迹，不补帧
            tmp = group.copy()
            tmp['interpolated'] = False
            filled_rows.extend(tmp.to_dict(orient='records'))
            continue

        frame_to_row = {row['frame_idx']: row for _, row in group.iterrows()}

        for i in range(len(frames) - 1):
            f_start = frames[i]
            f_end = frames[i + 1]
            row_start = frame_to_row[f_start]
            row_end = frame_to_row[f_end]

            # 提取起点终点的中心和宽高
            cx1, cy1, w1, h1 = get_box_center_width_height(row_start)
            cx2, cy2, w2, h2 = get_box_center_width_height(row_end)

            # 起点帧，直接加进去（真实检测帧）
            # 起点帧（真实检测帧），添加 interpolated 标记
            real_row = row_start.to_dict()
            real_row['interpolated'] = False  # 标记真实框
            filled_rows.append(real_row)

            # 插值中间帧
            for f in range(f_start + 1, f_end):

                alpha = (f - f_start) / (f_end - f_start)

                # 线性插值中心和宽高
                cx_f = cx1 + alpha * (cx2 - cx1)
                cy_f = cy1 + alpha * (cy2 - cy1)
                w_f = w1 + alpha * (w2 - w1)
                h_f = h1 + alpha * (h2 - h1)

                # 生成新 box
                new_box = build_box_from_center(cx_f, cy_f, w_f, h_f)

                # 基于起点拷贝一行，并替换数据
                fake_row = row_start.copy()
                fake_row['frame_idx'] = f
                fake_row['x1'] = new_box['x1']
                fake_row['y1'] = new_box['y1']
                fake_row['x2'] = new_box['x2']
                fake_row['y2'] = new_box['y2']
                fake_row['filename'] = frame_to_filename.get(f, None)  # 没有就置 None 或者直接跳过
                if fake_row['filename'] is None:
                    # 如果没有对应图片，既然可视化会按 filename 画，就别放进 filled_rows
                    # 也可以选择保留到表里，但可视化时过滤掉 filename 为空的行
                    pass
                else:
                    filled_rows.append(fake_row)
                #修正 filename（可选，如果想保证文件名同步变化）
                match = re.search(r'(\d{6})(?=\.jpg)', row_start['filename'])
                if match:
                    prev_frame_number = int(match.group(1))
                    offset = f - f_start
                    new_frame_number = prev_frame_number + offset
                    fake_row['filename'] = re.sub(r'\d{6}(?=\.jpg)', f"{new_frame_number:06d}", row_start['filename'])

                # 添加插值标记
                fake_row['interpolated'] = True  # 标记插值框
                fake_row['bubble_id'] = bubble_id
                filled_rows.append(fake_row)
                # # 在补帧循环中
                # if fake_row['bubble_id'] == -1:
                #     print("警告：插值框 bubble_id=-1 出现在帧", f)
                # track_ends_by_frame[f].append((bubble_id, fake_row))

        real_row = row_end.to_dict()
        real_row['interpolated'] = False  # 标记真实框
        # 补上最后一帧（终点检测帧）
        filled_rows.append(real_row)
    filled_df = pd.DataFrame(filled_rows)
    # 在 track_bubbles_as_dataframe_fast 内的多轨迹匹配（融合）逻辑下添加：
    # chosen_tid 就是融合后选中的轨迹 ID
    filled_df['merged_flag'] = filled_df['bubble_id'].apply(lambda x: 1 if x in merged_track_ids else 0)
    # filled_rows.append(row_end.to_dict())
    # 在函数结尾，添加标记列
    filled_df = filled_df.sort_values(['bubble_id', 'frame_idx']).reset_index(drop=True)
    filter_iou_thresh = 0.7  # IoU 过滤阈值
    filtered_rows = []
    grouped_by_frame = filled_df.groupby('frame_idx')

    for frame_idx, group in grouped_by_frame:
        real_boxes = group[group['interpolated'] == False]
        interp_boxes = group[group['interpolated'] == True]

        if real_boxes.empty and interp_boxes.empty:
            continue  # 如果该帧没有任何框，跳过

        # 提取所有真实框和插值框的坐标
        real_boxes_arr = real_boxes[['x1', 'y1', 'x2', 'y2']].values
        interp_boxes_arr = interp_boxes[['x1', 'y1', 'x2', 'y2']].values
        kept_interp_indices = []  # 记录保留的插值框索引

        # 遍历所有插值框
        for i in range(len(interp_boxes)):
            interp_box = interp_boxes_arr[i]
            interp_id = interp_boxes.iloc[i]['bubble_id']

            # 1. 检查插值框与所有真实框（无论是否同轨迹）的IoU
            if len(real_boxes_arr) > 0:
                ious_with_real = compute_iou_batch(interp_box, real_boxes_arr)
                max_iou_with_real = np.max(ious_with_real)
            else:
                max_iou_with_real = 0

            if max_iou_with_real >= filter_iou_thresh :
                # 跳过此插值框
                interp_id = interp_boxes.iloc[i]['bubble_id']
                continue
            else:
                kept_interp_indices.append(i)
                filtered_rows.append(interp_boxes.iloc[i].to_dict())

        # 最后把该帧的真实框也加进去
        filtered_rows.extend(real_boxes.to_dict(orient='records'))
    filled_df = pd.DataFrame(filtered_rows).sort_values(['bubble_id', 'frame_idx']).reset_index(drop=True)

    return filled_df


def visualize_bubble_tracking(bubble_df, image_folder=None, output_folder=None):
    """
    可视化每一帧的气泡识别与追踪结果，绘制检测框 + 气泡 ID
    :param bubble_df: 带 bubble_id 的 DataFrame
    :param image_folder: 原始图片文件夹
    :param output_folder: 可视化输出路径
    """
    os.makedirs(output_folder, exist_ok=True)
    grouped = bubble_df.groupby('filename')
    if image_folder is None:
        return
    if not os.path.exists(image_folder):
        print(f"⚠️ 图像文件夹不存在: {image_folder}")
        return
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"✅ 创建输出文件夹: {output_folder}")
    grouped = bubble_df.groupby('filename')
    for filename, group in tqdm(grouped, desc='🎨 可视化气泡追踪'):
        image_path = os.path.join(image_folder, filename)
        img = cv2.imread(image_path)
        if img is None:
            print(f"⚠️ 未找到图像文件: {image_path}")
            continue
        for _, row in group.iterrows():
            # 添加ID过滤条件
            x1, y1, x2, y2 = map(int, [row['x1'], row['y1'], row['x2'], row['y2']])
            bubble_id = int(row['bubble_id'])
            # # 过滤条件：跳过右上角和右下角的框
            # 获取是否插值标记（默认False）
            is_interpolated = row.get('interpolated', False)

            # 用红色表示插值框，绿色表示真实框

            color = (0, 255, 0)
            # 绘制框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # # #标注ID
            cv2.putText(img, f"ID:{bubble_id}", (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), )

        out_path = os.path.join(output_folder, filename)
        cv2.imwrite(out_path, img)


# ✅ 测试主入口
if __name__ == "__main__":
    if __name__ == "__main__":
        result_df = read_pkl_dict(r"E:\bubble_top\test2\2/out/all_results.pkl")
        bubble_df = track_bubbles_as_dataframe_fast(result_df, iou_thresh=0.2, max_backtrack=30, radius_thresh=60)
        # target_ids = [8229,11152,10889,36442,1070,22274,15835,2381,40571,45004,1154,14154,41722,10273,38239,16884,

        print(bubble_df.head())
        print(f"✅ 共检测到气泡轨迹数：{bubble_df['bubble_id'].nunique()}")
        bubble_df.to_csv("bubble_tracking_results.csv", index=False)

        # 查看第一个气泡轨迹
        #     bubble_id = bubble_df['bubble_id'].unique()[0]
        #     bubble_track = bubble_df[bubble_df['bubble_id'] == bubble_id]
        #     print(bubble_track)
        # 可视化保存到 out/
        visualize_bubble_tracking(bubble_df, image_folder=r'E:\bubble_top\test2\2/',
                                  output_folder=r'E:\bubble_top\test2\2/out/output_x1')