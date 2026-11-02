import argparse
import copy
import json
import shutil
import math
import random
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor
import heapq
import os
from multiprocessing import current_process
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

# Stage_MODE: 0=仅评估baseline布局并输出result/png, 1=只进行一阶段优化, 2=只进行二阶段优化, 3=两个阶段都进行
Stage_MODE = 1

# Defer importing post_process until needed in main() to avoid import-time
# dependency errors when this module is imported for unit tests or tooling.
# The functions `run_post_process` and `draw_post_layout` will be imported
# lazily inside main().

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image, ImageDraw, ImageFont

num_of_col = 2
num_single_col = 3

MAX_TOTAL_T_HOURS = 10000.0
MAX_AREA_M2 = 3000000.0
MAX_FEASIBLE_RETRIES = 300
# 1=旧CBS-lite快速方案；2=新CBS-lite方案（重排8次并选择makespan最小）
CURRENT_PATH_PLANNER_MODE = 1

# Simple in-memory cache for evaluate() to avoid recomputing identical layouts.
# Key is derived from (rows_by_station, station_cols) in station_order order.
_EVAL_CACHE = OrderedDict()
_EVAL_CACHE_MAX_SIZE = 16

# Global maximum generation cap is read from config (data.json) at startup.
# The top-level key "max_generations" controls the hard cap; this module does
# not set a compile-time default so the value is determined by the loaded cfg.
MAX_GENERATIONS_LIMIT = 600
ENDPOINT_HOLD_LIMIT = 80

@dataclass
class Individual:
    rows_by_station: dict
    station_cols: dict
    objectives: tuple = (math.inf, math.inf)
    rank: int = 0
    crowding: float = 0.0


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_station_order_from_schedule(schedule: dict) -> list:
    """从 schedule 的 operation_index 推断完整 station_order（含 209 系列与缓冲区）。"""
    snapshots = schedule.get("takt_snapshots", [])
    first_index = {}
    for snap in snapshots:
        for rec in snap.get("positions", []):
            st_name = rec.get("station_name", "")
            # station_name 形如 "209F_0" 或 "BUFFER_209_200_0"，取工位类型前缀
            st = st_name.rsplit("_", 1)[0]
            op_idx = rec.get("operation_index", 0)
            if st not in first_index or op_idx < first_index[st]:
                first_index[st] = op_idx
    station_counts = schedule.get("station_counts", {})
    stations = [st for st in station_counts if st in first_index]
    # 按首次出现的 operation_index 升序排列；同 index 按名称稳定排序
    stations.sort(key=lambda st: (first_index[st], st))
    return stations


def _table_to_individual(rows, cols, table_cells, station_order, station_counts):
    """从二维表格文本构建 Individual。"""
    temp = {st: [] for st in station_order}
    station_cols = {st: [] for st in station_order}
    for r in range(rows):
        for c in range(cols):
            txt = str(table_cells[r][c]).strip()
            if not txt:
                continue
            if "-" in txt:
                st, sid_str = txt.split("-", 1)
                sid = int(sid_str) - 1
            else:
                st, sid = txt, 0
            if st not in station_order:
                continue
            temp[st].append((sid, r))
            station_cols[st].append(c)

    rows_by_station = {}
    final_cols = {}
    for st in station_order:
        temp[st].sort()
        expected = station_counts.get(st, 0)
        while len(temp[st]) < expected:
            last_r = temp[st][-1][1] if temp[st] else 0
            temp[st].append((len(temp[st]), last_r))
        temp[st] = temp[st][:expected]
        rows_by_station[st] = [r for sid, r in temp[st]]
        cols_list = station_cols[st]
        if not cols_list:
            cols_list = [0]
        final_cols[st] = [cols_list[min(i, len(cols_list) - 1)] for i in range(expected)]

    return Individual(rows_by_station=rows_by_station, station_cols=final_cols)


def baseline_to_individual(baseline_path: Path, station_order, station_counts):
    """把 baseline_result.json 或 baseline_2.docx 的布局重建为 Individual 对象。

    Baseline 布局被视为已压缩/固定，evaluate 时跳过 repair_station_cols，
    从而完整保留原始 docx 表格中的空列和工位排列。
    """
    if baseline_path.suffix.lower() == ".docx":
        from docx import Document
        doc = Document(baseline_path)
        table = doc.tables[0]
        rows = len(table.rows)
        cols = len(table.columns)
        table_cells = [[cell.text for cell in row.cells] for row in table.rows]
        ind = _table_to_individual(rows, cols, table_cells, station_order, station_counts)
        object.__setattr__(ind, "compressed", True)
        return ind

    baseline = load_json(baseline_path)
    rows, cols = baseline["layout_shape"]
    lv = baseline["layout_vector"]
    station_cols_data = baseline["station_cols"]

    # 列 -> 工位类型
    col_to_station = {}
    for st, cval in station_cols_data.items():
        c_list = cval if isinstance(cval, list) else [cval]
        for c in c_list:
            col_to_station[int(c)] = st

    temp = {st: [] for st in station_order}
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            val = lv[idx]
            if val == -1 or val == 0 or val == "0":
                continue
            if c not in col_to_station:
                continue
            st = col_to_station[c]
            if st not in station_order:
                continue
            sid = int(val) - 1
            temp[st].append((sid, r))

    rows_by_station = {}
    station_cols = {}
    for st in station_order:
        temp[st].sort()
        expected = station_counts.get(st, 0)
        while len(temp[st]) < expected:
            last_r = temp[st][-1][1] if temp[st] else 0
            temp[st].append((len(temp[st]), last_r))
        temp[st] = temp[st][:expected]
        rows_by_station[st] = [r for sid, r in temp[st]]
        cval = station_cols_data.get(st)
        if cval is None:
            # baseline 缺失该工位时使用默认列：BUFFER_209_200 放最右，其余按 station_order 从右向左
            idx = station_order.index(st)
            if st == "BUFFER_209_200":
                col = cols - 1
            else:
                col = max(0, cols - 1 - idx)
        else:
            col = int(cval) if isinstance(cval, int) else int(cval[0])
        station_cols[st] = [col] * expected

    ind = Individual(rows_by_station=rows_by_station, station_cols=station_cols)
    object.__setattr__(ind, "compressed", True)
    object.__setattr__(ind, "actual_cols", cols)
    return ind


def _baseline_layout_shape(baseline_path: Path):
    """返回 baseline 布局的行数/列数（用于设置 grid_rows）。"""
    if baseline_path.suffix.lower() == ".docx":
        from docx import Document
        doc = Document(baseline_path)
        table = doc.tables[0]
        return [len(table.rows), len(table.columns)]
    baseline = load_json(baseline_path)
    return baseline.get("layout_shape", [None, None])


def evaluate_baseline(baseline_path: Path, schedule_path: Path, data_path: Path):
    """用当前 Area 的 evaluate 计算 baseline 的 T 和 S。"""
    cfg = load_json(data_path)
    schedule = load_json(schedule_path)
    station_counts = schedule["station_counts"]
    ind = baseline_to_individual(baseline_path, cfg["station_order"], station_counts)
    # 不再把 grid_rows 强制改为 baseline 自身行数：当前 schedule 的实例数可能超过
    # baseline 布局行数（例如 baseline 只有 4 行但 station_counts 为 5），强行压缩会
    # 导致布局过密、路径规划失败甚至长时间卡住。
    return evaluate(ind, cfg, schedule, station_counts)


def build_result_payload(payload, cfg, schedule, pareto_payloads, pareto_groups, hv_history, best_hv_generation):
    return {
        "algorithm": "nsga2",
        "objectives": {
            "T_hours": payload["objectives"][0],
            "S_m2": payload["objectives"][1],
        },
        "layout_vector": payload["layout_vector"],
        "layout_shape": [cfg["grid_rows"], payload.get("actual_cols", cfg["grid_cols"])],
        "station_cols": payload.get("station_cols", {}),
        "station_positions": payload.get("station_positions", {}),
        "pareto_front": [
            {"T_hours": p["objectives"][0], "S_m2": p["objectives"][1]}
            for p in pareto_payloads
        ],
        "pareto_groups": pareto_groups,
        "frontier_counts": hv_history,
        "hypervolume": hv_history,
        "best_hypervolume_generation": best_hv_generation,
        "move_plans": [
            {
                "P": rec[0],
                "path": rec[1],
                "timestamps": rec[2],
                "global_id": rec[3],
                "kind": rec[4] if len(rec) > 4 else "main",
                "from": rec[5] if len(rec) > 5 else None,
                "to": rec[6] if len(rec) > 6 else None,
            }
            for rec in payload["move_records"]
        ],
        "aircraft_sequences": {
            str(k): v for k, v in build_aircraft_timeline(schedule, payload["move_records"]).items()
        },
        "used_cells": payload["used_cells"],
        "station_counts": schedule.get("station_counts", {}),
    }


def non_dominated_sort(items):
    fronts = []
    dominates = [set() for _ in items]
    dominated_count = [0 for _ in items]
    for i, a in enumerate(items):
        for j, b in enumerate(items):
            if i == j:
                continue
            if dominates_obj(a.objectives, b.objectives):
                dominates[i].add(j)
            elif dominates_obj(b.objectives, a.objectives):
                dominated_count[i] += 1
    first = [i for i, c in enumerate(dominated_count) if c == 0]
    for i in first:
        items[i].rank = 0
    fronts.append(first)
    rank = 0
    while fronts[-1]:
        next_front = []
        for i in fronts[-1]:
            for j in dominates[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    items[j].rank = rank + 1
                    next_front.append(j)
        rank += 1
        fronts.append(next_front)
    if not fronts[-1]:
        fronts.pop()
    return fronts


def crowding_distance(items, front):
    if not front:
        return
    for idx in front:
        items[idx].crowding = 0.0
    for dim in (0, 1):
        ordered = sorted(front, key=lambda i: items[i].objectives[dim])
        items[ordered[0]].crowding = math.inf
        items[ordered[-1]].crowding = math.inf
        lo = items[ordered[0]].objectives[dim]
        hi = items[ordered[-1]].objectives[dim]
        if hi == lo:
            continue
        for i in range(1, len(ordered) - 1):
            prev_val = items[ordered[i - 1]].objectives[dim]
            next_val = items[ordered[i + 1]].objectives[dim]
            items[ordered[i]].crowding += (next_val - prev_val) / (hi - lo)


def dominates_obj(a, b):
    return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])


def weighted_score(obj, weights):
    return weights["T"] * obj[0] + weights["S"] * obj[1]


def calculate_column_utilization(grid, station_cols, station_counts, station_order, m, n):
    """
    计算各列的利用率，定义为该列工位数量与当前车间列高度的比值
    Returns: dict mapping column index to utilization ratio
    """
    column_utilization = {}
    column_station_count = {}
    
    # 统计每列的工位数量
    for col in range(n):
        column_station_count[col] = 0
        for r in range(m):
            if grid[r][col] != 0:
                column_station_count[col] += 1
    
    # 计算利用率 = 工位数量 / 列高度
    for col in range(n):
        if column_station_count[col] > 0:
            column_utilization[col] = column_station_count[col] / m
        else:
            column_utilization[col] = 0.0
    
    return column_utilization


def get_columns_by_utilization(column_utilization, ascending=True):
    """
    按利用率排序列，ascending=True时从低到高（优先压缩利用率低的列）
    """
    sorted_columns = sorted(column_utilization.items(), key=lambda x: x[1], reverse=not ascending)
    return [col for col, util in sorted_columns]


def manhattan_distance(pos1, pos2):
    """计算曼哈顿距离"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def find_nearest_empty_column(target_pos, grid, station_cols, station_order, m, n, max_station_types_per_column=3):
    """
    为工位寻找最近的相邻列空位，考虑工位类型约束
    """
    target_row, target_col = target_pos

    # 扩展搜索：在整个 grid 中寻找最近的可用空位（而不是仅限左右相邻列）。
    # 仍然遵守每列允许的不同工位类型上限。
    candidates = []
    for col in range(n):
        empty_positions = []
        column_station_types = set()
        for r in range(m):
            val = grid[r][col]
            if val == 0:
                empty_positions.append((r, col))
            else:
                # 期望 val 是像 "140C-1" 的字符串编码，安全地提取类型
                try:
                    stype = str(val).split('-')[0]
                    column_station_types.add(stype)
                except Exception:
                    continue

        # 只有当该列有空位且类型数未达上限时才作为候选
        if empty_positions and len(column_station_types) < max_station_types_per_column:
            min_dist = min(manhattan_distance(target_pos, pos) for pos in empty_positions)
            candidates.append((col, min_dist, empty_positions))

    if not candidates:
        return None, []

    candidates.sort(key=lambda x: x[1])
    return candidates[0][0], candidates[0][2]


def check_process_ordering_constraint(station_a, station_b, station_order):
    """
    检查工位顺序约束：对任意一对工位 a(前序工艺)、b(后序工艺)，要求 col(b) ≥ col(a)
    """
    try:
        idx_a = station_order.index(station_a)
        idx_b = station_order.index(station_b)
        return idx_b >= idx_a
    except ValueError:
        return True  # 如果找不到工位，默认允许


def migrate_stations_from_column(target_col, grid, station_cols, station_order, m, n, max_station_types_per_column=3):
    """
    从目标列迁移工位到相邻列，自上而下进行迁移
    Returns: (success, new_grid, new_station_cols, migrated_positions)
    """
    new_grid = [row[:] for row in grid]
    new_station_cols = copy.deepcopy(station_cols)
    migrated_positions = []
    
    # 获取该列所有工位位置（自上而下）
    column_stations = []
    for r in range(m):
        if new_grid[r][target_col] != 0:
            station_info = str(new_grid[r][target_col])
            station_type = station_info.split('-')[0]
            station_id = int(station_info.split('-')[1]) - 1  # 转换为0-based
            column_stations.append((r, station_type, station_id))
    
    # 按行号排序（自上而下）
    column_stations.sort(key=lambda x: x[0])
    
    migration_success = True
    
    for row, station_type, station_id in column_stations:
        target_pos = (row, target_col)

        # ── 先由工艺顺序约束确定合法列范围 [col_lo, col_hi] ──────────────────
        col_lo, col_hi = 0, n - 1
        for other_station in station_order:
            if other_station == station_type:
                continue
            other_cols = new_station_cols.get(other_station, [])
            if not other_cols:
                continue
            if check_process_ordering_constraint(station_type, other_station, station_order):
                # station_type 是前序 → 新列 <= min(后序列)
                col_hi = min(col_hi, min(other_cols))
            elif check_process_ordering_constraint(other_station, station_type, station_order):
                # station_type 是后序 → 新列 >= max(前序列)
                col_lo = max(col_lo, max(other_cols))

        if col_lo > col_hi:
            # 无合法列，迁移失败
            migration_success = False
            break

        # ── 在合法列范围内寻找最近空位（排除 target_col 自身）─────────────────
        best_pos = None
        best_dist = float('inf')
        for col in range(col_lo, col_hi + 1):
            if col == target_col:
                continue
            # 检查该列工位类型数量是否超限
            col_types = {str(new_grid[r2][col]).split('-')[0]
                         for r2 in range(m) if new_grid[r2][col] != 0}
            if len(col_types) >= max_station_types_per_column:
                continue
            for r2 in range(m):
                if new_grid[r2][col] == 0:
                    d = manhattan_distance(target_pos, (r2, col))
                    if d < best_dist:
                        best_dist = d
                        best_pos = (r2, col)

        if best_pos is None:
            migration_success = False
            break

        nearest_col = best_pos[1]
        chosen_empty_pos = best_pos

        # 执行迁移
        new_grid[chosen_empty_pos[0]][chosen_empty_pos[1]] = new_grid[row][target_col]
        new_grid[row][target_col] = 0
        migrated_positions.append((chosen_empty_pos, station_type, station_id))
        
        # 更新 station_cols
        if station_type in new_station_cols:
            # 优先使用 sid 直接索引更新以提高稳健性
            if 0 <= station_id < len(new_station_cols[station_type]):
                new_station_cols[station_type][station_id] = nearest_col
            else:
                # 回退：尝试匹配等于 target_col 的项并替换（防御性处理）
                for i, col in enumerate(new_station_cols[station_type]):
                    if col == target_col:
                        new_station_cols[station_type][i] = nearest_col
                        break
    
    return migration_success, new_grid, new_station_cols, migrated_positions


def compress_column(target_col, grid, station_cols, station_order, m, n, max_station_types_per_column=3):
    """
    压缩指定列：尝试迁移该列所有工位到相邻列，成功后删除该列
    Returns: (success, new_grid, new_station_cols, width_reduction)
    """
    # 首先检查该列是否为空列
    column_empty = True
    for r in range(m):
        if grid[r][target_col] != 0:
            column_empty = False
            break
    
    if column_empty:
        # 空列直接删除，不需要迁移工位
        #print(f"[Stage2] 发现空列{target_col}，直接压缩")
        new_grid = grid
        new_station_cols = station_cols
    else:
        # 非空列，需要先迁移工位
        success, new_grid, new_station_cols, migrated_positions = migrate_stations_from_column(
            target_col, grid, station_cols, station_order, m, n, max_station_types_per_column
        )
        
        if not success:
            print(f"[Stage2] 列{target_col}工位迁移失败（工艺约束或目标列已满）")
            return False, grid, station_cols, 0
        
        # 检查该列是否已完全清空
        column_empty = True
        for r in range(m):
            if new_grid[r][target_col] != 0:
                column_empty = False
                break
        
        if not column_empty:
            return False, grid, station_cols, 0
    
    # 空列压缩：完全删除空列，右侧列向左移动填补，减少实际列数
    final_grid = []
    for r in range(m):
        new_row = []
        for c in range(n):
            if c < target_col:
                # 空列左侧的列保持不变
                new_row.append(new_grid[r][c])
            elif c > target_col:
                # 空列右侧的列向左移动填补空缺
                new_row.append(new_grid[r][c])
            # c == target_col 的列被删除（跳过）
        final_grid.append(new_row)
    
    # 更新 station_cols 中的列索引
    final_station_cols = {}
    for station_type, cols in new_station_cols.items():
        updated_cols = []
        for col in cols:
            if col < target_col:
                # 空列左侧的列保持不变
                updated_cols.append(col)
            elif col > target_col:
                # 空列右侧的列向左移动一位
                updated_cols.append(col - 1)
            # col == target_col 的列被删除（但应该不存在，因为该列已清空）
        final_station_cols[station_type] = updated_cols
    
    return True, final_grid, final_station_cols, 1


def remove_all_empty_columns(grid, station_cols, m, n):
    """
    找出车间使用区域内的所有空列，一次性全部删除，右侧列向左移填补。
    Returns: (new_grid, new_station_cols, removed_count)
    """
    # 确定实际使用区域的列范围
    used_col_set = set()
    for r in range(m):
        for c in range(n):
            if grid[r][c] != 0:
                used_col_set.add(c)

    if not used_col_set:
        return grid, station_cols, 0

    min_c = min(used_col_set)
    max_c = max(used_col_set)

    # 找出使用区域内的空列
    empty_cols = sorted(
        [c for c in range(min_c, max_c + 1) if c not in used_col_set],
        reverse=True   # 从右到左删，避免索引漂移
    )

    if not empty_cols:
        return grid, station_cols, 0

    # 依次删除每个空列（右侧列向左移）
    new_grid = [row[:] for row in grid]
    new_station_cols = copy.deepcopy(station_cols)

    for target_col in empty_cols:
        cur_n = len(new_grid[0])
        new_grid = [
            [new_grid[r][c] for c in range(cur_n) if c != target_col]
            for r in range(m)
        ]
        updated = {}
        for st, cols in new_station_cols.items():
            updated[st] = [col if col < target_col else col - 1 for col in cols]
        new_station_cols = updated

    return new_grid, new_station_cols, len(empty_cols)


def remove_empty_columns_keep_leftmost(grid, station_cols, m, n):
    """
    找出车间使用区域内的所有空列，删除除最左侧空列之外的所有空列。
    保留最左侧空列，其余右侧列向左移填补。
    Returns: (new_grid, new_station_cols, removed_count)
    """
    used_col_set = set()
    for r in range(m):
        for c in range(n):
            if grid[r][c] != 0:
                used_col_set.add(c)

    if not used_col_set:
        return grid, station_cols, 0

    min_c = min(used_col_set)
    max_c = max(used_col_set)

    # 只在车间区域 [min_c, max_c] 内查找空列
    empty_cols = [c for c in range(min_c, max_c + 1) if c not in used_col_set]

    if not empty_cols:
        return grid, station_cols, 0

    # 保留最左侧的空列，删除其余空列
    keep_col = min(empty_cols)  # 最左侧空列
    cols_to_remove = sorted([c for c in empty_cols if c != keep_col], reverse=True)

    if not cols_to_remove:
        return grid, station_cols, 0

    new_grid = [row[:] for row in grid]
    new_station_cols = copy.deepcopy(station_cols)

    for target_col in cols_to_remove:
        cur_n = len(new_grid[0])
        new_grid = [
            [new_grid[r][c] for c in range(cur_n) if c != target_col]
            for r in range(m)
        ]
        updated = {}
        for st, cols in new_station_cols.items():
            updated[st] = [col if col < target_col else col - 1 for col in cols]
        new_station_cols = updated

    return new_grid, new_station_cols, len(cols_to_remove)


def remove_empty_rows(grid):
    """删除网格上下边缘的空行（无工位的行），返回(new_grid, removed_top, removed_bottom)"""
    m = len(grid)
    if m == 0:
        return grid, 0, 0
    used_rows = set()
    for r in range(m):
        for val in grid[r]:
            if val != 0:
                used_rows.add(r)
    if not used_rows:
        return grid, 0, 0
    min_r = min(used_rows)
    max_r = max(used_rows)
    new_grid = [grid[r][:] for r in range(min_r, max_r + 1)]
    return new_grid, min_r, m - 1 - max_r


def trim_left_empty_columns(grid, keep=1):
    """保留第一个工位左侧恰好 keep 列空列，多余的从左侧裁掉。
    Returns: (new_grid, removed_count)
    """
    if not grid or not grid[0]:
        return grid, 0
    min_c = None
    for r in range(len(grid)):
        for c in range(len(grid[r])):
            if grid[r][c] != 0:
                if min_c is None or c < min_c:
                    min_c = c
    if min_c is None or min_c <= keep:
        return grid, 0
    trim = min_c - keep
    new_grid = [row[trim:] for row in grid]
    return new_grid, trim


def to_station_key(rec):
    return rec["station_name"]


def parse_movements(schedule, station_order):
    snapshots = schedule["takt_snapshots"]
    phases = []
    last_station = {}
    last_op_index = {}
    for snap in snapshots:
        for rec in snap["positions"]:
            gid = rec["global_id"]
            last_station[gid] = to_station_key(rec)
            last_op_index[gid] = rec["operation_index"]

    for i in range(len(snapshots) - 1):
        prev_pos = {r["global_id"]: r for r in snapshots[i]["positions"]}
        next_pos = {r["global_id"]: r for r in snapshots[i + 1]["positions"]}
        reqs = []
        static_cells = set()
        for gid, rec in prev_pos.items():
            if gid in next_pos:
                cur = to_station_key(rec)
                nxt = to_station_key(next_pos[gid])
                if cur != nxt:
                    reqs.append({"gid": gid, "start": cur, "end": nxt, "phase": i + 1})
                else:
                    static_cells.add(cur)
            else:
                # 从最后工位离开产线
                if rec.get("operation") == station_order[-1]:
                    reqs.append({"gid": gid, "start": to_station_key(rec), "end": "OUT", "phase": i + 1})
                else:
                    static_cells.add(to_station_key(rec))
        phases.append({"index": i + 1, "requests": reqs, "static": static_cells})
    return phases


def random_station_cols(station_counts, station_order, n):
    # 从最右列开始分配。前 num_single_col 种工位每列一种，其余每列最多 num_of_col 种
    raw_cols = {}
    queue = list(station_order)
    if not queue:
        return repair_station_cols(raw_cols, station_counts, station_order, n)

    # 检查是否能在 n 列内容纳所有工位类型
    _n_single = min(num_single_col, len(queue))
    _n_multi = len(queue) - _n_single
    if _n_single > n or _n_multi > num_of_col * (n - _n_single):
        # 退回到原随机模式以保证可行性
        raw_cols = {}
        upper = n - 1
        for st in station_order:
            k = station_counts[st]
            if upper < 0:
                upper = 0
            cols = [random.randrange(upper + 1) for _ in range(k)]
            raw_cols[st] = cols
            upper = min(cols) if cols else upper
        return repair_station_cols(raw_cols, station_counts, station_order, n)

    # 从最右列开始分配
    col = n - 1
    idx = 0
    _first3_set = set(station_order[:num_single_col])
    while col >= 0 and idx < len(queue):
        remaining_stations = len(queue) - idx
        remaining_columns = col + 1
        if queue[idx] in _first3_set:
            # 前 num_single_col 种工位：每列只放一种
            k = 1
        else:
            min_needed = max(0, remaining_stations - (remaining_columns - 1) * num_of_col)
            max_assignable = min(num_of_col, remaining_stations)
            if min_needed >= max_assignable:
                k = max_assignable
            else:
                k = random.randint(min_needed, max_assignable)

        if k == 0:
            col -= 1
            continue

        for _ in range(k):
            st = queue[idx]
            raw_cols[st] = [col for _ in range(station_counts.get(st, 0))]
            idx += 1
        col -= 1

    # 如仍有未分配（理论上不应出现），把它们放到最左列
    while idx < len(queue):
        st = queue[idx]
        raw_cols[st] = [0 for _ in range(station_counts.get(st, 0))]
        idx += 1

    return repair_station_cols(raw_cols, station_counts, station_order, n)


def repair_station_cols(raw_cols, station_counts, station_order, n):
    repaired = {}
    col_station_types = {c: set() for c in range(n)}
    upper = n - 1
    first3_set = set(station_order[:num_single_col])

    for idx, st in enumerate(station_order):
        k = station_counts[st]
        if st == "BUFFER_209_200":
            # 虚拟外部缓冲列，不分配车间列
            repaired[st] = [n] * k
            continue

        values = raw_cols.get(st, [])
        if not isinstance(values, list):
            values = [values for _ in range(k)]
        if len(values) < k:
            fill = values[-1] if values else upper
            values = values + [fill for _ in range(k - len(values))]
        preferred = min(max(int(values[0] if values else upper), 0), n - 1, upper)
        remaining_types = len(station_order) - idx - 1
        candidates = sorted(range(upper + 1), key=lambda c: (abs(c - preferred), c))
        chosen_col = upper
        found = False
        for candidate in candidates:
            if st in first3_set:
                if len(col_station_types[candidate]) > 0:
                    continue
            else:
                if col_station_types[candidate] & first3_set:
                    continue
                if st not in col_station_types[candidate] and len(col_station_types[candidate]) >= num_of_col:
                    continue
            after_counts = {
                c: len(types) + (1 if c == candidate and st not in types else 0)
                for c, types in col_station_types.items()
            }
            free_type_slots = sum(
                0 if (col_station_types[c] & first3_set) or (c == candidate and st in first3_set)
                else max(0, num_of_col - after_counts[c])
                for c in range(candidate + 1)
            )
            if free_type_slots >= remaining_types:
                chosen_col = candidate
                found = True
                break
        if not found:
            # 放宽剩余类型容量检查，仍保证列本身可放
            for candidate in candidates:
                if st in first3_set:
                    if len(col_station_types[candidate]) > 0:
                        continue
                else:
                    if col_station_types[candidate] & first3_set:
                        continue
                    if st not in col_station_types[candidate] and len(col_station_types[candidate]) >= num_of_col:
                        continue
                chosen_col = candidate
                break
        cols = [chosen_col for _ in range(k)]
        col_station_types[chosen_col].add(st)
        upper = min(upper, chosen_col)
        repaired[st] = cols
    return repaired


def random_layout(station_counts, station_order, m, n):
    rows_by_station = {}
    for st in station_order:
        k = station_counts[st]
        if st == "BUFFER_209_200":
            # 虚拟缓冲列不占 grid，占位行列表用 0 初始化，后续由 _align_buffer_rows 同步
            rows_by_station[st] = [0] * k
            continue
        picked = random.sample(range(m), k)
        random.shuffle(picked)
        rows_by_station[st] = picked
    _align_buffer_rows(rows_by_station, station_counts)
    station_cols = random_station_cols(station_counts, station_order, n)
    return Individual(rows_by_station=rows_by_station, station_cols=station_cols)


def decode_layout(individual, station_counts, station_order, m, n, skip_repair=False):
    grid = [[0 for _ in range(n)] for _ in range(m)]
    station_coord = {}
    used_cols = {}
    if skip_repair or getattr(individual, 'compressed', False):
        # 使用原始 station_cols，仅做基础格式化和边界检查
        raw = individual.station_cols or {st: [n - 1 - i for _ in range(station_counts[st])] for i, st in enumerate(station_order)}
        station_cols = {}
        for st in station_order:
            vals = raw.get(st, [])
            if not isinstance(vals, list):
                vals = [vals]
            k = station_counts[st]
            if len(vals) < k:
                fill = vals[-1] if vals else n - 1
                vals = vals + [fill for _ in range(k - len(vals))]
            station_cols[st] = [min(max(int(c), 0), n - 1) for c in vals[:k]]
    else:
        station_cols = repair_station_cols(
            individual.station_cols or {st: [n - 1 - i for _ in range(station_counts[st])] for i, st in enumerate(station_order)},
            station_counts,
            station_order,
            n,
        )
    for idx, st in enumerate(station_order):
        cols = station_cols[st]
        # BUFFER_209_200 是车间外部虚拟缓冲列，不写入 grid，坐标在下方单独处理
        if st == "BUFFER_209_200":
            used_cols[st] = cols
            continue
        used_cols[st] = cols
        rows = individual.rows_by_station[st]
        for sid in range(station_counts[st]):
            # 边界检查：确保rows列表长度足够
            if sid >= len(rows):
                while len(rows) <= sid:
                    rows.append(0)
            r = rows[sid]
            # 边界检查：确保cols列表长度足够
            if sid >= len(cols):
                while len(cols) <= sid:
                    cols.append(cols[-1] if cols else 0)
            col = cols[sid]
            # 边界检查
            if r >= m:
                r = m - 1
                rows[sid] = r
            if col >= n:
                col = n - 1
                cols[sid] = col
            if grid[r][col] != 0:
                candidates = sorted(range(m), key=lambda rr: (abs(rr - r), rr))
                for rr in candidates:
                    if grid[rr][col] == 0:
                        r = rr
                        rows[sid] = rr
                        break
            grid[r][col] = f"{st}-{sid + 1}"
            station_coord[f"{st}_{sid}"] = (r, col)

    # BUFFER_209_200 虚拟外部坐标：200 过渡缓冲区（索引 6..10）行与 200C 对应实例行相同，
    # 列使用 n（车间右侧第一个虚拟列）；209 缓冲区（索引 0..5）列使用 n+1。
    if "BUFFER_209_200" in station_order:
        num_209_lines = station_counts.get("209A", 0)
        num_209_buf = num_209_lines * 2
        buf_count = station_counts.get("BUFFER_209_200", 0)
        rows_200C = individual.rows_by_station.get("200C", [])
        rows_buf = individual.rows_by_station.get("BUFFER_209_200", [])
        for sid in range(buf_count):
            if sid < num_209_buf:
                # 209 缓冲区：行按 209 产线编号（0..num_209_lines-1），列 = n+1（更外侧）
                r_virt = sid % num_209_lines
                station_coord[f"BUFFER_209_200_{sid}"] = (r_virt, n + 1)
            else:
                # 200 过渡缓冲区：行 = 对应 200C 实例行，列 = n
                k = sid - num_209_buf
                r_virt = rows_200C[k] if k < len(rows_200C) else k
                station_coord[f"BUFFER_209_200_{sid}"] = (r_virt, n)

    return grid, station_coord, used_cols


def neighbors(cell, m, n):
    r, c = cell
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < m and 0 <= nc < n:
            yield (nr, nc)


def low_level_astar(start, goal, blocked, reserve_v, reserve_e, m, n, max_t=180, require_goal_hold=False):
    # 使用 heapq 实现的 A*，保留时间维约束以避免每轮排序开销
    # 元组: (f, g, cell, parent_state)
    h0 = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    heap = [(h0, 0, start, None)]
    parent = {}
    best_g = {(start, 0): 0}

    while heap:
        f, g, cell, prev = heapq.heappop(heap)
        state = (cell, g)

        # 如果该 state 已经被记录为访问过（有前驱），跳过重复处理
        if state in parent:
            # 已经处理过这个 (cell, time) 状态
            continue
        parent[state] = prev

        if cell == goal:
            if require_goal_hold:
                hold_start = g + 1
                if any((goal, hold) in reserve_v for hold in range(hold_start, ENDPOINT_HOLD_LIMIT)):
                    continue
            path = []
            cur = state
            while cur is not None:
                path.append(cur[0])
                cur = parent[cur]
            return list(reversed(path))

        if g >= max_t:
            continue

        # 从虚拟外部列n（200过渡缓冲区）出发时，允许在起点停留等待。
        neighbor_cells = list(neighbors(cell, m, n))
        if cell == start and start[1] == n:
            neighbor_cells.append(start)

        for nb in neighbor_cells:
            ng = g + 1
            # Treat 'blocked' as hard obstacles (cells occupied by static aircraft
            # during this phase). Do not allow entering them, except allow the
            # trivial stay-in-place case when start == goal == nb.
            if nb in blocked:
                if not (start == goal and nb == goal):
                    continue
            if (nb, ng) in reserve_v:
                continue
            if ((cell, nb), g) in reserve_e or ((nb, cell), g) in reserve_e:
                continue
            key = (nb, ng)
            prev_best = best_g.get(key)
            if prev_best is not None and prev_best <= ng:
                continue
            best_g[key] = ng
            h = abs(nb[0] - goal[0]) + abs(nb[1] - goal[1])
            heapq.heappush(heap, (ng + h, ng, nb, state))
    return None


def cbs_plan_seed(seed, requests, blocked_cells, station_coord, m, n):
    """Module-level plan function suitable for ProcessPoolExecutor.
    Returns (score, paths) or None on failure.
    """
    rng = random.Random(seed)
    # 200过渡缓冲区->200C的请求放在最后规划，避免阻塞其他主路径飞机。
    buffer_requests = [rq for rq in requests if rq["start"].startswith("BUFFER_209_200_")]
    other_requests = [rq for rq in requests if not rq["start"].startswith("BUFFER_209_200_")]
    order = other_requests[:]
    rng.shuffle(order)
    order.extend(buffer_requests)
    reserve_v = set()
    reserve_e = set()
    paths = {}
    for rq in order:
        s = station_coord[rq["start"]]
        if rq["end"] == "OUT":
            # For OUT requests we treat the exit as instantaneous: the aircraft
            # finishes at its station and in the next move disappears from the
            # grid. Therefore we set the planning goal to the station cell
            # itself (no in-grid exit step) and do not reserve any extra
            # endpoint timestep.
            g = s
        else:
            g = station_coord.get(rq["end"])
        if g is None:
            return None
        path = low_level_astar(s, g, blocked_cells, reserve_v, reserve_e, m, n, require_goal_hold=(rq["end"] != "OUT"))
        if not path:
            return None
        if rq["end"] != "OUT":
            for hold in range(len(path), ENDPOINT_HOLD_LIMIT):
                if (path[-1], hold) in reserve_v:
                    return None
        for t, cell in enumerate(path):
            reserve_v.add((cell, t))
            if t < len(path) - 1:
                reserve_e.add(((path[t], path[t + 1]), t))
        if rq["end"] != "OUT":
            for hold in range(len(path), ENDPOINT_HOLD_LIMIT):
                reserve_v.add((path[-1], hold))
        # Reserve the endpoint after arrival so later aircraft cannot pass through it.
        paths[rq["gid"]] = path
    score = max((len(path) for path in paths.values()), default=0)
    return score, paths


# Planner ProcessPoolExecutor shared for the module (created lazily in main process)
_PLANNER_EXECUTOR = None

def get_planner_executor():
    global _PLANNER_EXECUTOR
    # Only create the shared ProcessPoolExecutor in the main process to avoid
    # worker processes creating further child processes (which can fail on Windows).
    try:
        if current_process().name != "MainProcess":
            return None
    except Exception:
        return None
    if _PLANNER_EXECUTOR is None:
        cpu_count = os.cpu_count() or 1
        _PLANNER_EXECUTOR = ProcessPoolExecutor(max_workers=cpu_count)
    return _PLANNER_EXECUTOR


# Shared ProcessPoolExecutor for evaluate() tasks. Created in the main process
# to avoid worker processes trying to spawn further child processes on Windows.
_EVAL_EXECUTOR = None


def get_eval_executor():
    global _EVAL_EXECUTOR
    try:
        if current_process().name != "MainProcess":
            return None
    except Exception:
        return None
    if _EVAL_EXECUTOR is None:
        cpu_count = os.cpu_count() or 1
        # Keep some cores for planner pool if used simultaneously; default to cpu_count
        _EVAL_EXECUTOR = ProcessPoolExecutor(max_workers=cpu_count)
    return _EVAL_EXECUTOR


def shutdown_executors():
    global _PLANNER_EXECUTOR, _EVAL_EXECUTOR
    for executor in (_PLANNER_EXECUTOR, _EVAL_EXECUTOR):
        if executor is not None:
            try:
                executor.shutdown(wait=True, cancel_futures=True)
            except TypeError:
                executor.shutdown(wait=True)
    _PLANNER_EXECUTOR = None
    _EVAL_EXECUTOR = None


def evaluate_worker(rows_by_station, station_cols, cfg, schedule, station_counts):
    """Top-level wrapper used when submitting evaluate() to a process pool.
    Accepts picklable primitive structures, reconstructs an Individual and
    returns the evaluate() payload. Kept at module top-level so it is
    picklable on Windows.
    """
    ind = Individual(rows_by_station=rows_by_station, station_cols=station_cols)
    return evaluate(ind, cfg, schedule, station_counts)


def cbs_like_plan_old(requests, blocked_cells, station_coord, m, n):
    # 使用“冲突后重排”的CBS-lite策略
    # 200过渡缓冲区->200C的请求放在最后规划，避免阻塞其他主路径飞机。
    buffer_requests = [rq for rq in requests if rq["start"].startswith("BUFFER_209_200_")]
    other_requests = [rq for rq in requests if not rq["start"].startswith("BUFFER_209_200_")]
    for _ in range(8):
        order = other_requests[:]
        random.shuffle(order)
        order.extend(buffer_requests)
        reserve_v = set()
        reserve_e = set()
        paths = {}
        ok = True
        for rq in order:
            s = station_coord[rq["start"]]
            if rq["end"] == "OUT":
                # Treat OUT as immediate disappearance after finishing at the
                # station: plan to remain on the station cell (the next move
                # is 'out' and is not modeled as a grid cell).
                g = s
            else:
                g = station_coord.get(rq["end"])
            if g is None:
                ok = False
                break
            path = low_level_astar(s, g, blocked_cells, reserve_v, reserve_e, m, n, require_goal_hold=(rq["end"] != "OUT"))
            if not path:
                ok = False
                break
            if rq["end"] != "OUT":
                for hold in range(len(path), ENDPOINT_HOLD_LIMIT):
                    if (path[-1], hold) in reserve_v:
                        ok = False
                        break
                if not ok:
                    break
            for t, cell in enumerate(path):
                reserve_v.add((cell, t))
                if t < len(path) - 1:
                    reserve_e.add(((path[t], path[t + 1]), t))
            if rq["end"] != "OUT":
                for hold in range(len(path), ENDPOINT_HOLD_LIMIT):
                    reserve_v.add((path[-1], hold))
            # Reserve the endpoint after arrival so later aircraft cannot pass through it.
            paths[rq["gid"]] = path
        if ok:
            return paths
    return {}


def cbs_like_plan_new(requests, blocked_cells, station_coord, m, n):
    # 使用“冲突后重排”的CBS-lite策略
    valid_requests = [rq for rq in requests if rq["start"] in station_coord]

    # 将单次尝试提升为模块级可序列化函数 cbs_plan_seed（见模块顶部），
    # 以便在 ProcessPoolExecutor 中使用进程并行加速 CPU 密集的 A* 调用。
    def cbs_plan_seed_wrapper(seed):
        # wrapper 保持本地签名以便内部调用 submit
        return cbs_plan_seed(seed, valid_requests, blocked_cells, station_coord, m, n)

    seeds = [random.randrange(2**32) for _ in range(8)]
    if not valid_requests:
        return {}

    cpu_count = os.cpu_count() or 1
    max_workers = min(len(seeds), cpu_count)

    best_result = None
    # Try to get shared planner executor (created only in main process). If not
    # available (e.g., in worker process), fall back to sequential execution.
    planner_exec = get_planner_executor()
    if planner_exec is not None:
        futures = [planner_exec.submit(cbs_plan_seed, s, valid_requests, blocked_cells, station_coord, m, n) for s in seeds]
        for fut in futures:
            try:
                res = fut.result()
            except Exception:
                res = None
            if res is not None and (best_result is None or res[0] < best_result[0]):
                best_result = res
    else:
        # Sequential fallback (no shared process pool available)
        for s in seeds:
            try:
                res = cbs_plan_seed(s, valid_requests, blocked_cells, station_coord, m, n)
            except Exception:
                res = None
            if res is not None and (best_result is None or res[0] < best_result[0]):
                best_result = res

    if best_result is None:
        return {}
    return best_result[1]


def my_path_planner1(requests, blocked_cells, station_coord, m, n):
    """Current default planner (same behavior as existing implementation)."""
    return cbs_like_plan_old(requests, blocked_cells, station_coord, m, n)


# 209 预装配线内部工位顺序（F->E->D->C->B->A），BUFFER_209_200 作为 209 区域出口
_209_STATION_PREFIXES = ["209F", "209E", "209D", "209C", "209B", "209A"]


def _build_209_coord_map(station_counts):
    """为 209 预装配区及缓冲区建立虚拟坐标 (line, col)。

    209 工位：209F=0 ... 209A=5。
    右上角 209 缓冲区：2 列 × 3 行 = 6 个位置（line 0..2，slot 0..1）。
    200 过渡缓冲区：5 个位置，在 209 缓冲区下方，与 200C 对齐。
    """
    coord_map = {}
    for idx, st in enumerate(_209_STATION_PREFIXES):
        for sid in range(station_counts.get(st, 0)):
            coord_map[f"{st}_{sid}"] = (sid, idx)
    buf_count = station_counts.get("BUFFER_209_200", 0)
    num_209_lines = station_counts.get("209A", 0)
    num_209_buf = num_209_lines * 2  # 2 slots per line
    base = len(_209_STATION_PREFIXES)
    for sid in range(buf_count):
        if sid < num_209_buf:
            # 209 缓冲区：按 line 优先排列，每个 line 两个 slot
            line = sid % num_209_lines
            slot = sid // num_209_lines
            coord_map[f"BUFFER_209_200_{sid}"] = (line, base + slot)
        else:
            # 200 过渡缓冲区：排在 209 缓冲区下方
            k = sid - num_209_buf
            coord_map[f"BUFFER_209_200_{sid}"] = (k, base + 2)
    return coord_map


def _209_distance(start_key, end_key, coord_map):
    """计算 209 区域两个工位之间的曼哈顿距离（cell2 单位），全部按地面移动。"""
    p1 = coord_map.get(start_key)
    p2 = coord_map.get(end_key)
    if p1 is None or p2 is None:
        return None
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def _is_209_movement(req, coord_map):
    return req["start"] in coord_map or req["end"] in coord_map


def _build_209_path(start_key, end_key, coord_map):
    """为 209 区域移动构建虚拟坐标路径（格点列表 [[r,c], ...]）。"""
    p1 = coord_map.get(start_key)
    p2 = coord_map.get(end_key)
    if p1 is None or p2 is None:
        return None
    path = [[p1[0], p1[1]]]
    r, c = p1
    while r != p2[0]:
        r += 1 if r < p2[0] else -1
        path.append([r, c])
    while c != p2[1]:
        c += 1 if c < p2[1] else -1
        path.append([r, c])
    return path


def _align_buffer_rows(rows_by_station, station_counts):
    """
    BUFFER_209_200 是车间外部虚拟缓冲列，不占 grid 行。
    此函数仅把 200 过渡缓冲区（索引 num_209_buf..10）的行号与 200C 对应实例行保持一致，
    以便 decode_layout 能直接用 rows_200C 推算虚拟外部坐标。
    209 缓冲区（索引 0..num_209_buf-1）的 row 字段不影响 grid，可任意值。
    """
    if "200C" not in rows_by_station or "BUFFER_209_200" not in rows_by_station:
        return rows_by_station

    rows_200C = list(rows_by_station["200C"])
    rows_buf = list(rows_by_station["BUFFER_209_200"])
    n_buf = len(rows_buf)
    num_209_buf = station_counts.get("209A", 0) * 2

    # 200 过渡缓冲区行 = 对应 200C 实例行
    for i, r in enumerate(rows_200C):
        buf_idx = num_209_buf + i
        if buf_idx < n_buf:
            rows_buf[buf_idx] = r

    rows_by_station["BUFFER_209_200"] = rows_buf
    return rows_by_station


def my_path_planner2(requests, blocked_cells, station_coord, m, n):
    """Placeholder planner with the same I/O signature as planner1."""
    return cbs_like_plan_new(requests, blocked_cells, station_coord, m, n)


def choose_path_planner(mode):
    if mode == 2:
        return my_path_planner2
    return my_path_planner1


def evaluate(individual, cfg, schedule, station_counts):
    station_order = cfg["station_order"]
    m = cfg["grid_rows"]
    n = cfg["grid_cols"]
    t0 = cfg["move_hours_per_cell"]

    # 根据individual实际列坐标确定压缩后的grid宽度（排除BUFFER_209_200的虚拟外部列）
    actual_n = getattr(individual, "actual_cols", None)
    if actual_n is None:
        all_ind_cols = [
            c for st, cols in individual.station_cols.items()
            if st != "BUFFER_209_200"
            for c in cols
        ]
        actual_n = max(all_ind_cols) + 1 if all_ind_cols else n
    # 保证有足够的列数容纳所有类型：前 num_single_col 种独占，其余每列最多 num_of_col 种
    min_n = num_single_col + math.ceil((len(station_order) - num_single_col) / num_of_col)
    actual_n = max(actual_n, min_n)

    grid, station_coord, used_cols = decode_layout(
        individual, station_counts, station_order, m, actual_n,
        skip_repair=getattr(individual, 'compressed', False)
    )
    station_positions = {st: [coord[0], coord[1]] for st, coord in station_coord.items()}
    # --- Fast path: return cached evaluate result for identical layout (rows+cols by station)
    try:
        key_rows = tuple(tuple(individual.rows_by_station[st]) for st in station_order)
    except Exception:
        key_rows = tuple(tuple(individual.rows_by_station.get(st, [])) for st in station_order)
    try:
        key_cols = tuple(tuple(used_cols.get(st, [])) for st in station_order)
    except Exception:
        key_cols = tuple(tuple([]) for _ in station_order)
    cache_key = (key_rows, key_cols)
    if cache_key in _EVAL_CACHE:
        cached_payload = _EVAL_CACHE.pop(cache_key)
        _EVAL_CACHE[cache_key] = cached_payload
        cached = copy.deepcopy(cached_payload)
        individual.objectives = cached.get("objectives", individual.objectives)
        return cached
    # 使用完整 station_order（含 209 与缓冲区）解析移动阶段，以便识别 209 区域内部移动。
    full_station_order = build_station_order_from_schedule(schedule)
    phases = parse_movements(schedule, full_station_order)
    cumulative_h = 0
    total_T = 0
    move_records = []
    # 虚拟外部坐标（列 >= n）不占 grid，不小入 used_cells
    used_cells = {c for c in station_coord.values() if c[1] < n}

    # 仅在一阶段优化时把 209 预装配区内部移动纳入总移动时间。
    include_209 = (Stage_MODE == 1)
    _209_coord_map = _build_209_coord_map(station_counts) if include_209 else {}

    planner = choose_path_planner(CURRENT_PATH_PLANNER_MODE)

    for phase in phases:
        blocked = set()
        for st in phase["static"]:
            coord = station_coord.get(st)
            if coord is not None and coord[1] < n:  # 只阻塞 grid 内工位
                blocked.add(coord)
        all_reqs = phase["requests"]
        # 将 start 在 grid 内以及 start 在200过渡缓冲区（虚拟外部列n）的请求交给路径规划器
        reqs_main = [r for r in all_reqs
                     if r["start"] in station_coord
                     and (station_coord[r["start"]][1] < n
                          or (r["start"].startswith("BUFFER_209_200_") and station_coord[r["start"]][1] == n))]
        reqs_209 = [r for r in all_reqs if include_209 and _is_209_movement(r, _209_coord_map)]
        try:
            paths = planner(reqs_main, blocked, station_coord, m, n)
        except Exception:
            # Fallback to the current stable planner when alternative planner fails.
            paths = my_path_planner1(reqs_main, blocked, station_coord, m, n)

        max_move_h = 0
        for rq in reqs_main:
            gid = rq["gid"]
            path = paths.get(gid)
            if not path:
                # 罚分：无法规划时给予大罚值
                total_T += 500
                continue
            steps = max(0, len(path) - 1)
            # If this request is an OUT (exit), treat the movement as a single
            # step that removes the aircraft from the grid on the next tick.
            if rq.get("end") == "OUT":
                steps = max(1, steps)
            move_h = steps * t0
            max_move_h = max(max_move_h, move_h)
            for cell in path:
                used_cells.add(cell)

            # 输出格式：[P, [path...], [timestamps...], gid, kind, from, to]
            ts = [cumulative_h + i * t0 for i in range(len(path))]
            path_out = path
            move_records.append([
                phase["index"],
                [[p[0], p[1]] for p in path_out],
                ts,
                gid,
                "main",
                rq["start"],
                rq.get("end"),
            ])

        # 209 预装配区内部移动按固定虚拟布局计算曼哈顿距离，全部按地面移动。
        max_209_h = 0
        if include_209:
            for rq in reqs_209:
                dist = _209_distance(rq["start"], rq["end"], _209_coord_map)
                if dist is None:
                    continue
                move_h = dist * t0
                max_209_h = max(max_209_h, move_h)

                # 生成 209/缓冲区地面移动路径记录
                path_209 = _build_209_path(rq["start"], rq["end"], _209_coord_map)
                if path_209:
                    ts_209 = [cumulative_h + i * t0 for i in range(len(path_209))]
                    move_records.append([
                        phase["index"], path_209, ts_209, rq["gid"],
                        "209", rq["start"], rq["end"],
                    ])

        phase_h = max(max_move_h, max_209_h)
        cumulative_h += phase_h
        total_T += phase_h

    # 将未使用的0标记为-1，使用字符串编码格式
    grid_used_cells = {
        cell for cell in used_cells
        if 0 <= cell[0] < m and 0 <= cell[1] < actual_n
    }
    layout_vec = []
    for r in range(m):
        for c in range(actual_n):
            val = grid[r][c]
            if val == 0 and (r, c) not in grid_used_cells:
                layout_vec.append(-1)
            else:
                layout_vec.append(val)

    # 车间区域定义为所有非-1格子的外接矩形
    if grid_used_cells:
        min_r = min(c[0] for c in grid_used_cells)
        max_r = max(c[0] for c in grid_used_cells)
        min_c = min(c[1] for c in grid_used_cells)
        max_c = max(c[1] for c in grid_used_cells)
        # 将外接矩形内的-1改为0（标为淡蓝色）
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                idx = r * actual_n + c
                if layout_vec[idx] == -1:
                    layout_vec[idx] = 0
        # 将外接矩形外的所有非-1格子改为-1，确保车间区域为严格矩形
        for r in range(m):
            for c in range(actual_n):
                idx = r * actual_n + c
                if r < min_r or r > max_r or c < min_c or c > max_c:
                    if layout_vec[idx] != -1:
                        layout_vec[idx] = -1
    else:
        min_r, max_r, min_c, max_c = 0, m - 1, 0, actual_n - 1

    # 以车间区域外接矩形计算面积
    n_rows_used = max_r - min_r + 1
    n_cols_used = max_c - min_c + 1

    _cl = cfg["layout_cell_length"]
    _cw = cfg["layout_cell_width"]
    _ch = cfg["layout_channel"]
    _um = cfg["layout_unit_m"]
    height_units = n_rows_used * _cl + (n_rows_used + 1) * _ch
    width_units  = n_cols_used * _cw + (n_cols_used + 1) * _ch
    area_m2 = height_units * width_units * (_um ** 2)

    raw_objectives = (float(total_T), float(area_m2))
    discarded = (raw_objectives[0] > MAX_TOTAL_T_HOURS) or (raw_objectives[1] > MAX_AREA_M2)
    individual.objectives = raw_objectives
    payload = {
        "objectives": individual.objectives,
        "raw_objectives": raw_objectives,
        "discarded": discarded,
        "layout_vector": layout_vec,
        "grid": grid,
        "station_cols": used_cols,
        "station_positions": station_positions,
        "move_records": move_records,
        "used_cells": [[c[0], c[1]] for c in sorted(used_cells)],
        "total_move_hours": total_T,
        "actual_cols": actual_n,
    }
    try:
        _EVAL_CACHE[cache_key] = copy.deepcopy(payload)
        while len(_EVAL_CACHE) > _EVAL_CACHE_MAX_SIZE:
            _EVAL_CACHE.popitem(last=False)
    except Exception:
        pass
    return payload


def mutate(individual, station_counts, m, n, station_order):
    child = copy.deepcopy(individual)
    roll = random.random()
    if roll < 0.7:
        st = random.choice(station_order)
        rows = child.rows_by_station[st]
        sid = random.randrange(len(rows))
        occupied = set(rows)
        free = [r for r in range(m) if r not in occupied or r == rows[sid]]
        rows[sid] = random.choice(free)
    elif roll < 0.85:
        st = random.choice(station_order)
        cols = child.station_cols[st]
        sid = random.randrange(len(cols))
        cols[sid] = random.randrange(n)
        child.station_cols = repair_station_cols(child.station_cols, station_counts, station_order, n)
    else:
        # Merge-column mutation: push two consecutive stations to share a column.
        # This directly explores the two-types-per-column search space.
        if len(station_order) > 1:
            idx = random.randrange(len(station_order) - 1)
            st_a = station_order[idx]
            st_b = station_order[idx + 1]
            cols_a = child.station_cols.get(st_a, [])
            if isinstance(cols_a, list) and cols_a:
                target_col = cols_a[0]
            elif isinstance(cols_a, int):
                target_col = cols_a
            else:
                target_col = 0
            cols_b = child.station_cols.get(st_b)
            if isinstance(cols_b, list):
                child.station_cols[st_b] = [target_col] * len(cols_b)
            else:
                child.station_cols[st_b] = [target_col]
            child.station_cols = repair_station_cols(child.station_cols, station_counts, station_order, n)
        else:
            st = random.choice(station_order)
            cols = child.station_cols[st]
            sid = random.randrange(len(cols))
            cols[sid] = random.randrange(n)
            child.station_cols = repair_station_cols(child.station_cols, station_counts, station_order, n)
    _align_buffer_rows(child.rows_by_station, station_counts)
    return child


def crossover(a, b, station_counts, station_order, n):
    rows = {}
    for st in a.rows_by_station:
        rows[st] = copy.deepcopy(a.rows_by_station[st] if random.random() < 0.5 else b.rows_by_station[st])
    _align_buffer_rows(rows, station_counts)
    raw_cols = {}
    for st in station_order:
        k = station_counts[st]
        raw_cols[st] = [
            (a.station_cols[st][sid] if random.random() < 0.5 else b.station_cols[st][sid])
            for sid in range(k)
        ]
    station_cols = repair_station_cols(raw_cols, station_counts, station_order, n)
    return Individual(rows_by_station=rows, station_cols=station_cols)


def compute_hypervolume_2d(points, ref_point):
    # 二维最小化问题的超体积（相对参考点）
    if not points:
        return 0.0
    px = [(float(p[0]), float(p[1])) for p in points]
    px = [p for p in px if p[0] <= ref_point[0] and p[1] <= ref_point[1]]
    if not px:
        return 0.0
    px.sort(key=lambda x: (x[0], x[1]))
    hv = 0.0
    prev_y = float(ref_point[1])
    for x, y in px:
        if y < prev_y:
            hv += max(0.0, float(ref_point[0]) - x) * (prev_y - y)
            prev_y = y
    return hv


def filter_non_dominated_payloads(payloads):
    if not payloads:
        return []
    selected = []
    for i, p in enumerate(payloads):
        oi = p["objectives"]
        dominated = False
        for j, q in enumerate(payloads):
            if i == j:
                continue
            oj = q["objectives"]
            if dominates_obj(oj, oi):
                dominated = True
                break
        if not dominated:
            selected.append(p)
    return selected


def payloads_to_pareto_points(payloads):
    return [{"T_hours": p["objectives"][0], "S_m2": p["objectives"][1]} for p in payloads]


def payloads_to_annotated_pareto_points(payloads):
    return [
        {
            "T_hours": p["objectives"][0],
            "S_m2": p["objectives"][1],
            "generation": p.get("source_generation"),
        }
        for p in payloads
    ]


def format_pareto_solutions(payloads):
    pairs = [f"({p['objectives'][0]:.1f},{p['objectives'][1]:.1f})" for p in payloads]
    return "[" + ", ".join(pairs) + "]"


def pareto_signature(payloads):
    """Build a stable signature for Pareto front changes using objective pairs only."""
    return tuple(sorted((float(p["objectives"][0]), float(p["objectives"][1])) for p in payloads))


def update_pareto_archive(archive_payloads, candidate_payloads):
    # Internal switch: whether to discard solutions that share the same objectives
    # but have different layouts. Default False means keep different layouts.
    discard_same_objectives_diff_layout = False

    merged = archive_payloads + candidate_payloads
    nd = filter_non_dominated_payloads(merged)
    # Deduplicate archive entries according to the switch above.
    seen = set()
    unique = []
    for p in nd:
        obj_key = (float(p["objectives"][0]), float(p["objectives"][1]))
        if discard_same_objectives_diff_layout:
            key = obj_key
        else:
            layout_sig = tuple(p.get("layout_vector", []))
            key = (obj_key, layout_sig)
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return sorted(unique, key=lambda p: (p["objectives"][0], p["objectives"][1]))


def run_aco(cfg, schedule, station_counts):
    params = cfg["aco"]
    station_order = cfg["station_order"]
    m, n = cfg["grid_rows"], cfg["grid_cols"]
    pheromone = {
        st: [[1.0 for _ in range(m)] for _ in range(station_counts[st])]
        for st in station_order
    }
    pheromone_col = {st: [1.0 for _ in range(n)] for st in station_order}

    hv_history = []
    pareto_archive_payloads = []
    pareto_groups = []
    layouts_by_generation = []
    best_payload = None
    best_score = math.inf
    ref_point = None
    best_hv = -1.0
    best_hv_generation = None

    stagnation_start = int(params["iterations"])
    stagnation_patience = int(params.get("pareto_stagnation_patience", 20))
    # Maximum extra generations to run after reaching stagnation_start (safety cap)
    max_extra = int(params.get("max_extra_generations", 1000))
    stagnation_count = 0
    last_pareto_sig = None

    g = 0
    extra_since_start = 0
    # Determine planner switch generation threshold.
    # For ACO runs use cfg['aco']['iterations'] as the switch point (if present);
    # generations before that will use planner mode 1, and generations >= that
    # will use planner mode 2. If the config key is missing or invalid, no
    # automatic switching will be performed.
    switch_gen = None
    try:
        switch_gen = int(cfg.get("aco", {}).get("iterations"))
    except Exception:
        switch_gen = None
    def sample_feasible_ant(seed):
        rng = random.Random(seed)
        for _retry in range(MAX_FEASIBLE_RETRIES):
            rows_by_station = {}
            station_cols = {}
            col_type_count = defaultdict(int)
            prev_col = n
            for idx_st, st in enumerate(station_order):
                remaining_after = len(station_order) - idx_st - 1
                # With <=2 station types per column, ceil(remaining_after/2)
                # distinct columns suffice for all remaining stations.
                lo = (remaining_after + 1) // 2
                # Allow sharing prev_col if it still has room (< 2 types).
                hi = prev_col if col_type_count[prev_col] < 2 else prev_col - 1
                hi = min(hi, n - 1)
                lo = max(lo, 0)
                if lo > hi:
                    lo = hi
                # Exclude prev_col-1: placing this station there would violate
                # the ordering constraint (B's col == A's col - 1).
                forbidden_col = prev_col - 1
                candidates = [
                    c for c in range(lo, hi + 1)
                    if col_type_count[c] < 2 and c != forbidden_col
                ]
                if not candidates:
                    candidates = [c for c in range(lo, hi + 1) if col_type_count[c] < 2]
                if not candidates:
                    candidates = [c for c in range(0, hi + 1) if col_type_count[c] < 2]
                if not candidates:
                    candidates = list(range(0, max(1, hi + 1)))
                weights = [pheromone_col[st][c] for c in candidates]
                # use local rng for thread-safety and reproducibility
                try:
                    chosen_col = rng.choices(candidates, weights=weights, k=1)[0]
                except Exception:
                    chosen_col = rng.choice(candidates) if candidates else 0
                station_cols[st] = chosen_col
                col_type_count[chosen_col] += 1
                prev_col = chosen_col

                assigned = []
                for sid in range(station_counts[st]):
                    probs = pheromone[st][sid][:]
                    for r in assigned:
                        probs[r] = 0.0
                    ssum = sum(probs)
                    if ssum <= 0:
                        choices = [r for r in range(m) if r not in assigned]
                        chosen = rng.choice(choices)
                    else:
                        pick = rng.random() * ssum
                        acc = 0.0
                        chosen = 0
                        for r, p in enumerate(probs):
                            acc += p
                            if acc >= pick:
                                chosen = r
                                break
                    assigned.append(chosen)
                rows_by_station[st] = assigned

            ind = Individual(rows_by_station=rows_by_station, station_cols=station_cols)
            payload = evaluate(ind, cfg, schedule, station_counts)
            if payload.get("discarded", False):
                continue
            return ind, payload
        # If retries exhausted
        raise ValueError("Unable to sample feasible ACO ant under hard limits after retries")

    while True:
        # Enforce optional hard cap on total generations.
        global MAX_GENERATIONS_LIMIT, CURRENT_PATH_PLANNER_MODE
        if MAX_GENERATIONS_LIMIT is not None and g >= MAX_GENERATIONS_LIMIT:
            print(f"[ACO] 达到最大代数限制 {MAX_GENERATIONS_LIMIT}，强制终止。")
            break

        # Decide planner mode for this generation based on config switch_gen
        if switch_gen is not None:
            if (g + 1) >= switch_gen:
                CURRENT_PATH_PLANNER_MODE = 2
            else:
                CURRENT_PATH_PLANNER_MODE = 1

        # Sample ants in parallel using threads. The heavy CPU work inside evaluate
        # will be handled by the planner's process pool when available.
        seeds = [random.randrange(2**32) for _ in range(params["ants"])]
        max_workers = min(params["ants"], (os.cpu_count() or 1) * 2)
        ants = []
        payloads = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(sample_feasible_ant, s) for s in seeds]
            for fut in futures:
                ind, payload = fut.result()
                ants.append(ind)
                payloads.append(payload)

        fronts = non_dominated_sort(ants)
        front0 = fronts[0] if fronts else []
        current_front_payloads = [payloads[idx] for idx in front0 if not payloads[idx].get("discarded", False)]
        front_objs = [p["objectives"] for p in current_front_payloads]
        if ref_point is None:
            if front_objs:
                ref_point = (
                    max(o[0] for o in front_objs) * 1.1 + 1.0,
                    max(o[1] for o in front_objs) * 1.1 + 1.0,
                )
            else:
                ref_point = (1.0, 1.0)
        for p in current_front_payloads:
            p.setdefault("source_generation", g + 1)
        if current_front_payloads:
            chosen = min(current_front_payloads, key=lambda p: (p["objectives"][0], p["objectives"][1]))
            layouts_by_generation.append({
                "generation": g + 1,
                "payload": {
                    "grid": chosen["grid"],
                    "objectives": chosen["objectives"],
                    "station_cols": chosen.get("station_cols", {}),
                },
            })
        pareto_archive_payloads = update_pareto_archive(pareto_archive_payloads, current_front_payloads)
        pareto_count = len(current_front_payloads)
        pareto_cum_count = len(pareto_archive_payloads)
        hv = compute_hypervolume_2d(front_objs, ref_point)
        hv_history.append(hv)
        if hv > best_hv:
            best_hv = hv
            best_hv_generation = g + 1
        print(
            f"[ACO] 第{g + 1}代 ParetoCount={pareto_count}, ParetoCumCount={pareto_cum_count}, "
            f"Hypervolume={hv_history[-1]:.4f}, ParetoCumSolutions={format_pareto_solutions(pareto_archive_payloads)}"
        )
        pareto_groups.append(
            {
                "generation": g + 1,
                "solutions": payloads_to_annotated_pareto_points(current_front_payloads),
            }
        )

        current_sig = pareto_signature(pareto_archive_payloads)
        if g + 1 >= stagnation_start:
            # We have reached the configured target; start counting extra generations
            extra_since_start += 1
            if last_pareto_sig is not None and current_sig == last_pareto_sig:
                stagnation_count += 1
            else:
                stagnation_count = 0
            last_pareto_sig = current_sig
            if stagnation_count >= stagnation_patience:
                print(
                    f"[ACO] 累计Pareto前沿连续{stagnation_patience}代无变化，提前终止于第{g + 1}代。"
                )
                break
            if extra_since_start >= max_extra:
                print(f"[ACO] 达到最大允许的额外代数 {max_extra}，提前终止以避免无限运行。")
                break
        else:
            last_pareto_sig = current_sig

        # 信息素更新
        ev = params["evaporation"]
        for st in station_order:
            for sid in range(station_counts[st]):
                for r in range(m):
                    pheromone[st][sid][r] *= (1.0 - ev)
                    pheromone[st][sid][r] = max(0.01, pheromone[st][sid][r])
            for c in range(n):
                pheromone_col[st][c] *= (1.0 - ev)
                pheromone_col[st][c] = max(0.01, pheromone_col[st][c])

        scored = sorted(list(zip(ants, payloads)), key=lambda x: (x[0].objectives[0], x[0].objectives[1]))
        feasible_scored = [(ind, p) for ind, p in scored if not p.get("discarded", False)]
        topk = max(1, len(feasible_scored) // 3) if feasible_scored else 0
        for ind, _payload in feasible_scored[:topk]:
            delta = params["q"] / (1.0 + ind.objectives[0])
            for st in station_order:
                for sid, row in enumerate(ind.rows_by_station[st]):
                    pheromone[st][sid][row] += delta
                pheromone_col[st][ind.station_cols[st]] += delta

        if feasible_scored:
            cur_best_ind, cur_best_payload = feasible_scored[0]
            cur_score = cur_best_ind.objectives[0]
            if cur_score < best_score:
                best_score = cur_score
                best_payload = cur_best_payload

        g += 1

    pareto_payloads = pareto_archive_payloads

    if best_payload is None:
        raise ValueError("No feasible solution found under hard limits: T<=5000H and S<=1500000m^2")

    return best_payload, pareto_payloads, hv_history, best_hv_generation, pareto_groups, layouts_by_generation


def tournament(pop):
    a, b = random.sample(pop, 2)
    if a.rank < b.rank:
        return a
    if b.rank < a.rank:
        return b
    return a if a.crowding >= b.crowding else b


def select_next_generation(pool, pop_size):
    fronts = non_dominated_sort(pool)
    for fr in fronts:
        crowding_distance(pool, fr)
    next_pop = []
    for fr in fronts:
        if len(next_pop) + len(fr) <= pop_size:
            next_pop.extend(pool[i] for i in fr)
        else:
            remain = pop_size - len(next_pop)
            ranked = sorted((pool[i] for i in fr), key=lambda x: x.crowding, reverse=True)
            next_pop.extend(ranked[:remain])
            break
    return next_pop, fronts


def run_nsga2(cfg, schedule, station_counts, baseline_ind=None):
    params = cfg["nsga2"]
    m, n = cfg["grid_rows"], cfg["grid_cols"]
    pop = []
    payload_map = {}
    hv_history = []
    ref_point = None
    best_hv = -1.0
    best_hv_generation = None
    pareto_groups = []
    layouts_by_generation = []
    pareto_archive_payloads = []

    stagnation_start = int(params["generations"])
    stagnation_patience = int(params.get("pareto_stagnation_patience", 20))
    stagnation_count = 0
    last_pareto_sig = None

    # Initialize NSGA-II population in parallel to speed up expensive evaluate calls.
    def sample_nsga_ind(_seed):
        for _retry in range(MAX_FEASIBLE_RETRIES):
            ind = random_layout(station_counts, cfg["station_order"], m, n)
            payload = evaluate(ind, cfg, schedule, station_counts)
            if payload.get("discarded", False):
                continue
            return ind, payload
        raise ValueError("Unable to initialize feasible NSGA-II population under hard limits")

    seeds = [random.randrange(2**32) for _ in range(params["population"])]
    max_workers = min(params["population"], (os.cpu_count() or 1) * 2)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(sample_nsga_ind, s) for s in seeds]
        for fut in futures:
            ind, payload = fut.result()
            pop.append(ind)
            payload_map[id(ind)] = payload

    # Seed baseline layout into initial population (replace first individual)
    if baseline_ind is not None:
        baseline_payload = evaluate(baseline_ind, cfg, schedule, station_counts)
        if pop:
            pop[0] = baseline_ind
        else:
            pop.append(baseline_ind)
        payload_map[id(baseline_ind)] = baseline_payload
        print(f"[BASELINE] seeded into initial population: "
              f"T={baseline_payload['objectives'][0]:.1f}h, "
              f"S={baseline_payload['objectives'][1]:.0f}m2")

    g = 0
    # Use the NSGA-II configured generations as the planner switch threshold.
    # When running NSGA-II we will use cfg['nsga2']['generations'] as the
    # switch point so that earlier generations use mode 1 and generations
    # >= that use mode 2.
    nsga_switch_gen = None
    try:
        nsga_switch_gen = int(cfg.get("nsga2", {}).get("generations"))
    except Exception:
        nsga_switch_gen = None
    while True:
        # Enforce optional hard cap on total generations for NSGA-II runs.
        global MAX_GENERATIONS_LIMIT, CURRENT_PATH_PLANNER_MODE
        if MAX_GENERATIONS_LIMIT is not None and g >= MAX_GENERATIONS_LIMIT:
            print(f"[NSGA-II] 达到最大代数限制 {MAX_GENERATIONS_LIMIT}，强制终止。")
            break


        # Set planner mode for this generation if configured
        if nsga_switch_gen is not None:
            if (g + 1) >= nsga_switch_gen:
                CURRENT_PATH_PLANNER_MODE = 2
            else:
                CURRENT_PATH_PLANNER_MODE = 1
        pool, fronts = select_next_generation(pop, params["population"])
        pop = pool
        front0 = fronts[0] if fronts else []
        current_front_payloads = [
            payload_map[id(pop[i])]
            for i in front0
            if not payload_map[id(pop[i])].get("discarded", False)
        ]
        front_objs = [p["objectives"] for p in current_front_payloads]
        if ref_point is None:
            if front_objs:
                ref_point = (
                    max(o[0] for o in front_objs) * 1.1 + 1.0,
                    max(o[1] for o in front_objs) * 1.1 + 1.0,
                )
            else:
                ref_point = (1.0, 1.0)
        pareto_count = len(current_front_payloads)
        hv = compute_hypervolume_2d(front_objs, ref_point)
        hv_history.append(hv)
        if hv > best_hv:
            best_hv = hv
            best_hv_generation = g + 1

        for p in current_front_payloads:
            p.setdefault("source_generation", g + 1)
        if current_front_payloads:
            chosen = min(current_front_payloads, key=lambda p: weighted_score(p["objectives"], cfg["objective_weights"]))
            layouts_by_generation.append({
                "generation": g + 1,
                "payload": {
                    "grid": chosen["grid"],
                    "objectives": chosen["objectives"],
                    "station_cols": chosen.get("station_cols", {}),
                },
            })
        pareto_archive_payloads = update_pareto_archive(pareto_archive_payloads, current_front_payloads)
        pareto_cum_count = len(pareto_archive_payloads)
        print(
            f"[NSGA-II] 第{g + 1}代 ParetoCount={pareto_count}, ParetoCumCount={pareto_cum_count}, "
            f"Hypervolume={hv_history[-1]:.4f}, ParetoCumSolutions={format_pareto_solutions(pareto_archive_payloads)}"
        )
        pareto_groups.append(
            {
                "generation": g + 1,
                "solutions": payloads_to_annotated_pareto_points(current_front_payloads),
            }
        )

        current_sig = pareto_signature(pareto_archive_payloads)
        if g + 1 >= stagnation_start:
            if last_pareto_sig is not None and current_sig == last_pareto_sig:
                stagnation_count += 1
            else:
                stagnation_count = 0
            last_pareto_sig = current_sig
            if stagnation_count >= stagnation_patience:
                print(
                    f"[NSGA-II] 累计Pareto前沿连续{stagnation_patience}代无变化，提前终止于第{g + 1}代。"
                )
                break
        else:
            last_pareto_sig = current_sig

        # 识别 Pareto 前沿边界点（rank=0 中 T最小 & S最小），已取消额外的父本选择权重
        _BOUNDARY_BOOST_WEIGHT = 1  # 边界点不再加权
        _front0_inds = [pop[i] for i in front0] if front0 else []
        _boundary_set = set()
        if _front0_inds:
            _min_T_ind = min(_front0_inds, key=lambda x: payload_map[id(x)]["objectives"][0])
            _min_S_ind = min(_front0_inds, key=lambda x: payload_map[id(x)]["objectives"][1])
            _boundary_set.add(id(_min_T_ind))
            _boundary_set.add(id(_min_S_ind))
        _pop_weights = [
            _BOUNDARY_BOOST_WEIGHT if id(ind) in _boundary_set else 1
            for ind in pop
        ]
        _weight_total = sum(_pop_weights)
        _pop_cum = []
        _acc = 0
        for _w in _pop_weights:
            _acc += _w
            _pop_cum.append(_acc)

        def _weighted_pick():
            _r = random.random() * _weight_total
            for _idx, _c in enumerate(_pop_cum):
                if _r <= _c:
                    return pop[_idx]
            return pop[-1]

        def _weighted_tournament():
            a = _weighted_pick()
            b = _weighted_pick()
            if a.rank < b.rank:
                return a
            if b.rank < a.rank:
                return b
            return a if a.crowding >= b.crowding else b

        children = []
        while len(children) < params["population"]:
            p1 = _weighted_tournament()
            p2 = _weighted_tournament()
            if random.random() < params["crossover_rate"]:
                c = crossover(p1, p2, station_counts, cfg["station_order"], n)
            else:
                c = copy.deepcopy(p1)
            if random.random() < params["mutation_rate"]:
                c = mutate(c, station_counts, m, n, cfg["station_order"])
            payload = evaluate(c, cfg, schedule, station_counts)
            if payload.get("discarded", False):
                continue
            payload_map[id(c)] = payload
            children.append(c)

        pool = pop + children
        pop, _ = select_next_generation(pool, params["population"])
        keep_ids = {id(ind) for ind in pop}
        payload_map = {key: payload for key, payload in payload_map.items() if key in keep_ids}

        g += 1

    final_pool, _final_fronts = select_next_generation(pop, params["population"])
    pop = final_pool

    feasible_pop = [ind for ind in pop if not payload_map[id(ind)].get("discarded", False)]
    if not feasible_pop:
        raise ValueError("No feasible solution found under hard limits: T<=5000H and S<=1500000m^2")

    best = min(feasible_pop, key=lambda x: weighted_score(x.objectives, cfg["objective_weights"]))
    best_payload = payload_map[id(best)]
    plot_payloads = pareto_archive_payloads

    return best_payload, plot_payloads, hv_history, best_hv_generation, pareto_groups, layouts_by_generation


def _grid_to_individual(new_grid, station_order):
    """从压缩后的grid重建Individual对象"""
    child_rows = {st: [] for st in station_order}
    child_cols = {st: [] for st in station_order}
    for r in range(len(new_grid)):
        for c in range(len(new_grid[0])):
            val = new_grid[r][c]
            if val != 0 and '-' in str(val):
                st = str(val).split('-')[0]
                sid = int(str(val).split('-')[1]) - 1
                if st in child_rows:
                    while len(child_rows[st]) <= sid:
                        child_rows[st].append(0)
                        child_cols[st].append(0)
                    child_rows[st][sid] = r
                    child_cols[st][sid] = c
    return Individual(rows_by_station=child_rows, station_cols=child_cols)


def run_stage2_optimization(initial_individuals, cfg, schedule, station_counts):
    """
    二阶段优化：对每个一阶段结果，删除外接矩形内除最左侧空列外的所有空列，
    重新评估移动时间。
    Returns: (best_payload, pareto_payloads, hv_history, best_hv_generation, pareto_groups, layouts_by_generation, all_payloads)
    """
    station_order = cfg["station_order"]
    m, n = cfg["grid_rows"], cfg["grid_cols"]

    all_payloads = []
    print("[Stage2] 处理每个一阶段结果，删除所有空列和空行...")

    for idx, ind in enumerate(initial_individuals, start=1):
        all_cols_a = [c for cols in ind.station_cols.values() for c in cols]
        cur_n_a = max(all_cols_a) + 1 if all_cols_a else n
        grid, _, used_cols = decode_layout(ind, station_counts, station_order, m, cur_n_a, skip_repair=True)
        new_grid, _, removed_cols = remove_all_empty_columns(grid, used_cols, m, cur_n_a)
        new_grid, removed_left = trim_left_empty_columns(new_grid, keep=1)
        new_grid, removed_top, removed_bottom = remove_empty_rows(new_grid)
        removed_rows = removed_top + removed_bottom
        cur_n_after = cur_n_a - removed_cols - removed_left
        if removed_cols > 0 or removed_left > 0:
            print(f"[Stage2] 结果{idx}: 删除 {removed_cols} 个内部空列 + {removed_left} 个左侧多余空列，列数 {cur_n_a} → {cur_n_after}")
        if removed_rows > 0:
            print(f"[Stage2] 结果{idx}: 删除 {removed_rows} 个空行（上{removed_top}行 下{removed_bottom}行），行数 {m} → {m - removed_rows}")
        if removed_cols == 0 and removed_left == 0 and removed_rows == 0:
            print(f"[Stage2] 结果{idx}: 无空列或空行可删除")
        ind = _grid_to_individual(new_grid, station_order)
        ind.compressed = True

        payload = evaluate(ind, cfg, schedule, station_counts)
        obj = payload["objectives"]
        print(f"[Stage2] 结果{idx}: T={obj[0]:.1f}h, S={obj[1]:.0f}m²")
        all_payloads.append(payload)

    feasible = [p for p in all_payloads if not p.get("discarded", False)]
    best_pool = feasible if feasible else all_payloads
    best_payload = min(best_pool, key=lambda p: (p["objectives"][1], p["objectives"][0]))

    pareto_archive_payloads = filter_non_dominated_payloads(
        [p for p in all_payloads if not p.get("discarded", False)]
    )

    print(f"[Stage2] 完成。最优解: T={best_payload['objectives'][0]:.1f}h, "
          f"S={best_payload['objectives'][1]:.0f}m²")

    return best_payload, pareto_archive_payloads, [], None, [], [], all_payloads


def draw_layout(cfg, grid, objectives, out_path: Path, station_cols=None):
    m = cfg["grid_rows"]
    n = cfg["grid_cols"]
    station_order = cfg["station_order"]
    fig, ax = plt.subplots(figsize=(18, 10))
    x_scale = 1.5
    cell_h = 1.0
    lane_h = 0.5
    row_pitch = cell_h + lane_h

    def row_y(r):
        return r * row_pitch

    def col_x(c):
        return c * x_scale

    palette = [
        "#fde2e2",
        "#e2f0cb",
        "#d9edff",
        "#fff1c9",
        "#e8dcff",
        "#cdeff2",
        "#ffe0b5",
        "#d8f3dc",
        "#f8d7ff",
    ]
    station_colors = {st: palette[i % len(palette)] for i, st in enumerate(station_order)}
    cell_to_station = {}
    col_to_station = {}
    if station_cols:
        for st, cols in station_cols.items():
            if not isinstance(cols, list):
                cols = [cols]
            for sid, col in enumerate(cols):
                col_to_station.setdefault(col, st)
                for r in range(m):
                    if grid[r][col] == sid + 1 and (r, col) not in cell_to_station:
                        cell_to_station[(r, col)] = st
                        break
    else:
        col_to_station = {n - 1 - i: st for i, st in enumerate(station_order)}

    # Draw AGV lanes as dedicated half-cell-height rows between station rows.
    for r in range(m - 1):
        y = row_y(r) + cell_h
        rect = patches.Rectangle((0, y), n * x_scale, lane_h, color="#d8ecff", alpha=0.9)
        ax.add_patch(rect)

    for r in range(m):
        for c in range(n):
            face = "white"
            cell_value = grid[r][c]
            # 处理字符串编码格式
            if isinstance(cell_value, str) and cell_value != "0":
                st = cell_value.split('-')[0]  # 提取工位类型
                face = station_colors.get(st, "#f8f4e6")
            elif isinstance(cell_value, int) and cell_value > 0:
                st = cell_to_station.get((r, c), col_to_station.get(c))
                face = station_colors.get(st, "#f8f4e6")
            y = row_y(r)
            rect = patches.Rectangle((col_x(c), y), x_scale, cell_h, edgecolor="black", facecolor=face, linewidth=0.8)
            ax.add_patch(rect)
            if isinstance(cell_value, str) and cell_value != "0":
                # 字符串格式：直接显示
                label = cell_value
                ax.text(col_x(c) + 0.5 * x_scale, y + 0.5, label, ha="center", va="center", fontsize=7)
            elif isinstance(cell_value, int) and cell_value > 0:
                st = cell_to_station.get((r, c), col_to_station.get(c))
                if st:
                    label = f"{st}-{cell_value}"
                else:
                    label = str(cell_value)
                ax.text(col_x(c) + 0.5 * x_scale, y + 0.5, label, ha="center", va="center", fontsize=7)

    ax.set_xlim(0, (n + 5) * x_scale)
    total_h = m * cell_h + (m - 1) * lane_h
    ax.set_ylim(total_h, 0)
    ax.set_aspect("equal")
    ax.set_title("Layout (origin at right-bottom in output semantics)")

    # 右侧显示目标值
    ax.text((n + 0.4) * x_scale, 2.0, f"T = {objectives[0]:.1f} H", fontsize=11)
    ax.text((n + 0.4) * x_scale, 3.0, f"S = {objectives[1]:.1f} m^2", fontsize=11)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def draw_curve(curve_values, out_path: Path, algo_name: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = list(range(1, len(curve_values) + 1))
    ax.plot(x, curve_values, linewidth=1.0, marker="o", markersize=3)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Hypervolume")
    ax.set_title(f"{algo_name} Hypervolume by Generation")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def draw_pareto_front(pareto_groups, out_path: Path, algo_name: str = "NSGA-II", baseline_obj=None):
    if not pareto_groups:
        return

    all_points = []
    for grp in pareto_groups:
        gen = grp.get("generation")
        for s in grp.get("solutions", []):
            all_points.append((float(s["T_hours"]), float(s["S_m2"]), int(gen) if gen is not None else None))
    if not all_points:
        return

    x_vals = [p[0] for p in all_points]
    y_vals = [p[1] for p in all_points]
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)
    x_span = max(1e-9, x_max - x_min)
    y_span = max(1e-9, y_max - y_min)

    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        cmap = plt.colormaps.get_cmap("viridis").resampled(max(2, len(pareto_groups)))
    except AttributeError:
        # older matplotlib compatibility
        cmap = plt.cm.get_cmap("viridis", max(2, len(pareto_groups)))

    # Keep only points that are far enough from already drawn points in normalized objective space.
    min_norm_dist = 0.035
    accepted_norm_pts = []

    for idx, grp in enumerate(pareto_groups):
        gen = grp.get("generation")
        sols = sorted(grp.get("solutions", []), key=lambda s: (s["T_hours"], s["S_m2"]))
        if not sols:
            continue

        color = cmap(idx)
        kept_x = []
        kept_y = []
        kept_gen = []
        for s in sols:
            x = float(s["T_hours"])
            y = float(s["S_m2"])
            nx = (x - x_min) / x_span
            ny = (y - y_min) / y_span
            if not accepted_norm_pts:
                keep = True
            else:
                min_d2 = min((nx - px) ** 2 + (ny - py) ** 2 for px, py in accepted_norm_pts)
                keep = min_d2 >= (min_norm_dist ** 2)
            if keep:
                accepted_norm_pts.append((nx, ny))
                kept_x.append(x)
                kept_y.append(y)
                kept_gen.append(gen)

        if kept_x:
            ax.scatter(kept_x, kept_y, s=20, color=[color], alpha=0.85)
            for x, y, g in zip(kept_x, kept_y, kept_gen):
                if g is not None:
                    ax.annotate(
                        f"G{g}",
                        (x, y),
                        textcoords="offset points",
                        xytext=(4, 3),
                        fontsize=8,
                        color="#333333",
                    )

    # Plot baseline point if provided
    if baseline_obj is not None:
        bx, by = float(baseline_obj[0]), float(baseline_obj[1])
        ax.scatter([bx], [by], s=120, marker="X", color="red", edgecolor="black", linewidth=1.5, zorder=10)
        ax.annotate(
            "Baseline",
            (bx, by),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
            color="red",
            fontweight="bold",
            zorder=11,
        )
        x_min = min(x_min, bx)
        x_max = max(x_max, bx)
        y_min = min(y_min, by)
        y_max = max(y_max, by)

    x_pad = max(1.0, (x_max - x_min) * 0.08 if x_max > x_min else max(1.0, x_max * 0.08))
    y_pad = max(1.0, (y_max - y_min) * 0.08 if y_max > y_min else max(1.0, y_max * 0.08))
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xlabel("T (hours)")
    ax.set_ylabel("S (m^2)")
    ax.set_title(f"{algo_name} Pareto Fronts by Generation")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def save_layouts_by_generation(cfg, layouts_by_generation, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in layouts_by_generation:
        payload = item.get("payload")
        if not payload:
            continue
        generation = item["generation"]
        draw_layout(
            cfg,
            payload["grid"],
            payload["objectives"],
            out_dir / f"{generation}.png",
            payload.get("station_cols"),
        )


def build_process_gif(image_dir: Path, out_gif: Path, fps: int = 2):
    frames = []
    files = sorted(
        image_dir.glob("*.png"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
    )
    if not files:
        return False

    total = len(files)
    # Frame text shown as title suffix.
    font_size = 36

    def load_large_font(size):
        candidates = [
            "arial.ttf",
            "Arial.ttf",
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\calibri.ttf",
            "DejaVuSans.ttf",
        ]
        for name in candidates:
            try:
                return ImageFont.truetype(name, size)
            except Exception:
                continue
        return ImageFont.load_default()

    font = load_large_font(font_size)
    max_gif_width = 1400
    for i, p in enumerate(files, start=1):
        img = Image.open(p).convert("RGBA")
        w, h = img.size
        canvas = img.copy()

        draw = ImageDraw.Draw(canvas)
        text = f"{i}/{total} frame"
        tw, th = draw.textbbox((0, 0), text, font=font)[2:4]
        # Place frame indicator right after the title area (title suffix style).
        tx = min(max(10, int(w * 0.57)), max(10, w - tw - 10))
        ty = max(6, int(h * 0.018))
        draw.rectangle([(tx - 6, ty - 2), (tx + tw + 6, ty + th + 2)], fill=(255, 255, 255, 220))
        draw.text((tx, ty), text, fill=(20, 20, 20, 255), font=font)
        if canvas.width > max_gif_width:
            ratio = max_gif_width / float(canvas.width)
            canvas = canvas.resize((max_gif_width, int(canvas.height * ratio)), Image.LANCZOS)
        frames.append(canvas.convert("P", palette=Image.ADAPTIVE))

    duration_ms = int(1000 / max(1, fps))
    frames[0].save(
        out_gif,
        save_all=True,
        append_images=frames[1:],
        optimize=True,
        duration=duration_ms,
        loop=0,
        disposal=2,
    )
    return True


def build_pareto_iter_gif(pareto_groups, out_gif: Path, fps: int = 2):
    if not pareto_groups:
        return False

    def update_archive_points(archive_points, candidate_points):
        merged = archive_points + [copy.deepcopy(p) for p in candidate_points]
        selected = []
        for i, p in enumerate(merged):
            oi = (p["T_hours"], p["S_m2"])
            dominated = False
            for j, q in enumerate(merged):
                if i == j:
                    continue
                oj = (q["T_hours"], q["S_m2"])
                if dominates_obj(oj, oi):
                    dominated = True
                    break
            if not dominated:
                selected.append(p)

        # Deduplicate by objective pair and keep earlier (archive-first) generation tag.
        seen = set()
        unique = []
        for p in selected:
            key = (float(p["T_hours"]), float(p["S_m2"]))
            if key in seen:
                continue
            seen.add(key)
            unique.append(p)
        return sorted(unique, key=lambda p: (p["T_hours"], p["S_m2"]))

    archive = []
    frame_groups = []
    for grp in pareto_groups:
        candidates = []
        for s in grp.get("solutions", []):
            candidates.append(
                {
                    "T_hours": float(s["T_hours"]),
                    "S_m2": float(s["S_m2"]),
                    "generation": s.get("generation", grp.get("generation")),
                }
            )
        archive = update_archive_points(archive, candidates)
        frame_groups.append({"generation": grp.get("generation"), "solutions": copy.deepcopy(archive)})

    all_points = [pt for grp in frame_groups for pt in grp.get("solutions", [])]
    if not all_points:
        return False

    xs = [pt["T_hours"] for pt in all_points]
    ys = [pt["S_m2"] for pt in all_points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_pad = max(1.0, (x_max - x_min) * 0.08 if x_max > x_min else max(1.0, x_max * 0.08))
    y_pad = max(1.0, (y_max - y_min) * 0.08 if y_max > y_min else max(1.0, y_max * 0.08))

    frames = []
    total = len(frame_groups)
    for grp in frame_groups:
        fig, ax = plt.subplots(figsize=(8, 6))
        sols = sorted(grp.get("solutions", []), key=lambda s: (s["T_hours"], s["S_m2"]))
        if sols:
            gx = [s["T_hours"] for s in sols]
            gy = [s["S_m2"] for s in sols]
            ax.scatter(gx, gy, s=28, color="#2a6fdb")
            for s in sols:
                gen_label = s.get("generation")
                if gen_label is not None:
                    ax.annotate(
                        str(gen_label),
                        (s["T_hours"], s["S_m2"]),
                        textcoords="offset points",
                        xytext=(5, 4),
                        fontsize=8,
                        color="#333333",
                    )

        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_xlabel("T (hours)")
        ax.set_ylabel("S (m^2)")
        ax.set_title(f"Pareto Archive Evolution - Generation {grp['generation']}/{total}")
        ax.grid(alpha=0.3)
        plt.tight_layout()

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        frame = Image.frombytes("RGB", (w, h), fig.canvas.tostring_rgb())
        frames.append(frame.convert("P", palette=Image.ADAPTIVE))
        plt.close(fig)

    duration_ms = int(1000 / max(1, fps))
    frames[0].save(
        out_gif,
        save_all=True,
        append_images=frames[1:],
        optimize=True,
        duration=duration_ms,
        loop=0,
        disposal=2,
    )
    return True


def build_aircraft_timeline(schedule, move_records):
    by_aircraft = defaultdict(list)
    snapshots = schedule["takt_snapshots"]

    # 计算每个 takt 快照对应的总时间起点
    takt_to_start_time = {}
    cumulative_takt_time = 0
    for snap in snapshots:
        takt = snap["takt"]
        if takt not in takt_to_start_time:
            takt_to_start_time[takt] = cumulative_takt_time
        cumulative_takt_time += snap["takt_time"]

    for snap in snapshots:
        takt = snap["takt"]
        for rec in snap["positions"]:
            by_aircraft[rec["global_id"]].append(
                {
                    "kind": "assembly",
                    "takt": takt,
                    "total_time": takt_to_start_time.get(takt, 0.0),
                    "station": rec["station_name"],
                    "status": rec["status"],
                    "operation": rec["operation"],
                }
            )
    for rec in move_records:
        phase = rec[0]
        path = rec[1]
        ts = rec[2]
        gid = rec[3]
        kind = rec[4] if len(rec) > 4 else "main"
        from_station = rec[5] if len(rec) > 5 else None
        to_station = rec[6] if len(rec) > 6 else None
        by_aircraft[gid].append(
            {
                "kind": "move",
                "phase": phase,
                "path": path,
                "timestamps": ts,
                "move_type": kind,
                "from": from_station,
                "to": to_station,
            }
        )
    for gid in by_aircraft:
        by_aircraft[gid].sort(key=lambda x: (
            x.get("total_time") if x["kind"] == "assembly" else (x.get("timestamps", [10**9])[0] if x.get("timestamps") else 10**9),
            x.get("phase", 0),
        ))
    return by_aircraft


def main():
    from datetime import datetime
    start_time = datetime.now()
    print(f"[TIME] Start: {start_time.isoformat()}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["nsga2"], default="nsga2")
    parser.add_argument("--data", default="data.json")
    parser.add_argument(
        "--schedule",
        default=None,
        help="指定排产文件名或路径（可选）；未指定时使用data.json中的io.schedule_file",
    )
    parser.add_argument("--quick", action="store_true", help="快速调试模式")
    parser.add_argument(
        "--redraw-pareto-iter",
        action="store_true",
        help="仅根据已有result.json重绘pareto_iter.gif，不重新运行优化",
    )
    parser.add_argument(
        "--no-image",
        dest="save_image",
        action="store_false",
        help="Do not create image/ and do not save per-generation layout PNGs (saves time/disk).",
        default=False,
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Path to baseline_result.json. If given, evaluate its T/S and seed it into the initial population.",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=None,
        help="Override Stage_MODE: 0=baseline only, 1=stage1, 2=stage2, 3=both",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    cfg = load_json(base / args.data)
    random.seed(cfg["seed"])
    global MAX_GENERATIONS_LIMIT
    mg = cfg.get("max_generations")
    try:
        MAX_GENERATIONS_LIMIT = int(mg) if mg is not None else None
    except Exception:
        MAX_GENERATIONS_LIMIT = None

    schedule_ref = args.schedule if args.schedule else cfg["io"]["schedule_file"]
    schedule_path = Path(schedule_ref)
    if not schedule_path.is_absolute():
        schedule_path = (base / schedule_path).resolve()
    output_root = schedule_path.parent / schedule_path.stem
    output_root.mkdir(parents=True, exist_ok=True)
    schedule_copy_dst = output_root / schedule_path.name
    if schedule_copy_dst.resolve() != schedule_path.resolve():
        shutil.copy2(schedule_path, schedule_copy_dst)

    try:
        if args.redraw_pareto_iter:
            result_file = output_root / cfg["io"]["result_file"]
            if not result_file.exists():
                raise FileNotFoundError(f"未找到结果文件: {result_file}")
            result = load_json(result_file)
            pareto_groups = result.get("pareto_groups", [])
            ok = build_pareto_iter_gif(pareto_groups, output_root / "pareto_iter.gif", fps=2)
            if not ok:
                raise ValueError("result.json中缺少可用于绘制pareto_iter.gif的pareto_groups数据")
            print(f"完成: pareto_iter.gif (from {result_file.name})")
            return

    
        if args.quick:
            cfg["nsga2"]["generations"] = min(8, cfg["nsga2"]["generations"])
            cfg["nsga2"]["population"] = min(10, cfg["nsga2"]["population"])

        schedule = load_json(schedule_path)
        station_counts = schedule["station_counts"]

        # 从 schedule 推断完整 station_order；主装配区布局使用排除 209 系列和外部虚拟缓冲区后的子集
        full_station_order = build_station_order_from_schedule(schedule)
        main_station_order = [st for st in full_station_order
                              if not st.startswith("209") and st != "BUFFER_209_200"]
        cfg["station_order"] = main_station_order
        print(f"[INFO] 推断 station_order（主装配区）: {main_station_order}")

        # 使用 data.json 中配置的 grid_rows；过度压缩（max_stations+1）会让工位过于密集，
        # 路径规划大量失败并显著拖慢评估。
        if not cfg.get("grid_rows"):
            main_counts = {st: station_counts.get(st, 0) for st in main_station_order}
            max_stations = max(main_counts.values()) if main_counts else 0
            cfg["grid_rows"] = max_stations + 1

        # 如果有 baseline，先计算其 T/S 并构建 Individual
        baseline_ind = None
        baseline_obj = None
        baseline_payload = None
        if args.baseline:
            baseline_path = Path(args.baseline)
            if not baseline_path.is_absolute():
                baseline_path = base / baseline_path
            baseline_payload = evaluate_baseline(baseline_path, schedule_path, base / args.data)
            baseline_obj = baseline_payload["objectives"]
            baseline_ind = baseline_to_individual(baseline_path, cfg["station_order"], station_counts)
            print(f"[BASELINE] evaluated: T={baseline_obj[0]:.1f}h, S={baseline_obj[1]:.0f}m2")

        pareto_file = cfg["io"].get("pareto_file", "pareto.png")

        global Stage_MODE
        if args.stage is not None:
            Stage_MODE = args.stage

        # Stage_MODE=0: 仅评估 baseline，输出 result/png 后退出
        if Stage_MODE == 0:
            if baseline_payload is None:
                print("[ERROR] Stage_MODE=0 需要通过 --baseline 指定 baseline 布局文件")
                return
            # 使用 baseline 自身的 layout_shape 作为 grid_rows，确保 result 里 layout_shape 正确
            _bl_shape = _baseline_layout_shape(baseline_path)
            cfg["grid_rows"] = _bl_shape[0]
            baseline_result = build_result_payload(
                baseline_payload, cfg, schedule,
                pareto_payloads=[baseline_payload],
                pareto_groups=[],
                hv_history=[],
                best_hv_generation=None,
            )
            baseline_out_json = output_root / "baseline_result.json"
            with baseline_out_json.open("w", encoding="utf-8") as f:
                json.dump(baseline_result, f, ensure_ascii=False, indent=2)
            print(f"[BASELINE] saved result: {baseline_out_json}")
            try:
                from post_process import post_process as run_post_process, draw_post_layout
            except Exception:
                run_post_process = None
                draw_post_layout = None
            if run_post_process is not None:
                post_file = run_post_process(baseline_out_json)
                if draw_post_layout is not None and post_file is not None:
                    baseline_out_png = output_root / "baseline_layout.png"
                    draw_post_layout(post_file, baseline_out_png)
                    print(f"[BASELINE] saved plot: {baseline_out_png}")
            return

        # 根据Stage_MODE执行不同的优化策略
        if Stage_MODE == 1:
            # 只进行一阶段优化
            print("[INFO] 执行一阶段优化 (Stage_MODE=1)")
            best_payload, pareto_payloads, hv_history, best_hv_generation, pareto_groups, layouts_by_generation = run_nsga2(
                cfg, schedule, station_counts, baseline_ind=baseline_ind
            )
            draw_pareto_front(pareto_groups, output_root / pareto_file, algo_name="NSGA-II Stage1", baseline_obj=baseline_obj)
            draw_curve(hv_history, output_root / cfg["io"]["curve_file"], "NSGA-II Stage1")
            
            # 保存一阶段结果
            stage_result_files = []
            for i, payload in enumerate(pareto_payloads, start=1):
                suffix_i = "" if i == 1 else str(i)
                file_name = f"result_stage1_{suffix_i}.json"
                out_file = output_root / file_name
                stage_result_files.append(file_name)
            
        elif Stage_MODE == 2:
            # 只进行二阶段优化，需要从文件加载一阶段结果
            print("[INFO] 执行二阶段优化 (Stage_MODE=2)")
            # 查找一阶段结果文件
            stage1_files = list(output_root.glob("result_stage1_*.json"))
            if not stage1_files:
                raise FileNotFoundError("未找到一阶段结果文件 result_stage1_*.json")
            
            # 加载一阶段Pareto前沿解
            initial_individuals = []
            for stage1_file in sorted(stage1_files):
                stage1_result = load_json(stage1_file)
                # 从结果中重建Individual
                layout_vector = stage1_result.get("layout_vector", [])
                station_cols_data = stage1_result.get("station_cols", {})
                
                # 转换为Individual格式
                rows_by_station = {}
                station_cols = {}
                station_order = cfg["station_order"]
                
                for station_type in station_order:
                    rows_by_station[station_type] = []
                    station_cols[station_type] = station_cols_data.get(station_type, [])
                
                # 从layout_vector重建rows_by_station，使用实际列数而非配置列数
                m = cfg["grid_rows"]
                actual_n = stage1_result.get("layout_shape", [m, cfg["grid_cols"]])[1]
                for idx, val in enumerate(layout_vector):
                    if val != -1 and val != 0:
                        r = idx // actual_n
                        c = idx % actual_n
                        if isinstance(val, str) and '-' in val:
                            station_type = val.split('-')[0]
                            station_id = int(val.split('-')[1]) - 1
                            if station_type in rows_by_station:
                                while len(rows_by_station[station_type]) <= station_id:
                                    rows_by_station[station_type].append(0)
                                rows_by_station[station_type][station_id] = r
                
                individual = Individual(rows_by_station=rows_by_station, station_cols=station_cols)
                initial_individuals.append(individual)
            
            # 执行二阶段优化
            best_payload, pareto_payloads, hv_history, best_hv_generation, pareto_groups, layouts_by_generation, all_payloads = run_stage2_optimization(
                initial_individuals, cfg, schedule, station_counts
            )
            draw_pareto_front(pareto_groups, output_root / pareto_file, algo_name="Stage2 Compression")
            draw_curve(hv_history, output_root / cfg["io"]["curve_file"], "Stage2")
            
        else:  # Stage_MODE == 3
            # 两个阶段都进行
            print("[INFO] 执行两阶段优化 (Stage_MODE=3)")
            
            # 一阶段优化
            print("[Stage 1] 开始一阶段优化...")
            stage1_best, stage1_pareto, stage1_hv_history, stage1_best_hv_gen, stage1_pareto_groups, stage1_layouts = run_nsga2(
                cfg, schedule, station_counts
            )
            
            # 保存一阶段结果
            for i, payload in enumerate(stage1_pareto, start=1):
                suffix_i = "" if i == 1 else str(i)
                file_name = f"result_stage1_{suffix_i}.json"
                out_file = output_root / file_name
                with out_file.open("w", encoding="utf-8") as f:
                    # 创建临时结果结构
                    temp_result = {
                        "algorithm": "nsga2_stage1",
                        "objectives": {
                            "T_hours": payload["objectives"][0],
                            "S_m2": payload["objectives"][1],
                        },
                        "layout_vector": payload["layout_vector"],
                        "layout_shape": [cfg["grid_rows"], payload.get("actual_cols", cfg["grid_cols"])],
                        "station_cols": payload.get("station_cols", {}),
                        "station_positions": payload.get("station_positions", {}),
                    }
                    json.dump(temp_result, f, ensure_ascii=False, indent=2)
            
            # 从一阶段结果构建初始种群
            initial_individuals = []
            for payload in stage1_pareto:
                station_cols_data = payload.get("station_cols", {})
                rows_by_station = {}
                station_cols = {}
                
                for station_type in cfg["station_order"]:
                    rows_by_station[station_type] = []
                    station_cols[station_type] = station_cols_data.get(station_type, [])
                
                # 从payload重建Individual
                individual = Individual(rows_by_station=rows_by_station, station_cols=station_cols)
                initial_individuals.append(individual)
            
            # 二阶段优化
            print("[Stage 2] 开始二阶段优化...")
            best_payload, pareto_payloads, hv_history, best_hv_generation, pareto_groups, layouts_by_generation, _all_stage2 = run_stage2_optimization(
                initial_individuals, cfg, schedule, station_counts
            )
            
            # 合并两阶段的Pareto前沿
            all_pareto_payloads = stage1_pareto + pareto_payloads
            merged_pareto = filter_non_dominated_payloads(all_pareto_payloads)
            
            # 使用合并后的结果
            pareto_payloads = merged_pareto
            if pareto_payloads:
                best_payload = min(pareto_payloads, key=lambda p: (p["objectives"][0], p["objectives"][1]))
            else:
                best_payload = stage1_best
            
            draw_pareto_front(pareto_groups, output_root / pareto_file, algo_name="Two-Stage Optimization")
            draw_curve(hv_history, output_root / cfg["io"]["curve_file"], "Two-Stage")
        
        # 通用后处理
        has_process_gif = False
        if args.save_image:
            save_layouts_by_generation(cfg, layouts_by_generation, output_root / "image")
            has_process_gif = build_process_gif(output_root / "image", output_root / "process.gif", fps=2)
        layouts_by_generation.clear()
        has_pareto_iter_gif = build_pareto_iter_gif(pareto_groups, output_root / "pareto_iter.gif", fps=2)

        def payload_signature(payload):
            return (
                float(payload["objectives"][0]),
                float(payload["objectives"][1]),
                tuple(payload.get("layout_vector", [])),
            )

        def build_result(payload):
            return build_result_payload(payload, cfg, schedule, pareto_payloads, pareto_groups, hv_history, best_hv_generation)

        # Save baseline result.json and layout.png into the schedule output directory
        if baseline_payload is not None:
            baseline_result = build_result_payload(
                baseline_payload, cfg, schedule,
                pareto_payloads=[baseline_payload],
                pareto_groups=[],
                hv_history=[],
                best_hv_generation=None,
            )
            baseline_out_json = output_root / "baseline_result.json"
            with baseline_out_json.open("w", encoding="utf-8") as f:
                json.dump(baseline_result, f, ensure_ascii=False, indent=2)
            print(f"[BASELINE] saved result: {baseline_out_json}")
            try:
                from post_process import post_process as run_post_process, draw_post_layout
            except Exception:
                run_post_process = None
                draw_post_layout = None
            if run_post_process is not None:
                post_file = run_post_process(baseline_out_json)
                if draw_post_layout is not None and post_file is not None:
                    baseline_out_png = output_root / "baseline_layout.png"
                    draw_post_layout(post_file, baseline_out_png)
                    print(f"[BASELINE] saved plot: {baseline_out_png}")

        # Export all final Pareto solutions with stage-based naming
        export_payloads = [best_payload]
        seen = {payload_signature(best_payload)}
        if len(pareto_payloads) > 1:
            for p in pareto_payloads:
                sig = payload_signature(p)
                if sig in seen:
                    continue
                seen.add(sig)
                export_payloads.append(p)

        written_files = []
        written_plots = []
        
        # 根据Stage_MODE决定文件命名
        if Stage_MODE == 1:
            # 一阶段优化：保存为 result_stage1_1.json, result_stage1_2.json...
            for i, payload in enumerate(export_payloads, start=1):
                file_name = f"result_stage1_{i}.json"
                out_file = output_root / file_name
                with out_file.open("w", encoding="utf-8") as f:
                    json.dump(build_result(payload), f, ensure_ascii=False, indent=2)
                written_files.append(out_file.name)
                
                # Import post_process lazily to avoid requiring it on module import.
                try:
                    from post_process import post_process as run_post_process, draw_post_layout
                except Exception:
                    run_post_process = None
                    draw_post_layout = None

                post_file = None
                if run_post_process is not None:
                    post_file = run_post_process(out_file)
                plot_name = f"plot_stage1_{i}.png"
                # If draw_post_layout is available, generate plot; otherwise skip.
                if draw_post_layout is not None and post_file is not None:
                    draw_post_layout(post_file, output_root / plot_name)
                else:
                    # create an empty placeholder file to keep downstream expectations
                    try:
                        (output_root / plot_name).write_text("")
                    except Exception:
                        pass
                written_plots.append(plot_name)
        elif Stage_MODE == 2:
            # 二阶段优化：对每个一阶段结果一对一保存为 result_stage2_1.json, result_stage2_2.json...
            # 同时将最优结果写入 result_compression.json 供 animation.py 使用
            comp_file = output_root / "result_compression.json"
            with comp_file.open("w", encoding="utf-8") as f:
                json.dump(build_result(best_payload), f, ensure_ascii=False, indent=2)
            written_files.append(comp_file.name)
            for i, payload in enumerate(all_payloads, start=1):
                file_name = f"result_stage2_{i}.json"
                out_file = output_root / file_name
                with out_file.open("w", encoding="utf-8") as f:
                    json.dump(build_result(payload), f, ensure_ascii=False, indent=2)
                written_files.append(out_file.name)

                # Import post_process lazily to avoid requiring it on module import.
                try:
                    from post_process import post_process as run_post_process, draw_post_layout
                except Exception:
                    run_post_process = None
                    draw_post_layout = None

                post_file = None
                if run_post_process is not None:
                    post_file = run_post_process(out_file)
                plot_name = f"plot_stage2_{i}.png"
                # If draw_post_layout is available, generate plot; otherwise skip.
                if draw_post_layout is not None and post_file is not None:
                    draw_post_layout(post_file, output_root / plot_name)
                else:
                    # create an empty placeholder file to keep downstream expectations
                    try:
                        (output_root / plot_name).write_text("")
                    except Exception:
                        pass
                written_plots.append(plot_name)
        else:  # Stage_MODE == 3
            # 两阶段优化：保存合并后的Pareto前沿
            for i, payload in enumerate(export_payloads, start=1):
                file_name = f"result_stage2_{i}.json"
                out_file = output_root / file_name
                with out_file.open("w", encoding="utf-8") as f:
                    json.dump(build_result(payload), f, ensure_ascii=False, indent=2)
                written_files.append(out_file.name)
                
                # Import post_process lazily to avoid requiring it on module import.
                try:
                    from post_process import post_process as run_post_process, draw_post_layout
                except Exception:
                    run_post_process = None
                    draw_post_layout = None

                post_file = None
                if run_post_process is not None:
                    post_file = run_post_process(out_file)
                plot_name = f"plot_stage2_{i}.png"
                # If draw_post_layout is available, generate plot; otherwise skip.
                if draw_post_layout is not None and post_file is not None:
                    draw_post_layout(post_file, output_root / plot_name)
                else:
                    # create an empty placeholder file to keep downstream expectations
                    try:
                        (output_root / plot_name).write_text("")
                    except Exception:
                        pass
                written_plots.append(plot_name)
        
        
        if baseline_obj is not None:
            bt, bs = baseline_obj
            better = [p for p in pareto_payloads if p["objectives"][0] < bt and p["objectives"][1] < bs]
            print(f"[BASELINE] reference T={bt:.1f}h, S={bs:.0f}m2")
            if better:
                best_better = min(better, key=lambda p: (p["objectives"][0], p["objectives"][1]))
                print(f"[RESULT] found {len(better)} solutions strictly better than baseline; "
                      f"best: T={best_better['objectives'][0]:.1f}h, S={best_better['objectives'][1]:.0f}m2")
            else:
                best_t = min(pareto_payloads, key=lambda p: p["objectives"][0])
                best_s = min(pareto_payloads, key=lambda p: p["objectives"][1])
                print(f"[RESULT] no solution strictly better than baseline in both objectives")
                print(f"          best T: T={best_t['objectives'][0]:.1f}h, S={best_t['objectives'][1]:.0f}m2")
                print(f"          best S: T={best_s['objectives'][0]:.1f}h, S={best_s['objectives'][1]:.0f}m2")

        msg = (
            f"完成: 输出目录 {output_root.name}/ -> "
            f"{', '.join(written_files)}, {', '.join(written_plots)}, {cfg['io']['curve_file']}"
        )
        msg += f", {pareto_file}"
        msg += ", image/*.png"
        if has_process_gif:
            msg += ", process.gif"
        if has_pareto_iter_gif:
            msg += ", pareto_iter.gif"
        print(msg)
    finally:
        shutdown_executors()
        _EVAL_CACHE.clear()
        end_time = datetime.now()
        print(f"[TIME] End: {end_time.isoformat()}")


if __name__ == "__main__":
    main()

