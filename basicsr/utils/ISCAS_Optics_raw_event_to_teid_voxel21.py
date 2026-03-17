import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    import torch
except Exception:
    torch = None


def parse_event_piece(stem: str) -> Tuple[Optional[str], Optional[int]]:
    if '_' not in stem:
        return None, None
    base, piece = stem.rsplit('_', 1)
    if not piece.isdigit():
        return None, None
    return base, int(piece)


def build_event_groups(event_dir: Path) -> Dict[str, List[Tuple[int, Path]]]:
    groups: Dict[str, List[Tuple[int, Path]]] = {}
    for path in sorted(event_dir.glob('*.npz')):
        base, piece_id = parse_event_piece(path.stem)
        if base is None:
            continue
        groups.setdefault(base, []).append((piece_id, path))

    for base_name in groups:
        groups[base_name].sort(key=lambda item: item[0])
    return groups


def infer_hw_from_image(image_path: Path) -> Tuple[int, int]:
    with Image.open(image_path) as img:
        width, height = img.size
    return height, width


def get_reference_frames(dataroot: Path, ref_dir_name: Optional[str] = None) -> Tuple[List[Path], str]:
    if ref_dir_name is None:
        if (dataroot / 'sharp').is_dir():
            ref_dir_name = 'sharp'
        elif (dataroot / 'blur').is_dir():
            ref_dir_name = 'blur'
        else:
            raise FileNotFoundError('未找到 sharp 或 blur 目录，请显式传入 --ref-dir-name。')

    ref_dir = dataroot / ref_dir_name
    if not ref_dir.is_dir():
        raise FileNotFoundError(f'参考帧目录不存在: {ref_dir}')

    frames = sorted(ref_dir.glob('*.png'))
    if len(frames) == 0:
        raise FileNotFoundError(f'参考帧目录中没有 png 文件: {ref_dir}')

    return frames, ref_dir_name


def window_piece_ids(piece_ids: Sequence[int], radius: int, strict_window: bool = True) -> Optional[List[int]]:
    num_pieces = len(piece_ids)
    center = num_pieces // 2

    if strict_window:
        left = center - radius
        right = center + radius
        if left < 0 or right >= num_pieces:
            return None
    else:
        left = max(0, center - radius)
        right = min(num_pieces - 1, center + radius)

    return list(piece_ids[left:right + 1])


def load_event_piece(npz_path: str, swap_xy: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(npz_path)

    x = np.asarray(data['x'], dtype=np.float32).reshape(-1)
    y = np.asarray(data['y'], dtype=np.float32).reshape(-1)
    t = np.asarray(data['timestamp'], dtype=np.float32).reshape(-1)
    p = np.asarray(data['polarity'], dtype=np.float32).reshape(-1)

    if swap_xy:
        x, y = y, x

    return t, x, y, p


def concat_piece_events(
    piece_cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    paths: Sequence[str],
    sort_by_time: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(paths) == 0:
        empty = np.zeros((0,), dtype=np.float32)
        return empty, empty, empty, empty

    ts = [piece_cache[path][0] for path in paths]
    xs = [piece_cache[path][1] for path in paths]
    ys = [piece_cache[path][2] for path in paths]
    ps = [piece_cache[path][3] for path in paths]

    ts = np.concatenate(ts, axis=0)
    xs = np.concatenate(xs, axis=0)
    ys = np.concatenate(ys, axis=0)
    ps = np.concatenate(ps, axis=0)

    if sort_by_time and ts.size > 1:
        order = np.argsort(ts)
        ts = ts[order]
        xs = xs[order]
        ys = ys[order]
        ps = ps[order]

    return ts, xs, ys, ps


def events_to_voxel_numpy(
    ts: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    ps: np.ndarray,
    num_bins: int,
    width: int,
    height: int,
) -> np.ndarray:
    total_size = num_bins * height * width
    voxel_flat = np.zeros((total_size,), dtype=np.float32)

    if ts.size == 0:
        return voxel_flat.reshape(num_bins, height, width).transpose(1, 2, 0)

    first_stamp = float(ts[0])
    last_stamp = float(ts[-1])
    delta_t = last_stamp - first_stamp
    if delta_t <= 0:
        delta_t = 1.0

    ts_norm = (num_bins - 1) * (ts - first_stamp) / delta_t

    xs_int = xs.astype(np.int32)
    ys_int = ys.astype(np.int32)
    tis = ts_norm.astype(np.int32)
    dts = ts_norm - tis

    pols = np.where(ps > 0, 1.0, -1.0).astype(np.float32)
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    in_range_xy = (
        (xs_int >= 0) & (xs_int < width) &
        (ys_int >= 0) & (ys_int < height)
    )

    idx_left = xs_int + ys_int * width + tis * width * height
    valid_left = in_range_xy & (tis >= 0) & (tis < num_bins)
    if np.any(valid_left):
        voxel_flat += np.bincount(
            idx_left[valid_left],
            weights=vals_left[valid_left],
            minlength=total_size,
        ).astype(np.float32)

    idx_right = xs_int + ys_int * width + (tis + 1) * width * height
    valid_right = in_range_xy & ((tis + 1) >= 0) & ((tis + 1) < num_bins)
    if np.any(valid_right):
        voxel_flat += np.bincount(
            idx_right[valid_right],
            weights=vals_right[valid_right],
            minlength=total_size,
        ).astype(np.float32)

    return voxel_flat.reshape(num_bins, height, width).transpose(1, 2, 0)


def events_to_voxel_torch(
    ts: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    ps: np.ndarray,
    num_bins: int,
    width: int,
    height: int,
    device: 'torch.device',
) -> np.ndarray:
    total_size = num_bins * height * width

    if ts.size == 0:
        voxel = torch.zeros((num_bins, height, width), dtype=torch.float32, device=device)
        return voxel.permute(1, 2, 0).cpu().numpy()

    t_t = torch.from_numpy(ts).to(device=device, dtype=torch.float32, non_blocking=True)
    x_t = torch.from_numpy(xs).to(device=device, dtype=torch.float32, non_blocking=True)
    y_t = torch.from_numpy(ys).to(device=device, dtype=torch.float32, non_blocking=True)
    p_t = torch.from_numpy(ps).to(device=device, dtype=torch.float32, non_blocking=True)

    first_stamp = t_t[0]
    last_stamp = t_t[-1]
    delta_t = last_stamp - first_stamp
    if torch.abs(delta_t).item() <= 0:
        delta_t = torch.tensor(1.0, device=device)

    ts_norm = (num_bins - 1) * (t_t - first_stamp) / delta_t

    xs_int = x_t.long()
    ys_int = y_t.long()
    tis = torch.floor(ts_norm).long()
    dts = ts_norm - tis.float()

    pols = torch.where(p_t > 0, torch.ones_like(p_t), -torch.ones_like(p_t))
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    in_range_xy = (
        (xs_int >= 0) & (xs_int < width) &
        (ys_int >= 0) & (ys_int < height)
    )

    voxel_flat = torch.zeros((total_size,), dtype=torch.float32, device=device)

    idx_left = xs_int + ys_int * width + tis * width * height
    valid_left = in_range_xy & (tis >= 0) & (tis < num_bins)
    if torch.any(valid_left):
        voxel_flat.index_add_(0, idx_left[valid_left], vals_left[valid_left])

    idx_right = xs_int + ys_int * width + (tis + 1) * width * height
    valid_right = in_range_xy & ((tis + 1) >= 0) & ((tis + 1) < num_bins)
    if torch.any(valid_right):
        voxel_flat.index_add_(0, idx_right[valid_right], vals_right[valid_right])

    voxel = voxel_flat.view(num_bins, height, width)
    return voxel.permute(1, 2, 0).contiguous().cpu().numpy()


def save_voxel(
    out_path: str,
    voxel_21: np.ndarray,
    voxel_l: np.ndarray,
    voxel_m: np.ndarray,
    voxel_s: np.ndarray,
    compressed: bool,
    save_separate: bool,
) -> None:
    if compressed:
        if save_separate:
            np.savez_compressed(
                out_path,
                voxel=voxel_21,
                voxel_long=voxel_l,
                voxel_medium=voxel_m,
                voxel_short=voxel_s,
            )
        else:
            np.savez_compressed(out_path, voxel=voxel_21)
    else:
        if save_separate:
            np.savez(
                out_path,
                voxel=voxel_21,
                voxel_long=voxel_l,
                voxel_medium=voxel_m,
                voxel_short=voxel_s,
            )
        else:
            np.savez(out_path, voxel=voxel_21)


def make_task(
    base_name: str,
    pieces: List[Tuple[int, Path]],
    out_path: Path,
    tl: int,
    tm: int,
    ts: int,
    strict_window: bool,
) -> Tuple[str, Optional[dict]]:
    piece_ids = [piece_id for piece_id, _ in pieces]
    piece_path_map = {piece_id: str(path) for piece_id, path in pieces}

    ids_l = window_piece_ids(piece_ids, tl, strict_window=strict_window)
    ids_m = window_piece_ids(piece_ids, tm, strict_window=strict_window)
    ids_s = window_piece_ids(piece_ids, ts, strict_window=strict_window)

    if ids_l is None or ids_m is None or ids_s is None:
        return 'insufficient_window', None

    task = {
        'base_name': base_name,
        'out_path': str(out_path),
        'paths_l': [piece_path_map[piece_id] for piece_id in ids_l],
        'paths_m': [piece_path_map[piece_id] for piece_id in ids_m],
        'paths_s': [piece_path_map[piece_id] for piece_id in ids_s],
    }
    return 'ok', task


def process_task_cpu(task: dict, cfg: dict) -> Tuple[str, str]:
    piece_cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    unique_paths = sorted(set(task['paths_l'] + task['paths_m'] + task['paths_s']))

    for path in unique_paths:
        piece_cache[path] = load_event_piece(path, swap_xy=cfg['swap_xy'])

    t_l, x_l, y_l, p_l = concat_piece_events(piece_cache, task['paths_l'], sort_by_time=cfg['sort_by_time'])
    t_m, x_m, y_m, p_m = concat_piece_events(piece_cache, task['paths_m'], sort_by_time=cfg['sort_by_time'])
    t_s, x_s, y_s, p_s = concat_piece_events(piece_cache, task['paths_s'], sort_by_time=cfg['sort_by_time'])

    voxel_l = events_to_voxel_numpy(t_l, x_l, y_l, p_l, cfg['bins'], cfg['width'], cfg['height'])
    voxel_m = events_to_voxel_numpy(t_m, x_m, y_m, p_m, cfg['bins'], cfg['width'], cfg['height'])
    voxel_s = events_to_voxel_numpy(t_s, x_s, y_s, p_s, cfg['bins'], cfg['width'], cfg['height'])

    voxel_21 = np.concatenate([voxel_l, voxel_m, voxel_s], axis=2).astype(np.float32)
    save_voxel(task['out_path'], voxel_21, voxel_l, voxel_m, voxel_s, cfg['compressed'], cfg['save_separate'])
    return task['base_name'], 'ok'


def process_tasks_gpu(tasks: List[dict], cfg: dict, gpu_id: int) -> Dict[str, int]:
    if torch is None:
        raise RuntimeError('GPU 后端需要 PyTorch，但当前环境无法导入 torch。')
    if not torch.cuda.is_available():
        raise RuntimeError('GPU 后端需要 CUDA，但当前环境中 CUDA 不可用。')

    device = torch.device(f'cuda:{gpu_id}')
    stats = {'ok': 0}

    torch.cuda.set_device(device)

    for idx, task in enumerate(tasks, start=1):
        piece_cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        unique_paths = sorted(set(task['paths_l'] + task['paths_m'] + task['paths_s']))

        for path in unique_paths:
            piece_cache[path] = load_event_piece(path, swap_xy=cfg['swap_xy'])

        t_l, x_l, y_l, p_l = concat_piece_events(piece_cache, task['paths_l'], sort_by_time=cfg['sort_by_time'])
        t_m, x_m, y_m, p_m = concat_piece_events(piece_cache, task['paths_m'], sort_by_time=cfg['sort_by_time'])
        t_s, x_s, y_s, p_s = concat_piece_events(piece_cache, task['paths_s'], sort_by_time=cfg['sort_by_time'])

        voxel_l = events_to_voxel_torch(t_l, x_l, y_l, p_l, cfg['bins'], cfg['width'], cfg['height'], device)
        voxel_m = events_to_voxel_torch(t_m, x_m, y_m, p_m, cfg['bins'], cfg['width'], cfg['height'], device)
        voxel_s = events_to_voxel_torch(t_s, x_s, y_s, p_s, cfg['bins'], cfg['width'], cfg['height'], device)

        voxel_21 = np.concatenate([voxel_l, voxel_m, voxel_s], axis=2).astype(np.float32)
        save_voxel(task['out_path'], voxel_21, voxel_l, voxel_m, voxel_s, cfg['compressed'], cfg['save_separate'])
        stats['ok'] += 1

        if idx % 50 == 0 or idx == len(tasks):
            print(f'[PROGRESS][GPU] {idx}/{len(tasks)} | ok={stats["ok"]}')

    return stats


_GLOBAL_CFG = None


def _init_worker(cfg: dict):
    global _GLOBAL_CFG
    _GLOBAL_CFG = cfg


def _worker_entry(task: dict) -> Tuple[str, str]:
    return process_task_cpu(task, _GLOBAL_CFG)


def resolve_backend(requested: str) -> str:
    if requested in ('cpu', 'gpu'):
        return requested
    if requested == 'auto':
        if torch is not None and torch.cuda.is_available():
            return 'gpu'
        return 'cpu'
    raise ValueError(f'不支持的 backend: {requested}')


def resolve_num_workers(num_workers: int) -> int:
    if num_workers <= 0:
        return max(1, os.cpu_count() or 1)
    return num_workers


def parse_args():
    parser = argparse.ArgumentParser(description='高性能: raw event -> TEID风格 21通道 voxel')
    parser.add_argument('--dataroot', type=str, required=True, help='数据集根目录，需包含 event 与 sharp/blur。')
    parser.add_argument('--event-dir-name', type=str, default='event', help='event 目录名。')
    parser.add_argument('--ref-dir-name', type=str, default=None, help='参考帧目录名，默认 sharp，其次 blur。')
    parser.add_argument('--output-root', type=str, default='/tmp', help='输出根目录，默认 /tmp。')
    parser.add_argument('--output-dir-name', type=str, default='voxel21_teid_b7_tl5_tm1_ts0', help='输出目录名。')

    parser.add_argument('--bins', type=int, default=7, help='每个时间尺度的 bin 数。')
    parser.add_argument('--tl', type=int, default=5, help='long-term 窗口半径。')
    parser.add_argument('--tm', type=int, default=1, help='medium-term 窗口半径。')
    parser.add_argument('--ts', type=int, default=0, help='short-term 窗口半径。')

    parser.add_argument('--height', type=int, default=None, help='输出高度，默认由参考帧推断。')
    parser.add_argument('--width', type=int, default=None, help='输出宽度，默认由参考帧推断。')

    parser.add_argument('--backend', type=str, default='auto', choices=['auto', 'cpu', 'gpu'], help='计算后端。')
    parser.add_argument('--num-workers', type=int, default=0, help='CPU后端并行进程数，<=0 表示使用全部CPU。')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU 后端时使用的 CUDA 设备号。')

    parser.add_argument('--max-samples', type=int, default=None, help='仅处理前 N 帧，便于快速验证。')
    parser.add_argument('--overwrite', action='store_true', help='输出已存在时覆盖。')
    parser.add_argument('--strict-window', action=argparse.BooleanOptionalAction, default=True,
                        help='是否要求窗口严格对称可用。默认启用。')
    parser.add_argument('--swap-xy', action=argparse.BooleanOptionalAction, default=True,
                        help='是否交换 x/y（HighREV 历史脚本默认交换）。')
    parser.add_argument('--sort-by-time', action=argparse.BooleanOptionalAction, default=False,
                        help='拼接后是否按 timestamp 全局排序。')
    parser.add_argument('--compressed', action=argparse.BooleanOptionalAction, default=False,
                        help='是否使用 np.savez_compressed（会更慢但更省磁盘）。')
    parser.add_argument('--save-separate', action='store_true',
                        help='额外保存 voxel_long/voxel_medium/voxel_short。')
    return parser.parse_args()


def main():
    args = parse_args()

    dataroot = Path(args.dataroot)
    event_dir = dataroot / args.event_dir_name
    if not event_dir.is_dir():
        raise FileNotFoundError(f'event 目录不存在: {event_dir}')

    ref_frames, chosen_ref_dir = get_reference_frames(dataroot, args.ref_dir_name)
    if args.max_samples is not None:
        ref_frames = ref_frames[:args.max_samples]

    output_dir = Path(args.output_root) / args.output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.height is None or args.width is None:
        h_ref, w_ref = infer_hw_from_image(ref_frames[0])
        height = h_ref if args.height is None else args.height
        width = w_ref if args.width is None else args.width
    else:
        height = args.height
        width = args.width

    if args.bins <= 0:
        raise ValueError('--bins 必须 > 0')
    if args.tl < 0 or args.tm < 0 or args.ts < 0:
        raise ValueError('tl/tm/ts 必须 >= 0')

    backend = resolve_backend(args.backend)
    num_workers = resolve_num_workers(args.num_workers)

    event_groups = build_event_groups(event_dir)

    stats = {
        'ok': 0,
        'skip_exists': 0,
        'missing_event_group': 0,
        'insufficient_window': 0,
    }
    tasks: List[dict] = []

    for frame_path in ref_frames:
        base_name = frame_path.stem
        out_path = output_dir / f'{base_name}.npz'

        if out_path.exists() and not args.overwrite:
            stats['skip_exists'] += 1
            continue

        if base_name not in event_groups:
            stats['missing_event_group'] += 1
            continue

        status, task = make_task(
            base_name=base_name,
            pieces=event_groups[base_name],
            out_path=out_path,
            tl=args.tl,
            tm=args.tm,
            ts=args.ts,
            strict_window=args.strict_window,
        )
        if status != 'ok' or task is None:
            stats['insufficient_window'] += 1
            continue

        tasks.append(task)

    cfg = {
        'bins': args.bins,
        'width': width,
        'height': height,
        'swap_xy': args.swap_xy,
        'sort_by_time': args.sort_by_time,
        'compressed': args.compressed,
        'save_separate': args.save_separate,
    }

    start_time = time.time()

    print(f'[INFO] dataroot: {dataroot}')
    print(f'[INFO] event_dir: {event_dir}')
    print(f'[INFO] ref_dir: {chosen_ref_dir}, frames={len(ref_frames)}')
    print(f'[INFO] output_dir: {output_dir}')
    print(f'[INFO] backend: {backend}')
    print(f'[INFO] windows: Tl={args.tl}, Tm={args.tm}, Ts={args.ts}')
    print(f'[INFO] voxel shape: H={height}, W={width}, C={args.bins * 3}')
    print(f'[INFO] valid tasks: {len(tasks)}')

    if backend == 'gpu':
        print(f'[INFO] gpu_id: {args.gpu_id}')
        gpu_stats = process_tasks_gpu(tasks, cfg, args.gpu_id)
        stats['ok'] += gpu_stats.get('ok', 0)
    else:
        print(f'[INFO] cpu workers: {num_workers}')
        if len(tasks) > 0:
            with ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=_init_worker,
                initargs=(cfg,),
            ) as executor:
                futures = [executor.submit(_worker_entry, task) for task in tasks]
                for idx, future in enumerate(as_completed(futures), start=1):
                    _, status = future.result()
                    if status in stats:
                        stats[status] += 1
                    else:
                        stats[status] = stats.get(status, 0) + 1

                    if idx % 100 == 0 or idx == len(futures):
                        print(
                            f'[PROGRESS][CPU] {idx}/{len(futures)} | '
                            f'ok={stats.get("ok", 0)} | '
                            f'skip={stats.get("skip_exists", 0)} | '
                            f'missing={stats.get("missing_event_group", 0)} | '
                            f'insufficient={stats.get("insufficient_window", 0)}'
                        )

    elapsed = time.time() - start_time

    print('[DONE]')
    print(f'[DONE] stats: {stats}')
    print(f'[DONE] elapsed: {elapsed:.2f}s')


if __name__ == '__main__':
    main()
