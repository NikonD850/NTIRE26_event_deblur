import importlib
import logging
import math
from collections import OrderedDict
from copy import deepcopy
from os import path as osp

import torch
import torch.nn.functional as F
from tqdm import tqdm

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_model_flops, get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import get_dist_info

loss_module = importlib.import_module("basicsr.models.losses")
metric_module = importlib.import_module("basicsr.metrics")
logger = logging.getLogger("basicsr")


class _ISCASImageEventRestorationModel(BaseModel):
    """Base Event-based deblur model for single image deblur."""

    def __init__(self, opt):
        super(_ISCASImageEventRestorationModel, self).__init__(opt)

        # define network
        self.net_g = define_network(deepcopy(opt["network_g"]))
        self.net_g = self.model_to_device(self.net_g)
        # parameters
        self.print_network(self.net_g)

        # flops
        if self.opt.get("print_flops", False):
            input_dim = self.opt.get(
                "flops_input_shape", [(3, 256, 256), (6, 256, 256)]
            )
            flops = get_model_flops(self.net_g, input_dim, False)
            flops = flops / 10**9
            logger.info("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

        # load pretrained models
        load_path = self.opt["path"].get("pretrain_network_g", None)
        if load_path is not None:
            self.load_network(
                self.net_g,
                load_path,
                self.opt["path"].get("strict_load_g", True),
                param_key=self.opt["path"].get("param_key", "params"),
            )

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt["train"]

        # define losses
        if train_opt.get("pixel_opt"):
            self.pixel_type = train_opt["pixel_opt"].pop("type")
            cri_pix_cls = getattr(loss_module, self.pixel_type)

            self.cri_pix = cri_pix_cls(**train_opt["pixel_opt"]).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get("perceptual_opt"):
            percep_type = train_opt["perceptual_opt"].pop("type")
            cri_perceptual_cls = getattr(loss_module, percep_type)
            self.cri_perceptual = cri_perceptual_cls(**train_opt["perceptual_opt"]).to(
                self.device
            )
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError("Both pixel and perceptual losses are None.")

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt["train"]
        optim_params = []
        optim_params_lowlr = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                if k.startswith("module.offsets") or k.startswith("module.dcns"):
                    optim_params_lowlr.append(v)
                else:
                    optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f"Params {k} will not be optimized.")
        ratio = 0.1

        optim_type = train_opt["optim_g"].pop("type")
        if optim_type == "Adam":
            self.optimizer_g = torch.optim.Adam(
                [
                    {"params": optim_params},
                    {
                        "params": optim_params_lowlr,
                        "lr": train_opt["optim_g"]["lr"] * ratio,
                    },
                ],
                **train_opt["optim_g"],
            )
        elif optim_type == "AdamW":
            self.optimizer_g = torch.optim.AdamW(
                [
                    {"params": optim_params},
                    {
                        "params": optim_params_lowlr,
                        "lr": train_opt["optim_g"]["lr"] * ratio,
                    },
                ],
                **train_opt["optim_g"],
            )

        else:
            raise NotImplementedError(f"optimizer {optim_type} is not supperted yet.")
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data["frame"].to(self.device)
        self.voxel = data["voxel"].to(self.device)
        if "mask" in data:
            self.mask = data["mask"].to(self.device)
        if "frame_gt" in data:
            self.gt = data["frame_gt"].to(self.device)
        if "image_name" in data:
            self.image_name = data["image_name"]

    def transpose(self, t, trans_idx):
        if trans_idx >= 4:
            t = torch.flip(t, [3])
        return torch.rot90(t, trans_idx % 4, [2, 3])

    def transpose_inverse(self, t, trans_idx):
        t = torch.rot90(t, 4 - trans_idx % 4, [2, 3])
        if trans_idx >= 4:
            t = torch.flip(t, [3])
        return t

    def grids_voxel(self):
        b, c, h, w = self.voxel.size()
        self.original_size_voxel = self.voxel.size()
        assert b == 1
        crop_size = self.opt["val"].get("crop_size")
        num_row = (h - 1) // crop_size + 1
        num_col = (w - 1) // crop_size + 1

        import math

        step_j = (
            crop_size
            if num_col == 1
            else math.ceil((w - crop_size) / (num_col - 1) - 1e-8)
        )
        step_i = (
            crop_size
            if num_row == 1
            else math.ceil((h - crop_size) / (num_row - 1) - 1e-8)
        )

        parts = []
        idxes = []

        i = 0
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size >= h:
                i = h - crop_size
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + crop_size >= w:
                    j = w - crop_size
                    last_j = True
                for trans_idx in range(self.opt["val"].get("trans_num", 1)):
                    parts.append(
                        self.transpose(
                            self.voxel[:, :, i : i + crop_size, j : j + crop_size],
                            trans_idx,
                        )
                    )
                    idxes.append({"i": i, "j": j, "trans_idx": trans_idx})
                j = j + step_j
            i = i + step_i
        if self.opt["val"].get("random_crop_num", 0) > 0:
            for _ in range(self.opt["val"].get("random_crop_num")):
                import random

                i = random.randint(0, h - crop_size)
                j = random.randint(0, w - crop_size)
                trans_idx = random.randint(0, self.opt["val"].get("trans_num", 1) - 1)
                parts.append(
                    self.transpose(
                        self.voxel[:, :, i : i + crop_size, j : j + crop_size],
                        trans_idx,
                    )
                )
                idxes.append({"i": i, "j": j, "trans_idx": trans_idx})

        self.origin_voxel = self.voxel
        self.voxel = torch.cat(parts, dim=0)
        print("----------parts voxel .. ", len(parts), self.voxel.size())
        self.idxes = idxes

    def grids(self):
        b, c, h, w = self.lq.size()
        self.original_size = self.lq.size()
        assert b == 1
        crop_size = self.opt["val"].get("crop_size")
        num_row = (h - 1) // crop_size + 1
        num_col = (w - 1) // crop_size + 1

        import math

        step_j = (
            crop_size
            if num_col == 1
            else math.ceil((w - crop_size) / (num_col - 1) - 1e-8)
        )
        step_i = (
            crop_size
            if num_row == 1
            else math.ceil((h - crop_size) / (num_row - 1) - 1e-8)
        )

        parts = []
        idxes = []

        i = 0
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size >= h:
                i = h - crop_size
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + crop_size >= w:
                    j = w - crop_size
                    last_j = True
                for trans_idx in range(self.opt["val"].get("trans_num", 1)):
                    parts.append(
                        self.transpose(
                            self.lq[:, :, i : i + crop_size, j : j + crop_size],
                            trans_idx,
                        )
                    )
                    idxes.append({"i": i, "j": j, "trans_idx": trans_idx})
                j = j + step_j
            i = i + step_i
        if self.opt["val"].get("random_crop_num", 0) > 0:
            for _ in range(self.opt["val"].get("random_crop_num")):
                import random

                i = random.randint(0, h - crop_size)
                j = random.randint(0, w - crop_size)
                trans_idx = random.randint(0, self.opt["val"].get("trans_num", 1) - 1)
                parts.append(
                    self.transpose(
                        self.lq[:, :, i : i + crop_size, j : j + crop_size], trans_idx
                    )
                )
                idxes.append({"i": i, "j": j, "trans_idx": trans_idx})

        self.origin_lq = self.lq
        self.lq = torch.cat(parts, dim=0)
        self.idxes = idxes

    def grids_inverse(self):
        preds = torch.zeros(self.original_size).to(self.device)
        b, c, h, w = self.original_size

        print("...", self.device)

        count_mt = torch.zeros((b, 1, h, w)).to(self.device)
        crop_size = self.opt["val"].get("crop_size")

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx["i"]
            j = each_idx["j"]
            trans_idx = each_idx["trans_idx"]
            preds[0, :, i : i + crop_size, j : j + crop_size] += self.transpose_inverse(
                self.output[cnt, :, :, :].unsqueeze(0), trans_idx
            ).squeeze(0)
            count_mt[0, 0, i : i + crop_size, j : j + crop_size] += 1.0

        self.output = preds / count_mt
        self.lq = self.origin_lq
        self.voxel = self.origin_voxel

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        if self.opt["datasets"]["train"].get("use_mask"):
            preds = self.net_g(x=self.lq, event=self.voxel, mask=self.mask)
        else:
            preds = self.net_g(x=self.lq, event=self.voxel)

        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1]

        l_total = 0
        loss_dict = OrderedDict()
        if self.cri_pix:
            l_pix = 0.0

            if self.pixel_type == "PSNRATLoss":
                l_pix += self.cri_pix(*preds, self.gt)

            elif self.pixel_type == "PSNRGateLoss":
                for pred in preds:
                    l_pix += self.cri_pix(pred, self.gt, self.mask)

            elif self.pixel_type == "PSNRLoss":
                for pred in preds:
                    l_pix += self.cri_pix(pred, self.gt)

            else:
                for pred in preds:
                    l_pix += self.cri_pix(pred, self.gt)

            l_total += l_pix
            loss_dict["l_pix"] = l_pix

        l_total = l_total + 0 * sum(p.sum() for p in self.net_g.parameters())

        l_total.backward()
        use_grad_clip = self.opt["train"].get("use_grad_clip", True)
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            n = self.lq.size(0)
            outs = []
            m = self.opt["val"].get("max_minibatch", n)
            i = 0

            while i < n:
                j = min(i + m, n)

                lq = self.lq[i:j]
                voxel = self.voxel[i:j]
                mask = self.mask[i:j] if hasattr(self, "mask") else None

                _, _, H, W = lq.shape

                H_pad = math.ceil(H / 64) * 64
                W_pad = math.ceil(W / 64) * 64

                pad_h = H_pad - H
                pad_w = W_pad - W
                pad = (0, pad_w, 0, pad_h)

                lq_pad = F.pad(lq, pad, mode="reflect")
                voxel_pad = F.pad(voxel, pad, mode="reflect")
                if mask is not None:
                    mask_pad = F.pad(mask, pad, mode="reflect")

                if "val" in self.opt["datasets"] and self.opt["datasets"]["val"].get(
                    "use_mask"
                ):
                    pred = self.net_g(x=lq_pad, event=voxel_pad, mask=mask_pad)
                elif "test" in self.opt["datasets"] and self.opt["datasets"][
                    "test"
                ].get("use_mask"):
                    pred = self.net_g(x=lq_pad, event=voxel_pad, mask=mask_pad)
                else:
                    pred = self.net_g(x=lq_pad, event=voxel_pad)

                if isinstance(pred, list):
                    pred = pred[-1]

                pred = pred[:, :, :H, :W]

                outs.append(pred)
                i = j

            self.output = torch.cat(outs, dim=0)

        self.net_g.train()

    def single_image_inference(self, img, voxel, save_path):
        self.feed_data(
            data={"frame": img.unsqueeze(dim=0), "voxel": voxel.unsqueeze(dim=0)}
        )
        if self.opt["val"].get("grids") is not None:
            self.grids()
            self.grids_voxel()

        self.test()

        if self.opt["val"].get("grids") is not None:
            self.grids_inverse()

        visuals = self.get_current_visuals()
        sr_img = tensor2img([visuals["result"]])
        imwrite(sr_img, save_path)

    def dist_validation(
        self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image
    ):
        dataset_name = dataloader.dataset.opt["name"]
        with_metrics = self.opt["val"].get("metrics") is not None
        if with_metrics:
            self.metric_results = {
                metric: 0 for metric in self.opt["val"]["metrics"].keys()
            }

        rank, world_size = get_dist_info()
        if rank == 0:
            pbar = tqdm(total=len(dataloader), unit="image")

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            if idx % world_size != rank:
                continue

            img_name = val_data["image_name"][0]
            print(img_name)

            self.feed_data(val_data)
            if self.opt["val"].get("grids", False):
                self.grids()

            self.test()

            if self.opt["val"].get("grids", False):
                self.grids_inverse()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals["result"]], rgb2bgr=rgb2bgr)
            if "gt" in visuals:
                gt_img = tensor2img([visuals["gt"]], rgb2bgr=rgb2bgr)
                del self.gt

            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if sr_img.shape[2] == 6:
                    L_img = sr_img[:, :, :3]
                    R_img = sr_img[:, :, 3:]

                    visual_dir = osp.join(
                        self.opt["path"]["visualization"], dataset_name
                    )

                    imwrite(L_img, osp.join(visual_dir, f"{img_name}_L.png"))
                    imwrite(R_img, osp.join(visual_dir, f"{img_name}_R.png"))
                else:
                    save_gt = self.opt["val"].get("save_gt", False)
                    if self.opt["is_train"]:

                        save_img_path = osp.join(
                            self.opt["path"]["visualization"],
                            img_name,
                            f"{img_name}_{current_iter}.png",
                        )

                        save_gt_img_path = osp.join(
                            self.opt["path"]["visualization"],
                            img_name,
                            f"{img_name}_{current_iter}_gt.png",
                        )
                    else:
                        save_img_path = osp.join(
                            self.opt["path"]["visualization"],
                            dataset_name,
                            f"{img_name}.png",
                        )
                        save_gt_img_path = osp.join(
                            self.opt["path"]["visualization"],
                            dataset_name,
                            f"{img_name}_gt.png",
                        )

                    imwrite(sr_img, save_img_path)
                    if save_gt:
                        imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                opt_metric = deepcopy(self.opt["val"]["metrics"])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop("type")
                        self.metric_results[name] += getattr(
                            metric_module, metric_type
                        )(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop("type")
                        self.metric_results[name] += getattr(
                            metric_module, metric_type
                        )(visuals["result"], visuals["gt"], **opt_)

            cnt += 1
            if rank == 0:
                for _ in range(world_size):
                    pbar.update(1)
                    pbar.set_description(f"Test {img_name}")
        if rank == 0:
            pbar.close()

        collected_metrics = OrderedDict()
        if with_metrics:
            for metric in self.metric_results.keys():
                collected_metrics[metric] = (
                    torch.tensor(self.metric_results[metric]).float().to(self.device)
                )
            collected_metrics["cnt"] = torch.tensor(cnt).float().to(self.device)

            self.collected_metrics = collected_metrics

        keys = []
        metrics = []
        for name, value in self.collected_metrics.items():
            keys.append(name)
            metrics.append(value)
        metrics = torch.stack(metrics, 0)
        torch.distributed.reduce(metrics, dst=0)
        if self.opt["rank"] == 0:
            metrics_dict = {}
            cnt = 0
            for key, metric in zip(keys, metrics):
                if key == "cnt":
                    cnt = float(metric)
                    continue
                metrics_dict[key] = float(metric)

            for key in metrics_dict:
                metrics_dict[key] /= cnt

            self._log_validation_metric_values(
                current_iter,
                dataloader.dataset.opt["name"],
                tb_logger,
                metrics_dict,
            )
        return 0.0

    def nondist_validation(self, *args, **kwargs):
        logger = get_root_logger()
        logger.warning("nondist_validation is not implemented. Run dist_validation.")
        self.dist_validation(*args, **kwargs)

    def _log_validation_metric_values(
        self, current_iter, dataset_name, tb_logger, metrics_to_log=None
    ):
        metric_values = self.metric_results if metrics_to_log is None else metrics_to_log
        log_str = f"Validation {dataset_name},\t"
        for metric, value in metric_values.items():
            log_str += f"\t # {metric}: {value:.4f}"
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in metric_values.items():
                tb_logger.add_scalar(f"metrics/{metric}", value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict["lq"] = self.lq.detach().cpu()
        out_dict["result"] = self.output.detach().cpu()
        if hasattr(self, "gt"):
            out_dict["gt"] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, "net_g", current_iter)
        self.save_training_state(epoch, current_iter)


class _ISCASRotateFlipTTAMixin:
    """Patch-level rotate/flip TTA helper."""

    default_tta_mode = "all8"

    _TTA_MODE_TO_INDICES = {
        "none": (0,),
        "identity": (0,),
        "flip2": (0, 4),
        "hflip2": (0, 4),
        "rot4": (0, 1, 2, 3),
        "rotate4": (0, 1, 2, 3),
        "all8": (0, 1, 2, 3, 4, 5, 6, 7),
        "all": (0, 1, 2, 3, 4, 5, 6, 7),
        "rotateflip": (0, 1, 2, 3, 4, 5, 6, 7),
        "x8": (0, 1, 2, 3, 4, 5, 6, 7),
    }

    def _resolve_tta_indices(self):
        val_opt = self.opt.get("val", {})

        trans_indices = val_opt.get("tta_trans_indices")
        if trans_indices is not None:
            if not isinstance(trans_indices, (list, tuple)):
                raise TypeError(
                    "tta_trans_indices 必须是列表或元组，"
                    f"当前类型: {type(trans_indices).__name__}"
                )

            indices = tuple(int(trans_idx) for trans_idx in trans_indices)
            if not indices:
                raise ValueError("tta_trans_indices 不能为空")

            for trans_idx in indices:
                if trans_idx < 0 or trans_idx > 7:
                    raise ValueError(
                        "tta_trans_indices 中的值必须在 [0, 7] 范围内，"
                        f"当前: {indices}"
                    )
            return indices

        tta_mode = str(val_opt.get("tta_mode", self.default_tta_mode)).lower()
        if tta_mode not in self._TTA_MODE_TO_INDICES:
            supported = ", ".join(sorted(self._TTA_MODE_TO_INDICES))
            raise ValueError(
                f"不支持的 tta_mode: {tta_mode}，支持的取值: {supported}"
            )
        return self._TTA_MODE_TO_INDICES[tta_mode]

    def _forward_tta_patch(self, forward_fn, lq_patch, voxel_patch, mask_patch=None):
        trans_indices = self._resolve_tta_indices()

        if trans_indices == (0,):
            return forward_fn(lq_patch, voxel_patch, mask_patch)

        pred_sum = None
        for trans_idx in trans_indices:
            lq_trans = self.transpose(lq_patch, trans_idx)
            voxel_trans = self.transpose(voxel_patch, trans_idx)
            mask_trans = (
                self.transpose(mask_patch, trans_idx)
                if mask_patch is not None
                else None
            )

            pred_trans = forward_fn(lq_trans, voxel_trans, mask_trans)
            pred_restore = self.transpose_inverse(pred_trans, trans_idx)

            if pred_sum is None:
                pred_sum = pred_restore
            else:
                pred_sum = pred_sum + pred_restore

        return pred_sum / float(len(trans_indices))


class _ISCASTestAllPatchSizeImageEventRestorationModel(
    _ISCASImageEventRestorationModel
):
    """支持任意 patch_size/stride 的测试模型。"""

    @staticmethod
    def _need_mask(datasets_opt):
        if "val" in datasets_opt and datasets_opt["val"].get("use_mask"):
            return True
        if "test" in datasets_opt and datasets_opt["test"].get("use_mask"):
            return True
        return False

    @staticmethod
    def _parse_hw_value(value, name):
        if value is None:
            return None

        if isinstance(value, (list, tuple)):
            if len(value) != 2:
                raise ValueError(f"{name} 必须是长度为2的列表/元组，当前: {value}")
            h_val = int(value[0])
            w_val = int(value[1])
        else:
            h_val = int(value)
            w_val = int(value)

        if h_val <= 0 or w_val <= 0:
            raise ValueError(f"{name} 必须为正数，当前: {value}")
        return h_val, w_val

    def _resolve_patch_hw(self, val_opt):
        patch_hw = self._parse_hw_value(val_opt.get("patch_size"), "patch_size")
        if patch_hw is not None:
            return patch_hw

        patch_h = val_opt.get("patch_h")
        patch_w = val_opt.get("patch_w")
        if patch_h is not None and patch_w is not None:
            return self._parse_hw_value([patch_h, patch_w], "patch_h/patch_w")

        patch_hw = self._parse_hw_value(
            val_opt.get("four_patch_size"), "four_patch_size"
        )
        if patch_hw is not None:
            return patch_hw

        patch_h = val_opt.get("four_patch_h")
        patch_w = val_opt.get("four_patch_w")
        if patch_h is not None and patch_w is not None:
            return self._parse_hw_value([patch_h, patch_w], "four_patch_h/four_patch_w")
        return None

    def _resolve_stride_hw(self, val_opt):
        stride_hw = self._parse_hw_value(val_opt.get("stride"), "stride")
        if stride_hw is not None:
            return stride_hw

        stride_h = val_opt.get("stride_h")
        stride_w = val_opt.get("stride_w")
        if stride_h is not None and stride_w is not None:
            return self._parse_hw_value([stride_h, stride_w], "stride_h/stride_w")

        stride_hw = self._parse_hw_value(
            val_opt.get("four_patch_stride"),
            "four_patch_stride",
        )
        if stride_hw is not None:
            return stride_hw

        stride_h = val_opt.get("four_patch_stride_h")
        stride_w = val_opt.get("four_patch_stride_w")
        if stride_h is not None and stride_w is not None:
            return self._parse_hw_value(
                [stride_h, stride_w],
                "four_patch_stride_h/four_patch_stride_w",
            )
        return None

    def _resolve_patch_and_stride(self, height, width):
        val_opt = self.opt.get("val", {})
        patch_hw = self._resolve_patch_hw(val_opt)
        if patch_hw is None:
            return None

        patch_h = max(1, min(int(patch_hw[0]), height))
        patch_w = max(1, min(int(patch_hw[1]), width))

        stride_hw = self._resolve_stride_hw(val_opt)
        if stride_hw is None:
            stride_h = patch_h
            stride_w = patch_w
        else:
            stride_h = max(1, int(stride_hw[0]))
            stride_w = max(1, int(stride_hw[1]))
        return patch_h, patch_w, stride_h, stride_w

    @staticmethod
    def _build_ranges_by_halves(length):
        mid = length // 2
        if mid == 0:
            return [(0, length)]
        return [(0, mid), (mid, length)]

    @staticmethod
    def _build_ranges_with_sliding(length, patch, stride):
        if patch >= length:
            return [(0, length)]

        starts = list(range(0, length - patch + 1, stride))
        last_start = length - patch
        if starts[-1] != last_start:
            starts.append(last_start)
        return [(start, min(start + patch, length)) for start in starts]

    @staticmethod
    def _get_pad_unit(val_opt):
        pad_unit = int(val_opt.get("pad_unit", 64))
        if pad_unit <= 0:
            raise ValueError(f"pad_unit 必须为正数，当前: {pad_unit}")
        return pad_unit

    @staticmethod
    def _calc_symmetric_pad(length, pad_unit):
        length_pad = math.ceil(length / pad_unit) * pad_unit
        pad_total = length_pad - length
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        return pad_before, pad_after

    @staticmethod
    def _remove_symmetric_pad(tensor, pad, original_h, original_w):
        left, right, top, bottom = pad
        h_start = top
        h_end = top + original_h
        w_start = left
        w_end = left + original_w
        return tensor[:, :, h_start:h_end, w_start:w_end]

    def _forward_patch(self, lq_patch, voxel_patch, mask_patch=None):
        if mask_patch is not None:
            pred_patch = self.net_g(x=lq_patch, event=voxel_patch, mask=mask_patch)
        else:
            pred_patch = self.net_g(x=lq_patch, event=voxel_patch)

        if isinstance(pred_patch, list):
            pred_patch = pred_patch[-1]
        return pred_patch

    def _all_patch_inference(self, lq, voxel, mask=None):
        _, _, height, width = lq.shape
        patch_stride = self._resolve_patch_and_stride(height, width)
        if patch_stride is None:
            h_ranges = self._build_ranges_by_halves(height)
            w_ranges = self._build_ranges_by_halves(width)
        else:
            patch_h, patch_w, stride_h, stride_w = patch_stride
            h_ranges = self._build_ranges_with_sliding(height, patch_h, stride_h)
            w_ranges = self._build_ranges_with_sliding(width, patch_w, stride_w)

        pred_sum = None
        pred_weight = None
        for h_start, h_end in h_ranges:
            for w_start, w_end in w_ranges:
                lq_patch = lq[:, :, h_start:h_end, w_start:w_end]
                voxel_patch = voxel[:, :, h_start:h_end, w_start:w_end]
                mask_patch = (
                    mask[:, :, h_start:h_end, w_start:w_end]
                    if mask is not None
                    else None
                )
                pred_patch = self._forward_patch(lq_patch, voxel_patch, mask_patch)

                if pred_sum is None:
                    pred_sum = pred_patch.new_zeros(
                        pred_patch.size(0), pred_patch.size(1), height, width
                    )
                    pred_weight = pred_patch.new_zeros(
                        pred_patch.size(0), 1, height, width
                    )

                pred_sum[:, :, h_start:h_end, w_start:w_end] += pred_patch
                pred_weight[:, :, h_start:h_end, w_start:w_end] += 1.0

        return pred_sum / pred_weight.clamp_min(1e-6)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            n = self.lq.size(0)
            outs = []
            val_opt = self.opt.get("val", {})
            max_minibatch = val_opt.get("max_minibatch", n)
            use_mask = self._need_mask(self.opt.get("datasets", {}))
            pad_unit = self._get_pad_unit(val_opt)

            i = 0
            while i < n:
                j = min(i + max_minibatch, n)

                lq = self.lq[i:j]
                voxel = self.voxel[i:j]
                mask = self.mask[i:j] if use_mask and hasattr(self, "mask") else None

                _, _, height, width = lq.shape
                pad_top, pad_bottom = self._calc_symmetric_pad(height, pad_unit)
                pad_left, pad_right = self._calc_symmetric_pad(width, pad_unit)
                pad = (pad_left, pad_right, pad_top, pad_bottom)

                lq_pad = F.pad(lq, pad, mode="reflect")
                voxel_pad = F.pad(voxel, pad, mode="reflect")
                mask_pad = F.pad(mask, pad, mode="reflect") if mask is not None else None

                pred = self._all_patch_inference(lq_pad, voxel_pad, mask_pad)
                pred = self._remove_symmetric_pad(pred, pad, height, width)

                outs.append(pred)
                i = j

            self.output = torch.cat(outs, dim=0)

        self.net_g.train()


class ISCASOpticsModel(
    _ISCASRotateFlipTTAMixin, _ISCASTestAllPatchSizeImageEventRestorationModel
):
    """基于 TestAllPatchSizeImageEventRestorationModel 的 patch-level 几何 TTA 模型。"""

    default_tta_mode = "rot4"
    _TTA_MODE_TO_INDICES = {
        "none": (0,),
        "identity": (0,),
        "rot2": (0, 1),
        "rot90pair": (0, 1),
        "input_rot90": (0, 1),
        "flip2": (0, 4),
        "hflip2": (0, 4),
        "input_flip": (0, 4),
        "rot4": (0, 1, 2, 3),
        "rotate4": (0, 1, 2, 3),
        "all8": (0, 1, 2, 3, 4, 5, 6, 7),
        "all": (0, 1, 2, 3, 4, 5, 6, 7),
        "rotateflip": (0, 1, 2, 3, 4, 5, 6, 7),
        "x8": (0, 1, 2, 3, 4, 5, 6, 7),
    }

    def _forward_patch(self, lq_patch, voxel_patch, mask_patch=None):
        return self._forward_tta_patch(
            super()._forward_patch,
            lq_patch,
            voxel_patch,
            mask_patch,
        )


class StandaloneISCASOpticsModel(ISCASOpticsModel):
    """Alias for the same single-file model."""
