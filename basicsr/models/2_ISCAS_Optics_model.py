import math

import torch
import torch.nn.functional as F

from basicsr.models.image_event_restoration_model import ImageEventRestorationModel


class ISCASOpticsModel(ImageEventRestorationModel):
    """Single-file test-all-patch + geometric TTA model.

    This is a self-contained replacement for the composed test model stack:
    - TestAllPatchSizeImageEventRestorationModel
    - TestAllPatchSizeRotateFlipExtendedImageEventRestorationModel

    It keeps the original inference behavior:
    - symmetric reflect padding to the configured pad unit
    - sliding-window patch inference with overlap averaging
    - patch-level geometric TTA with inverse restoration and averaging
    """

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
                raise ValueError(
                    f"{name} must be a list/tuple with length 2, got: {value}"
                )
            h_val = int(value[0])
            w_val = int(value[1])
        else:
            h_val = int(value)
            w_val = int(value)

        if h_val <= 0 or w_val <= 0:
            raise ValueError(f"{name} must be positive, got: {value}")
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
            val_opt.get("four_patch_size"),
            "four_patch_size",
        )
        if patch_hw is not None:
            return patch_hw

        patch_h = val_opt.get("four_patch_h")
        patch_w = val_opt.get("four_patch_w")
        if patch_h is not None and patch_w is not None:
            return self._parse_hw_value(
                [patch_h, patch_w],
                "four_patch_h/four_patch_w",
            )

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
            raise ValueError(f"pad_unit must be positive, got: {pad_unit}")
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

    def _resolve_tta_indices(self):
        val_opt = self.opt.get("val", {})

        trans_indices = val_opt.get("tta_trans_indices")
        if trans_indices is not None:
            if not isinstance(trans_indices, (list, tuple)):
                raise TypeError(
                    "tta_trans_indices must be a list or tuple, "
                    f"got: {type(trans_indices).__name__}"
                )

            indices = tuple(int(trans_idx) for trans_idx in trans_indices)
            if not indices:
                raise ValueError("tta_trans_indices cannot be empty")

            for trans_idx in indices:
                if trans_idx < 0 or trans_idx > 7:
                    raise ValueError(
                        "tta_trans_indices values must be in [0, 7], "
                        f"got: {indices}"
                    )
            return indices

        tta_mode = str(val_opt.get("tta_mode", self.default_tta_mode)).lower()
        if tta_mode not in self._TTA_MODE_TO_INDICES:
            supported = ", ".join(sorted(self._TTA_MODE_TO_INDICES))
            raise ValueError(
                f"Unsupported tta_mode: {tta_mode}. Supported: {supported}"
            )
        return self._TTA_MODE_TO_INDICES[tta_mode]

    def _forward_single_patch(self, lq_patch, voxel_patch, mask_patch=None):
        if mask_patch is not None:
            pred_patch = self.net_g(x=lq_patch, event=voxel_patch, mask=mask_patch)
        else:
            pred_patch = self.net_g(x=lq_patch, event=voxel_patch)

        if isinstance(pred_patch, list):
            pred_patch = pred_patch[-1]
        return pred_patch

    def _forward_tta_patch(self, lq_patch, voxel_patch, mask_patch=None):
        trans_indices = self._resolve_tta_indices()
        if trans_indices == (0,):
            return self._forward_single_patch(lq_patch, voxel_patch, mask_patch)

        pred_sum = None
        for trans_idx in trans_indices:
            lq_trans = self.transpose(lq_patch, trans_idx)
            voxel_trans = self.transpose(voxel_patch, trans_idx)
            mask_trans = (
                self.transpose(mask_patch, trans_idx)
                if mask_patch is not None
                else None
            )

            pred_trans = self._forward_single_patch(
                lq_trans,
                voxel_trans,
                mask_trans,
            )
            pred_restore = self.transpose_inverse(pred_trans, trans_idx)

            if pred_sum is None:
                pred_sum = pred_restore
            else:
                pred_sum = pred_sum + pred_restore

        return pred_sum / float(len(trans_indices))

    def _forward_patch(self, lq_patch, voxel_patch, mask_patch=None):
        return self._forward_tta_patch(lq_patch, voxel_patch, mask_patch)

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
                        pred_patch.size(0),
                        pred_patch.size(1),
                        height,
                        width,
                    )
                    pred_weight = pred_patch.new_zeros(
                        pred_patch.size(0),
                        1,
                        height,
                        width,
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


class StandaloneISCASOpticsModel(ISCASOpticsModel):
    """Alias for the single-file patch + TTA model."""
