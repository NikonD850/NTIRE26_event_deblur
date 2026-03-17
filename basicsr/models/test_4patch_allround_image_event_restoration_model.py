import math

import torch
import torch.nn.functional as F

from basicsr.models.test_4patch_image_event_restoration_model import (
    Test4PatchImageEventRestorationModel,
)


class Test4PatchAllRoundImageEventRestorationModel(Test4PatchImageEventRestorationModel):
    """8方向增强 + 4-patch 推理并反变换平均。"""

    def _all_round_four_patch_inference(self, lq, voxel, mask=None):
        trans_indices = (0, 5, 4, 1, 3, 6, 2, 7)
        pred_sum = None

        for trans_idx in trans_indices:
            lq_trans = self.transpose(lq, trans_idx)
            voxel_trans = self.transpose(voxel, trans_idx)
            mask_trans = self.transpose(mask, trans_idx) if mask is not None else None

            pred_trans = self._four_patch_inference(lq_trans, voxel_trans, mask_trans)
            pred_restore = self.transpose_inverse(pred_trans, trans_idx)

            if pred_sum is None:
                pred_sum = pred_restore
            else:
                pred_sum = pred_sum + pred_restore

        return pred_sum / float(len(trans_indices))

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            n = self.lq.size(0)
            outs = []
            m = self.opt["val"].get("max_minibatch", n)
            i = 0

            use_mask = False
            if "val" in self.opt["datasets"] and self.opt["datasets"]["val"].get(
                "use_mask"
            ):
                use_mask = True
            elif "test" in self.opt["datasets"] and self.opt["datasets"]["test"].get(
                "use_mask"
            ):
                use_mask = True

            while i < n:
                j = min(i + m, n)

                lq = self.lq[i:j]
                voxel = self.voxel[i:j]
                mask = self.mask[i:j] if use_mask and hasattr(self, "mask") else None

                _, _, height, width = lq.shape

                height_pad = math.ceil(height / 64) * 64
                width_pad = math.ceil(width / 64) * 64
                pad_h = height_pad - height
                pad_w = width_pad - width
                pad = (0, pad_w, 0, pad_h)

                lq_pad = F.pad(lq, pad, mode="reflect")
                voxel_pad = F.pad(voxel, pad, mode="reflect")
                mask_pad = F.pad(mask, pad, mode="reflect") if mask is not None else None

                pred = self._all_round_four_patch_inference(lq_pad, voxel_pad, mask_pad)
                pred = pred[:, :, :height, :width]

                outs.append(pred)
                i = j

            self.output = torch.cat(outs, dim=0)

        self.net_g.train()
