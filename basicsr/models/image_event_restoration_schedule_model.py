import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import logging

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img, get_model_flops
from basicsr.utils.dist_util import get_dist_info
import math
import torch.nn.functional as F

loss_module = importlib.import_module("basicsr.models.losses")
metric_module = importlib.import_module("basicsr.metrics")
logger = logging.getLogger("basicsr")


class ImageEventRestorationScheduleModel(BaseModel):
    """Event-based deblur model with loss schedule rotation."""

    def __init__(self, opt):
        super(ImageEventRestorationScheduleModel, self).__init__(opt)

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
        self.use_loss_schedule = False
        self.loss_schedule = []
        self.loss_schedule_cum_iters = []
        self.loss_schedule_cycle = False
        self.active_loss_segment_idx = None

        if train_opt.get("loss_schedule"):
            self.use_loss_schedule = True
            self.loss_schedule_cycle = bool(train_opt.get("loss_schedule_cycle", False))
            cum_iters = 0
            for seg in train_opt["loss_schedule"]:
                seg_pix = seg.get("pixel_opt")
                if seg_pix is None:
                    raise ValueError("Each loss_schedule segment must define pixel_opt.")
                seg_pix = deepcopy(seg_pix)
                pixel_type = seg_pix.pop("type")

                seg_percep = seg.get("perceptual_opt")
                percep_type = None
                if seg_percep is not None:
                    seg_percep = deepcopy(seg_percep)
                    percep_type = seg_percep.pop("type")

                iters = int(seg.get("iters", 0))
                if iters <= 0:
                    raise ValueError("Each loss_schedule segment must set positive iters.")
                cum_iters += iters
                self.loss_schedule.append(
                    {
                        "iters": iters,
                        "pixel_type": pixel_type,
                        "pixel_opt": seg_pix,
                        "percep_type": percep_type,
                        "perceptual_opt": seg_percep,
                    }
                )
                self.loss_schedule_cum_iters.append(cum_iters)

            if not self.loss_schedule:
                raise ValueError("loss_schedule is enabled but no valid segment is found.")

            self.cri_pix = None
            self.cri_perceptual = None
            self.pixel_type = None
            self._activate_loss_segment(0)
        else:
            if train_opt.get("pixel_opt"):
                pixel_opt = deepcopy(train_opt["pixel_opt"])
                self.pixel_type = pixel_opt.pop("type")
                cri_pix_cls = getattr(loss_module, self.pixel_type)
                self.cri_pix = cri_pix_cls(**pixel_opt).to(self.device)
            else:
                self.cri_pix = None

            if train_opt.get("perceptual_opt"):
                perceptual_opt = deepcopy(train_opt["perceptual_opt"])
                percep_type = perceptual_opt.pop("type")
                cri_perceptual_cls = getattr(loss_module, percep_type)
                self.cri_perceptual = cri_perceptual_cls(**perceptual_opt).to(
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
        # print(optim_params)
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
        # print('transpose jt .. ', t.size())
        if trans_idx >= 4:
            t = torch.flip(t, [3])
        return torch.rot90(t, trans_idx % 4, [2, 3])

    def transpose_inverse(self, t, trans_idx):
        # print( 'inverse transpose .. t', t.size())
        t = torch.rot90(t, 4 - trans_idx % 4, [2, 3])
        if trans_idx >= 4:
            t = torch.flip(t, [3])
        return t

    def grids_voxel(self):
        b, c, h, w = self.voxel.size()
        self.original_size_voxel = self.voxel.size()
        assert b == 1
        crop_size = self.opt["val"].get("crop_size")
        # step_j = self.opt['val'].get('step_j', crop_size)
        # step_i = self.opt['val'].get('step_i', crop_size)
        ##adaptive step_i, step_j
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

        # print('step_i, stepj', step_i, step_j)
        # exit(0)

        parts = []
        idxes = []

        # cnt_idx = 0

        i = 0  # 0~h-1
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
                # from i, j to i+crop_szie, j + crop_size
                # print(' trans 8')
                for trans_idx in range(self.opt["val"].get("trans_num", 1)):
                    parts.append(
                        self.transpose(
                            self.voxel[:, :, i : i + crop_size, j : j + crop_size],
                            trans_idx,
                        )
                    )
                    idxes.append({"i": i, "j": j, "trans_idx": trans_idx})
                    # cnt_idx += 1
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
        b, c, h, w = (
            self.lq.size()
        )  # lq is after data augment (for example, crop, if have)
        self.original_size = self.lq.size()
        assert b == 1
        crop_size = self.opt["val"].get("crop_size")
        # step_j = self.opt['val'].get('step_j', crop_size)
        # step_i = self.opt['val'].get('step_i', crop_size)
        ##adaptive step_i, step_j
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

        # print('step_i, stepj', step_i, step_j)
        # exit(0)

        parts = []
        idxes = []

        # cnt_idx = 0

        i = 0  # 0~h-1
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
                # from i, j to i+crop_szie, j + crop_size
                # print(' trans 8')
                for trans_idx in range(self.opt["val"].get("trans_num", 1)):
                    parts.append(
                        self.transpose(
                            self.lq[:, :, i : i + crop_size, j : j + crop_size],
                            trans_idx,
                        )
                    )
                    idxes.append({"i": i, "j": j, "trans_idx": trans_idx})
                    # cnt_idx += 1
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
        # print('parts .. ', len(parts), self.lq.size())
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

        if self.use_loss_schedule:
            seg_idx = self._select_loss_segment_idx(current_iter)
            self._activate_loss_segment(seg_idx)

        # pixel loss
        cri_pix = self.cri_pix
        pixel_type = self.pixel_type

        if cri_pix:
            l_pix = 0.0

            if pixel_type == "PSNRATLoss":
                l_pix += cri_pix(*preds, self.gt)
            elif pixel_type == "PSNRGateLoss":
                for pred in preds:
                    l_pix += cri_pix(pred, self.gt, self.mask)
            elif pixel_type == "PSNRLoss":
                for pred in preds:
                    l_pix += cri_pix(pred, self.gt)
            else:
                for pred in preds:
                    l_pix += cri_pix(pred, self.gt)

            l_total += l_pix
            loss_dict["l_pix"] = l_pix
        # perceptual loss
        # if self.cri_perceptual:
        #
        #
        #     l_percep, l_style = self.cri_perceptual(self.output, self.gt)
        #
        #     if l_percep is not None:
        #         l_total += l_percep
        #         loss_dict['l_percep'] = l_percep
        #     if l_style is not None:
        #         l_total += l_style
        #         loss_dict['l_style'] = l_style

        l_total = l_total + 0 * sum(p.sum() for p in self.net_g.parameters())

        l_total.backward()
        use_grad_clip = self.opt["train"].get("use_grad_clip", True)
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

    # def test(self):
    #     self.net_g.eval()
    #     with torch.no_grad():
    #         n = self.lq.size(0)  # n: batch size
    #         outs = []
    #         m = self.opt['val'].get('max_minibatch', n)  # m is the minibatch, equals to batch size or mini batch size
    #         i = 0
    #         while i < n:
    #             j = i + m
    #             if j >= n:
    #                 j = n

    #             if 'val' in self.opt['datasets'] and self.opt['datasets']['val'].get('use_mask'):
    #                 pred = self.net_g(x = self.lq[i:j, :, :, :], event = self.voxel[i:j, :, :, :], mask = self.mask[i:j, :, :, :])  # mini batch all in
    #             elif 'test' in self.opt['datasets'] and self.opt['datasets']['test'].get('use_mask'):
    #                 pred = self.net_g(x = self.lq[i:j, :, :, :], event = self.voxel[i:j, :, :, :], mask = self.mask[i:j, :, :, :])  # mini batch all in
    #             else:
    #                 pred = self.net_g(x = self.lq[i:j, :, :, :], event = self.voxel[i:j, :, :, :])  # mini batch all in

    #             if isinstance(pred, list):
    #                 pred = pred[-1]
    #             outs.append(pred)
    #             i = j

    #         self.output = torch.cat(outs, dim=0)  # all mini batch cat in dim0
    #     self.net_g.train()
    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            n = self.lq.size(0)  # batch size
            outs = []
            m = self.opt["val"].get("max_minibatch", n)
            i = 0

            while i < n:
                j = min(i + m, n)

                lq = self.lq[i:j]
                voxel = self.voxel[i:j]
                mask = self.mask[i:j] if hasattr(self, "mask") else None

                _, _, H, W = lq.shape

                # -------- pad to 64 multiple --------
                H_pad = math.ceil(H / 64) * 64
                W_pad = math.ceil(W / 64) * 64

                pad_h = H_pad - H
                pad_w = W_pad - W

                # pad format: (left, right, top, bottom)
                pad = (0, pad_w, 0, pad_h)

                lq_pad = F.pad(lq, pad, mode="reflect")
                voxel_pad = F.pad(voxel, pad, mode="reflect")
                if mask is not None:
                    mask_pad = F.pad(mask, pad, mode="reflect")

                # -------- inference --------
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

                # -------- remove pad --------
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
            # self.grids_inverse_voxel()

        visuals = self.get_current_visuals()
        sr_img = tensor2img([visuals["result"]])
        imwrite(sr_img, save_path)

    # def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
    #     logger = get_root_logger()
    #     # logger.info('Only support single GPU validation.')
    #     #import os
    #     #if os.environ['LOCAL_RANK'] == '0':
    #     #    return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
    #     #else:
    #     #    return 0.
    #     import torch.distributed as dist
    #     rank = dist.get_rank()
    #     if rank == 0:
    #         metric = self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
    #     else:
    #         metric = None
    #     dist.barrier()
    #     return metric

    # def dist_validation(self, dataloader, current_iter, tb_logger,
    #                save_img, rgb2bgr, use_image):
    #
    #        import torch.distributed as dist
    #        rank = dist.get_rank()
    #        world_size = dist.get_world_size()
    #
    #        dataset_name = self.opt.get('name')
    #        with_metrics = self.opt['val'].get('metrics') is not None
    #
    #        if with_metrics:
    #            self.metric_results = {
    #                metric: 0.0
    #                for metric in self.opt['val']['metrics'].keys()
    #            }
    #
    #        cnt = 0

    #       for idx, val_data in enumerate(dataloader):
    #            self.feed_data(val_data)
    #            self.test()
    #
    #            visuals = self.get_current_visuals()
    #            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
    #            if 'gt' in visuals:
    #                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
    #
    # 只有 rank0 保存图像
    #            if save_img and rank == 0:
    #                save_img_path = osp.join(
    #                    self.opt['path']['visualization'],
    #                    dataset_name,
    #                    f'{self.image_name}.png'
    #                )
    #                imwrite(sr_img, save_img_path)

    #            if with_metrics:
    #                opt_metric = deepcopy(self.opt['val']['metrics'])
    #                for name, opt_ in opt_metric.items():
    #                    metric_type = opt_.pop('type')
    #                    self.metric_results[name] += getattr(
    #                        metric_module, metric_type)(
    #                            sr_img, gt_img, **opt_)
    #
    #            cnt += 1

    # ========== 核心：metric all_reduce ==========
    #        if with_metrics:
    #            for k in self.metric_results:
    #                tensor = torch.tensor(
    #                    self.metric_results[k], device='cuda')
    #                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    # self.metric_results[k] = tensor.item() / cnt / world_size

    #        dist.barrier()

    #        if with_metrics and rank == 0:
    #            self._log_validation_metric_values(
    #                current_iter, dataset_name, tb_logger)
    #            return list(self.metric_results.values())[0]
    #        else:
    #            return 0.

    # def nondist_validation(self, dataloader, current_iter, tb_logger,
    #                        save_img, rgb2bgr, use_image):
    #     dataset_name = self.opt.get('name') # !

    #     with_metrics = self.opt['val'].get('metrics') is not None
    #     if with_metrics:
    #         self.metric_results = {
    #             metric: 0
    #             for metric in self.opt['val']['metrics'].keys()
    #         }
    #     pbar = tqdm(total=len(dataloader), unit='image')

    #     cnt = 0

    #     for idx, val_data in enumerate(dataloader):

    #         self.feed_data(val_data)
    #         if self.opt['val'].get('grids') is not None:
    #             self.grids()
    #             self.grids_voxel()

    #         self.test()

    #         if self.opt['val'].get('grids') is not None:
    #             self.grids_inverse()

    #         visuals = self.get_current_visuals()
    #         sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
    #         if 'gt' in visuals:
    #             gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
    #             del self.gt

    #         # tentative for out of GPU memory
    #         del self.lq
    #         del self.output
    #         torch.cuda.empty_cache()

    #         if save_img:

    #             if self.opt['is_train']:
    #                 if cnt == 1: # visualize cnt=1 image every time
    #                     save_img_path = osp.join(self.opt['path']['visualization'],
    #                                             self.image_name,
    #                                             f'{self.image_name}_{current_iter}.png')

    #                     save_gt_img_path = osp.join(self.opt['path']['visualization'],
    #                                             self.image_name,
    #                                             f'{self.image_name}_{current_iter}_gt.png')
    #             else:
    #                 print('Save path:{}'.format(self.opt['path']['visualization']))
    #                 print('Dataset name:{}'.format(dataset_name))
    #                 print('Img_name:{}'.format(self.image_name))
    #                 save_img_path = osp.join(
    #                     self.opt['path']['visualization'], dataset_name,
    #                     f'{self.image_name}.png')
    #                 save_gt_img_path = osp.join(
    #                     self.opt['path']['visualization'], dataset_name,
    #                     f'{self.image_name}_gt.png')

    #             imwrite(sr_img, save_img_path)
    #             imwrite(gt_img, save_gt_img_path)

    #         if with_metrics:
    #             # calculate metrics
    #             opt_metric = deepcopy(self.opt['val']['metrics'])
    #             if use_image:
    #                 for name, opt_ in opt_metric.items():
    #                     metric_type = opt_.pop('type')
    #                     self.metric_results[name] += getattr(
    #                         metric_module, metric_type)(sr_img, gt_img, **opt_)
    #             else:
    #                 for name, opt_ in opt_metric.items():
    #                     metric_type = opt_.pop('type')
    #                     self.metric_results[name] += getattr(
    #                         metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

    #         pbar.update(1)
    #         pbar.set_description(f'Test {self.image_name}')
    #         cnt += 1
    #     pbar.close()

    #     current_metric = 0.
    #     if with_metrics:
    #         for metric in self.metric_results.keys():
    #             self.metric_results[metric] /= cnt
    #             current_metric = self.metric_results[metric]

    #         self._log_validation_metric_values(current_iter, dataset_name,
    #                                            tb_logger)
    #     return current_metric

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

            img_name = val_data['image_name'][0]
            print(img_name) #osp.splitext(osp.basename(val_data["blur_path"][0]))[0]

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

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if sr_img.shape[2] == 6:
                    L_img = sr_img[:, :, :3]
                    R_img = sr_img[:, :, 3:]

                    # visual_dir = osp.join('visual_results', dataset_name, self.opt['name'])
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
                # calculate metrics
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

        # current_metric = 0.
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

    def _activate_loss_segment(self, seg_idx):
        if not self.use_loss_schedule:
            return
        if seg_idx == self.active_loss_segment_idx:
            return

        old_cri_pix = self.cri_pix
        old_cri_perceptual = self.cri_perceptual

        seg = self.loss_schedule[seg_idx]
        self.pixel_type = seg["pixel_type"]
        cri_pix_cls = getattr(loss_module, self.pixel_type)
        self.cri_pix = cri_pix_cls(**seg["pixel_opt"]).to(self.device)

        percep_type = seg.get("percep_type")
        if percep_type is not None:
            cri_perceptual_cls = getattr(loss_module, percep_type)
            self.cri_perceptual = cri_perceptual_cls(**seg["perceptual_opt"]).to(
                self.device
            )
        else:
            self.cri_perceptual = None

        self.active_loss_segment_idx = seg_idx

        if old_cri_pix is not None:
            del old_cri_pix
        if old_cri_perceptual is not None:
            del old_cri_perceptual

    def _select_loss_segment_idx(self, current_iter):
        if not self.use_loss_schedule:
            return None
        current_iter = max(int(current_iter), 1)
        if self.loss_schedule_cycle:
            total = self.loss_schedule_cum_iters[-1]
            current_iter = ((current_iter - 1) % total) + 1
        for idx, end_iter in enumerate(self.loss_schedule_cum_iters):
            if current_iter <= end_iter:
                return idx
        return len(self.loss_schedule) - 1
