import copy
import functools
import os

import blobfile as bf
import torch as th
# import torch.distributed as dist
# from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

# from . import dist_util, 
from . import logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
import wandb
import random

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0
# SNR (var): 1 (0.9) 5 (0.6) 10 (0.36) 15 (0.22) 20 (0.13) 25 (0.08) 30 (0.05) 100 (0.0)
SNR_DICT = {100: 0.0,
            30: 0.05,
            25: 0.08,
            20: 0.13,
            15: 0.22,
            10: 0.36,
            5: 0.6,
            1: 0.9}

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        num_classes,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        drop_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        checkpoint_dir="./checkpoints",
        one_hot_label=True,
        snr=100,
        add_noise=False,
        noise_to="semantics",
        unet_model="unet"
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.drop_rate = drop_rate
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.checkpoint_dir = checkpoint_dir
        self.one_hot_label = one_hot_label
        self.snr = snr
        self.add_noise = add_noise
        self.noise_to = noise_to
        self.unet_model = unet_model

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size # * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            # self.use_ddp = True
            # self.use_ddp = False
        #     self.ddp_model = DDP(
        #         self.model,
        #         device_ids=[dist_util.dev()],
        #         output_device=dist_util.dev(),
        #         broadcast_buffers=False,
        #         bucket_cap_mb=128,
        #         find_unused_parameters=False,
        #     )
        # else:
            # if dist.get_world_size() > 1:
            #     logger.warn(
            #         "Distributed training requires CUDA. "
            #         "Gradients will not be synchronized properly!"
            #     )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            # if dist.get_rank() == 0:
                # logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            print(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                th.load(
                    # resume_checkpoint, map_location=dist_util.dev()
                    resume_checkpoint, map_location="cuda"
                )
            )

        # dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            # if dist.get_rank() == 0:
                # logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            print(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = th.load(
                # ema_checkpoint, map_location=dist_util.dev()
                ema_checkpoint, map_location="cuda"
            )
            ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        # dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            # logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            print(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = th.load(
                # opt_checkpoint, map_location=dist_util.dev()
                opt_checkpoint, map_location="cuda"
            )
            self.opt.load_state_dict(state_dict)

            if self.opt.param_groups[0]['lr'] != self.lr:
                self.opt.param_groups[0]['lr'] = self.lr

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            cond = self.preprocess_input(cond, one_hot_label=self.one_hot_label, add_noise=self.add_noise, noise_to=self.noise_to)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                pass
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            # micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro = batch[i : i + self.microbatch].to("cuda")
            micro_cond = {
                # k: v[i : i + self.microbatch].to(dist_util.dev())
                k: v[i : i + self.microbatch].to("cuda")
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            # t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            t, weights = self.schedule_sampler.sample(micro.shape[0], "cuda")

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

            wandb.log({"Loss": loss})

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        wandb.log({"step": self.step + self.resume_step,
                   "samples": (self.step + self.resume_step + 1) * self.global_batch})

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            print(f"saving model {rate}...")
            if not rate:
                filename = f"model{(self.step+self.resume_step):06d}.pt"
            else:
                filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
            with bf.BlobFile(bf.join(self.checkpoint_dir, filename), "wb") as f:
                th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        with bf.BlobFile(
            bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
            "wb",
        ) as f:
            th.save(self.opt.state_dict(), f)

    def preprocess_input(self, data, one_hot_label=True, add_noise=False, noise_to="semantics"):
        # move to GPU and change data types
        data['label'] = data['label'].long()

        if one_hot_label:
            # create one-hot label map
            label_map = data['label']
            bs, _, h, w = label_map.size()
            nc = self.num_classes
            input_label = th.FloatTensor(bs, nc, h, w).zero_()
            input_semantics = input_label.scatter_(1, label_map, 1.0)

            # concatenate instance map if it exists
            if 'instance' in data:
                inst_map = data['instance']
                instance_edge_map = self.get_edges(inst_map)
                input_semantics = th.cat((input_semantics, instance_edge_map), dim=1)

        else:
            label_map = data['label']
            if 'instance' in data:
                inst_map = data['instance']
                instance_edge_map = self.get_edges(inst_map)
                input_semantics = th.cat((label_map, instance_edge_map), dim=1)

        if add_noise:
            snr = random.choice([0.6, 0.36, 0.22, 0.13, 0.08, 0.05, 0.0, 0.0, 0.0])
            # SNR (var): 1 (0.9) 5 (0.6) 10 (0.36) 15 (0.22) 20 (0.13) 25 (0.08) 30 (0.05) 100 (0.0)
            noise = th.randn(data["label"].shape, device=data["label"].device)*snr
            if noise_to == "semantics":
                input_semantics += noise
            elif noise_to == "label":
                data["label"] = (data["label"] - th.min(data["label"])) / (th.max(data["label"]) - th.min(data["label"]))
                data["label"] += noise

        if self.drop_rate > 0.0:
            mask = (th.rand([input_semantics.shape[0], 1, 1, 1]) > self.drop_rate).float()
            input_semantics = input_semantics * mask

        cond = {key: value for key, value in data.items() if key not in ['label', 'instance', 'path', 'label_ori']}
        cond['y'] = input_semantics
        cond['label'] = data['label']
        return cond

    def get_edges(self, t):
        edge = th.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        wandb.log({key: values.mean().item()})
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            wandb.log({f"{key}_q{quartile}": sub_loss})

