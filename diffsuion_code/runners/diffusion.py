import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data

from models.diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import get_dataset

from skimage.exposure import rescale_intensity

import torchvision.utils as tvu

import nibabel as nib
import imageio

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps, s=0.008):  #获得β
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    elif beta_schedule == 'cosine':
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = num_diffusion_timesteps + 1
        x = torch.linspace(0, num_diffusion_timesteps, steps)
        alphas_cumprod = torch.cos(((x / num_diffusion_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0] #这求得是哪一个值呢？
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.0001, 0.9999).numpy()
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def scale_data(data): 
    p10 = np.percentile(data, 10)
    data[data<p10] = p10
    data -= p10
    
    p99 = np.percentile(data, 99.5)
    data[data>p99] = p99
    data /= p99
    data *= 255

    return data

def save_nifti(data, ref_path, save_path): 
    data = data[0].detach().cpu().numpy()
    ref_data = nib.load(ref_path)
    nib.Nifti1Image(data, ref_data.affine).to_filename(save_path)


def proc_nib_data(nib_data):
    p10 = np.percentile(nib_data, 10)
    p99 = np.percentile(nib_data, 99.9)

    nib_data[nib_data<p10] = p10
    nib_data[nib_data>p99] = p99

    m = np.mean(nib_data, axis=(0, 1, 2))
    s = np.std(nib_data, axis=(0, 1, 2))
    nib_data = (nib_data - m) / s

    nib_data = torch.tensor(nib_data, dtype=torch.float32)

    return nib_data


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
            
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]
        self.num_denoising_timesteps=config.diffusion.num_denoising_timesteps

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def train(self):
        args, config = self.args, self.config
        # tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        model = Model(config) #Unet

        model = model.to(self.device)
        model = torch.nn.DataParallel(model) 

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load("")
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])
            print("load weight sucess") 

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            for i, (x, fn) in enumerate(train_loader):
                # print('shape', x.shape, c1.shape, c2.shape)
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                x = x.to(self.device)
                # x = data_transform(self.config, x)
                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss = loss_registry[config.model.type](model, x, t, e, b)

                # tb_logger.add_scalar("loss", loss, global_step=step)

                logging.info(
                    f"step: {step}, loss: {loss.item():12.4f}, data time: {data_time / (i+1)}"
                )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()

    def inference(self):
        model = Model(self.config) 

        states = torch.load(
            "/public/home/wangkd2023/Denghw/diffsuion_code/exp/logs/diffusion_initial/ckpt_275000.pth",
            map_location=self.config.device,
        )

        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0], strict=True)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(model)
        else:
            ema_helper = None
        
        model.eval()

        c1 = nib.load(self.args.reference_img).get_fdata()
        p10 = np.percentile(c1, 10)
        p99 = np.percentile(c1, 99)
        c1 = rescale_intensity(c1, in_range=(p10, p99), out_range=(0, 1))
        m = np.mean(c1, axis=(0, 1, 2))
        s = np.std(c1, axis=(0, 1, 2))
        c1 = (c1 - m) / s
        c1 = torch.tensor(c1, dtype=torch.float32)
        c1 = c1.unsqueeze(0).unsqueeze(0)
        c1 = c1.to(self.device)


        b = self.betas
        e = torch.randn_like(c1)
        logging.info(self.num_timesteps)
        t = torch.randint(
            low=0, high=self.num_timesteps, size=(1 // 2 + 1,)
        ).to(self.device)
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:1]        
        a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1, 1)
        x = c1 * a.sqrt() + e * (1.0 - a).sqrt()
        save_nifti(
            x[0], self.args.reference_img, os.path.join(self.args.inference_folder, f"noise_xt.nii.gz"))
        x = self.inference_image(x, model, (not self.args.inference_gif)) 
        
        save_nifti(
            x[0], self.args.reference_img, os.path.join(self.args.inference_folder, f"fake_initial.nii.gz"))
    def inference_all(self):
        args, config = self.args, self.config
        # tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        inference_loader = data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )
        model = Model(self.config)

        states = torch.load(
            "/public/home/wangkd2023/Denghw/diffsuion_code/exp/logs/diffusion_initial/ckpt_275000.pth",
            map_location=self.config.device,
        )

        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0], strict=True)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(model)
        else:
            ema_helper = None
        
        model.eval()
        b = self.betas
        t = torch.randint(
            low=0, high=self.num_timesteps, size=(1 // 2 + 1,)
        ).to(self.device)
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:1]        
        a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1, 1)
        logging.info(a)
        for i, (t1, fn) in enumerate(inference_loader):
            logging.info("Run to:")
            logging.info(i)
            fn1 = "/".join(fn[0].split("/")[:-1])
            logging.info("The current result will be saved to:")
            logging.info(fn1)
            t1 = t1.to(self.device)
            e = torch.randn_like(t1)
            x = t1 * a.sqrt() + e * (1.0 - a).sqrt()
            x = self.inference_image(x, model, (not self.args.inference_gif)) 

            save_nifti(
                x[0], fn[0], os.path.join(fn1, f"T1_initial.nii.gz"))
            logging.info("Result output")

    def inference_image(self, x, model, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":#true
            if self.args.skip_type == "uniform":#true
                skip = self.num_denoising_timesteps // self.args.timesteps
                seq = range(0, self.num_denoising_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_denoising_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import inference_steps

            xs = inference_steps(x, seq, model, self.betas, eta=self.args.eta)
            x = xs
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x

    def test(self):
        pass
