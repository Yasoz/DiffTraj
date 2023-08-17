import torch
import torch.nn as nn
import numpy as np
import math
import datetime
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from types import SimpleNamespace
from utils.config import args
from utils.EMA import EMAHelper
from utils.Traj_UNet import *
from utils.logger import Logger, log_info
from pathlib import Path
import shutil


# This code part from https://github.com/sunlin-ai/diffusion_tutorial


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1)


def main(config, logger, exp_dir):

    # Modified to return the noise itself as well
    def q_xt_x0(x0, t):
        mean = gather(alpha_bar, t)**0.5 * x0
        var = 1 - gather(alpha_bar, t)
        eps = torch.randn_like(x0).to(x0.device)
        return mean + (var**0.5) * eps, eps  # also returns noise

    # Create the model
    unet = Guide_UNet(config).cuda()
    # print(unet)
    traj = np.load('./xxxxxx',
                   allow_pickle=True)
    traj = traj[:, :, :2]
    head = np.load('./xxxxxx',
                   allow_pickle=True)
    traj = np.swapaxes(traj, 1, 2)
    traj = torch.from_numpy(traj).float()
    head = torch.from_numpy(head).float()
    dataset = TensorDataset(traj, head)
    dataloader = DataLoader(dataset,
                            batch_size=config.training.batch_size,
                            shuffle=True,
                            num_workers=8)

    # Training params
    # Set up some parameters
    n_steps = config.diffusion.num_diffusion_timesteps
    beta = torch.linspace(config.diffusion.beta_start,
                          config.diffusion.beta_end, n_steps).cuda()
    alpha = 1. - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    lr = 2e-4  # Explore this - might want it lower when training on the full dataset

    losses = []  # Store losses for later plotting
    # optimizer
    optim = torch.optim.AdamW(unet.parameters(), lr=lr)  # Optimizer

    # EMA
    if config.model.ema:
        ema_helper = EMAHelper(mu=config.model.ema_rate)
        ema_helper.register(unet)
    else:
        ema_helper = None

    # new filefold for save model pt
    model_save = exp_dir / 'models' / (timestamp + '/')
    if not os.path.exists(model_save):
        os.makedirs(model_save)

    # config.training.n_epochs = 1
    for epoch in range(1, config.training.n_epochs + 1):
        logger.info("<----Epoch-{}---->".format(epoch))
        for _, (trainx, head) in enumerate(dataloader):
            x0 = trainx.cuda()
            head = head.cuda()
            t = torch.randint(low=0, high=n_steps,
                              size=(len(x0) // 2 + 1, )).cuda()
            t = torch.cat([t, n_steps - t - 1], dim=0)[:len(x0)]
            # Get the noised images (xt) and the noise (our target)
            xt, noise = q_xt_x0(x0, t)
            # Run xt through the network to get its predictions
            pred_noise = unet(xt.float(), t, head)
            # Compare the predictions with the targets
            loss = F.mse_loss(noise.float(), pred_noise)
            # Store the loss for later viewing
            losses.append(loss.item())
            optim.zero_grad()
            loss.backward()
            optim.step()
            if config.model.ema:
                ema_helper.update(unet)
        if (epoch) % 10 == 0:
            m_path = model_save / f"unet_{epoch}.pt"
            torch.save(unet.state_dict(), m_path)
            m_path = exp_dir / 'results' / f"loss_{epoch}.npy"
            np.save(m_path, np.array(losses))


if __name__ == "__main__":
    # Load configuration
    temp = {}
    for k, v in args.items():
        temp[k] = SimpleNamespace(**v)
    config = SimpleNamespace(**temp)

    root_dir = Path(__name__).resolve().parents[0]
    result_name = '{}_steps={}_len={}_{}_bs={}'.format(
        config.data.dataset, config.diffusion.num_diffusion_timesteps,
        config.data.traj_length, config.diffusion.beta_end,
        config.training.batch_size)
    exp_dir = root_dir / "DiffTraj" / result_name
    for d in ["results", "models", "logs","Files"]:
        os.makedirs(exp_dir / d, exist_ok=True)
    print("All files saved path ---->>", exp_dir)
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
    files_save = exp_dir / 'Files' / (timestamp + '/')
    if not os.path.exists(files_save):
        os.makedirs(files_save)
    shutil.copy('./utils/config.py', files_save)
    shutil.copy('./utils/Traj_UNet.py', files_save)

    logger = Logger(
        __name__,
        log_path=exp_dir / "logs" / (timestamp + '.log'),
        colorize=True,
    )
    log_info(config, logger)
    main(config, logger, exp_dir)
