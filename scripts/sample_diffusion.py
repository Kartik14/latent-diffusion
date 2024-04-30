import argparse, os, sys, glob, datetime, yaml
import torch
import time
import numpy as np
from tqdm import trange
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid

from omegaconf import OmegaConf
from PIL import Image

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config

rescale = lambda x: (x + 1.0) / 2.0


def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1.0, 1.0)
    x = (x + 1.0) / 2.0
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def custom_to_np(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample


def logs2pil(logs, keys=["sample"]):
    imgs = dict()
    for k in logs:
        try:
            if len(logs[k].shape) == 4:
                img = custom_to_pil(logs[k][0, ...])
            elif len(logs[k].shape) == 3:
                img = custom_to_pil(logs[k])
            else:
                print(f"Unknown format for key {k}. ")
                img = None
        except:
            img = None
        imgs[k] = img
    return imgs


@torch.no_grad()
def convsample(
    model, shape, return_intermediates=True, verbose=True, make_prog_row=False
):

    if not make_prog_row:
        return model.p_sample_loop(
            None, shape, return_intermediates=return_intermediates, verbose=verbose
        )
    else:
        return model.progressive_denoising(None, shape, verbose=True)


@torch.no_grad()
def convsample_ddim(model, steps, shape, eta=1.0):
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = ddim.sample(
        steps,
        batch_size=bs,
        shape=shape,
        eta=eta,
        verbose=False,
    )
    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(
    model,
    batch_size,
    vanilla=False,
    custom_steps=None,
    eta=1.0,
):

    log = dict()

    shape = [
        batch_size,
        model.model.diffusion_model.in_channels,
        model.model.diffusion_model.image_size,
        model.model.diffusion_model.image_size,
    ]

    with model.ema_scope("Plotting"):
        t0 = time.time()
        if vanilla:
            sample, progrow = convsample(model, shape, make_prog_row=True)
        else:
            sample, intermediates = convsample_ddim(
                model, steps=custom_steps, shape=shape, eta=eta
            )

        t1 = time.time()

    x_sample = model.decode_first_stage(sample)
    log["sample_emb"] = sample
    # print(sample.shape)
    log["sample"] = x_sample
    log["time"] = t1 - t0
    log["throughput"] = sample.shape[0] / (t1 - t0)
    print(f'Throughput for this batch: {log["throughput"]}')
    return log


def run(
    model,
    logdir,
    batch_size=50,
    vanilla=False,
    custom_steps=None,
    eta=None,
    n_samples=50000,
    nplog=None,
    ckpt=None,
):
    if vanilla:
        print(f"Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.")
    else:
        print(f"Using DDIM sampling with {custom_steps} sampling steps and eta={eta}")

    tstart = time.time()
    n_saved = len(glob.glob(os.path.join(logdir, "*.png"))) - 1
    # path = logdir
    if model.cond_stage_model is None:
        all_img = []
        all_embs = []
        print(f"Running unconditional sampling for {n_samples} samples")
        for _ in trange(
            n_samples // batch_size, desc="Sampling Batches (unconditional)"
        ):
            logs = make_convolutional_sample(
                model,
                batch_size=batch_size,
                vanilla=vanilla,
                custom_steps=custom_steps,
                eta=eta,
            )
            n_saved = save_logs(logs, logdir, n_saved=n_saved, key="sample")
            all_img.extend([custom_to_np(logs["sample"])])
            if n_saved >= n_samples:
                print(f"Finish after generating {n_saved} samples")
                break
            all_embs.append(logs["sample_emb"])
        all_img = np.concatenate(all_img, axis=0)
        all_img = all_img[:n_samples]
        shape_str = "x".join([str(x) for x in all_img.shape])
        nppath = os.path.join(nplog, f"{shape_str}-samples.npz")
        np.savez(nppath, all_img)

        all_embs = torch.cat(all_embs, dim=0).cpu().numpy()
        nppath_emb = os.path.join(nplog, f"{shape_str}-samples_emb.npz")
        np.savez(nppath_emb, all_embs)

    else:
        # raise NotImplementedError(
        #     "Currently only sampling for unconditional models supported."
        # )
        classes = [0, 1, 2]  # define classes to be sampled here
        n_samples_per_class = n_samples // len(classes)
        print(f"Running conditional sampling for {n_samples} samples")

        ddim_steps = custom_steps
        ddim_eta = eta
        scale = 3.0  # for unconditional guidance
        sampler = DDIMSampler(model)
        all_img = {}
        with torch.no_grad():
            with model.ema_scope():
                for i in range(n_samples_per_class // batch_size):
                    print(f"Sampling batch {i+1}/{n_samples_per_class // batch_size}")
                    uc = model.get_learned_conditioning(
                        {
                            model.cond_stage_key: torch.tensor(batch_size * [2]).to(
                                "cuda"
                            )
                        }
                    )

                    for class_label in classes:
                        xc = torch.tensor(batch_size * [class_label])
                        xc = xc.to("cuda")
                        c = model.get_learned_conditioning({model.cond_stage_key: xc})

                        samples_ddim, _ = sampler.sample(
                            S=ddim_steps,
                            conditioning=c,
                            batch_size=batch_size,
                            shape=[16, 16, 16],
                            verbose=False,
                            unconditional_guidance_scale=scale,
                            unconditional_conditioning=uc,
                            eta=ddim_eta,
                        )

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        if class_label not in all_img:
                            all_img[class_label] = []
                        all_img[class_label].append(x_samples_ddim)

        # all_img = torch.cat(all_img, dim=0)
        for k in all_img:
            all_img[k] = torch.cat(all_img[k], dim=0)
        # save images as png file
        for k in all_img:
            for i, img in enumerate(all_img[k]):
                img = custom_to_pil(img)
                os.makedirs(f"conditional_images/{ckpt}/{k}", exist_ok=True)
                imgpath = os.path.join(
                    f"conditional_images/{ckpt}/{k}", f"sample_{i}.png"
                )
                img.save(imgpath)
                n_saved += 1

    print(
        f"sampling of {n_saved} images finished in {(time.time() - tstart) / 60.:.2f} minutes."
    )


def save_logs(logs, path, n_saved=0, key="sample", np_path=None):
    for k in logs:
        if k == key:
            batch = logs[key]
            if np_path is None:
                for x in batch:
                    img = custom_to_pil(x)
                    imgpath = os.path.join(path, f"{key}_{n_saved:06}.png")
                    img.save(imgpath)
                    n_saved += 1
            else:
                npbatch = custom_to_np(batch)
                shape_str = "x".join([str(x) for x in npbatch.shape])
                nppath = os.path.join(np_path, f"{n_saved}-{shape_str}-samples.npz")
                np.savez(nppath, npbatch)
                n_saved += npbatch.shape[0]
    return n_saved


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        nargs="?",
        help="number of samples to draw",
        default=50000,
    )
    parser.add_argument(
        "--ckpt",
        type=int,
        nargs="?",
        help="checkpoint to load",
        default=10,
    )
    parser.add_argument(
        "-a",
        "--allcheckpoints",
        default=False,
        action="store_true",
        help="sample all checkpoints in the logdir?",
    )
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        nargs="?",
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
        default=1.0,
    )
    parser.add_argument(
        "-v",
        "--vanilla_sample",
        default=False,
        action="store_true",
        help="vanilla sampling (default option is DDIM sampling)?",
    )
    parser.add_argument(
        "-l", "--logdir", type=str, nargs="?", help="extra logdir", default="none"
    )
    parser.add_argument(
        "-c",
        "--custom_steps",
        type=int,
        nargs="?",
        help="number of steps for ddim and fastdpm sampling",
        default=50,
    )
    parser.add_argument("--batch_size", type=int, nargs="?", help="the bs", default=10)
    return parser


def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def load_model(config, ckpt, gpu, eval_mode):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model, pl_sd["state_dict"])

    return model, global_step


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    sys.path.append(os.getcwd())
    command = " ".join(sys.argv)

    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    ckpt = None

    logdir = opt.resume
    print(logdir)
    if opt.allcheckpoints:
        checkpoints = glob.glob(os.path.join(logdir, "checkpoints", "*.ckpt"))
    else:
        filename = (
            logdir
            + "/checkpoints/"
            + f"epoch={''.join(['0']*(6 - len(str(opt.ckpt))))+ str(opt.ckpt)}.ckpt"
        )
        checkpoints = [filename]
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Checkpoint {filename} not found.")

    base_configs = sorted(glob.glob(os.path.join(logdir, "configs", "*-project.yaml")))
    opt.base = base_configs

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    gpu = True
    eval_mode = True

    if opt.logdir != "none":
        locallog = logdir.split(os.sep)[-1]
        if locallog == "":
            locallog = logdir.split(os.sep)[-2]
        print(
            f"Switching logdir from '{logdir}' to '{os.path.join(opt.logdir, locallog)}'"
        )
        logdir = os.path.join(opt.logdir, locallog)

    print(config)
    sdir = os.path.join(logdir, "samples_" + now)
    os.makedirs(sdir)
    samplesdir = os.path.join(sdir, "img")
    npsamplesdir = os.path.join(sdir, "numpy")
    os.makedirs(samplesdir, exist_ok=True)
    os.makedirs(npsamplesdir, exist_ok=True)
    # write config out
    sampling_file = os.path.join(sdir, "sampling_config.yaml")
    sampling_conf = vars(opt)

    with open(sampling_file, "w") as f:
        yaml.dump(sampling_conf, f, default_flow_style=False)
    print(sampling_conf)
    for ckpt in checkpoints:
        if "last" in ckpt or ("0079" in ckpt):
            print(f"Skipping {ckpt}")
            continue
        model, global_step = load_model(config, ckpt, gpu, eval_mode)
        print(f"global step: {global_step}")
        print(75 * "=")
        print("logging to:")
        ckpt_num = ckpt.split("/")[-1].split(".")[0].strip("0")
        sampledir = samplesdir + f"/{ckpt_num}"
        nplogdir = npsamplesdir + f"/{ckpt_num}"
        os.makedirs(sampledir)
        os.makedirs(nplogdir)

        print(75 * "=")

        run(
            model,
            sampledir,
            eta=opt.eta,
            vanilla=opt.vanilla_sample,
            n_samples=opt.n_samples,
            custom_steps=opt.custom_steps,
            batch_size=opt.batch_size,
            nplog=nplogdir,
            ckpt=opt.ckpt,
        )

    print("done.")
