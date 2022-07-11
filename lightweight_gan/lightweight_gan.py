import os
import json
import multiprocessing
import math
from math import log2, floor
from functools import lru_cache
from contextlib import contextmanager, ExitStack
from pathlib import Path
from shutil import rmtree
from einops import rearrange, reduce, repeat
import torchvision
from PIL import Image

import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import grad as torch_grad
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from lightweight_gan.version import __version__
from lightweight_gan.iclr_2021 import LightweightGAN
from lightweight_gan.aug import ImageDataset

from tqdm import tqdm

# asserts
assert torch.cuda.is_available(), 'You need to have an Nvidia GPU with CUDA installed.'

# constants
NUM_CORES = multiprocessing.cpu_count()

@contextmanager
def null_context():
    yield

# helpers
def combine_contexts(contexts):
    @contextmanager
    def multi_contexts():
        with ExitStack() as stack:
            yield [stack.enter_context(ctx()) for ctx in contexts]
    return multi_contexts

def cycle(iterable):
    while True:
        for i in iterable:
            yield i

def raise_if_nan(t):
    if torch.isnan(t):
        raise NanException

def gradient_accumulate_contexts(gradient_accumulate_every, is_ddp, ddps):
    if is_ddp:
        num_no_syncs = gradient_accumulate_every - 1
        head = [combine_contexts(map(lambda ddp: ddp.no_sync, ddps))] * num_no_syncs
        tail = [null_context]
        contexts =  head + tail
    else:
        contexts = [null_context] * gradient_accumulate_every

    for context in contexts:
        with context():
            yield

def evaluate_in_chunks(max_batch_size, model, *args):
    split_args = list(zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))
    chunked_outputs = [model(*i) for i in split_args]
    if len(chunked_outputs) == 1:
        return chunked_outputs[0]
    return torch.cat(chunked_outputs, dim=0)

def slerp(val, low, high):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (torch.sin(val * omega) / so).unsqueeze(1) * high
    return res

def safe_div(n, d):
    try:
        res = n / d
    except ZeroDivisionError:
        prefix = '' if int(n >= 0) else '-'
        res = float(f'{prefix}inf')
    return res

# loss functions
def gen_hinge_loss(fake, real):
    return fake.mean()

def hinge_loss(real, fake):
    return (F.relu(1 + real) + F.relu(1 - fake)).mean()

def dual_contrastive_loss(real_logits, fake_logits):
    device = real_logits.device
    real_logits, fake_logits = map(lambda t: rearrange(t, '... -> (...)'), (real_logits, fake_logits))

    def loss_half(t1, t2):
        t1 = rearrange(t1, 'i -> i ()')
        t2 = repeat(t2, 'j -> i j', i = t1.shape[0])
        t = torch.cat((t1, t2), dim = -1)
        return F.cross_entropy(t, torch.zeros(t1.shape[0], device = device, dtype = torch.long))

    return loss_half(real_logits, fake_logits) + loss_half(-fake_logits, -real_logits)

@lru_cache(maxsize=10)
def det_randn(*args):
    """
    deterministic random to track the same latent vars (and images) across training steps
    helps to visualize same image over training steps
    """
    return torch.randn(*args)

def interpolate_between(a, b, *, num_samples, dim):
    assert num_samples > 2
    samples = []
    step_size = 0
    for _ in range(num_samples):
        sample = torch.lerp(a, b, step_size)
        samples.append(sample)
        step_size += 1 / (num_samples - 1)
    return torch.stack(samples, dim=dim)

# helper classes

class NanException(Exception):
    pass


# dataset


class identity(object):
    def __call__(self, tensor):
        return tensor


# trainer
class Trainer():
    def __init__(
        self,
        name = 'default',
        results_dir = 'results',
        models_dir = 'models',
        base_dir = './',
        optimizer = 'adam',
        num_workers = None,
        latent_dim = 256,
        image_size = 128,
        num_image_tiles = 8,
        fmap_max = 512,
        transparent = False,
        greyscale = False,
        batch_size = 4,
        gp_weight = 10,
        gradient_accumulate_every = 1,
        attn_res_layers = [],
        freq_chan_attn = False,
        disc_output_size = 5,
        dual_contrast_loss = False,
        antialias = False,
        lr = 2e-4,
        lr_mlp = 1.,
        ttur_mult = 1.,
        save_every = 1000,
        evaluate_every = 1000,
        aug_prob = None,
        aug_types = ['translation', 'cutout'],
        dataset_aug_prob = 0.,
        calculate_fid_every = None,
        calculate_fid_num_images = 12800,
        clear_fid_cache = False,
        is_ddp = False,
        rank = 0,
        world_size = 1,
        log = False,
        amp = False,
        hparams = None,
        use_aim = True,
        aim_repo = None,
        aim_run_hash = None,
        load_strict = True,
        *args,
        **kwargs
    ):
        self.GAN_params = [args, kwargs]
        self.GAN = None

        self.name = name

        base_dir = Path(base_dir)
        self.base_dir = base_dir
        self.results_dir = base_dir / results_dir
        self.models_dir = base_dir / models_dir
        self.fid_dir = base_dir / 'fid' / name

        self.config_path = self.models_dir / name / '.config.json'
        is_power_of_two = lambda x: log2(x).is_integer()

        assert is_power_of_two(image_size), 'image size must be a power of 2 (64, 128, 256, 512, 1024)'
        assert all(map(is_power_of_two, attn_res_layers)), 'resolution layers of attention must all be powers of 2 (16, 32, 64, 128, 256, 512)'

        assert not (dual_contrast_loss and disc_output_size > 1), 'discriminator output size cannot be greater than 1 if using dual contrastive loss'

        self.image_size = image_size
        self.num_image_tiles = num_image_tiles

        self.latent_dim = latent_dim
        self.fmap_max = fmap_max
        self.transparent = transparent
        self.greyscale = greyscale

        assert (int(self.transparent) + int(self.greyscale)) < 2, 'you can only set either transparency or greyscale'

        self.aug_prob = aug_prob
        self.aug_types = aug_types

        self.lr = lr
        self.optimizer = optimizer
        self.num_workers = num_workers
        self.ttur_mult = ttur_mult
        self.batch_size = batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.gp_weight = gp_weight

        self.evaluate_every = evaluate_every
        self.save_every = save_every
        self.steps = 0

        self.attn_res_layers = attn_res_layers
        self.freq_chan_attn = freq_chan_attn

        self.disc_output_size = disc_output_size
        self.antialias = antialias

        self.dual_contrast_loss = dual_contrast_loss

        self.d_loss = 0
        self.g_loss = 0
        self.last_gp_loss = None
        self.last_recon_loss = None
        self.last_fid = None

        self.init_folders()

        self.loader = None
        self.dataset_aug_prob = dataset_aug_prob

        self.calculate_fid_every = calculate_fid_every
        self.calculate_fid_num_images = calculate_fid_num_images
        self.clear_fid_cache = clear_fid_cache

        self.is_ddp = is_ddp
        self.is_main = rank == 0
        self.rank = rank
        self.world_size = world_size

        self.syncbatchnorm = is_ddp

        self.load_strict = load_strict

        self.amp = amp
        self.G_scaler = GradScaler(enabled = self.amp)
        self.D_scaler = GradScaler(enabled = self.amp)

        self.run = None
        self.hparams = hparams

        if self.is_main and use_aim:
            try:
                import aim
                self.aim = aim
            except ImportError:
                print('unable to import aim experiment tracker - please run `pip install aim` first')

            self.run = self.aim.Run(run_hash=aim_run_hash, repo=aim_repo)
            self.run['hparams'] = hparams

    @property
    def image_extension(self):
        return 'jpg' if not self.transparent else 'png'

    @property
    def checkpoint_num(self):
        return floor(self.steps // self.save_every)
        
    def init_GAN(self):
        args, kwargs = self.GAN_params

        # set some global variables before instantiating GAN

        global norm_class
        global Blur

        norm_class = nn.SyncBatchNorm if self.syncbatchnorm else nn.BatchNorm2d
        Blur = nn.Identity if not self.antialias else Blur

        # handle bugs when
        # switching from multi-gpu back to single gpu

        if self.syncbatchnorm and not self.is_ddp:
            import torch.distributed as dist
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            dist.init_process_group('nccl', rank=0, world_size=1)

        # instantiate GAN

        self.GAN = LightweightGAN(
            optimizer=self.optimizer,
            lr = self.lr,
            latent_dim = self.latent_dim,
            attn_res_layers = self.attn_res_layers,
            freq_chan_attn = self.freq_chan_attn,
            image_size = self.image_size,
            ttur_mult = self.ttur_mult,
            fmap_max = self.fmap_max,
            disc_output_size = self.disc_output_size,
            transparent = self.transparent,
            greyscale = self.greyscale,
            rank = self.rank,
            *args,
            **kwargs
        )

        if self.is_ddp:
            ddp_kwargs = {'device_ids': [self.rank], 'output_device': self.rank, 'find_unused_parameters': True}

            self.G_ddp = DDP(self.GAN.G, **ddp_kwargs)
            self.D_ddp = DDP(self.GAN.D, **ddp_kwargs)
            self.D_aug_ddp = DDP(self.GAN.D_aug, **ddp_kwargs)

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config()))

    def load_config(self):
        config = self.config() if not self.config_path.exists() else json.loads(self.config_path.read_text())
        self.image_size = config['image_size']
        self.transparent = config['transparent']
        self.syncbatchnorm = config['syncbatchnorm']
        self.disc_output_size = config['disc_output_size']
        self.greyscale = config.pop('greyscale', False)
        self.attn_res_layers = config.pop('attn_res_layers', [])
        self.freq_chan_attn = config.pop('freq_chan_attn', False)
        self.optimizer = config.pop('optimizer', 'adam')
        self.fmap_max = config.pop('fmap_max', 512)
        del self.GAN
        self.init_GAN()

    def config(self):
        return {
            'image_size': self.image_size,
            'transparent': self.transparent,
            'greyscale': self.greyscale,
            'syncbatchnorm': self.syncbatchnorm,
            'disc_output_size': self.disc_output_size,
            'optimizer': self.optimizer,
            'attn_res_layers': self.attn_res_layers,
            'freq_chan_attn': self.freq_chan_attn
        }

    def set_data_src(self, folder):
        num_workers = self.num_workers
        if num_workers is None:
            num_workers = math.ceil(NUM_CORES / self.world_size)
        self.dataset = ImageDataset(folder, self.image_size, transparent = self.transparent, greyscale = self.greyscale, aug_prob = self.dataset_aug_prob)
        sampler = DistributedSampler(self.dataset, rank=self.rank, num_replicas=self.world_size, shuffle=True) if self.is_ddp else None
        dataloader = DataLoader(self.dataset, num_workers = num_workers, batch_size = math.ceil(self.batch_size / self.world_size), sampler = sampler, shuffle = not self.is_ddp, drop_last = True, pin_memory = True)
        self.loader = cycle(dataloader)

        # auto set augmentation prob for user if dataset is detected to be low
        num_samples = len(self.dataset)
        if self.aug_prob is None and num_samples < 1e5:
            self.aug_prob = min(0.5, (1e5 - num_samples) * 3e-6)
            print(f'autosetting augmentation probability to {round(self.aug_prob * 100)}%')

    def train(self):
        assert self.loader is not None, 'You must first initialize the data source with `.set_data_src(<folder of images>)`'
        device = torch.device(f'cuda:{self.rank}')

        if self.GAN is None:
            self.init_GAN()

        self.GAN.train()
        total_disc_loss = torch.zeros([], device=device)
        total_gen_loss = torch.zeros([], device=device)

        batch_size = math.ceil(self.batch_size / self.world_size)

        image_size = self.GAN.image_size
        latent_dim = self.GAN.latent_dim

        aug_prob = 0 if self.aug_prob is None else self.aug_prob
        aug_types  = self.aug_types
        aug_kwargs = {'prob': aug_prob, 'types': aug_types}

        G = self.GAN.G if not self.is_ddp else self.G_ddp
        D = self.GAN.D if not self.is_ddp else self.D_ddp
        D_aug = self.GAN.D_aug if not self.is_ddp else self.D_aug_ddp

        apply_gradient_penalty = self.steps % 4 == 0

        # amp related contexts and functions

        amp_context = autocast if self.amp else null_context

        # discriminator loss fn

        if self.dual_contrast_loss:
            D_loss_fn = dual_contrastive_loss
        else:
            D_loss_fn = hinge_loss

        # train discriminator

        self.GAN.D_opt.zero_grad()
        for i in gradient_accumulate_contexts(self.gradient_accumulate_every, self.is_ddp, ddps=[D_aug, G]):
            latents = torch.randn(batch_size, latent_dim).cuda(self.rank)
            image_batch = next(self.loader).cuda(self.rank)
            image_batch.requires_grad_()

            with amp_context():
                with torch.no_grad():
                    generated_images = G(latents)

                fake_output, fake_output_32x32, _ = D_aug(generated_images, detach = True, **aug_kwargs)

                real_output, real_output_32x32, real_aux_loss = D_aug(image_batch,  calc_aux_loss = True, **aug_kwargs)

                real_output_loss = real_output
                fake_output_loss = fake_output

                divergence = D_loss_fn(real_output_loss, fake_output_loss)
                divergence_32x32 = D_loss_fn(real_output_32x32, fake_output_32x32)
                disc_loss = divergence + divergence_32x32

                aux_loss = real_aux_loss
                disc_loss = disc_loss + aux_loss

            if apply_gradient_penalty:
                outputs = [real_output, real_output_32x32]
                outputs = list(map(self.D_scaler.scale, outputs)) if self.amp else outputs

                scaled_gradients = torch_grad(outputs=outputs, inputs=image_batch,
                                       grad_outputs=list(map(lambda t: torch.ones(t.size(), device = image_batch.device), outputs)),
                                       create_graph=True, retain_graph=True, only_inputs=True)[0]

                inv_scale = safe_div(1., self.D_scaler.get_scale()) if self.amp else 1.

                if inv_scale != float('inf'):
                    gradients = scaled_gradients * inv_scale

                    with amp_context():
                        gradients = gradients.reshape(batch_size, -1)
                        gp =  self.gp_weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

                        if not torch.isnan(gp):
                            disc_loss = disc_loss + gp
                            self.last_gp_loss = gp.clone().detach().item()

            with amp_context():
                disc_loss = disc_loss / self.gradient_accumulate_every

            disc_loss.register_hook(raise_if_nan)
            self.D_scaler.scale(disc_loss).backward()
            total_disc_loss += divergence

        self.last_recon_loss = aux_loss.item()
        self.d_loss = float(total_disc_loss.item() / self.gradient_accumulate_every)
        self.D_scaler.step(self.GAN.D_opt)
        self.D_scaler.update()

        # generator loss fn

        if self.dual_contrast_loss:
            G_loss_fn = dual_contrastive_loss
            G_requires_calc_real = True
        else:
            G_loss_fn = gen_hinge_loss
            G_requires_calc_real = False

        # train generator

        self.GAN.G_opt.zero_grad()

        for i in gradient_accumulate_contexts(self.gradient_accumulate_every, self.is_ddp, ddps=[G, D_aug]):
            latents = torch.randn(batch_size, latent_dim).cuda(self.rank)

            if G_requires_calc_real:
                image_batch = next(self.loader).cuda(self.rank)
                image_batch.requires_grad_()

            with amp_context():
                generated_images = G(latents)

                fake_output, fake_output_32x32, _ = D_aug(generated_images, **aug_kwargs)
                real_output, real_output_32x32, _ = D_aug(image_batch, **aug_kwargs) if G_requires_calc_real else (None, None, None)

                loss = G_loss_fn(fake_output, real_output)
                loss_32x32 = G_loss_fn(fake_output_32x32, real_output_32x32)

                gen_loss = loss + loss_32x32

                gen_loss = gen_loss / self.gradient_accumulate_every

            gen_loss.register_hook(raise_if_nan)
            self.G_scaler.scale(gen_loss).backward()
            total_gen_loss += loss

        self.g_loss = float(total_gen_loss.item() / self.gradient_accumulate_every)
        self.G_scaler.step(self.GAN.G_opt)
        self.G_scaler.update()

        # calculate moving averages

        if self.is_main and self.steps % 10 == 0 and self.steps > 20000:
            self.GAN.EMA()

        if self.is_main and self.steps <= 25000 and self.steps % 1000 == 2:
            self.GAN.reset_parameter_averaging()

        # save from NaN errors

        if any(torch.isnan(l) for l in (total_gen_loss, total_disc_loss)):
            print(f'NaN detected for generator or discriminator. Loading from checkpoint #{self.checkpoint_num}')
            self.load(self.checkpoint_num)
            raise NanException

        del total_disc_loss
        del total_gen_loss

        # periodically save results

        if self.is_main:
            if self.steps % self.save_every == 0:
                self.save(self.checkpoint_num)

            if self.steps % self.evaluate_every == 0 or (self.steps % 100 == 0 and self.steps < 20000):
                self.evaluate(floor(self.steps / self.evaluate_every), num_image_tiles = self.num_image_tiles)

            if self.calculate_fid_every is not None and self.steps % self.calculate_fid_every == 0 and self.steps != 0:
                num_batches = math.ceil(self.calculate_fid_num_images / self.batch_size)
                fid = self.calculate_fid(num_batches)
                self.last_fid = fid

                with open(str(self.results_dir / self.name / f'fid_scores.txt'), 'a') as f:
                    f.write(f'{self.steps},{fid}\n')

        self.steps += 1

    @torch.no_grad()
    def evaluate(self, num = 0, num_image_tiles = 4):
        self.GAN.eval()

        ext = self.image_extension
        num_rows = num_image_tiles
    
        latent_dim = self.GAN.latent_dim
        image_size = self.GAN.image_size

        # latents and noise
        def image_to_pil(image):
            ndarr = image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(ndarr)
            return im

        latents = det_randn((num_rows ** 2, latent_dim)).cuda(self.rank)
        interpolate_latents = interpolate_between(latents[:num_rows], latents[-num_rows:],
                                                  num_samples=num_rows,
                                                  dim=0).flatten(end_dim=1)

        generate_interpolations = self.generate_(self.GAN.G, interpolate_latents)
        if self.run is not None:
            grouped = generate_interpolations.view(num_rows, num_rows, *generate_interpolations.shape[1:])
            for idx, images in enumerate(grouped):
                alpha = idx / (len(grouped) - 1)
                aim_images = []
                for image in images:
                    im = image_to_pil(image)
                    aim_images.append(self.aim.Image(im, caption=f'#{idx}'))

                self.run.track(value=aim_images, name='generated',
                               step=self.steps,
                               context={'interpolated': True,
                                        'alpha': alpha})
        torchvision.utils.save_image(generate_interpolations, str(self.results_dir / self.name / f'{str(num)}-interp.{ext}'), nrow=num_rows)
        # regular

        generated_images = self.generate_(self.GAN.G, latents)

        if self.run is not None:
            aim_images = []
            for idx, image in enumerate(generated_images):
                im = image_to_pil(image)
                aim_images.append(self.aim.Image(im, caption=f'#{idx}'))

            self.run.track(value=aim_images, name='generated',
                           step=self.steps,
                           context={'ema': False})
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}.{ext}'), nrow=num_rows)

        # moving averages

        generated_images = self.generate_(self.GAN.GE, latents)
        if self.run is not None:
            aim_images = []
            for idx, image in enumerate(generated_images):
                im = image_to_pil(image)
                aim_images.append(self.aim.Image(im, caption=f'EMA #{idx}'))

            self.run.track(value=aim_images, name='generated',
                           step=self.steps,
                           context={'ema': True})
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}-ema.{ext}'), nrow=num_rows)

    @torch.no_grad()
    def generate(self, num=0, num_image_tiles=4, checkpoint=None, types=['default', 'ema']):
        self.GAN.eval()

        latent_dim = self.GAN.latent_dim
        dir_name = self.name + str('-generated-') + str(checkpoint)
        dir_full = Path().absolute() / self.results_dir / dir_name
        ext = self.image_extension

        if not dir_full.exists():
            os.mkdir(dir_full)

        # regular
        if 'default' in types:
            for i in tqdm(range(num_image_tiles), desc='Saving generated default images'):
                latents = torch.randn((1, latent_dim)).cuda(self.rank)
                generated_image = self.generate_(self.GAN.G, latents)
                path = str(self.results_dir / dir_name / f'{str(num)}-{str(i)}.{ext}')
                torchvision.utils.save_image(generated_image[0], path, nrow=1)

        # moving averages
        if 'ema' in types:
            for i in tqdm(range(num_image_tiles), desc='Saving generated EMA images'):
                latents = torch.randn((1, latent_dim)).cuda(self.rank)
                generated_image = self.generate_(self.GAN.GE, latents)
                path = str(self.results_dir / dir_name / f'{str(num)}-{str(i)}-ema.{ext}')
                torchvision.utils.save_image(generated_image[0], path, nrow=1)

        return dir_full

    @torch.no_grad()
    def show_progress(self, num_images=4, types=['default', 'ema']):
        checkpoints = self.get_checkpoints()
        assert checkpoints is not None, 'cannot find any checkpoints to create a training progress video for'

        dir_name = self.name + str('-progress')
        dir_full = Path().absolute() / self.results_dir / dir_name
        ext = self.image_extension
        latents = None

        zfill_length = math.ceil(math.log10(len(checkpoints)))

        if not dir_full.exists():
            os.mkdir(dir_full)

        for checkpoint in tqdm(checkpoints, desc='Generating progress images'):
            self.load(checkpoint, print_version=False)
            self.GAN.eval()

            if checkpoint == 0:
                latents = torch.randn((num_images, self.GAN.latent_dim)).cuda(self.rank)

            # regular
            if 'default' in types:
                generated_image = self.generate_(self.GAN.G, latents)
                path = str(self.results_dir / dir_name / f'{str(checkpoint).zfill(zfill_length)}.{ext}')
                torchvision.utils.save_image(generated_image, path, nrow=num_images)

            # moving averages
            if 'ema' in types:
                generated_image = self.generate_(self.GAN.GE, latents)
                path = str(self.results_dir / dir_name / f'{str(checkpoint).zfill(zfill_length)}-ema.{ext}')
                torchvision.utils.save_image(generated_image, path, nrow=num_images)

    @torch.no_grad()
    def calculate_fid(self, num_batches):
        from pytorch_fid import fid_score
        torch.cuda.empty_cache()

        real_path = self.fid_dir / 'real'
        fake_path = self.fid_dir / 'fake'

        # remove any existing files used for fid calculation and recreate directories
        if not real_path.exists() or self.clear_fid_cache:
            rmtree(real_path, ignore_errors=True)
            os.makedirs(real_path)

            for batch_num in tqdm(range(num_batches), desc='calculating FID - saving reals'):
                real_batch = next(self.loader)
                for k, image in enumerate(real_batch.unbind(0)):
                    ind = k + batch_num * self.batch_size
                    torchvision.utils.save_image(image, real_path / f'{ind}.png')

        # generate a bunch of fake images in results / name / fid_fake

        rmtree(fake_path, ignore_errors=True)
        os.makedirs(fake_path)

        self.GAN.eval()
        ext = self.image_extension

        latent_dim = self.GAN.latent_dim
        image_size = self.GAN.image_size

        for batch_num in tqdm(range(num_batches), desc='calculating FID - saving generated'):
            # latents and noise
            latents = torch.randn(self.batch_size, latent_dim).cuda(self.rank)

            # moving averages
            generated_images = self.generate_(self.GAN.GE, latents)

            for j, image in enumerate(generated_images.unbind(0)):
                ind = j + batch_num * self.batch_size
                torchvision.utils.save_image(image, str(fake_path / f'{str(ind)}-ema.{ext}'))

        return fid_score.calculate_fid_given_paths([str(real_path), str(fake_path)], 256, latents.device, 2048)

    @torch.no_grad()
    def generate_(self, G, style, num_image_tiles = 8):
        generated_images = evaluate_in_chunks(self.batch_size, G, style)
        return generated_images.clamp_(0., 1.)

    @torch.no_grad()
    def generate_interpolation(self, num = 0, num_image_tiles = 8, num_steps = 100, save_frames = False):
        self.GAN.eval()
        ext = self.image_extension
        num_rows = num_image_tiles

        latent_dim = self.GAN.latent_dim
        image_size = self.GAN.image_size

        # latents and noise

        latents_low = torch.randn(num_rows ** 2, latent_dim).cuda(self.rank)
        latents_high = torch.randn(num_rows ** 2, latent_dim).cuda(self.rank)

        ratios = torch.linspace(0., 8., num_steps)

        frames = []
        for ratio in tqdm(ratios):
            interp_latents = slerp(ratio, latents_low, latents_high)
            generated_images = self.generate_(self.GAN.GE, interp_latents)
            images_grid = torchvision.utils.make_grid(generated_images, nrow = num_rows)
            pil_image = torchvision.transforms.ToPILImage()(images_grid.cpu())
            
            if self.transparent:
                background = Image.new('RGBA', pil_image.size, (255, 255, 255))
                pil_image = Image.alpha_composite(background, pil_image)
                
            frames.append(pil_image)

        frames[0].save(str(self.results_dir / self.name / f'{str(num)}.gif'), save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)

        if save_frames:
            folder_path = (self.results_dir / self.name / f'{str(num)}')
            folder_path.mkdir(parents=True, exist_ok=True)
            for ind, frame in enumerate(frames):
                frame.save(str(folder_path / f'{str(ind)}.{ext}'))

    def print_log(self):
        data = [
            ('G', self.g_loss),
            ('D', self.d_loss),
            ('GP', self.last_gp_loss),
            ('SS', self.last_recon_loss),
            ('FID', self.last_fid)
        ]

        data = [d for d in data if d[1] is not None]
        log = ' | '.join(map(lambda n: f'{n[0]}: {n[1]:.2f}', data))
        print(log)

        if self.run is not None:
            for key, value in data:
                self.run.track(value, key, step=self.steps)

        return data

    def model_name(self, num):
        return str(self.models_dir / self.name / f'model_{num}.pt')

    def init_folders(self):
        (self.results_dir / self.name).mkdir(parents=True, exist_ok=True)
        (self.models_dir / self.name).mkdir(parents=True, exist_ok=True)

    def clear(self):
        rmtree(str(self.models_dir / self.name), True)
        rmtree(str(self.results_dir / self.name), True)
        rmtree(str(self.fid_dir), True)
        rmtree(str(self.config_path), True)
        self.init_folders()

    def save(self, num):
        save_data = {
            'GAN': self.GAN.state_dict(),
            'version': __version__,
            'G_scaler': self.G_scaler.state_dict(),
            'D_scaler': self.D_scaler.state_dict()
        }

        torch.save(save_data, self.model_name(num))
        self.write_config()

    def load(self, num=-1, print_version=True):
        self.load_config()

        name = num
        if num == -1:
            checkpoints = self.get_checkpoints()

            if checkpoints is None:
                return

            name = checkpoints[-1]
            print(f'continuing from previous epoch - {name}')

        self.steps = name * self.save_every

        load_data = torch.load(self.model_name(name))

        if print_version and 'version' in load_data and self.is_main:
            print(f"loading from version {load_data['version']}")

        try:
            self.GAN.load_state_dict(load_data['GAN'], strict = self.load_strict)
        except Exception as e:
            saved_version = load_data['version']
            print('unable to load save model. please try downgrading the package to the version specified by the saved model (to do so, just run `pip install lightweight-gan=={saved_version}`')
            raise e

        if 'G_scaler' in load_data:
            self.G_scaler.load_state_dict(load_data['G_scaler'])
        if 'D_scaler' in load_data:
            self.D_scaler.load_state_dict(load_data['D_scaler'])

    def get_checkpoints(self):
        file_paths = [p for p in Path(self.models_dir / self.name).glob('model_*.pt')]
        saved_nums = sorted(map(lambda x: int(x.stem.split('_')[1]), file_paths))

        if len(saved_nums) == 0:
            return None

        return saved_nums
