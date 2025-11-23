import random
import logging
from typing import Literal

import easydel as ed
import jax
import jax.experimental.multihost_utils as mh
import tqdm.auto as tqdm

from gidd_easydel.loading import load_checkpoint
from gidd_easydel.sampling import generate
from gidd_easydel.likelihood import likelihood

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.utils import simple_parse_args_string


logger = logging.getLogger(__name__)

@register_model("gidd")
class GiddModel(LM):
    def __init__(
        self,
        checkpoint_dir: str,
        num_layers: int,
        hidden_size: int,
        num_attn_heads: int,
        hybrid_mixing_shift: float,
        prior_distribution: Literal["mask", "uniform"],
        max_sequence_length: int = 2048,
        sharding: str = "1,-1,1,1,1",
        num_denoising_steps: int = 128,
        min_completion_length: int = 128,
        max_completion_length: int = 128,
        noise_schedule: Literal["linear", "cosine"] = "cosine",
        sampler: Literal["ancestral", "adaptive"] = "adaptive",
        top_k: int = 1,
        temperature: float = 0.0,
        batch_size: str | int = 1,
        seed: int = 0,
        completion_only: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        num_procs = jax.process_count()
        num_local_devices = jax.local_device_count()
        num_devices = jax.device_count()
        logger.info("Process count: %d, local device count: %d, process index: %d, global device count: %d",
                num_procs, num_local_devices, jax.process_index(), num_devices)
        
        self.is_main_process = jax.process_index() == 0

        # attributes
        self.checkpoint_dir = checkpoint_dir
        self.max_sequence_length = int(max_sequence_length)
        self.hybrid_mixing_shift = float(hybrid_mixing_shift)
        self.prior_distribution = prior_distribution
        self.noise_schedule = noise_schedule
        self.sampler = sampler
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self.num_denoising_steps = int(num_denoising_steps)
        self.min_completion_length = int(min_completion_length)
        self.max_completion_length = int(max_completion_length)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.completion_only = completion_only
        
        sharding_axis_dims = [int(s) for s in sharding.split(",")]
        self.mesh, self.module, self.tokenizer, self.dtype = load_checkpoint(
            checkpoint_dir,
            num_layers=int(num_layers),
            hidden_size=int(hidden_size),
            num_attn_heads=int(num_attn_heads),
            max_seq_len=self.max_sequence_length,
            sharding_axis_dims=sharding_axis_dims,
        )

        self.rng = random.Random(self.seed)

        if self.completion_only:
            raise NotImplementedError("completion_only is not supported")


    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        logger.info(additional_config)
        args = {}
        if additional_config is not None:
            args = additional_config
        args.update(simple_parse_args_string(arg_string))
        return cls(**args)

    def generate_until(self, requests, disable_tqdm: bool = False):
        res = []

        # logger.info(requests[0])

        stop_sequences = requests[0].args[1].get("until", [])

        # first_iteration = False

        # res = [f"#### {i}\n" for i in range(len(requests))]
        # return res

        strings = [r.args[0] for r in requests]
        batched = [strings[i:i + self.batch_size] for i in range(0, len(strings), self.batch_size)]
        with tqdm.tqdm(total=len(batched) * self.batch_size, disable=disable_tqdm or not self.is_main_process) as pbar:
            for batch in batched:
                bs = len(batch)
                if bs < self.batch_size:
                    # pad to make sure we keep the same shape
                    batch += [""] * (self.batch_size - bs)

                seed_i = self.rng.randint(0, 2**32 - 1)

                # if first_iteration:
                #     logger.info(batch[0])

                completions = generate(
                    self.mesh,
                    self.module,
                    self.tokenizer,
                    prompts=batch,
                    hybrid_mixing_shift=self.hybrid_mixing_shift,
                    prior=self.prior_distribution,
                    num_denoising_steps=self.num_denoising_steps,
                    max_sequence_length=self.max_sequence_length,
                    min_completion_length=self.min_completion_length,
                    max_completion_length=self.max_completion_length,
                    noise_schedule=self.noise_schedule,
                    sampler=self.sampler,
                    top_k=self.top_k,
                    temperature=self.temperature,
                    seed=seed_i,
                    show_progress=False,
                )

                for i in range(bs):
                    for stop_seq in stop_sequences:
                        if stop_seq in completions[i]:
                            completions[i] = completions[i].split(stop_seq)[0]
                    res.append(completions[i])

                # if first_iteration:
                #     logger.info(completions[0])
                #     first_iteration = False

                pbar.update(self.batch_size)

        return res

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        res = []

        strings = [(r.args[0], r.args[1]) for r in requests]
        batched = [strings[i:i + self.batch_size] for i in range(0, len(strings), self.batch_size)]
        with tqdm.tqdm(total=len(batched) * self.batch_size, disable=disable_tqdm or not self.is_main_process) as pbar:
            for batch in batched:
                bs = len(batch)
                if bs < self.batch_size:
                    # pad to make sure we keep the same shape
                    batch += [("", "")] * (self.batch_size - bs)

                seed_i = self.rng.randint(0, 2**32 - 1)

                prompts = [b[0] for b in batch]
                completions = [b[1] for b in batch]

                metrics = likelihood(
                    self.mesh,
                    self.module,
                    self.tokenizer,
                    prompts=prompts,
                    completions=completions,
                    hybrid_mixing_shift=self.hybrid_mixing_shift,
                    prior=self.prior_distribution,
                    num_denoising_steps=self.num_denoising_steps,
                    max_sequence_length=self.max_sequence_length,
                    noise_schedule="linear",
                    seed=seed_i,
                    show_progress=False,
                )

                for i in range(bs):
                    log_likelihood = -metrics["total_nll"][i].item()
                    is_greedy = False  # impossible to tell
                    answer = (log_likelihood, is_greedy)
                    res.append(answer)

                pbar.update(self.batch_size)

        return res

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("loglikelihood_rolling is not implemented for GiddModel")
        res = []

        for _ in tqdm.tqdm(requests, disable=disable_tqdm):
            res.append(-random.random())

        return res

    @property
    def accelerator(self):
        return self._Accelerator()

    class _Accelerator:
        def wait_for_everyone(self):
            # sync devices
            mh.sync_global_devices()

        def gather(self, local_tensor):
            # input: local tensor of shape (N, ...)
            # output: tensor of shape (world_size * N, ...)
            # edge case where local tensor is a scalar: output shape (world_size,)
            # ==> jax already handles this for us
            return local_tensor
