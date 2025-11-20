import functools
import os
import random
import logging
from typing import Literal

from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from typing import Union
from lm_eval.utils import simple_parse_args_string

import torch
import torch.distributed as dist

from gidd.utils import parse_dtype
from gidd.loss import get_loss
from gidd.checkpoints import load_checkpoint
from gidd.likelihood import ELBO, compute_elbo, compute_causal_nll


logger = logging.getLogger(__name__)

@register_model("gidd")
class GiddModel(LM):
    def __init__(
        self,
        model_path: str,
        num_samples: int = 32,
        completion_only: bool = False,
        device: Union[str, torch.device] | None = None,
        batch_size: str | int = 1,
        **kwargs,
    ) -> None:
        # super init
        super().__init__()

        # attributes
        self.model_path = model_path
        self.completion_only = completion_only
        self.num_samples = num_samples
        self.batch_size = int(batch_size)
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.completion_only:
            raise NotImplementedError("completion_only is not supported")

        if "LOCAL_RANK" in os.environ:
            self._rank = int(os.environ["LOCAL_RANK"])
            self._world_size = int(os.environ["WORLD_SIZE"])
            dist.init_process_group(backend="nccl", rank=self.rank, world_size=self.world_size)
            torch.cuda.set_device(self.rank)
            self.device = torch.device("cuda", self.rank)

        # print the model path
        logger.info(f"[RANK: {self.rank}] Loading model from {model_path}")

        # # torch stuff
        # torch.set_float32_matmul_precision('high')

        # load the model
        model, noise_schedule, tokenizer, config = load_checkpoint(model_path, device=self.device)
        model.eval()
        tokenizer.truncation_side = "left"
        tokenizer.padding_side = "right"

        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.dtype = parse_dtype(self.config.training.dtype)

        # parse the dtype the model was trained in
        self.dtype = parse_dtype(self.config.training.dtype)

        # construct the likelihood estimator
        if config.model.type == "autoregressive":
            model = torch.compile(model)
            self.nll_func = functools.partial(
                compute_causal_nll,
                model,
                return_token_nlls=True,
            )
        else:
            loss_fn = get_loss(config, tokenizer, noise_schedule)
            likelihood = ELBO(config, model, noise_schedule, loss_fn)
            likelihood = likelihood.to(self.device)

            # compile for better efficiency
            self.likelihood = torch.compile(likelihood)

            self.nll_func = functools.partial(
                compute_elbo,
                self.likelihood,
                num_samples=self.num_samples,  # number of inner samples in the estimator, higher number has lower bias and variance
                t_eps=self.config.model.get("t_eps", 1e-5),  # time epsilong for the noise schedule
                show_progress=False,  # turn on/off the progress bar
                return_token_nlls=True,  # set to True to return the token-level nlls
            )

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        logger.info(additional_config)
        args = {}
        if additional_config is not None:
            args = additional_config
        args.update(simple_parse_args_string(arg_string))
        return cls(**args)
    
    def prepare_tokens(self, xs, ys):
        texts = [x + y for x, y in zip(xs, ys)]
        batch = self.tokenizer(texts, padding="max_length", truncation=True, max_length=self.config.model.max_seq_len, return_tensors="pt")
        if batch["input_ids"].shape[1] > self.config.model.max_seq_len:
            batch["input_ids"] = batch["input_ids"][:, :self.config.model.max_seq_len]
        batch["loss_mask"] = batch["attention_mask"]
        return batch

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        res = []
        
        # create the batches input
        strings = [(r.args[0], r.args[1]) for r in requests]
        batched = [strings[i:i + self.batch_size] for i in range(0, len(strings), self.batch_size)]
        for batch in tqdm(batched, disable=disable_tqdm):
            # load a batch
            bs = len(batch)
            if bs < self.batch_size:
                # pad to make sure we keep the same shape
                batch += [("", "")] * (self.batch_size - bs)
            # batch = self.tokenizer(batch, return_tensors="pt", padding="max_length", truncation=True, max_length=self.config.model.max_seq_len)
            batch = self.prepare_tokens(*zip(*batch))
            batch = batch.to(self.device)

            with torch.no_grad(), torch.autocast(self.device.type, self.dtype):
                _, token_nlls = self.nll_func(batch)

                # shape of token_nlls: (batch_size, max_seq_len)
                # also includes NLL for padding tokens, can be masked like this:
                if self.completion_only:
                    loss_mask = batch["loss_mask"][..., :token_nlls.size(-1)]
                else:
                    loss_mask = batch["attention_mask"][..., :token_nlls.size(-1)]
                nll = (token_nlls * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1)

                # remove batch padding
                ll = -nll[:bs]
                is_greedy = True
                for x in ll.cpu().numpy():
                    res.append((x, is_greedy))
        
        return res

    def generate_until(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("generate_until is not implemented for semantic_diffusion")
        res = []

        for request in tqdm(requests, disable=disable_tqdm):
            res.append("lol")
            assert request.arguments[0].strip() != ""

        return res

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("loglikelihood_rolling is not implemented for semantic_diffusion")
        res = []

        for _ in tqdm(requests, disable=disable_tqdm):
            res.append(-random.random())

        return res

    @property
    def accelerator(self):
        return self._Accelerator(self.world_size)

    class _Accelerator:
        def __init__(self, world_size):
            self.world_size = world_size

        def wait_for_everyone(self):
            dist.barrier()

        def gather(self, local_tensor):
            gathered_tensors = [
                torch.zeros_like(local_tensor)
                for _ in range(self.world_size)
            ]
            dist.all_gather(gathered_tensors, local_tensor)
            if local_tensor.dim() < 1:
                return torch.stack(gathered_tensors)
            return torch.cat(gathered_tensors)
