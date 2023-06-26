from pathlib import Path
import pickle
from typing import Optional, Mapping
from copy import deepcopy
from multiprocessing import Array
from itertools import cycle, chain
from functools import partial
import uuid
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from datasets import load_dataset, Dataset, IterableDataset
from datasets.info import DatasetInfo
from datasets.iterable_dataset import ExamplesIterable, RandomlyCyclingMultiSourcesExamplesIterable
from transformers import AutoTokenizer, default_data_collator
import torch.distributed as dist
from omegaconf import OmegaConf as om
from tqdm import tqdm
import math
from datasets.filesystems import _reset_fsspec_lock
from datasets.utils.logging import get_logger
from datasets import load_from_disk
import shutil
import transformers
from llmfoundry.data.text_data import StreamingDataset
from streaming import Stream
import os

logger = get_logger(__name__)


class UpdatableRandomlyCyclingMultiSourcesExamplesIterable(
        RandomlyCyclingMultiSourcesExamplesIterable):

    def __init__(self,
                 ex_iterables,
                 generator,
                 probabilities=None,
                 probabilities_file=None,
                 stopping_strategy="all_exhausted"):
        '''
        probabilities: vector of static probabilities over training
        probabilities_file: tmp file to store dynamically changing probabilities
        '''
        super().__init__(ex_iterables,
                         generator,
                         stopping_strategy=stopping_strategy)
        self.probabilities_file = probabilities_file
        self.probabilities = probabilities

    @staticmethod
    def _iter_random_indices(rng,
                             num_sources,
                             probabilities_file=None,
                             probabilities=None,
                             random_batch_size=8192):
        while True:
            # read domain weights
            if probabilities_file is not None:
                with open(probabilities_file, 'rb') as f:
                    probabilities = pickle.load(f)

            yield from (int(i) for i in rng.choice(
                num_sources, size=random_batch_size, p=probabilities))

    def _give_indice_iterator(self):
        rng = deepcopy(self.generator)
        return self._iter_random_indices(
            rng,
            len(self.ex_iterables),
            probabilities_file=self.probabilities_file,
            probabilities=self.probabilities)

    def shard_data_sources(self, shard_indices):
        return self

    @property
    def n_shards(self):
        return 1

    def shuffle_data_sources(self, seed):
        self.ex_iterables = [
            ex_iterable.shuffle_data_sources(seed)
            for ex_iterable in self.ex_iterables
        ]
        return self


def interleave_datasets(datasets,
                        probabilities=None,
                        probabilities_file=None,
                        seed=None,
                        stopping_strategy='all_exhausted'):
    iterable_datasets = []
    for dataset in datasets:
        if not isinstance(dataset, IterableDataset):
            iterable_datasets.append(dataset.to_iterable_dataset())
        else:
            iterable_datasets.append(dataset)

    ex_iterables = [d._ex_iterable for d in iterable_datasets]

    generator = np.random.default_rng(seed)
    ex_iterable = UpdatableRandomlyCyclingMultiSourcesExamplesIterable(
        ex_iterables,
        generator=generator,
        probabilities=probabilities,
        probabilities_file=probabilities_file,
        stopping_strategy=stopping_strategy)

    return IterableDataset(ex_iterable=ex_iterable)


def get_dataset(pile_dir, domain_ids_dir, cache_dir=None, split='train'):
    # initialize streaming datasets from pile_dir
    pile_dir = Path(pile_dir)
    domain_ids_split_dir = Path(domain_ids_dir) / split
    if split == 'train':
        data_files = [
            str(pile_dir / f"{subset}.jsonl.zst") for subset in PILE_SUBSETS
        ]
        domain_ids = [
            np.load(domain_ids_split_dir / f"{subset}_domain_ids.npy")
            for subset in PILE_SUBSETS
        ]
    elif split == 'validation':
        data_files = [str(pile_dir / "val.jsonl.zst")]
        domain_ids = [
            np.load(domain_ids_split_dir / f"{split}_domain_ids.npy")
        ]
    else:
        data_files = [str(pile_dir / f"test.jsonl.zst")]
        domain_ids = [
            np.load(domain_ids_split_dir / f"{split}_domain_ids.npy")
        ]

    ds_ls = []
    for data_file in data_files:
        ds = load_dataset('json',
                          data_files=[data_file],
                          cache_dir=cache_dir,
                          streaming=True)['train']
        ds_ls.append(ds)
    return ds_ls, domain_ids


def get_pile_sharded_datasets(
    preprocessed_dir,
    cache_dir=None,
    split='train',
    sharded=True,
):
    preprocessed_dir = Path(preprocessed_dir) / split
    first_domain_dir = list(preprocessed_dir.iterdir())[0]
    if sharded:
        num_shards = len(list(first_domain_dir.iterdir()))
    else:
        num_shards = 1

    all_ds_shards = [{} for _ in range(num_shards)]
    for domain_dir in preprocessed_dir.iterdir():
        domain_shard_ds_ls = []
        if sharded:
            for shard_idx, shard_dir in enumerate(domain_dir.iterdir()):
                ds = load_from_disk(dataset_path=str(shard_dir))
                all_ds_shards[shard_idx][domain_dir.name] = ds
        else:
            all_ds_shards[0][domain_dir.name] = load_from_disk(
                dataset_path=str(domain_dir))
    return all_ds_shards


def get_perdomain_sharded_datasets(preprocessed_dir,
                                   domain_weights_dict,
                                   cache_dir=None):
    preprocessed_dir = Path(preprocessed_dir)
    num_shards = 1

    def data_gen(shards):
        for shard in shards:
            for ex in shard:
                yield ex

    domains = list(sorted(domain_weights_dict.keys()))

    all_ds_shards = [{} for _ in range(num_shards)]
    for domain in domains:
        domain_dir = preprocessed_dir / domain

        if (domain_dir / 'dataset_info.json').exists():
            print(f"Loading {domain_dir}")
            ds = load_from_disk(dataset_path=str(domain_dir))
            print(f"Length of {domain_dir}: {len(ds)}")
        else:
            curr_shards = []
            for shard_dir in domain_dir.iterdir():
                print(f"Loading {shard_dir}")
                curr_shards.append(load_from_disk(dataset_path=str(shard_dir)))
            ds = IterableDataset.from_generator(
                data_gen, gen_kwargs={'shards': curr_shards})
        all_ds_shards[0][domain] = ds
    return all_ds_shards


PILE_DATA_SOURCES = [
    "Pile-CC", "PubMed Central", "Books3", "OpenWebText2", "ArXiv", "Github",
    "FreeLaw", "StackExchange", "USPTO Backgrounds", "PubMed Abstracts",
    "Gutenberg (PG-19)", "OpenSubtitles", "Wikipedia (en)", "DM Mathematics",
    "Ubuntu IRC", "BookCorpus2", "EuroParl", "HackerNews", "YoutubeSubtitles",
    "PhilPapers", "NIH ExPorter", "Enron Emails"
]


class StreamingTextDataset(StreamingDataset):
    """Generic text dataset using MosaicML's StreamingDataset.

    Args:
        tokenizer (Tokenizer): HuggingFace tokenizer to
            tokenize samples.
        max_seq_len (int): The max sequence length of each sample.
        streams (Sequence[Stream], optional): One or more Streams to stream/cache samples from,
            which may be upsampled or downsampled. StreamingDataset uses either ``streams`` or
            ``remote``/``local``. Defaults to ``None``.
        remote (str, optional): Remote path or directory to download the dataset from. If ``None``,
            its data must exist locally. StreamingDataset uses either ``streams`` or
            ``remote``/``local``. Defaults to ``None``.
        local (str, optional): Local working directory to download shards to. This is where shards
            are cached while they are being used. Uses a temp directory if not set.
            StreamingDataset uses either ``streams`` or ``remote``/``local``. Defaults to ``None``.
        split (str, optional): Which dataset split to use, if any. If provided, we stream from/to
            the ``split`` subdirs of  ``remote`` and ``local``. Defaults to ``None``.
        download_retry (int): Number of download re-attempts before giving up. Defaults to ``2``.
        download_timeout (float): Number of seconds to wait for a shard to download before raising
            an exception. Defaults to ``60``.
        validate_hash (str, optional): Optional hash or checksum algorithm to use to validate
            shards. Defaults to ``None``.
        keep_zip (bool): Whether to keep or delete the compressed form when decompressing
            downloaded shards. If ``False``, keep iff remote is local or no remote. Defaults to
            `False``.
        keep_raw (bool): Whether to keep or delete the decompressed form (or only form)
            of shards after all their samples have been yielded this epoch. If ``False``, keep iff
            remote is local or no remote and no compression. Defaults to ``True``.
        samples_per_epoch (int, optional): Provide this field iff you are weighting sub-datasets
            proportionally. Defaults to ``None``.
        predownload (int, optional): Target number of samples ahead to download the shards of while
            iterating. Defaults to ``100_000``.
        partition_algo (str): Which partitioning algorithm to use. Defaults to ``orig``.
        num_canonical_nodes (int, optional): Canonical number of nodes for shuffling with
            resumption. Defaults to ``None``, which is interpreted as the number of nodes of the
            initial run.
        batch_size (int, optional): Batch size of its DataLoader, which affects how the dataset is
            partitioned over the workers. Defaults to ``None``.
        shuffle (bool): Whether to iterate over the samples in randomized order. Defaults to
            ``False``.
        shuffle_algo (str): Which shuffling algorithm to use. Defaults to ``py1b``.
        shuffle_seed (int): Seed for Deterministic data shuffling. Defaults to ``9176``.
        shuffle_block_size (int): Unit of shuffle. Defaults to ``1 << 18``.
    """

    def __init__(self,
                 tokenizer,
                 max_seq_len: int,
                 streams=None,
                 remote: Optional[str] = None,
                 local: Optional[str] = None,
                 split: Optional[str] = None,
                 download_retry: int = 2,
                 download_timeout: float = 60,
                 validate_hash: Optional[str] = None,
                 keep_zip: bool = False,
                 keep_raw: bool = True,
                 samples_per_epoch: Optional[int] = None,
                 predownload: int = 100_000,
                 partition_algo: str = 'orig',
                 num_canonical_nodes: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 shuffle: bool = False,
                 shuffle_algo: str = 'py1b',
                 shuffle_seed: int = 9176,
                 shuffle_block_size: int = 1 << 18,
                 **kwargs):

        group_method = kwargs.pop('group_method', None)
        if group_method is not None:
            raise NotImplementedError(
                'group_method is deprecated and has been removed.\nTo ' +
                'concatenate, use the --concat_tokens ' +
                'argument when creating your MDS dataset with concat_c4.py')

        if kwargs is not None and len(kwargs) > 0:
            raise ValueError(
                f'StreamingTextDataset() got an unexpected keyword argument: {kwargs}'
            )

        if local is not None and (remote is None or (local == remote)):
            if os.path.isdir(local):
                contents = set(os.listdir(local))
                if split not in contents:
                    raise ValueError(
                        f'local directory {local} does not contain split {split}'
                    )

        # Build Dataset
        super().__init__(
            streams=streams,
            remote=remote,
            local=local,
            split=split,
            download_retry=download_retry,
            download_timeout=download_timeout,
            validate_hash=validate_hash,
            keep_zip=keep_zip,
            keep_raw=keep_raw,
            samples_per_epoch=samples_per_epoch,
            predownload=predownload,
            partition_algo=partition_algo,
            num_canonical_nodes=num_canonical_nodes,
            batch_size=batch_size,
            shuffle=shuffle,
            shuffle_algo=shuffle_algo,
            shuffle_seed=shuffle_seed,
        )
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    # How to tokenize a text sample to a token sample
    def _tokenize(self, text_sample):
        if self.tokenizer._pad_token is None:
            # Some tokenizers (e.g. GPT2 tokenizer) have no padding token which causes bugs
            raise RuntimeError(
                'If tokenizing on-the-fly, tokenizer must have a pad_token_id')

        return self.tokenizer(text_sample['text'],
                              truncation=True,
                              padding='max_length',
                              max_length=self.max_seq_len)

    def _read_binary_tokenized_sample(self, sample):
        return torch.from_numpy(
            np.frombuffer(sample['tokens'],
                          dtype=np.int64)[:self.max_seq_len].copy())

    def _read_binary_reference_losses(self, sample):
        losses = np.frombuffer(sample['ref_losses'],
                               dtype=np.float16)[:self.max_seq_len].copy()
        return torch.from_numpy(losses).type(torch.float32)

    # How to process a sample
    def __getitem__(self, idx: int):
        sample = super().__getitem__(idx)
        if 'text' in sample:
            token_sample = self._tokenize(sample)
        elif 'domain_idx' in sample:
            token_sample = {
                "tokens": self._read_binary_tokenized_sample(sample),
                "domain_idx": sample["domain_idx"]
            }
        elif 'tokens' in sample:
            token_sample = self._read_binary_tokenized_sample(sample)
        else:
            raise RuntimeError(
                'StreamingTextDataset needs samples to have a `text` or `tokens` column'
            )
        return token_sample


def get_preprocessed_mixed_dataset(split,
                                   tokenizer,
                                   device_batch_size: int,
                                   uniform=False):

    cfg = {
        "streams":
        [{
            "remote":
            f"oci://mosaicml-internal-doremi/pile/pre-concat/gpt-neox-20b-seqlen-2048/data-sources/baseline-100K-samples/domain-{domain_idx}",
            "local": f"/tmp/streaming/domains/domain-{domain_idx}",
            "split": split,
            "proportion": (1 / 22) if uniform else None
        } for domain_idx in range(22)
         ],  # Skip proportion for now since fixing to be baseline dataset like they do
        "shuffle":
        True,
        "shuffle_seed":
        17,
        "eos_token_id":
        0,
        "max_seq_len":
        2048
    }  # Hard setting to seed 17 for now
    cfg = om.create(cfg)

    # build streams
    streams_dict = cfg.get('streams', None)
    streams = None
    if streams_dict is not None:
        streams = []
        for stream in streams_dict:
            streams.append(**stream)
            #streams.append(
            #    Stream(
            #        remote=stream.get('remote', None)
            #        or cfg.get('remote', None),
            #        local=stream.get('local', None) or cfg.get('local', None),
            #        split=stream.get('split', None) or cfg.get('split', None),
            #        proportion=stream.get('proportion', None),
            #        repeat=stream.get('repeat', None),
            #        download_retry=stream.get('download_retry', None)
            #        or cfg.get('download_retry', 2),
            #        download_timeout=stream.get('download_timeout', None)
            #        or cfg.get('download_timeout', 60),
            #        validate_hash=stream.get('validate_hash', None)
            #        or cfg.get('validate_hash', None),
            #        keep_zip=stream.get('keep_zip', None)
            #        or cfg.get('keep_zip', False),
            #        keep_raw=stream.get('keep_raw', None)
            #        or cfg.get('keep_raw', True),
            #    ))

    # build dataset potentially with streams
    dataset = StreamingTextDataset(
        tokenizer=tokenizer,
        max_seq_len=cfg.max_seq_len,
        streams=streams,
        remote=cfg.get('remote', None),
        local=cfg.get('local', None),
        split=cfg.get('split', None),
        download_retry=cfg.get('download_retry', 2),
        download_timeout=cfg.get('download_timeout', 60),
        validate_hash=cfg.get('validate_hash', None),
        keep_zip=cfg.get('keep_zip', False),
        keep_raw=cfg.get('keep_raw', True),
        samples_per_epoch=cfg.get('samples_per_epoch', None),
        predownload=cfg.get('predownload', 100_000),
        partition_algo=cfg.get('partition_algo', 'orig'),
        num_canonical_nodes=cfg.get('num_canonical_nodes', 128),
        batch_size=device_batch_size,
        shuffle=cfg.get('shuffle', False),
        shuffle_algo=cfg.get('shuffle_algo', 'py1b'),
        shuffle_seed=cfg.get('shuffle_seed', 9176),
        shuffle_block_size=cfg.get('shuffle_block_size', 1 << 18),
    )

    return dataset


class ConcatenatedSequenceCollatorWrapper:
    """Collator wrapper to add sequence_id to batch."""

    def __init__(
        self,
        base_collator,
        eos_token_id=None,
        bos_token_id=None,
    ):
        self.base_collator = base_collator
        if (eos_token_id is None) and (bos_token_id is None):
            raise ValueError(
                'Must supply a value for either eos_token_id or bos_token_id, but got None for both.'
            )
        if (eos_token_id is not None) and (bos_token_id is not None):
            raise ValueError(
                'Cannot use *both* EOS and BOS tokens for detecting sequence boundaries. ' +\
                'Please supply `eos_token_id` if sequences end with an EOS token, or use ' +\
                '`bos_token_id` if sequences start with a BOS token.'
            )

        self.split_token_id = eos_token_id
        self.bos_mode = False
        if eos_token_id is None:
            self.split_token_id = bos_token_id
            self.bos_mode = True

    def __call__(self, examples):
        if isinstance(examples[0], Mapping):
            batch = self.base_collator(
                [example["tokens"] for example in examples])
        else:
            batch = self.base_collator(examples)
        batch['sequence_id'] = self.get_sequence_id_from_batch(batch)
        if isinstance(examples[0], Mapping):
            batch["domain_idx"] = torch.tensor(
                [example["domain_idx"] for example in examples])
        return batch

    def get_sequence_id_from_batch(self, batch) -> torch.Tensor:
        is_separator = torch.eq(batch['input_ids'],
                                self.split_token_id)  # type: ignore
        cumulative_sep = torch.cumsum(is_separator,
                                      dim=1).to(batch['input_ids'].dtype)
        # If separator token is bos, we're already done
        if self.bos_mode:
            return cumulative_sep

        # If separator token is eos, right shift 1 space
        left_zeros = cumulative_sep.new_zeros((cumulative_sep.shape[0], 1))
        return torch.cat([left_zeros, cumulative_sep[:, :-1]], dim=1)


def get_data_collator(tokenizer,
                      eos_token_id,
                      return_tensors='pt',
                      do_padding=False):

    collate_fn = transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, mlm_probability=None)

    eos_token_id = 0  # Hard set because using neo-x tokenizer
    bos_token_id = None
    collate_fn = ConcatenatedSequenceCollatorWrapper(base_collator=collate_fn,
                                                     eos_token_id=eos_token_id,
                                                     bos_token_id=bos_token_id)

    return collate_fn


if __name__ == "__main__":
    # a short test

    PILE_DOMAINS = [
        'ArXiv', 'BookCorpus2', 'Books3', 'DM Mathematics', 'Enron Emails',
        'EuroParl', 'FreeLaw', 'Github', 'Gutenberg (PG-19)', 'HackerNews',
        'NIH ExPorter', 'OpenSubtitles', 'OpenWebText2', 'PhilPapers',
        'Pile-CC', 'PubMed Abstracts', 'PubMed Central', 'StackExchange',
        'USPTO Backgrounds', 'Ubuntu IRC', 'Wikipedia (en)', 'YoutubeSubtitles'
    ]

    DOMAIN_TO_IDX = {name: idx for idx, name in enumerate(PILE_DOMAINS)}

    PILE_SUBSETS = [f'0{i}' if i < 10 else str(i) for i in range(0, 30)]

    domain_weights_dict = {domain: 1 for domain in PILE_DOMAINS}
    ds, domain_weights = get_preprocessed_mixed_dataset(
        preprocessed_dir=
        '/path/to/preprocessed',  # run filter_domains.py in scripts/
        domain_weights_dict=domain_weights_dict,
        cache_dir='/path/to/cache',
        split='train',
        sharded=True)

    tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    dataloader = DataLoader(ds,
                            batch_size=512,
                            num_workers=1,
                            collate_fn=get_data_collator(tokenizer))

    domain_weights_dict_2 = {
        domain: 1 if domain == 'Books3' else 0
        for domain in PILE_DOMAINS
    }
    domain_weights_2_vec = torch.tensor(list(domain_weights_dict_2.values()))
    domain_weights_2_vec = domain_weights_2_vec / domain_weights_2_vec.sum()
    phase_1_domains = [0] * len(PILE_DOMAINS)
    phase_2_domains = [0] * len(PILE_DOMAINS)
    for i, batch in tqdm(enumerate(dataloader)):
        if i < 500:
            for domain_id in batch['domain_ids']:
                phase_1_domains[domain_id] += 1
        elif i < 1000:
            if i == 500:
                # dataloader.dataset._ex_iterable.set_domain_weights(domain_weights_2_vec)
                domain_weights[:] = domain_weights_2_vec[:]
            for domain_id in batch['domain_ids']:
                phase_2_domains[domain_id] += 1
        else:
            break

    phase_1_domains = np.asarray(phase_1_domains)
    phase_2_domains = np.asarray(phase_2_domains)
    print("Phase 1")
    print({
        domain: count / phase_1_domains.sum()
        for domain, count in zip(PILE_DOMAINS, phase_1_domains)
    })

    print("Phase 2")
    print({
        domain: count / phase_2_domains.sum()
        for domain, count in zip(PILE_DOMAINS, phase_2_domains)
    })
