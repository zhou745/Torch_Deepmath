import TorchDeepmath as td
import torch
from tqdm import tqdm
import os

td.Data.gen_from_tfrecord.tfgen("/mnt/lustre/zhoujingqiu/deephol-data/deepmath/deephol/proofs/human/",
                                "/mnt/cache/zhoujingqiu/data_goal_lr")
# td.Data.gen_from_tfrecord.text_gen("/mnt/lustre/zhoujingqiu/deephol-data/deepmath/deephol/proofs/human/",
#                                    "/mnt/cache/zhoujingqiu/data_thmlr")