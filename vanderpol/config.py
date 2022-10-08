# my_project/config.py
from yacs.config import CfgNode as CN

_C = CN()

_C.SYSTEM = CN()
_C.SYSTEM.DEVICE = 'cpu'
# _C.SYSTEM.D_TYPE = torch.float64

_C.META = CN()
_C.META.META_LR = 1e-3
_C.META.ADAPT_LR = 1e-5
_C.META.ADAPT_ITER = 10
_C.META.ADAPT_BATCH_SZ = 20
_C.META.META_ITER = 2
_C.META.META_BATCH_SZ = 10
_C.META.USE_ALL_TASKS_PER_EPOCH = False

_C.EMBEDDING = CN()

_C.ENCODER = CN()
_C.ENCODER.RNN = CN()
_C.ENCODER.RNN.HIDDEN_SZ = 16
_C.ENCODER.RNN.HIDDEN_NUM_LAYERS = 1

_C.ENCODER.MLP = CN()
_C.ENCODER.MLP.HIDDEN_SZ = [16]

_C.DECODER = CN()
_C.DECODER.MLP = CN()
_C.DECODER.MLP.HIDDEN_SZ = [16]


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()