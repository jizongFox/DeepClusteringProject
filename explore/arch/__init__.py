from deepclustering.arch import _register_arch

from .net5g_two_head import ClusterNet5gTwoHead
from .net6c_two_head import ClusterNet6cTwoHead

_register_arch("clusternet5gtwohead_sn", ClusterNet5gTwoHead)
_register_arch("clusternet6ctwohead_sn", ClusterNet6cTwoHead)
