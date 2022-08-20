from typing import Union

from src.train import TrainerSeparate, TrainerAttention
from src.collection import CollectionSeparate, CollectionAttention

TrainerType = Union[TrainerSeparate, TrainerAttention]
CollectionType = Union[CollectionSeparate, CollectionAttention]
