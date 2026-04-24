"""Vesicle object with biophysical state and transition properties."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from npq.docking import DockingSite


class VesicleState(Enum):
    FREE = auto()
    DOCKED = auto()
    FUSED = auto()


ReleaseRateFn = Callable[[float], float]


@dataclass
class Vesicle:
    docking_rate: float
    fusion_rate: ReleaseRateFn
    quantal_size: float
    recovery_rate: float
    state: VesicleState = VesicleState.FREE
    site: Optional["DockingSite"] = field(default=None, repr=False)

    @property
    def is_free(self) -> bool:
        return self.state is VesicleState.FREE

    @property
    def is_docked(self) -> bool:
        return self.state is VesicleState.DOCKED

    @property
    def is_fused(self) -> bool:
        return self.state is VesicleState.FUSED
