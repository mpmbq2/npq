"""DockingSite: a position in the active zone that holds at most one vesicle."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from npq.vesicle import Vesicle, VesicleState


@dataclass
class DockingSite:
    occupant: Optional[Vesicle] = field(default=None)

    @property
    def is_empty(self) -> bool:
        return self.occupant is None

    def bind(self, vesicle: Vesicle) -> None:
        if self.occupant is not None:
            raise RuntimeError("docking site is already occupied")
        if vesicle.state is not VesicleState.FREE:
            raise RuntimeError("only FREE vesicles may dock")
        self.occupant = vesicle
        vesicle.site = self
        vesicle.state = VesicleState.DOCKED

    def release(self) -> Vesicle:
        if self.occupant is None:
            raise RuntimeError("docking site is empty")
        v = self.occupant
        v.site = None
        self.occupant = None
        return v
