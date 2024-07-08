from dataclasses import dataclass


@dataclass
class LadderOperator:
    particle_type: int
    mode: int
    creation: bool
