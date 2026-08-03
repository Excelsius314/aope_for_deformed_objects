from dataclasses import dataclass
from enum import Enum

@dataclass
class TrainConfig:
    num_epochs : int = 200
    log_frequency : int = 500 # log every log_frequency iterations of training loop
    batch_size = 10

    compare_idx = 10

    runs_dir = "/home/marek/aope_for_deformed_objects/aope/runs"
    run_name = "valve_no_handle"

    similarity_weight = 1#0.5
    base_rotation_weight= 3
    part_rotation_weight= 2
    canonical_zero_centered_weight= 2
    canonical_normed_scale_weight = 0#3#10.0
    canonical_consistency_weight= 0.5
