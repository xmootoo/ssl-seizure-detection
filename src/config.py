from pydantic import BaseModel, Field
from typing import List

from pydantic import BaseModel, Field
from typing import List, Union, Optional

class ModelConfig(BaseModel):
    num_node_features: int
    num_edge_features: int
    hidden_channels: List[int]
    classify: Optional[str] = None
    head: Optional[str] = None

class LossConfig(BaseModel):
    # Define loss config parameters here
    pass

class TrainingConfig(BaseModel):
    data_path: str
    logdir: str
    patient_id: str
    epochs: int
    config: ModelConfig
    data_size: float = Field(..., gt=0)
    val_ratio: float
    test_ratio: float
    batch_size: int
    num_workers: int
    lr: Union[List[float], float]
    weight_decay: float
    model_id: str
    timing: bool
    dropout: bool
    datetime_id: Optional[str] = None
    run_type: str
    requires_grad: bool
    model_path: Optional[str] = None
    model_dict_path: Optional[str] = None
    transfer_id: Optional[str] = None
    train_ratio: Optional[float] = None
    loss_config: Optional[LossConfig] = None
    project_id: str
    patience: int
    eta_min: float
