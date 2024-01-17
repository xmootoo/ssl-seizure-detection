from pydantic import BaseModel, Field
from typing import List, Union, Optional

class ModelConfig(BaseModel):
    num_node_features: int
    num_edge_features: int
    hidden_channels: List[int]
    dropout: Optional[float] = None
    p: Optional[float] = None
    batch_norm: Optional[bool] = False

class LossConfig(BaseModel):
    loss_coeffs: List[float]
    TSF_scale: bool=True 
    gamma: float=1
    epsilon: float=1e-4
class TrainConfig(BaseModel):
    data_path: str
    logdir: str
    patient_id: str
    epochs: int
    data_size: float = Field(..., gt=0)
    val_ratio: float
    test_ratio: float
    batch_size: int
    num_workers: int
    lr: Union[List[float], float]
    weight_decay: float
    model_id: str
    timing: bool
    project_id: str
    patience: int
    eta_min: float
    exp_id: float
    run_type: str
    datetime_id: Optional[str] = None
    model_path: Optional[str] = None
    model_dict_path: Optional[str] = None
    transfer_id: Optional[str] = None
    train_ratio: Optional[float] = None
    requires_grad: Optional[bool] = None
    classify: Optional[str] = None
    head: Optional[str] = None
    
class ExperimentalConfig(BaseModel):
    pass


