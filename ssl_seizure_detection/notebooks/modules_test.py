from ssl_seizure_detection.src.config import ModelConfig


model_config = ModelConfig(
    num_node_features=9,
    num_edge_features=3,
    hidden_channels=[64, 128, 128, 512, 512, 512],
    batch_norm=True,
    dropout=True,
    p=0.1,
)

print(model_config)