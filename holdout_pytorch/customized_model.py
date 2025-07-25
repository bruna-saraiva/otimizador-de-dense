# customized_model.py
from model import DenseNet  # Assuming your DenseNet class is in model.py

def get_model(input_shape, num_blocks, num_layers_per_block, growth_rate,
              dropout_rate, compress_factor, num_filters, num_classes, se_config):
    """
    Wrapper function to create and return the DenseNet model
    """
    model = DenseNet(
        input_shape=input_shape,
        num_blocks=num_blocks,
        num_layers_per_block=num_layers_per_block,
        growth_rate=growth_rate,
        dropout_rate=dropout_rate,
        compress_factor=compress_factor,
        num_filters=num_filters,
        num_classes=num_classes,
        se_config=se_config
    )
    return model