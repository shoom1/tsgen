from tsgen.models.unet import UNet1D
from tsgen.models.transformer import DiffusionTransformer
from tsgen.models.baselines import GBMGenerativeModel, BootstrapGenerativeModel, MultivariateLogNormalModel
from tsgen.models.timevae import TimeVAE

def create_model(config):
    """
    Factory function to create models based on configuration.
    
    Args:
        config (dict): Configuration dictionary. 
                       Must contain 'model' key with 'type' and other parameters.
    
    Returns:
        GenerativeModel: An instantiated model.
    """
    # Support flat config for now (backward compatibility) or nested 'model' config
    model_type = config.get('model_type', 'unet')
    
    # Extract common params safely
    seq_len = config.get('sequence_length')
    features = None
    if 'tickers' in config:
        features = len(config['tickers'])
    
    if model_type == 'unet':
        base_channels = config.get('base_channels', 64)
        return UNet1D(sequence_length=seq_len, features=features, base_channels=base_channels)
    
    elif model_type == 'transformer':
        return DiffusionTransformer(
            sequence_length=seq_len,
            features=features,
            dim=config.get('dim', 64),
            depth=config.get('depth', 4),
            heads=config.get('heads', 4),
            mlp_dim=config.get('mlp_dim', 128),
            dropout=config.get('dropout', 0.0),
            num_classes=config.get('num_classes', 0)
        )
    
    elif model_type == 'gbm':
        return GBMGenerativeModel(features=features)

    elif model_type == 'bootstrap':
        return BootstrapGenerativeModel(features=features, sequence_length=seq_len)

    elif model_type == 'multivariate_lognormal':
        return MultivariateLogNormalModel(features=features)

    elif model_type == 'timevae':
        return TimeVAE(
            features=features,
            sequence_length=seq_len,
            hidden_dim=config.get('hidden_dim', 64),
            latent_dim=config.get('latent_dim', 16),
            encoder_type=config.get('encoder_type', 'lstm'),
            num_layers=config.get('num_layers', 2)
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}")
