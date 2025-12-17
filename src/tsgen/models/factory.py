from tsgen.models.unet import UNet1D
from tsgen.models.transformer import DiffusionTransformer
from tsgen.models.mamba import MambaDiffusion
from tsgen.models.baselines import MultivariateGBM, BootstrapGenerativeModel
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
    
    # Extract params source: check config['model']['params'] first, then config root
    params = config
    if 'model' in config and isinstance(config['model'], dict) and 'params' in config['model']:
        params = config['model']['params']
        
    # Extract common params safely
    # data section might be in config['data'] or root
    data_conf = config.get('data', config)
    seq_len = data_conf.get('sequence_length')
    
    features = None
    if 'tickers' in data_conf:
        features = len(data_conf['tickers'])
    # Fallback for when tickers might be in root
    if features is None and 'tickers' in config:
        features = len(config['tickers'])
    
    if model_type == 'unet':
        base_channels = params.get('base_channels', 64)
        return UNet1D(sequence_length=seq_len, features=features, base_channels=base_channels)
    
    elif model_type == 'transformer':
        return DiffusionTransformer(
            sequence_length=seq_len,
            features=features,
            dim=params.get('dim', 64),
            depth=params.get('depth', 4),
            heads=params.get('heads', 4),
            mlp_dim=params.get('mlp_dim', 128),
            dropout=params.get('dropout', 0.0),
            num_classes=params.get('num_classes', 0)
        )
    
    elif model_type == 'mamba':
        return MambaDiffusion(
            sequence_length=seq_len,
            features=features,
            dim=params.get('dim', 128),
            depth=params.get('depth', 4),
            d_state=params.get('d_state', 16),
            d_conv=params.get('d_conv', 4),
            expand=params.get('expand', 2),
            num_classes=params.get('num_classes', 0)
        )
    
    elif model_type == 'multivariate_gbm':
        # New unified model with configurable covariance
        full_covariance = params.get('full_covariance', True)
        return MultivariateGBM(features=features, full_covariance=full_covariance)

    elif model_type == 'gbm':
        # Backward compatibility: independent sampling (no correlations)
        return MultivariateGBM(features=features, full_covariance=False)

    elif model_type == 'multivariate_lognormal':
        # Backward compatibility: full covariance (with correlations)
        return MultivariateGBM(features=features, full_covariance=True)

    elif model_type == 'bootstrap':
        return BootstrapGenerativeModel(features=features, sequence_length=seq_len)

    elif model_type == 'timevae':
        return TimeVAE(
            features=features,
            sequence_length=seq_len,
            hidden_dim=params.get('hidden_dim', 64),
            latent_dim=params.get('latent_dim', 16),
            encoder_type=params.get('encoder_type', 'lstm'),
            num_layers=params.get('num_layers', 2)
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}")
