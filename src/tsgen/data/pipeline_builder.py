"""
Pipeline builder for composable data preprocessing.

Provides functional composition pattern (like torchvision.transforms.Compose)
for data pipelines with YAML configuration support.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Any, List, Optional, Type
import inspect


@dataclass
class PipelineStep:
    """Wrapper for composable pipeline functions.

    Wraps a function with metadata for validation and parameter handling.
    Supports merging config-level and runtime-level parameters.

    Attributes:
        name: Step name (e.g., 'load_prices', 'clean_data')
        fn: The actual function to execute
        params: Parameters from config (e.g., {' column': 'adj_close'})
        input_type: Expected input type (None for first step)
        output_type: Output type for validation
        required_params: List of required parameter names
        optional_params: List of optional parameter names

    Example:
        >>> step = PipelineStep(
        ...     name='clean_data',
        ...     fn=clean_data,
        ...     params={'strategy': 'ffill_drop'},
        ...     input_type=pd.DataFrame,
        ...     output_type=pd.DataFrame
        ... )
        >>> result = step(df)  # Execute with DataFrame input
    """
    name: str
    fn: Callable
    params: Dict[str, Any]
    input_type: Optional[Type] = None
    output_type: Optional[Type] = None
    required_params: List[str] = field(default_factory=list)
    optional_params: List[str] = field(default_factory=list)
    _fn_params: frozenset = field(init=False, repr=False)

    def __post_init__(self):
        self._fn_params = frozenset(inspect.signature(self.fn).parameters)

    def __call__(self, data, **runtime_params):
        """Execute step with merged config and runtime params.

        Args:
            data: Input data from previous step (None for first step)
            **runtime_params: Runtime parameters (e.g., tickers, dates, processor)

        Returns:
            Output from function execution

        Raises:
            ValueError: If required parameters are missing
        """
        # Merge config params with runtime params (runtime takes precedence)
        all_params = {**self.params, **runtime_params}

        # Validate required parameters
        missing = [p for p in self.required_params if p not in all_params]
        if missing:
            raise ValueError(
                f"Step '{self.name}' missing required parameters: {missing}\n"
                f"Available params: {list(all_params.keys())}"
            )

        # Extract only parameters that the function accepts
        fn_params = {k: v for k, v in all_params.items() if k in self._fn_params}

        # Execute function
        if data is None:
            # First step in pipeline (e.g., load_prices)
            return self.fn(**fn_params)
        else:
            # Subsequent steps (pass data as first argument)
            return self.fn(data, **fn_params)


class DataPipeline:
    """Composable data pipeline from function sequence.

    Builds a pipeline from a sequence of PipelineStep objects, validates
    type compatibility, and executes the pipeline with runtime parameters.

    Similar to torchvision.transforms.Compose but for data preprocessing.

    Attributes:
        steps: List of PipelineStep objects to execute in sequence

    Example:
        >>> pipeline = DataPipeline([
        ...     PipelineStep('load', load_prices, {'column': 'adj_close'}),
        ...     PipelineStep('clean', clean_data, {'strategy': 'ffill_drop'}),
        ...     PipelineStep('window', create_windows, {'sequence_length': 64})
        ... ])
        >>> dataloader = pipeline.execute(
        ...     tickers=['AAPL', 'MSFT'],
        ...     start_date='2020-01-01',
        ...     end_date='2024-12-31'
        ... )
    """

    def __init__(self, steps: List[PipelineStep]):
        """Initialize pipeline with steps.

        Args:
            steps: List of PipelineStep objects

        Raises:
            TypeError: If step types are incompatible
        """
        self.steps = steps
        self._validate()

    def _validate(self):
        """Validate that step output/input types are compatible.

        Checks that the output type of step N matches the input type
        of step N+1 throughout the pipeline.

        Raises:
            TypeError: If adjacent steps have incompatible types
        """
        for i in range(len(self.steps) - 1):
            curr = self.steps[i]
            next_step = self.steps[i+1]

            # Only validate if both types are specified
            if curr.output_type and next_step.input_type:
                # Check for exact type match or tuple (for split_temporal)
                if curr.output_type != next_step.input_type:
                    # Special case: split_temporal outputs tuple, but we can extract first element
                    if curr.output_type == tuple:
                        continue  # Allow tuple → any (user must handle in config)

                    raise TypeError(
                        f"Type mismatch in pipeline:\n"
                        f"  Step {i} ('{curr.name}') outputs {curr.output_type.__name__}\n"
                        f"  Step {i+1} ('{next_step.name}') expects {next_step.input_type.__name__}\n"
                        f"  These types are incompatible."
                    )

    def execute(self, **params):
        """Execute pipeline with runtime parameters.

        Runs all pipeline steps in sequence, passing output of each step
        as input to the next.

        Special handling for tuple outputs:
        - clean_data(strategy='mask') returns (data, mask): mask is stored in
          params['mask'] and propagated to subsequent steps
        - split_temporal returns (train, test): only train is passed to next step

        Args:
            **params: Runtime parameters to pass to all steps (e.g., tickers,
                     start_date, end_date, processor instance)

        Returns:
            Final output from last step in pipeline

        Example:
            >>> result = pipeline.execute(
            ...     tickers=['AAPL', 'MSFT'],
            ...     start_date='2020-01-01',
            ...     processor=LogReturnProcessor()
            ... )
        """
        result = None

        for i, step in enumerate(self.steps):
            try:
                result = step(result, **params)

                # Handle tuple outputs
                if isinstance(result, tuple) and i < len(self.steps) - 1:
                    if step.name == 'clean_data' and len(result) == 2:
                        # clean_data with mask strategy: (data, mask)
                        # Store mask in params for subsequent steps
                        data, mask = result
                        params['mask'] = mask
                        result = data
                    elif step.name == 'split_temporal' and len(result) == 2:
                        # split_temporal: (train_df, test_df)
                        # If mask exists, split it too
                        train_df, test_df = result
                        if 'mask' in params and params['mask'] is not None:
                            mask = params['mask']
                            split_idx = len(train_df)
                            train_mask = mask.iloc[:split_idx]
                            params['mask'] = train_mask
                        result = train_df
                    elif step.name == 'process_prices' and len(result) == 2:
                        # process_prices with mask: (data, mask)
                        # Update mask in params (mask may have changed shape)
                        data, mask = result
                        params['mask'] = mask
                        result = data
                    elif step.name == 'create_windows' and len(result) == 2:
                        # create_windows with mask: (data_windows, mask_windows)
                        data, mask = result
                        params['mask'] = mask
                        result = data
                    else:
                        # Other tuples: take first element
                        result = result[0]

            except Exception as e:
                raise RuntimeError(
                    f"Error in pipeline step {i} ('{step.name}'): {e}"
                ) from e

        return result

    @classmethod
    def from_config(cls, config) -> 'DataPipeline':
        """Build pipeline from YAML configuration.

        Parses the 'DataPipeline' section from config and builds a DataPipeline
        using the PIPELINE_REGISTRY for step metadata.

        Args:
            config: ExperimentConfig or dictionary from YAML

        Returns:
            Configured DataPipeline instance

        Raises:
            ValueError: If pipeline config is invalid or missing
            ValueError: If unknown step name encountered

        Example Config:
            ```yaml
            data_pipeline:
              - load_prices:
                  column: 'adj_close'
              - clean_data:
                  strategy: 'ffill_drop'
              - create_windows:
                  sequence_length: 64
            ```

        Example Usage:
            >>> config = yaml.safe_load(open('config.yaml'))
            >>> pipeline = DataPipeline.from_config(config)
            >>> result = pipeline.execute(tickers=['AAPL'], ...)
        """
        from .pipeline_registry import PIPELINE_REGISTRY

        # Support both ExperimentConfig (attribute access) and dict
        if hasattr(config, 'data_pipeline'):
            step_configs = config.data_pipeline or []
        else:
            step_configs = config.get('data_pipeline', [])

        if not step_configs:
            raise ValueError(
                "No pipeline steps defined in config.\n"
                "Add a 'data_pipeline' list to your YAML config.\n"
                "Example:\n"
                "  data_pipeline:\n"
                "    - load_prices:\n"
                "        column: 'adj_close'\n"
                "    - clean_data:\n"
                "        strategy: 'ffill_drop'"
            )

        steps = []
        for idx, step_config in enumerate(step_configs):
            # Each step should be a dict with single key (step name)
            if not isinstance(step_config, dict):
                raise ValueError(
                    f"Invalid step config at index {idx}: {step_config}\n"
                    f"Each step must be a dictionary with format: {{step_name: {{params}}}}"
                )

            if len(step_config) != 1:
                raise ValueError(
                    f"Invalid step config at index {idx}: {step_config}\n"
                    f"Each step must have exactly one key (the step name)"
                )

            step_name, step_params = list(step_config.items())[0]

            if step_name not in PIPELINE_REGISTRY:
                available = ', '.join(PIPELINE_REGISTRY.keys())
                raise ValueError(
                    f"Unknown pipeline step: '{step_name}'\n"
                    f"Available steps: {available}"
                )

            fn, metadata = PIPELINE_REGISTRY[step_name]

            steps.append(PipelineStep(
                name=step_name,
                fn=fn,
                params=step_params or {},  # Handle None params
                **metadata
            ))

        return cls(steps)

    def __repr__(self):
        """String representation showing pipeline steps."""
        step_names = ' → '.join(step.name for step in self.steps)
        return f"DataPipeline({step_names})"
