"""
Tests for pipeline builder (functional composition).

Tests the PipelineStep, DataPipeline, and PIPELINE_REGISTRY components.
"""

import pytest
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from tsgen.data.pipeline_builder import PipelineStep, DataPipeline
from tsgen.data.pipeline_registry import PIPELINE_REGISTRY, get_available_steps, get_step_info
from tsgen.data.processor import LogReturnProcessor


class TestPipelineStep:
    """Tests for PipelineStep wrapper."""

    def test_step_initialization(self):
        """Test creating a PipelineStep."""
        def dummy_fn(data, param1=None):
            return data

        step = PipelineStep(
            name='test_step',
            fn=dummy_fn,
            params={'param1': 'value1'},
            required_params=['param1']
        )

        assert step.name == 'test_step'
        assert step.fn == dummy_fn
        assert step.params == {'param1': 'value1'}
        assert step.required_params == ['param1']

    def test_step_execution_with_data(self):
        """Test executing step with input data."""
        def add_column(df, column_name='new_col'):
            df = df.copy()
            df[column_name] = 1
            return df

        step = PipelineStep(
            name='add_col',
            fn=add_column,
            params={'column_name': 'test'}
        )

        df = pd.DataFrame({'a': [1, 2, 3]})
        result = step(df)

        assert 'test' in result.columns
        assert result['test'].tolist() == [1, 1, 1]

    def test_step_execution_without_data(self):
        """Test executing first step (no input data)."""
        def create_df(rows=3):
            return pd.DataFrame({'a': list(range(rows))})

        step = PipelineStep(
            name='create',
            fn=create_df,
            params={'rows': 5}
        )

        result = step(None)  # No input data

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5

    def test_step_parameter_merging(self):
        """Test that runtime params override config params."""
        def get_value(value=None):  # First-step function (no data param)
            return value

        step = PipelineStep(
            name='getter',
            fn=get_value,
            params={'value': 'config_value'}
        )

        # With runtime param
        result = step(None, value='runtime_value')
        assert result == 'runtime_value'

        # Without runtime param (uses config)
        result = step(None)
        assert result == 'config_value'

    def test_step_missing_required_params(self):
        """Test that missing required params raises ValueError."""
        def needs_param(data, required_param):
            return required_param

        step = PipelineStep(
            name='needs',
            fn=needs_param,
            params={},
            required_params=['required_param']
        )

        with pytest.raises(ValueError, match="missing required parameters"):
            step(None)  # No required_param provided

    def test_step_filters_params_by_signature(self):
        """Test that step only passes params the function accepts."""
        def simple_fn(param1):  # First-step function (no data param)
            return param1

        step = PipelineStep(
            name='simple',
            fn=simple_fn,
            params={'param1': 'a', 'param2': 'b'}  # param2 not in signature
        )

        # Should not raise error about unexpected param2
        result = step(None)
        assert result == 'a'


class TestDataPipeline:
    """Tests for DataPipeline composition."""

    def test_pipeline_initialization(self):
        """Test creating a pipeline."""
        steps = [
            PipelineStep('step1', lambda x: x, {}),
            PipelineStep('step2', lambda x: x, {})
        ]

        pipeline = DataPipeline(steps)

        assert len(pipeline.steps) == 2
        assert pipeline.steps[0].name == 'step1'
        assert pipeline.steps[1].name == 'step2'

    def test_pipeline_execution(self):
        """Test executing a simple pipeline."""
        def create_list():
            return [1, 2, 3]

        def double_values(data):
            return [x * 2 for x in data]

        def sum_values(data):
            return sum(data)

        steps = [
            PipelineStep('create', create_list, {}),
            PipelineStep('double', double_values, {}),
            PipelineStep('sum', sum_values, {})
        ]

        pipeline = DataPipeline(steps)
        result = pipeline.execute()

        assert result == 12  # (1+2+3)*2 = 12

    def test_pipeline_with_runtime_params(self):
        """Test pipeline with runtime parameters."""
        def create_df(rows=3):
            return pd.DataFrame({'a': list(range(rows))})

        def filter_df(df, threshold=2):
            return df[df['a'] > threshold]

        steps = [
            PipelineStep('create', create_df, {'rows': 5}),
            PipelineStep('filter', filter_df, {})
        ]

        pipeline = DataPipeline(steps)
        result = pipeline.execute(threshold=3)  # Runtime param

        assert len(result) == 1  # Only row with a=4 passes threshold>3
        assert result['a'].iloc[0] == 4

    def test_pipeline_repr(self):
        """Test string representation."""
        steps = [
            PipelineStep('load', lambda: None, {}),
            PipelineStep('clean', lambda x: x, {}),
            PipelineStep('process', lambda x: x, {})
        ]

        pipeline = DataPipeline(steps)

        assert repr(pipeline) == "DataPipeline(load → clean → process)"


class TestPipelineTypeValidation:
    """Tests for type validation in pipelines."""

    def test_valid_type_sequence(self):
        """Test that valid type sequence passes validation."""
        steps = [
            PipelineStep('step1', lambda: None, {},
                        input_type=None, output_type=pd.DataFrame),
            PipelineStep('step2', lambda x: x, {},
                        input_type=pd.DataFrame, output_type=pd.DataFrame),
            PipelineStep('step3', lambda x: None, {},
                        input_type=pd.DataFrame, output_type=np.ndarray)
        ]

        # Should not raise
        pipeline = DataPipeline(steps)
        assert len(pipeline.steps) == 3

    def test_invalid_type_sequence(self):
        """Test that invalid type sequence raises TypeError."""
        steps = [
            PipelineStep('step1', lambda: None, {},
                        input_type=None, output_type=pd.DataFrame),
            PipelineStep('step2', lambda x: x, {},
                        input_type=np.ndarray,  # Wrong! expects ndarray but gets DataFrame
                        output_type=np.ndarray)
        ]

        with pytest.raises(TypeError, match="Type mismatch"):
            DataPipeline(steps)

    def test_tuple_output_allowed(self):
        """Test that tuple output type is allowed (for split_temporal)."""
        steps = [
            PipelineStep('step1', lambda: None, {},
                        input_type=None, output_type=pd.DataFrame),
            PipelineStep('split', lambda x: (x, x), {},
                        input_type=pd.DataFrame, output_type=tuple)
        ]

        # Should not raise (tuple is allowed as output)
        pipeline = DataPipeline(steps)
        assert len(pipeline.steps) == 2


class TestPipelineFromConfig:
    """Tests for building pipelines from YAML config."""

    def test_from_config_simple(self):
        """Test building pipeline from simple config."""
        config = {
            'data_pipeline': [
                {'load_prices': {'column': 'adj_close'}},
                {'clean_data': {'strategy': 'ffill_drop'}}
            ]
        }

        pipeline = DataPipeline.from_config(config)

        assert len(pipeline.steps) == 2
        assert pipeline.steps[0].name == 'load_prices'
        assert pipeline.steps[0].params == {'column': 'adj_close'}
        assert pipeline.steps[1].name == 'clean_data'
        assert pipeline.steps[1].params == {'strategy': 'ffill_drop'}

    def test_from_config_full_pipeline(self):
        """Test building full data pipeline from config."""
        config = {
            'data_pipeline': [
                {'load_prices': {'column': 'adj_close'}},
                {'clean_data': {'strategy': 'ffill_drop'}},
                {'process_prices': {'fit': True}},
                {'create_windows': {'sequence_length': 64}},
                {'create_dataloader': {'batch_size': 32, 'shuffle': True}}
            ]
        }

        pipeline = DataPipeline.from_config(config)

        assert len(pipeline.steps) == 5
        assert pipeline.steps[0].name == 'load_prices'
        assert pipeline.steps[2].name == 'process_prices'
        assert pipeline.steps[4].params['batch_size'] == 32

    def test_from_config_empty_params(self):
        """Test step with empty params dict."""
        config = {
            'data_pipeline': [
                {'load_prices': {}},  # Empty params
                {'clean_data': None}   # None params
            ]
        }

        pipeline = DataPipeline.from_config(config)

        assert pipeline.steps[0].params == {}
        assert pipeline.steps[1].params == {}

    def test_from_config_missing_pipeline_key(self):
        """Test that missing data_pipeline key raises ValueError."""
        config = {}  # No 'data_pipeline' key

        with pytest.raises(ValueError, match="No pipeline steps defined"):
            DataPipeline.from_config(config)

    def test_from_config_empty_pipeline(self):
        """Test that empty data_pipeline list raises ValueError."""
        config = {'data_pipeline': []}  # Empty list

        with pytest.raises(ValueError, match="No pipeline steps defined"):
            DataPipeline.from_config(config)

    def test_from_config_invalid_step_format(self):
        """Test that invalid step format raises ValueError."""
        config = {
            'data_pipeline': [
                'invalid_string',  # Should be dict
            ]
        }

        with pytest.raises(ValueError, match="Invalid step config"):
            DataPipeline.from_config(config)

    def test_from_config_multiple_keys_per_step(self):
        """Test that step with multiple keys raises ValueError."""
        config = {
            'data_pipeline': [
                {'load_prices': {}, 'clean_data': {}}  # Two keys!
            ]
        }

        with pytest.raises(ValueError, match="exactly one key"):
            DataPipeline.from_config(config)

    def test_from_config_unknown_step(self):
        """Test that unknown step name raises ValueError."""
        config = {
            'data_pipeline': [
                {'unknown_step': {}}
            ]
        }

        with pytest.raises(ValueError, match="Unknown pipeline step"):
            DataPipeline.from_config(config)


class TestPipelineRegistry:
    """Tests for PIPELINE_REGISTRY utilities."""

    def test_registry_has_all_steps(self):
        """Test that registry contains expected steps."""
        expected_steps = {
            'load_prices',
            'clean_data',
            'split_temporal',
            'process_prices',
            'create_windows',
            'create_dataloader'
        }

        assert set(PIPELINE_REGISTRY.keys()) == expected_steps

    def test_get_available_steps(self):
        """Test get_available_steps() returns sorted list."""
        steps = get_available_steps()

        assert isinstance(steps, list)
        assert len(steps) == 6
        assert steps == sorted(steps)  # Should be sorted
        assert 'load_prices' in steps

    def test_get_step_info_valid(self):
        """Test getting step info for valid step."""
        info = get_step_info('load_prices')

        assert 'function' in info
        assert 'input_type' in info
        assert 'output_type' in info
        assert 'required_params' in info
        assert 'optional_params' in info

        assert info['input_type'] is None  # First step
        assert info['output_type'] == pd.DataFrame
        assert 'tickers' in info['required_params']

    def test_get_step_info_invalid(self):
        """Test that invalid step name raises KeyError."""
        with pytest.raises(KeyError, match="Unknown pipeline step"):
            get_step_info('nonexistent_step')

    def test_registry_metadata_complete(self):
        """Test that all registry entries have complete metadata."""
        for step_name, (fn, metadata) in PIPELINE_REGISTRY.items():
            assert callable(fn), f"{step_name} function not callable"
            assert 'input_type' in metadata, f"{step_name} missing input_type"
            assert 'output_type' in metadata, f"{step_name} missing output_type"
            assert 'required_params' in metadata, f"{step_name} missing required_params"
            assert 'optional_params' in metadata, f"{step_name} missing optional_params"
            assert isinstance(metadata['required_params'], list)
            assert isinstance(metadata['optional_params'], list)


@pytest.mark.integration
class TestPipelineIntegration:
    """Integration tests for full pipeline execution (requires findata database)."""

    def test_minimal_pipeline_execution(self):
        """Test executing minimal real pipeline."""
        config = {
            'data_pipeline': [
                {'load_prices': {'column': 'adj_close'}},
                {'clean_data': {'strategy': 'ffill_drop'}},
            ]
        }

        pipeline = DataPipeline.from_config(config)

        # Execute with real data
        result = pipeline.execute(
            tickers=['AAPL', 'MSFT'],
            start_date='2024-01-01',
            end_date='2024-12-31'
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'AAPL' in result.columns or result.index.names[0] is not None

    def test_full_pipeline_execution(self):
        """Test executing full data pipeline."""
        config = {
            'data_pipeline': [
                {'load_prices': {'column': 'adj_close'}},
                {'clean_data': {'strategy': 'ffill_drop'}},
                {'process_prices': {'fit': True}},
                {'create_windows': {'sequence_length': 10}},
                {'create_dataloader': {'batch_size': 4, 'shuffle': False}}
            ]
        }

        pipeline = DataPipeline.from_config(config)

        # Execute with real data and processor
        processor = LogReturnProcessor()
        result = pipeline.execute(
            tickers=['AAPL', 'MSFT'],
            start_date='2024-01-01',
            end_date='2024-12-28',
            processor=processor
        )

        assert isinstance(result, DataLoader)
        assert result.batch_size == 4

        # Check that we can iterate
        batch = next(iter(result))
        assert isinstance(batch, (tuple, list))  # DataLoader returns tuple/list

    def test_pipeline_error_handling(self):
        """Test that pipeline errors provide context."""
        config = {
            'data_pipeline': [
                {'load_prices': {}},  # Missing required params
            ]
        }

        pipeline = DataPipeline.from_config(config)

        # Should raise with step context
        with pytest.raises(RuntimeError, match="Error in pipeline step 0"):
            pipeline.execute()  # Missing tickers, start_date, end_date


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
