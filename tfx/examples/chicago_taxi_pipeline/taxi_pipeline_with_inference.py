# Lint as: python2, python3
# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Chicago taxi example pipeline for training and offline inference."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import List, Text

import absl
import tensorflow_model_analysis as tfma

from tfx.components import BulkInferrer
from tfx.components import CsvExampleGen
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.dsl.components.common import resolver
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.proto import bulk_inferrer_pb2
from tfx.proto import example_gen_pb2
from tfx.proto import trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing

_pipeline_name = 'chicago_taxi_with_inference'

# This example assumes that the taxi data is stored in ~/taxi/data and the
# taxi utility function is in ~/taxi.  Feel free to customize this as needed.
_taxi_root = os.path.join(os.environ['HOME'], 'taxi')
_training_data_root = os.path.join(_taxi_root, 'data', 'simple')
_inference_data_root = os.path.join(_taxi_root, 'data', 'unlabelled')
# Python module file to inject customized logic into the TFX components. The
# Transform and Trainer both require user-defined functions to run successfully.
_module_file = os.path.join(_taxi_root, 'taxi_utils.py')

# Directory and data locations.  This example assumes all of the chicago taxi
# example code and metadata library is relative to $HOME, but you can store
# these files anywhere on your local filesystem.
_tfx_root = os.path.join(os.environ['HOME'], 'tfx')
_pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)
# Sqlite ML-metadata db path.
_metadata_path = os.path.join(_tfx_root, 'metadata', _pipeline_name,
                              'metadata.db')

# Pipeline arguments for Beam powered Components.
_beam_pipeline_args = [
    '--direct_running_mode=multi_processing',
    # 0 means auto-detect based on on the number of CPUs available
    # during execution time.
    '--direct_num_workers=0',
]


def _create_pipeline(pipeline_name: Text, pipeline_root: Text,
                     training_data_root: Text, inference_data_root: Text,
                     module_file: Text, metadata_path: Text,
                     beam_pipeline_args: List[Text]) -> pipeline.Pipeline:
  """Implements the chicago taxi pipeline with TFX."""
  # Brings training data into the pipeline or otherwise joins/converts
  # training data.
  training_example_gen = CsvExampleGen(
      input_base=training_data_root, instance_name='training_example_gen')

  # Computes statistics over data for visualization and example validation.
  statistics_gen = StatisticsGen(
      examples=training_example_gen.outputs['examples'])

  # Generates schema based on statistics files.
  schema_gen = SchemaGen(
      statistics=statistics_gen.outputs['statistics'],
      infer_feature_shape=False)

  # Performs anomaly detection based on statistics and data schema.
  example_validator = ExampleValidator(
      statistics=statistics_gen.outputs['statistics'],
      schema=schema_gen.outputs['schema'])

  # Performs transformations and feature engineering in training and serving.
  transform = Transform(
      examples=training_example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'],
      module_file=module_file)

  # Uses user-provided Python function that implements a model using TF-Learn.
  trainer = Trainer(
      module_file=module_file,
      transformed_examples=transform.outputs['transformed_examples'],
      schema=schema_gen.outputs['schema'],
      transform_graph=transform.outputs['transform_graph'],
      train_args=trainer_pb2.TrainArgs(num_steps=10000),
      eval_args=trainer_pb2.EvalArgs(num_steps=5000))

  # Get the latest blessed model for model validation.
  model_resolver = resolver.Resolver(
      instance_name='latest_blessed_model_resolver',
      resolver_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
      model=Channel(type=Model),
      model_blessing=Channel(type=ModelBlessing))

  # Uses TFMA to compute a evaluation statistics over features of a model and
  # perform quality validation of a candidate model (compared to a baseline).
  eval_config = tfma.EvalConfig(
      model_specs=[tfma.ModelSpec(signature_name='eval')],
      slicing_specs=[
          tfma.SlicingSpec(),
          tfma.SlicingSpec(feature_keys=['trip_start_hour'])
      ],
      metrics_specs=[
          tfma.MetricsSpec(
              thresholds={
                  'accuracy':
                      tfma.config.MetricThreshold(
                          value_threshold=tfma.GenericValueThreshold(
                              lower_bound={'value': 0.6}),
                          # Change threshold will be ignored if there is no
                          # baseline model resolved from MLMD (first run).
                          change_threshold=tfma.GenericChangeThreshold(
                              direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                              absolute={'value': -1e-10}))
              })
      ])
  evaluator = Evaluator(
      examples=training_example_gen.outputs['examples'],
      model=trainer.outputs['model'],
      baseline_model=model_resolver.outputs['model'],
      eval_config=eval_config)

  # Brings inference data into the pipeline.
  inference_example_gen = CsvExampleGen(
      input_base=inference_data_root,
      output_config=example_gen_pb2.Output(
          split_config=example_gen_pb2.SplitConfig(splits=[
              example_gen_pb2.SplitConfig.Split(
                  name='unlabelled', hash_buckets=100)
          ])),
      instance_name='inference_example_gen')

  # Performs offline batch inference over inference examples.
  bulk_inferrer = BulkInferrer(
      examples=inference_example_gen.outputs['examples'],
      model=trainer.outputs['model'],
      model_blessing=evaluator.outputs['blessing'],
      # Empty data_spec.example_splits will result in using all splits.
      data_spec=bulk_inferrer_pb2.DataSpec(),
      model_spec=bulk_inferrer_pb2.ModelSpec())

  return pipeline.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=[
          training_example_gen, inference_example_gen, statistics_gen,
          schema_gen, example_validator, transform, trainer, model_resolver,
          evaluator, bulk_inferrer
      ],
      enable_cache=True,
      metadata_connection_config=metadata.sqlite_metadata_connection_config(
          metadata_path),
      beam_pipeline_args=beam_pipeline_args)


# To run this pipeline from the python CLI:
#   $python taxi_pipeline_with_inference.py
if __name__ == '__main__':
  absl.logging.set_verbosity(absl.logging.INFO)

  BeamDagRunner().run(
      _create_pipeline(
          pipeline_name=_pipeline_name,
          pipeline_root=_pipeline_root,
          training_data_root=_training_data_root,
          inference_data_root=_inference_data_root,
          module_file=_module_file,
          metadata_path=_metadata_path,
          beam_pipeline_args=_beam_pipeline_args))
