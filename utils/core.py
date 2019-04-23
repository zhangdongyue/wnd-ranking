# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Public interface for flag definition.

See _example.py for detailed instructions on defining flags.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import sys
import codecs

from absl import app as absl_app
from absl import flags
import tensorflow as tf
import multiprocessing

from . import hooks_helper

# ---------------- base --------------------------
def define_base(data_dir=True, model_dir=True, clean=True, train_epochs=True,
                epochs_between_evals=True, stop_threshold=True, batch_size=True,
                num_gpu=True, hooks=True, export_dir=True):
  """Register base flags.

  Args:
    data_dir: Create a flag for specifying the input data directory.
    model_dir: Create a flag for specifying the model file directory.
    train_epochs: Create a flag to specify the number of training epochs.
    epochs_between_evals: Create a flag to specify the frequency of testing.
    stop_threshold: Create a flag to specify a threshold accuracy or other
      eval metric which should trigger the end of training.
    batch_size: Create a flag to specify the batch size.
    num_gpu: Create a flag to specify the number of GPUs used.
    hooks: Create a flag to specify hooks for logging.
    export_dir: Create a flag to specify where a SavedModel should be exported.

  Returns:
    A list of flags for core.py to marks as key flags.
  """
  key_flags = []

  if data_dir:
    flags.DEFINE_string(
        name="data_dir", short_name="dd", default="/tmp",
        help=help_wrap("The location of the input data."))
    key_flags.append("data_dir")

  if model_dir:
    flags.DEFINE_string(
        name="model_dir", short_name="md", default="/tmp",
        help=help_wrap("The location of the model checkpoint files."))
    key_flags.append("model_dir")

  if clean:
    flags.DEFINE_boolean(
        name="clean", default=False,
        help=help_wrap("If set, model_dir will be removed if it exists."))
    key_flags.append("clean")

  if train_epochs:
    flags.DEFINE_integer(
        name="train_epochs", short_name="te", default=1,
        help=help_wrap("The number of epochs used to train."))
    key_flags.append("train_epochs")

  if epochs_between_evals:
    flags.DEFINE_integer(
        name="epochs_between_evals", short_name="ebe", default=1,
        help=help_wrap("The number of training epochs to run between "
                       "evaluations."))
    key_flags.append("epochs_between_evals")

  if stop_threshold:
    flags.DEFINE_float(
        name="stop_threshold", short_name="st",
        default=None,
        help=help_wrap("If passed, training will stop at the earlier of "
                       "train_epochs and when the evaluation metric is  "
                       "greater than or equal to stop_threshold."))

  if batch_size:
    flags.DEFINE_integer(
        name="batch_size", short_name="bs", default=32,
        help=help_wrap("Batch size for training and evaluation. When using "
                       "multiple gpus, this is the global batch size for "
                       "all devices. For example, if the batch size is 32 "
                       "and there are 4 GPUs, each GPU will get 8 examples on "
                       "each step."))
    key_flags.append("batch_size")

  if num_gpu:
    flags.DEFINE_integer(
        name="num_gpus", short_name="ng",
        default=1 if tf.test.is_gpu_available() else 0,
        help=help_wrap(
            "How many GPUs to use with the DistributionStrategies API. The "
            "default is 1 if TensorFlow can detect a GPU, and 0 otherwise."))

  if hooks:
    # Construct a pretty summary of hooks.
    hook_list_str = (
        u"\ufeff  Hook:\n" + u"\n".join([u"\ufeff    {}".format(key) for key
                                         in hooks_helper.HOOKS]))
    flags.DEFINE_list(
        name="hooks", short_name="hk", default="LoggingTensorHook",
        help=help_wrap(
            u"A list of (case insensitive) strings to specify the names of "
            u"training hooks.\n{}\n\ufeff  Example: `--hooks ProfilerHook,"
            u"ExamplesPerSecondHook`\n See official.utils.logs.hooks_helper "
            u"for details.".format(hook_list_str))
    )
    key_flags.append("hooks")

  if export_dir:
    flags.DEFINE_string(
        name="export_dir", short_name="ed", default=None,
        help=help_wrap("If set, a SavedModel serialization of the model will "
                       "be exported to this directory at the end of training. "
                       "See the README for more details and relevant links.")
    )
    key_flags.append("export_dir")

  return key_flags


def get_num_gpus(flags_obj):
  """Treat num_gpus=-1 as 'use all'."""
  if flags_obj.num_gpus != -1:
    return flags_obj.num_gpus

  from tensorflow.python.client import device_lib  # pylint: disable=g-import-not-at-top
  local_device_protos = device_lib.list_local_devices()
  return sum([1 for d in local_device_protos if d.device_type == "GPU"])

# ---------------- benchmark ---------------------
def define_benchmark(benchmark_log_dir=True, bigquery_uploader=True):
  """Register benchmarking flags.

  Args:
    benchmark_log_dir: Create a flag to specify location for benchmark logging.
    bigquery_uploader: Create flags for uploading results to BigQuery.

  Returns:
    A list of flags for core.py to marks as key flags.
  """

  key_flags = []

  flags.DEFINE_enum(
      name="benchmark_logger_type", default="BaseBenchmarkLogger",
      enum_values=["BaseBenchmarkLogger", "BenchmarkFileLogger",
                   "BenchmarkBigQueryLogger"],
      help=help_wrap("The type of benchmark logger to use. Defaults to using "
                     "BaseBenchmarkLogger which logs to STDOUT. Different "
                     "loggers will require other flags to be able to work."))
  flags.DEFINE_string(
      name="benchmark_test_id", short_name="bti", default=None,
      help=help_wrap("The unique test ID of the benchmark run. It could be the "
                     "combination of key parameters. It is hardware "
                     "independent and could be used compare the performance "
                     "between different test runs. This flag is designed for "
                     "human consumption, and does not have any impact within "
                     "the system."))

  if benchmark_log_dir:
    flags.DEFINE_string(
        name="benchmark_log_dir", short_name="bld", default=None,
        help=help_wrap("The location of the benchmark logging.")
    )

  if bigquery_uploader:
    flags.DEFINE_string(
        name="gcp_project", short_name="gp", default=None,
        help=help_wrap(
            "The GCP project name where the benchmark will be uploaded."))

    flags.DEFINE_string(
        name="bigquery_data_set", short_name="bds", default="test_benchmark",
        help=help_wrap(
            "The Bigquery dataset name where the benchmark will be uploaded."))

    flags.DEFINE_string(
        name="bigquery_run_table", short_name="brt", default="benchmark_run",
        help=help_wrap("The Bigquery table name where the benchmark run "
                       "information will be uploaded."))

    flags.DEFINE_string(
        name="bigquery_run_status_table", short_name="brst",
        default="benchmark_run_status",
        help=help_wrap("The Bigquery table name where the benchmark run "
                       "status information will be uploaded."))

    flags.DEFINE_string(
        name="bigquery_metric_table", short_name="bmt",
        default="benchmark_metric",
        help=help_wrap("The Bigquery table name where the benchmark metric "
                       "information will be uploaded."))

  @flags.multi_flags_validator(
      ["benchmark_logger_type", "benchmark_log_dir"],
      message="--benchmark_logger_type=BenchmarkFileLogger will require "
              "--benchmark_log_dir being set")
  def _check_benchmark_log_dir(flags_dict):
    benchmark_logger_type = flags_dict["benchmark_logger_type"]
    if benchmark_logger_type == "BenchmarkFileLogger":
      return flags_dict["benchmark_log_dir"]
    return True

  return key_flags

# ---------------- conventions -------------------
# This codifies help string conventions and makes it easy to update them if
# necessary. Currently the only major effect is that help bodies start on the
# line after flags are listed. All flag definitions should wrap the text bodies
# with help wrap when calling DEFINE_*.
_help_wrap = functools.partial(flags.text_wrap, length=80, indent="",
                               firstline_indent="\n")


# Pretty formatting causes issues when utf-8 is not installed on a system.
try:
  codecs.lookup("utf-8")
  help_wrap = _help_wrap
except LookupError:
  def help_wrap(text, *args, **kwargs):
    return _help_wrap(text, *args, **kwargs).replace("\ufeff", "")


# Replace None with h to also allow -h
absl_app.HelpshortFlag.SHORT_NAME = "h"

# ------------------- device ------------------------
def require_cloud_storage(flag_names):
  """Register a validator to check directory flags.
  Args:
    flag_names: An iterable of strings containing the names of flags to be
      checked.
  """
  msg = "TPU requires GCS path for {}".format(", ".join(flag_names))
  @flags.multi_flags_validator(["tpu"] + flag_names, message=msg)
  def _path_check(flag_values):  # pylint: disable=missing-docstring
    if flag_values["tpu"] is None:
      return True

    valid_flags = True
    for key in flag_names:
      if not flag_values[key].startswith("gs://"):
        tf.logging.error("{} must be a GCS path.".format(key))
        valid_flags = False

    return valid_flags


def define_device(tpu=True):
  """Register device specific flags.
  Args:
    tpu: Create flags to specify TPU operation.
  Returns:
    A list of flags for core.py to marks as key flags.
  """

  key_flags = []

  if tpu:
    flags.DEFINE_string(
        name="tpu", default=None,
        help=help_wrap(
            "The Cloud TPU to use for training. This should be either the name "
            "used when creating the Cloud TPU, or a "
            "grpc://ip.address.of.tpu:8470 url. Passing `local` will use the"
            "CPU of the local instance instead. (Good for debugging.)"))
    key_flags.append("tpu")

    flags.DEFINE_string(
        name="tpu_zone", default=None,
        help=help_wrap(
            "[Optional] GCE zone where the Cloud TPU is located in. If not "
            "specified, we will attempt to automatically detect the GCE "
            "project from metadata."))

    flags.DEFINE_string(
        name="tpu_gcp_project", default=None,
        help=help_wrap(
            "[Optional] Project name for the Cloud TPU-enabled project. If not "
            "specified, we will attempt to automatically detect the GCE "
            "project from metadata."))

    flags.DEFINE_integer(name="num_tpu_shards", default=8,
                         help=help_wrap("Number of shards (TPU chips)."))

  return key_flags

# ------------------ misc -----------------
def define_image(data_format=True):
  """Register image specific flags.

  Args:
    data_format: Create a flag to specify image axis convention.

  Returns:
    A list of flags for core.py to marks as key flags.
  """

  key_flags = []

  if data_format:
    flags.DEFINE_enum(
        name="data_format", short_name="df", default=None,
        enum_values=["channels_first", "channels_last"],
        help=help_wrap(
            "A flag to override the data format used in the model. "
            "channels_first provides a performance boost on GPU but is not "
            "always compatible with CPU. If left unspecified, the data format "
            "will be chosen automatically based on whether TensorFlow was "
            "built for CPU or GPU."))
    key_flags.append("data_format")

  return key_flags

# ------------------ performance -----------------
# Map string to (TensorFlow dtype, default loss scale)
DTYPE_MAP = {
    "fp16": (tf.float16, 128),
    "fp32": (tf.float32, 1),
}


def get_tf_dtype(flags_obj):
  return DTYPE_MAP[flags_obj.dtype][0]


def get_loss_scale(flags_obj):
  if flags_obj.loss_scale is not None:
    return flags_obj.loss_scale
  return DTYPE_MAP[flags_obj.dtype][1]


def define_performance(num_parallel_calls=True, inter_op=True, intra_op=True,
                       synthetic_data=True, max_train_steps=True, dtype=True,
                       all_reduce_alg=True, tf_gpu_thread_mode=False,
                       datasets_num_private_threads=False,
                       datasets_num_parallel_batches=False):
  """Register flags for specifying performance tuning arguments.

  Args:
    num_parallel_calls: Create a flag to specify parallelism of data loading.
    inter_op: Create a flag to allow specification of inter op threads.
    intra_op: Create a flag to allow specification of intra op threads.
    synthetic_data: Create a flag to allow the use of synthetic data.
    max_train_steps: Create a flags to allow specification of maximum number
      of training steps
    dtype: Create flags for specifying dtype.
    all_reduce_alg: If set forces a specific algorithm for multi-gpu.
    tf_gpu_thread_mode: gpu_private triggers us of private thread pool.
    datasets_num_private_threads: Number of private threads for datasets.
    datasets_num_parallel_batches: Determines how many batches to process in
    parallel when using map and batch from tf.data.

  Returns:
    A list of flags for core.py to marks as key flags.
  """

  key_flags = []
  if num_parallel_calls:
    flags.DEFINE_integer(
        name="num_parallel_calls", short_name="npc",
        default=multiprocessing.cpu_count(),
        help=help_wrap("The number of records that are  processed in parallel "
                       "during input processing. This can be optimized per "
                       "data set but for generally homogeneous data sets, "
                       "should be approximately the number of available CPU "
                       "cores. (default behavior)"))

  if inter_op:
    flags.DEFINE_integer(
        name="inter_op_parallelism_threads", short_name="inter", default=0,
        help=help_wrap("Number of inter_op_parallelism_threads to use for CPU. "
                       "See TensorFlow config.proto for details.")
    )

  if intra_op:
    flags.DEFINE_integer(
        name="intra_op_parallelism_threads", short_name="intra", default=0,
        help=help_wrap("Number of intra_op_parallelism_threads to use for CPU. "
                       "See TensorFlow config.proto for details."))

  if synthetic_data:
    flags.DEFINE_bool(
        name="use_synthetic_data", short_name="synth", default=False,
        help=help_wrap(
            "If set, use fake data (zeroes) instead of a real dataset. "
            "This mode is useful for performance debugging, as it removes "
            "input processing steps, but will not learn anything."))

  if max_train_steps:
    flags.DEFINE_integer(
        name="max_train_steps", short_name="mts", default=None, help=help_wrap(
            "The model will stop training if the global_step reaches this "
            "value. If not set, training will run until the specified number "
            "of epochs have run as usual. It is generally recommended to set "
            "--train_epochs=1 when using this flag."
        ))

  if dtype:
    flags.DEFINE_enum(
        name="dtype", short_name="dt", default="fp32",
        enum_values=DTYPE_MAP.keys(),
        help=help_wrap("The TensorFlow datatype used for calculations. "
                       "Variables may be cast to a higher precision on a "
                       "case-by-case basis for numerical stability."))

    flags.DEFINE_integer(
        name="loss_scale", short_name="ls", default=None,
        help=help_wrap(
            "The amount to scale the loss by when the model is run. Before "
            "gradients are computed, the loss is multiplied by the loss scale, "
            "making all gradients loss_scale times larger. To adjust for this, "
            "gradients are divided by the loss scale before being applied to "
            "variables. This is mathematically equivalent to training without "
            "a loss scale, but the loss scale helps avoid some intermediate "
            "gradients from underflowing to zero. If not provided the default "
            "for fp16 is 128 and 1 for all other dtypes."))

    loss_scale_val_msg = "loss_scale should be a positive integer."
    @flags.validator(flag_name="loss_scale", message=loss_scale_val_msg)
    def _check_loss_scale(loss_scale):  # pylint: disable=unused-variable
      if loss_scale is None:
        return True  # null case is handled in get_loss_scale()

      return loss_scale > 0

  if all_reduce_alg:
    flags.DEFINE_string(
        name="all_reduce_alg", short_name="ara", default=None,
        help=help_wrap("Defines the algorithm to use for performing all-reduce."
                       "See tf.contrib.distribute.AllReduceCrossTowerOps for "
                       "more details and available options."))

  if tf_gpu_thread_mode:
    flags.DEFINE_string(
        name="tf_gpu_thread_mode", short_name="gt_mode", default=None,
        help=help_wrap(
            "Whether and how the GPU device uses its own threadpool.")
    )

  if datasets_num_private_threads:
    flags.DEFINE_integer(
        name="datasets_num_private_threads",
        default=None,
        help=help_wrap(
            "Number of threads for a private threadpool created for all"
            "datasets computation..")
    )

  if datasets_num_parallel_batches:
    flags.DEFINE_integer(
        name="datasets_num_parallel_batches",
        default=None,
        help=help_wrap(
            "Determines how many batches to process in parallel when using "
            "map and batch from tf.data.")
    )

  return key_flags

# ---------------------- performance end --------------------------

def set_defaults(**kwargs):
  for key, value in kwargs.items():
    flags.FLAGS.set_default(name=key, value=value)


def parse_flags(argv=None):
  """Reset flags and reparse. Currently only used in testing."""
  flags.FLAGS.unparse_flags()
  absl_app.parse_flags_with_usage(argv or sys.argv)


def register_key_flags_in_core(f):
  """Defines a function in core.py, and registers its key flags.

  absl uses the location of a flags.declare_key_flag() to determine the context
  in which a flag is key. By making all declares in core, this allows model
  main functions to call flags.adopt_module_key_flags() on core and correctly
  chain key flags.

  Args:
    f:  The function to be wrapped

  Returns:
    The "core-defined" version of the input function.
  """

  def core_fn(*args, **kwargs):
    key_flags = f(*args, **kwargs)
    [flags.declare_key_flag(fl) for fl in key_flags]  # pylint: disable=expression-not-assigned
  return core_fn


define_base = register_key_flags_in_core(define_base)
# Remove options not relevant for Eager from define_base().
define_base_eager = register_key_flags_in_core(functools.partial(
    define_base, epochs_between_evals=False, stop_threshold=False,
    hooks=False))
define_benchmark = register_key_flags_in_core(define_benchmark)
define_device = register_key_flags_in_core(define_device)
define_image = register_key_flags_in_core(define_image)
define_performance = register_key_flags_in_core(define_performance)
