/*
 * Copyright (C) 2017 Seoul National University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package edu.snu.cay.dolphin;

import edu.snu.cay.services.et.configuration.parameters.NumTotalBlocks;
import org.apache.reef.tang.annotations.Name;
import org.apache.reef.tang.annotations.NamedParameter;

/**
 * PS specific parameters.
 */
public final class DolphinParameters {
  @NamedParameter(doc = "input table identifier", default_value = InputTableId.DEFAULT_VALUE)
  public final class InputTableId implements Name<String> {
    public static final String DEFAULT_VALUE = "input_table";

    private InputTableId() {
    }
  }

  @NamedParameter(doc = "model table identifier", default_value = ModelTableId.DEFAULT_VALUE)
  public final class ModelTableId implements Name<String> {
    public static final String DEFAULT_VALUE = "model_table";

    private ModelTableId() {
    }
  }

  @NamedParameter(doc = "Local model table identifier", default_value = LocalModelTableId.DEFAULT_VALUE)
  public final class LocalModelTableId implements Name<String> {
    public static final String DEFAULT_VALUE = "local_model_table";

    private LocalModelTableId() {
    }
  }

  @NamedParameter(doc = "Whether this app's input data has key or not", default_value = "false")
  public final class HasInputDataKey implements Name<Boolean> {
  }

  @NamedParameter(doc = "Whether this app has local model table or not", default_value = "false")
  public final class HasLocalModelTable implements Name<Boolean> {
  }

  @NamedParameter(doc = "Starting epoch index for each worker")
  public final class StartingEpochIdx implements Name<Integer> {
  }

  @NamedParameter(doc = "Maximum number of epochs to run before termination",
                  short_name = "max_num_epochs")
  public final class MaxNumEpochs implements Name<Integer> {
  }

  @NamedParameter(doc = "The total number of batches in an epoch",
      short_name = "num_mini_batches")
  public final class NumTotalMiniBatches implements Name<Integer> {
  }

  @NamedParameter(doc = "The total number of blocks for worker table",
      short_name = "num_worker_blocks", default_value = NumTotalBlocks.DEFAULT_VALUE_STR)
  public final class NumWorkerBlocks implements Name<Integer> {
  }

  @NamedParameter(doc = "The total number of blocks for server table",
      short_name = "num_server_blocks", default_value = NumTotalBlocks.DEFAULT_VALUE_STR)
  public final class NumServerBlocks implements Name<Integer> {
  }

  @NamedParameter(doc = "Whether the model cache is enabled.",
      short_name = "model_cache_enabled", default_value = "false")
  public final class ModelCacheEnabled implements Name<Boolean> {
  }

  @NamedParameter(doc = "Desired memory size for each worker evaluator (MBs)",
                  short_name = "dolphin_worker_mem_size",
                  default_value = "128")
  public final class WorkerMemSize implements Name<Integer> {
  }

  @NamedParameter(doc = "Desired memory size for each server evaluator (MBs)",
                  short_name = "server_mem_size",
                  default_value = "128")
  public final class ServerMemSize implements Name<Integer> {
  }

  @NamedParameter(doc = "The number of workers",
                  short_name = "num_dolphin_workers",
                  default_value = "1")
  public final class NumWorkers implements Name<Integer> {
  }

  @NamedParameter(doc = "The number of servers. 0 means PS-collocation",
                  short_name = "num_servers",
                  default_value = "1")
  public final class NumServers implements Name<Integer> {
  }

  @NamedParameter(doc = "The number of CPU cores for each worker evaluator",
                  short_name = "num_dolphin_worker_cores",
                  default_value = "1")
  public final class NumWorkerCores implements Name<Integer> {
  }

  @NamedParameter(doc = "The number of CPU cores for each server evaluator",
                  short_name = "num_server_cores",
                  default_value = "1")
  public final class NumServerCores implements Name<Integer> {
  }

  @NamedParameter(doc = "The number of trainer threads",
                  short_name = "num_trainer_threads",
                  default_value = NumTrainerThreads.UNSET_VALUE)
  public final class NumTrainerThreads implements Name<Integer> {
    public static final String UNSET_VALUE = "0";

    private NumTrainerThreads() {

    }
  }

  @NamedParameter(doc = "A slack of clock gap between workers. Each worker's clock increments upon every mini-batch.",
                  short_name = "clock_slack",
                  default_value = "0")
  public final class ClockSlack implements Name<Integer> {
  }

  @NamedParameter(doc = "The queue size to handle remote access messages at server",
                  short_name = "server_handler_queue_size",
                  default_value = "500000")
  public final class ServerHandlerQueueSize implements Name<Integer> {
  }

  @NamedParameter(doc = "The queue size to handle remote access messages at worker",
                  short_name = "worker_handler_queue_size",
                  default_value = "500000")
  public final class WorkerHandlerQueueSize implements Name<Integer> {
  }

  @NamedParameter(doc = "The queue size to send remote access messages at server",
                  short_name = "server_sender_queue_size",
                  default_value = "500000")
  public final class ServerSenderQueueSize implements Name<Integer> {
  }

  @NamedParameter(doc = "The queue size to send remote access messages at worker",
                  short_name = "worker_sender_queue_size",
                  default_value = "500000")
  public final class WorkerSenderQueueSize implements Name<Integer> {
  }

  @NamedParameter(doc = "The number of threads to handle remote access messages at server",
                  short_name = "num_server_handler_threads",
                  default_value = "2")
  public final class NumServerHandlerThreads implements Name<Integer> {
  }

  @NamedParameter(doc = "The number of threads to handle remote access messages at worker",
                  short_name = "num_worker_handler_threads",
                  default_value = "2")
  public final class NumWorkerHandlerThreads implements Name<Integer> {
  }

  @NamedParameter(doc = "The number of threads to send remote access messages at server",
                  short_name = "num_server_sender_threads",
                  default_value = "2")
  public final class NumServerSenderThreads implements Name<Integer> {
  }

  @NamedParameter(doc = "The number of threads to send remote access messages at worker",
                  short_name = "num_worker_sender_threads",
                  default_value = "2")
  public final class NumWorkerSenderThreads implements Name<Integer> {
  }

  @NamedParameter(doc = "Whether to load model",
      short_name = "load_model", default_value = "false")
  public final class LoadModel implements Name<Boolean> {
  }

  @NamedParameter(doc = "The path of checkpointed model to load",
      short_name = "model_path")
  public final class ModelPath implements Name<String> {
  }

  @NamedParameter(doc = "The path of checkpointed local model to load",
      short_name = "local_model_path")
  public final class LocalModelPath implements Name<String> {
  }

  @NamedParameter(doc = "The path of checkpointed input to load",
      short_name = "input_chkp_path")
  public final class InputChkpPath implements Name<String> {
  }

  @NamedParameter(doc = "The path of test data",
                  short_name = "test_data_path",
                  default_value = TestDataPath.NONE)
  public final class TestDataPath implements Name<String> {
    public static final String NONE = "";
    private TestDataPath() {
    }
  }

  @NamedParameter(doc = "Whether to evaluate the trained model or train a new model.",
      short_name = "model_eval", default_value = "false")
  public final class ModelEvaluation implements Name<Boolean> {
  }

  @NamedParameter(doc = "Whether to evaluate a trained model in offline or online.",
      short_name = "offline_model_eval", default_value = "false")
  public final class OfflineModelEvaluation implements Name<Boolean> {
  }

  // below is commonly used ML app parameters, not all apps use them
  @NamedParameter(doc = "input dimension", short_name = "features", default_value = "0")
  public static final class NumFeatures implements Name<Integer> {
  }

  @NamedParameter(doc = "step size for stochastic gradient descent", short_name = "step_size", default_value = "0.0")
  public static final class StepSize implements Name<Float> {
  }

  @NamedParameter(doc = "regularization constant value", short_name = "lambda", default_value = "0.0")
  public static final class Lambda implements Name<Float> {
  }

  @NamedParameter(doc = "ratio which learning rate decreases by (multiplicative). this value must be larger than 0 " +
                 "and less than or equal to 1. if decay_rate=1.0, decaying process is turned off.",
                 short_name = "decay_rate",
                 default_value = "0.9")
  public static final class DecayRate implements Name<Float> {
  }

  @NamedParameter(doc = "number of epochs to wait until learning rate decreases (periodic). this value must be " +
                  "a positive value.",
                  short_name = "decay_period",
                  default_value = "5")
  public static final class DecayPeriod implements Name<Integer> {
  }

  @NamedParameter(doc = "standard deviation of the gaussian distribution used for initializing model parameters",
                  short_name = "model_gaussian",
                  default_value = "0.001")
  public static final class ModelGaussian implements Name<Double> {
  }

  @NamedParameter(doc = "number of features for each model partition",
                  short_name = "features_per_partition",
                  default_value = "0")
  public static final class NumFeaturesPerPartition implements Name<Integer> {
  }
}
