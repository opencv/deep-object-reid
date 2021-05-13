import io
import math
from typing import List, Optional
from copy import deepcopy
from pathlib import Path
import tempfile
import shutil

import torch
from torchvision import transforms

import torchreid
from torchreid.engine import build_engine
from torchreid.ops import DataParallel
from torchreid.utils import load_pretrained_weights, set_random_seed
from scripts.default_config import (engine_run_kwargs, get_default_config,
                                    imagedata_kwargs, lr_finder_run_kwargs,
                                    lr_scheduler_kwargs, model_kwargs,
                                    optimizer_kwargs)
from torchreid.integration.sc.monitors import PerformanceMonitor, StopCallback, DefaultMetricsMonitor
from torchreid.integration.sc.utils import (ClassificationImageFolder, CannotLoadModelException,
                                              generate_batch_indices, predict, list_available_models)
from torchreid.integration.sc.parameters import ClassificationParameters

from sc_sdk.entities.analyse_parameters import AnalyseParameters
from sc_sdk.entities.datasets import Dataset, DatasetItem, Subset
from sc_sdk.entities.label_relations import LabelGroupType
from sc_sdk.entities.metrics import Performance, MetricsGroup, CurveMetric, LineChartInfo
from sc_sdk.entities.model import Model, NullModel
from sc_sdk.entities.result_media import ResultMedia
from sc_sdk.entities.resultset import ResultSetEntity
from sc_sdk.entities.task_environment import TaskEnvironment
from sc_sdk.entities.train_parameters import TrainParameters
from sc_sdk.logging import logger_factory
from sc_sdk.usecases.evaluation.metrics_helper import MetricsHelper
from sc_sdk.usecases.reporting.time_monitor_callback import TimeMonitorCallback
from sc_sdk.usecases.repos import ModelRepo
from sc_sdk.usecases.tasks.image_deep_learning_task import ImageDeepLearningTask
from sc_sdk.usecases.tasks.interfaces.configurable_parameters_interface import IConfigurableParameters

logger = logger_factory.get_logger("TorchClassificationTask")


class TorchClassificationTask(ImageDeepLearningTask, IConfigurableParameters):
    """
    Start by making a class. Since we will make a deep learning task with 2d support, we inherit from ImageDeepLearningTask.
    Additionally, this task will support configurable parameters. So we add the interface IConfigurableParameters as well.

    Let's save the task environment given. The task environment is very important, it contains the context of the task.
    Think of the loaded project, the labels, the task configuration and stored configurable parameters.
    """

    def __init__(self, task_environment: TaskEnvironment, configs_root: str =''):
        logger.info(f"Loading classification task with task id {task_environment.task_node.id}")

        self.task_environment = task_environment

        self.model_repository = ModelRepo(task_environment.project)

        # Define monitors
        self.stop_callback = StopCallback()
        self.metrics_monitor = DefaultMetricsMonitor()
        self.perf_monitor = PerformanceMonitor()

        # Initialize and load models
        self.cfg = get_default_config()
        self.configs_root = configs_root
        if not configs_root:
            self.configs_root = Path(__file__).parent.parent.parent.parent / 'configs/ote_custom_classification/'
        self.all_models = list_available_models(str(self.configs_root))
        configurable_parameters = self.get_configurable_parameters(self.task_environment)
        self.model_name = configurable_parameters.learning_architecture.model_architecture.value
        self.switch_arch(self.model_name)

        self.device = torch.device("cuda:0") if torch.cuda.device_count() else torch.device("cpu")
        self.model = self.create_model().to(self.device)
        self.load_model(self.task_environment)

    def switch_arch(self, new_arch_name):
        model_info = filter(lambda x: x['name'] == new_arch_name, self.all_models)
        model_dir = list(model_info)[0]['dir']
        cfg_path = Path(model_dir) / 'main_model.yaml'

        self.cfg = get_default_config()
        self.cfg.merge_from_file(str(cfg_path))
        self.cfg.use_gpu = torch.cuda.device_count() > 0
        self.num_devices = 1 if self.cfg.use_gpu else 0
        self.cfg.model.classification = True
        self.cfg.custom_datasets.types = ['external_classification_wrapper', 'external_classification_wrapper']
        self.cfg.custom_datasets.names = ['train', 'val']
        self.cfg.custom_datasets.roots = ['']*2
        self.cfg.data.sources = ['train']
        self.cfg.data.targets = ['val']
        self.num_classes = len(self.task_environment.labels)

        for i, conf in enumerate(self.cfg.mutual_learning.aux_configs):
            if str(model_dir) not in conf:
                self.cfg.mutual_learning.aux_configs[i] = Path(model_dir) / conf

    def cancel_training(self):
        """
        Called when the user wants to abort training.
        In this example, this is not implemented.

        :return: None
        """
        self.stop_callback.stop()

    def get_training_progress(self) -> float:
        """
        Returns the progress for training. Returns -1 if this is not known.

        :return: Float with progress [0.0-100.0] or -1.0 if unknown
        """
        return self.perf_monitor.get_training_progress()

    @staticmethod
    def get_configurable_parameters(task_environment: TaskEnvironment) -> ClassificationParameters:
        """
        Returns the configurable parameters.

        :param task_environment: Current task environment

        :return: Instance of ClassificationParameters
        """
        return task_environment.get_configurable_parameters(instance_of=ClassificationParameters)

    def update_configurable_parameters(self, task_environment: TaskEnvironment):
        """
        Called when the user changes the configurable parameters in the UI.

        :param task_environment: New task environment with updated configurable parameters
        """
        self.task_environment = task_environment

    def load_model(self, task_environment: TaskEnvironment):
        """
        Load the given model.

        This function provides an updated task environment with the model to load.

        :param task_environment: New task environment with the updated model.
        """
        self.task_environment = task_environment

        model = task_environment.model

        if model != NullModel():
            weights = io.BytesIO(model.data)
            logger.info(f"Loading model from: {model.data_url}")
            try:
                torch_model = self.create_model()
                state_dict = torch.load(weights)
                load_pretrained_weights(torch_model, pretrained_dict=state_dict)
                self.model = torch_model
            except BaseException as ex:
                raise CannotLoadModelException("Could not load the saved model. The model file structure is invalid.") \
                    from ex

    def get_model_bytes(self) -> bytes:
        """
        Returns the data of the current model

        :return: data of current model in bytes
        """
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        return buffer.getvalue()

    def create_model(self):
        group = self.task_environment.label_relations.get_groups_by_task_id(
            task_id=self.task_environment.task_node.id)[0]
        label_relation_type = group.group_type
        if label_relation_type == LabelGroupType.EXCLUSIVE:
            print('creating exclusive model')
            num_train_classes = len(self.task_environment.labels)
            model = torchreid.models.build_model(**model_kwargs(self.cfg, num_train_classes))
            return model
        else:
            raise ValueError(f"This task does not support label relations of type {label_relation_type.name}")

    def analyse(self, dataset: Dataset, analyse_parameters: Optional[AnalyseParameters] = None) -> Dataset:
        """
        Perform inference on the given dataset.

        :param dataset: Dataset entity to analyse
        :param analyse_parameters: Additional parameters for inference.
            For example, when results are generated for evaluation purposes, Saliency maps can be turned off.
        :return: Dataset that also includes the classification results
        """
        configurable_parameters = self.get_configurable_parameters(self.task_environment)
        batch_size = configurable_parameters.learning_parameters.test_batch_size.value
        alllabels = self.task_environment.labels
        datamanager = torchreid.data.ImageDataManager(**imagedata_kwargs(self.cfg))

        for ii, batch_slice in enumerate(generate_batch_indices(len(dataset), batch_size)):
            """
            Iterate over slices, with a maximum size of the batch size
            """
            dataset_slice: List[DatasetItem] = dataset[batch_slice]

            # Get labels and saliency for given slice
            outputs_batch = predict(dataset_slice=dataset_slice, labels=alllabels, model=self.model,
                                    transform=datamanager.transform_te,
                                    device=self.device)

            # Iterate result slice, and append the labels and saliency to each dataset item.
            for j, (dataset_row, scored_labels) in enumerate(outputs_batch):
                dataset_slice[j].append_labels(labels=scored_labels)

        return dataset

    def generate_training_metrics_group(self) -> List[MetricsGroup]:
        """
        Create additional metrics for the Dashboard.

        :return: List of MetricGroup
        """
        output = []
        if self.metrics_monitor is not None:
            loss = CurveMetric(ys=self.metrics_monitor.get_metric_values('Loss/loss'), name="Training")
            visualization_info = LineChartInfo(name="Loss curve", x_axis_label="Iteration", y_axis_label="Loss value")
            output.append(MetricsGroup(metrics=[loss], visualization_info=visualization_info))
        return output

    def train(self, dataset: Dataset, train_parameters: Optional[TrainParameters] = None) -> Model:
        """
        Train the model given the Dataset.
        If the task finds it was able to improve the model, it will return a new Model entity,
        otherwise if the task finds the model was not able to improve, it can return a NullModel.
        If this happens, NOUS will not update the model repository.

        :param dataset: Dataset to use for training. Contains properly split subsets of Train, Validation, Testing.
        :param train_parameters: Define additional parameters, such as train_on_empty_model.
            train_on_empty_model means that a pre-trained model or
            randomised weights should be used instead of the latest model.
        :return: Model used by the task at the end of training.
            This might be a new model or an old one if training failed.
        """
        configurable_parameters = self.get_configurable_parameters(self.task_environment)
        if configurable_parameters.learning_architecture.model_architecture.value != self.model_name:
            self.model_name = configurable_parameters.learning_architecture.model_architecture.value
            self.switch_arch(self.model_name)

        if train_parameters is not None and train_parameters.train_on_empty_model:
            train_model = self.create_model()
        else:
            train_model = deepcopy(self.model)
        self.cfg.data.save_dir = tempfile.mkdtemp()

        self.cfg.train.batch_size = configurable_parameters.learning_parameters.batch_size.value
        self.cfg.train.lr = configurable_parameters.learning_parameters.base_learning_rate.value
        self.cfg.train.max_epoch = configurable_parameters.learning_parameters.max_num_epochs.value

        train_steps = math.ceil(len(dataset.get_subset(Subset.TRAINING)) / self.cfg.train.batch_size)
        validation_steps = math.ceil((len(dataset.get_subset(Subset.VALIDATION)) / self.cfg.test.batch_size))
        self.perf_monitor.init(self.cfg.train.max_epoch, train_steps, validation_steps)
        self.metrics_monitor = DefaultMetricsMonitor()
        self.stop_callback.reset()

        set_random_seed(self.cfg.train.seed)
        labels = self.task_environment.labels
        train_subset = dataset.get_subset(Subset.TRAINING)
        val_subset = dataset.get_subset(Subset.VALIDATION)
        self.cfg.custom_datasets.roots = [ClassificationImageFolder(train_subset, labels), ClassificationImageFolder(val_subset, labels)]
        datamanager = torchreid.data.ImageDataManager(**imagedata_kwargs(self.cfg))

        if self.num_classes != datamanager.num_train_pids:
            self.num_classes = datamanager.num_train_pids
            train_model = self.create_model()

        num_aux_models = len(self.cfg.mutual_learning.aux_configs)

        if self.cfg.use_gpu:
            main_device_ids = list(range(self.num_devices))
            extra_device_ids = [main_device_ids for _ in range(num_aux_models)]
            self.model = DataParallel(self.model, device_ids=main_device_ids, output_device=0).cuda(main_device_ids[0])
        else:
            extra_device_ids = [None for _ in range(num_aux_models)]

        optimizer = torchreid.optim.build_optimizer(self.model, **optimizer_kwargs(self.cfg))

        if self.cfg.lr_finder.enable and self.cfg.lr_finder.mode == 'automatic': # and not parameters.resume_from:
            scheduler = None
        else:
            scheduler = torchreid.optim.build_lr_scheduler(optimizer, **lr_scheduler_kwargs(self.cfg))
        '''
        if parameters.resume_from:
            self.cfg.train.start_epoch = resume_from_checkpoint(
                parameters.resume_from, self.model, optimizer=optimizer, scheduler=scheduler
            )
        '''

        lr = None # placeholder, needed for aux models
        if self.cfg.lr_finder.enable: # and not parameters.resume_from:
            if num_aux_models:
                print("Mutual learning is enabled. Learning rate will be estimated for the main model only.")

            # build new engine
            engine = build_engine(self.cfg, datamanager, self.model, optimizer, scheduler)
            lr = engine.find_lr(**lr_finder_run_kwargs(self.cfg), stop_callback=self.stop_callback)

            print(f"Estimated learning rate: {lr}")
            if self.cfg.lr_finder.stop_after:
                print("Finding learning rate finished. Terminate the training process")
                return

            # reload random seeds, optimizer with new lr and scheduler for it
            self.cfg.train.lr = lr
            self.cfg.lr_finder.enable = False
            set_random_seed(self.cfg.train.seed)

            optimizer = torchreid.optim.build_optimizer(train_model, **optimizer_kwargs(self.cfg))
            scheduler = torchreid.optim.build_lr_scheduler(optimizer, **lr_scheduler_kwargs(self.cfg))

        if num_aux_models:
            print('Enabled mutual learning between {} models.'.format(num_aux_models + 1))

            models, optimizers, schedulers = [train_model], [optimizer], [scheduler]
            for config_file, device_ids in zip(self.cfg.mutual_learning.aux_configs, extra_device_ids):
                aux_model, aux_optimizer, aux_scheduler = build_auxiliary_model(
                    config_file, self.num_classes, self.cfg.use_gpu, device_ids, lr
                )

                models.append(aux_model)
                optimizers.append(aux_optimizer)
                schedulers.append(aux_scheduler)
        else:
            models, optimizers, schedulers = train_model, optimizer, scheduler

        print('Building {}-engine for {}-reid'.format(self.cfg.loss.name, self.cfg.data.type))
        engine = build_engine(self.cfg, datamanager, models, optimizers, schedulers)
        engine.run(**engine_run_kwargs(self.cfg), tb_writer=self.metrics_monitor, perf_monitor=performance_monitor,
                   stop_callback=self.stop_callback)

        train_model = train_model.module
        #self.metrics_monitor.close()
        if self.stop_callback.check_stop():
            print('Training has been canceled')

        logger.info("Training finished, and it has an improved model")
        self.model = deepcopy(train_model)
        model_data = self.get_model_bytes()
        model = Model(project=self.task_environment.project,
                        task_node=self.task_environment.task_node,
                        configuration=self.task_environment.get_model_configuration(),
                        data=model_data,
                        tags=["classification_model"],
                        train_dataset=dataset)

        self.task_environment.model = model

        self.progress_monitor = None
        shutil.rmtree(self.cfg.data.save_dir)

        return self.task_environment.model

    def compute_performance(self, resultset: ResultSetEntity) -> Performance:
        """
        Compute the performance over a given resultset.
        Adds additional dashboard metrics.

        :param resultset: ResultSet to evaluate
        :return: Performance entity
        """
        performance = MetricsHelper.compute_accuracy(resultset).get_performance()
        performance.dashboard_metrics = self.generate_training_metrics_group()
        logger.info(f"Computes performance of {performance}")
        return performance
