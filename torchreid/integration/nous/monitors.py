"""
 Copyright (c) 2021 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import abc
import time

from torch.utils.tensorboard import SummaryWriter


class IMetricsMonitor(metaclass=abc.ABCMeta):
    """
    Interface for providing Tensorboard-style logging
    """

    @abc.abstractmethod
    def add_scalar(self, capture: str, value: float, timestamp: int):
        """
        Similar to Tensorboard method that allows to log values of named scalar variables
        """

    @abc.abstractmethod
    def close(self):
        """
        Flushes the collected scalar values to log, closes corresponding file,
        then resets the monitor to default state
        """


class IPerformanceMonitor(metaclass=abc.ABCMeta):
    """
    Interface for collecting training performance numbers
    """
    @abc.abstractmethod
    def init(self, total_epochs: int, num_train_steps: int, num_validation_steps: int):
        """
        Initializes the monitor with the training parameters
        """

    @abc.abstractmethod
    def on_train_batch_begin(self):
        """
        Method starts timer that measures batch forward-backward time during training
        """

    @abc.abstractmethod
    def on_train_batch_end(self):
        """
        Method stops timer that measures batch forward-backward time during training
        """

    @abc.abstractmethod
    def on_val_batch_begin(self):
        """
        Method starts timer that measures batch forward-backward time during evaluation
        """

    @abc.abstractmethod
    def on_val_batch_end(self):
        """
        Method stops timer that measures batch forward-backward time during evaluation
        """

    @abc.abstractmethod
    def on_train_begin(self):
        """
        Method notifies the monitor that training has begun
        """

    @abc.abstractmethod
    def on_train_end(self):
        """
        Method notifies the monitor that training has finished
        """

    @abc.abstractmethod
    def on_train_epoch_begin(self):
        """
        Method notifies the monitor that the next training epoch has begun
        """

    @abc.abstractmethod
    def on_train_epoch_end(self):
        """
        Method notifies the monitor that training epoch has finished
        """

    @abc.abstractmethod
    def get_training_progress(self) -> int:
        """
        Returns progress of the training process in range of 0 to 100
        """

    @abc.abstractmethod
    def get_val_progress(self) -> int:
        """
        Returns progress of the evaluation process in range of 0 to 100
        """

    @abc.abstractmethod
    def get_val_eta(self) -> int:
        """
        Returns ETA of the evaluation process in seconds
        """

    @abc.abstractmethod
    def get_training_eta(self) -> int:
        """
        Returns ETA of the training process in seconds
        """


class IStopCallback(metaclass=abc.ABCMeta):
    """
    Interface for wrapping a stop signal.
    By default an object implementing this interface should be in permissive state
    """

    @abc.abstractmethod
    def check_stop(self) -> bool:
        """
        Method returns True if the object of this class is in stopping state,
        otherwise it returns False
        """

    @abc.abstractmethod
    def reset(self):
        """
        Method resets the internal state of the object to permissive
        """

class StopCallback(IStopCallback):
    def __init__(self):
        self.stop_flag = False

    def stop(self):
        self.stop_flag = True

    def check_stop(self):
        return self.stop_flag

    def reset(self):
        self.stop_flag = False


class MetricsMonitor(IMetricsMonitor):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.tb = None

    def add_scalar(self, capture: str, value: float, timestamp: int):
        if not self.tb:
            self.tb = SummaryWriter(self.log_dir)
        self.tb.add_scalar(capture, value, timestamp)

    def close(self):
        if self.tb:
            self.tb.close()
            self.tb = None


class PerformanceMonitor(IPerformanceMonitor):
    def init(self, total_epochs: int, num_train_steps: int, num_validation_steps: int):
        self.total_epochs = total_epochs
        self.num_train_steps = num_train_steps
        self.num_validation_steps = num_validation_steps

        self.train_steps_passed = 0
        self.val_steps_passed = 0
        self.train_epochs_passed = 0

        self.granularity = 1. / total_epochs
        self.start_epoch_time = 0
        self.start_batch_time = 0
        self.avg_epoch_time = 0
        self.avg_batch_time = 0
        self.avg_val_batch_time = 0

    def on_train_batch_begin(self):
        self.start_batch_time = time.time()

    def on_train_batch_end(self):
        self.train_steps_passed += 1
        batch_time = time.time() - self.start_batch_time
        self.avg_batch_time = (self.avg_batch_time * \
            (self.train_steps_passed - 1) + batch_time) / self.train_steps_passed

    def on_val_batch_begin(self):
        self.start_batch_time = time.time()

    def on_val_batch_end(self):
        self.val_steps_passed += 1
        batch_time = time.time() - self.start_batch_time
        self.avg_val_batch_time = (self.avg_val_batch_time * \
            (self.val_steps_passed - 1) + batch_time) / self.val_steps_passed

    def on_train_begin(self):
        self.train_steps_passed = 0
        self.val_steps_passed = 0

    def on_train_end(self):
        pass

    def on_train_epoch_begin(self):
        self.start_epoch_time = time.time()

    def on_train_epoch_end(self):
        self.val_steps_passed = 0
        self.train_epochs_passed += 1
        epoch_time = time.time() - self.start_epoch_time
        self.avg_epoch_time = (self.avg_epoch_time * \
            (self.train_epochs_passed - 1) + epoch_time) / self.train_epochs_passed

    def get_training_progress(self) -> int:
        return int(self.train_epochs_passed / self.total_epochs * 100) + \
               int(self.granularity * (self.train_steps_passed % self.num_train_steps) / self.num_train_steps * 100)

    def get_val_progress(self) -> int:
        print(self.val_steps_passed, self.num_validation_steps)
        return int(self.val_steps_passed / self.num_validation_steps * 100)

    def get_val_eta(self) -> int:
        remaining_steps = self.num_validation_steps - self.val_steps_passed
        remaining_batch_time = remaining_steps * self.avg_batch_time
        return int(remaining_batch_time)

    def get_training_eta(self) -> int:
        remaining_steps = self.num_train_steps - self.train_steps_passed % self.num_train_steps
        remaining_epochs = self.total_epochs - self.train_epochs_passed

        remaining_batch_time = remaining_steps * self.avg_batch_time
        remaining_epoch_time = remaining_epochs * self.avg_epoch_time

        if remaining_epoch_time == 0:
            remaining_time = remaining_batch_time * 1.5
        else:
            remaining_time = remaining_epoch_time
            epoch_completion = (self.train_steps_passed % self.num_train_steps) / self.num_train_steps
            remaining_time -= epoch_completion * self.avg_epoch_time

        return int(remaining_time)


class DefaultStopCallback(IStopCallback):
    def stop(self):
        pass

    def check_stop(self):
        pass

    def reset(self):
        pass


class DefaultMetricsMonitor(IMetricsMonitor):
    def __init__(self):
        self.metrics_dict = {}

    def add_scalar(self, capture: str, value: float, timestamp: int):
        if capture in self.metrics_dict:
            self.metrics_dict[capture].append(value)
        else:
            self.metrics_dict[capture] = [value,]

    def get_metric_values(self, capture):
        return self.metrics_dict[capture]

    def close(self):
        pass


class DefaultPerformanceMonitor(IPerformanceMonitor):
    def init(self, total_epochs: int, num_train_steps: int, num_validation_steps: int):
        pass

    def on_train_batch_begin(self):
        pass

    def on_train_batch_end(self):
        pass

    def on_val_batch_begin(self):
        pass

    def on_val_batch_end(self):
        pass

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

    def on_train_epoch_begin(self):
        pass

    def on_train_epoch_end(self):
        pass

    def get_training_progress(self) -> int:
        return 0

    def get_val_progress(self) -> int:
        return 0

    def get_val_eta(self) -> int:
        return 0

    def get_training_eta(self) -> int:
        return 0
