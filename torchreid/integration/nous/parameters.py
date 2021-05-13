from pathlib import Path

from torchreid.integration.nous.utils import list_available_models

from sc_sdk.configuration.configurable_parameters import Group, Integer, Float, Object, Selectable, Option
from sc_sdk.configuration.deep_learning_configurable_parameters import DeepLearningConfigurableParameters


class ClassificationParameters(DeepLearningConfigurableParameters):
    class __LearningParameters(Group):
        header = "Learning parameters"
        description = header

        batch_size = Integer(header="Batch size",
                             default_value=32,
                             min_value=1,
                             max_value=512,
                             editable=True)

        test_batch_size = Integer(header="Test batch size",
                                  default_value=32,
                                  min_value=1,
                                  max_value=512,
                                  editable=True)

        max_num_epochs = Integer(header="Maximum number of epochs",
                                 default_value=5,
                                 min_value=1,
                                 max_value=1000,
                                 editable=True)

        base_learning_rate = Float(header="Learning rate",
                                   default_value=0.01,
                                   min_value=1e-06,
                                   max_value=1e-01,
                                   editable=True)


    class __LearningArchitecture(Group):
        header = "Learning Architecture"
        description = header
        base_models_dir = Path(__file__).parent.parent.parent.parent / 'configs/ote_custom_classification/'
        available_models = list_available_models(str(base_models_dir))
        model_architecture = Selectable(header="Model architecture",
                                        default_value=available_models[0]['name'],
                                        options=[Option(key=x['name'], value=x['name'], description='') for x in available_models],
                                        description="Specify learning architecture for the the task.",
                                        editable=True)

    learning_parameters: __LearningParameters = Object(__LearningParameters)
    learning_architecture: __LearningArchitecture = Object(__LearningArchitecture)
