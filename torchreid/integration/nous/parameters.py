from sc_sdk.configuration.configurable_parameters import Group, Integer, Float, Object, Boolean
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

    learning_parameters: __LearningParameters = Object(__LearningParameters)
