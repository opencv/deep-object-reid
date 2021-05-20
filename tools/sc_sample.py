import os
import sys
import os.path as osp
import random
from typing import Tuple

import cv2 as cv
import matplotlib.pyplot as plt
import seaborn as sns  # optional: to show confusion matrix
import sklearn.metrics as skm

from sc_sdk.entities.analyse_parameters import AnalyseParameters
from sc_sdk.entities.annotation import Annotation, AnnotationKind
from sc_sdk.entities.datasets import Dataset, Subset
from sc_sdk.entities.image import Image
from sc_sdk.entities.label import ScoredLabel
from sc_sdk.entities.project import Project
from sc_sdk.entities.resultset import ResultSet
from sc_sdk.entities.shapes.box import Box
from sc_sdk.entities.task_environment import TaskEnvironment
from sc_sdk.entities.url import URL
from sc_sdk.tests.test_helpers import generate_training_dataset_of_all_annotated_media_in_project
from sc_sdk.usecases.repos import *
from sc_sdk.usecases.repos import BinaryRepo
from sc_sdk.usecases.repos.optimized_model_repo import OptimizedModelRepo
from sc_sdk.utils.project_factory import ProjectFactory

from torchreid.integration.sc.task import TorchClassificationTask


def load_annotation(data_dir, filter_classes=None, dataset_id=0):
    ALLOWED_EXTS = ('.jpg', '.jpeg', '.png', '.gif')
    def is_valid(filename):
        return not filename.startswith('.') and filename.lower().endswith(ALLOWED_EXTS)

    def find_classes(dir, filter_names=None):
        if filter_names:
            classes = [d.name for d in os.scandir(dir) if d.is_dir() and d.name in filter_names]
        else:
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return class_to_idx

    class_to_idx = find_classes(data_dir, filter_classes)

    out_data = []
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = osp.join(data_dir, target_class)
        if not osp.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = osp.join(root, fname)
                if is_valid(path):
                    out_data.append((path, class_index, 0, dataset_id, '', -1, -1))\

    if not len(out_data):
        print('Failed to locate images in folder ' + data_dir + f' with extensions {ALLOWED_EXTS}')

    return out_data, class_to_idx


def createproject(projectname, taskname, basedir):
	anno, classes = load_annotation(basedir)
	project = ProjectFactory().create_project_single_task(name=projectname, description="",
														  label_names=list(classes.values()), task_name=taskname)
	for i, item in enumerate(anno):
		imdata = cv.imread(item[0])
		imdata = cv.cvtColor(imdata, cv.COLOR_RGB2BGR)
		image = Image(name=f"{os.path.basename(item[0])}", project=project, numpy=imdata)
		ImageRepo(project).save(image)
		label = [label for label in project.get_labels() if label.name==item[1]][0]
		shapes = [Box.generate_full_box(labels=[ScoredLabel(label)])]
		annotation = Annotation(kind=AnnotationKind.ANNOTATION, media_identifier=image.media_identifier, shapes=shapes)
		AnnotationRepo(project).save(annotation)

	print('Data loaded')
	ProjectRepo().save(project)
	dataset = generate_training_dataset_of_all_annotated_media_in_project(project)
	DatasetRepo(project).save(dataset)
	print('Dataset generated')
	return project, dataset


basedir = os.path.expanduser(sys.argv[1])
projectname = "Torch classification"
project, dataset = createproject(projectname, "ClassificationCLASS", basedir)
print([task.task_name for task in project.tasks])
classification_environment = TaskEnvironment(project=project, task_node=project.tasks[-1])

classification_task = TorchClassificationTask(task_environment=classification_environment)

classification_model = classification_task.train(dataset=dataset)
ModelRepo(project).save(classification_model)
DatasetRepo(project).save(dataset)


validation_dataset = dataset.get_subset(Subset.VALIDATION)
print(f"validation dataset: {len(validation_dataset)} items")

predicted_validation_dataset = classification_task.analyse(
    validation_dataset.with_empty_annotations(), AnalyseParameters(is_evaluation=True))

resultset = ResultSet(
    model=classification_model,
    ground_truth_dataset=validation_dataset,
    prediction_dataset=predicted_validation_dataset,
)
ResultSetRepo(project).save(resultset)

performance = classification_task.compute_performance(resultset)
resultset.performance = performance
ResultSetRepo(project).save(resultset)
optimezed_model = classification_task.optimize_loaded_model()

print(resultset.performance)
