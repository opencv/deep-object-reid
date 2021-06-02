import abc
import os
import os.path as osp
import argparse
import json
from typing import Union

import cv2 as cv
from zipfile import ZipFile

from sc_sdk.entities.analyse_parameters import AnalyseParameters
from sc_sdk.entities.annotation import Annotation, AnnotationKind
from sc_sdk.entities.datasets import Subset
from sc_sdk.entities.image import Image
from sc_sdk.entities.label import ScoredLabel
from sc_sdk.entities.project import Project
from sc_sdk.entities.resultset import ResultSet
from sc_sdk.entities.shapes.box import Box
from sc_sdk.entities.task_environment import TaskEnvironment
from sc_sdk.entities.url import URL
from sc_sdk.tests.test_helpers import generate_training_dataset_of_all_annotated_media_in_project
from sc_sdk.usecases.repos import *
from sc_sdk.utils.project_factory import ProjectFactory
from sc_sdk.usecases.adapters.binary_interpreters import RAWBinaryInterpreter
from sc_sdk.entities.optimized_model import OptimizedModel
from sc_sdk.communication.mappers.mongodb_mapper import LabelToMongo
from sc_sdk.logging import logger_factory

from torchreid.integration.sc.task import TorchClassificationTask

logger = logger_factory.get_logger("Classification_sample")

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
		if i > 1000:
			break

	print('Data loaded')
	ProjectRepo().save(project)
	dataset = generate_training_dataset_of_all_annotated_media_in_project(project)
	DatasetRepo(project).save(dataset)
	print('Dataset generated')
	return project, dataset

def zip_folder_or_file(zip_file: ZipFile, path: str, folder_prefix: str = "",
                       skip_parent: bool = False):
    """
    Recursively (in case path is a folder) put the content of file_path into zip_file.
    :param zip_file: the zip file object
    :param path: path to the folder/file to be added to the zip file
    :param folder_prefix: the directory inside the zip file in which the data will be stored
    :param skip_parent: if this is set to True, only the content of the path will be stored inside folder_prefix,
        not the directory.
    """
    dir_, original_filename = os.path.split(path)
    if not os.path.isdir(path):
        zip_file.write(path, os.path.join(folder_prefix, original_filename))
    else:
        if not skip_parent:
            # only update folder prefix if skip_parent is False
            folder_prefix = os.path.join(folder_prefix, original_filename)

        for filename in os.listdir(path):
            zip_folder_or_file(zip_file, os.path.join(path, filename),
                               folder_prefix, skip_parent=False)

class IZippedObjectEntry:
    """
    Abstract class representing objects which can be written to a zip file.
    """

    @abc.abstractmethod
    def __repr__(self):
        pass

    @abc.abstractmethod
    def write_to_zip_file(self, zip_file: ZipFile):
        """
        Write the object to a given zip file.
        :param zip_file: Zip file
        """
        pass


class ZippedFileEntry(IZippedObjectEntry):
    """
    This class represents one file which will be zipped and exported.
    :param filepath: the filepath inside the zip file where data can be found.
    :param data: this data will be stored inside a file whose path as defined in filepath.
    """

    def __init__(self, filepath: str, data: Union[str, bytes]):
        self.filepath: str = filepath
        self.data: Union[str, bytes] = data

    def __repr__(self):
        return f"ZippedFileEntry({self.filepath}, {len(self.data)} bytes)"

    def write_to_zip_file(self, zip_file: ZipFile):
        zip_file.writestr(self.filepath, self.data)


class OptimizedModelExporter:
    @staticmethod
    def export_optimized_model(root_project: Project, optimized_model: OptimizedModel):
        try:
            binary_interpreter = RAWBinaryInterpreter()
            openvino_xml_data = BinaryRepo(root_project).get_by_url(optimized_model.openvino_xml_url,
                                                                    binary_interpreter=binary_interpreter)
            openvino_bin_data = BinaryRepo(root_project).get_by_url(optimized_model.openvino_bin_url,
                                                                    binary_interpreter=binary_interpreter)
            yield ZippedFileEntry(f"optimized_models/{optimized_model.precision.name}/inference_model.xml",
                                    openvino_xml_data)
            yield ZippedFileEntry(f"optimized_models/{optimized_model.precision.name}/inference_model.bin",
                                    openvino_bin_data)
            label_data = OptimizedModelExporter.generate_label_data(optimized_model.model)
            yield ZippedFileEntry(f"optimized_models/{optimized_model.precision.name}/labels.json", label_data)
        except FileNotFoundError:
            logger.warning(f"Failed to export the optimized model {optimized_model.name} "
                            f"because the file is no longer available.")

    @staticmethod
    def generate_label_data(model) -> str:
        labels = model.configuration.labels
        mapped_labels = []
        for label in labels:
            # FIXME.
            mapped_labels.append(LabelToMongo().forward(label))
        for label in mapped_labels:
            label["_id"] = str(label["_id"])
            label["task_id"] = str(label["task_id"])
            label["creation_date"] = label["creation_date"].isoformat()
        return json.dumps(mapped_labels)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datadir', type=str, default='')
    parser.add_argument('--export_zip_file_path', type=str,default='')
    args = parser.parse_args()

    projectname = "Torch classification"
    project, dataset = createproject(projectname, "ClassificationCLASS", args.datadir)
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

    optimized_model = classification_task.optimize_loaded_model()[0]

    print(resultset.performance)

    if args.export_zip_file_path:
        with ZipFile(args.export_zip_file_path, "w") as zip_file:
            for file_entry in OptimizedModelExporter.export_optimized_model(project, optimized_model):
                file_entry.write_to_zip_file(zip_file)
            #file_entry = ZippedFileEntry(f"optimized_models/{optimized_model.precision.name}/configurable_parameters.json",
            #                            json.dumps(task.get_configurable_parameters(environment).serialize()))
            #file_entry.write_to_zip_file(zip_file)


if __name__ == '__main__':
    main()
