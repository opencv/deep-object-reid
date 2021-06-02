import os
import logging as log

import cv2 as cv
import numpy as np

try:
    from openvino.inference_engine.ie_api import IECore
except ImportError:
    # If OpenVINO import fails, add the library dir to PATH
    # https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_pip.html
    import os
    import sys
    if "win" in sys.platform:
        try:
            library_dir = os.path.dirname(sys.executable) + "\..\Library\\bin"
            current_path = os.environ.get("PATH", "")
            os.environ["PATH"] = f"{library_dir};{current_path}"
            from openvino.inference_engine.ie_api import IECore
        except ImportError:
            raise ImportError(
                "Please add the library dir to the system PATH. See https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_pip.html")
    else:
        raise ImportError(
            "Please add the library dir to the system PATH. See https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_pip.html")

class IEModel:
    """Class for inference of models in the Inference Engine format"""
    def __init__(self, exec_net, inputs_info, input_key, output_key):
        self.net = exec_net
        self.inputs_info = inputs_info
        self.input_key = input_key
        self.output_key = output_key
        self.reqs_ids = []

    def _preprocess(self, img):
        _, _, h, w = self.get_input_shape()
        img = np.expand_dims(cv.resize(img, (w, h)).transpose(2, 0, 1), axis=0)
        return img

    def forward(self, img):
        """Performs forward pass of the wrapped IE model"""
        res = self.net.infer(inputs={self.input_key: self._preprocess(img)})
        return list(res.values())

    def forward_async(self, img):
        id_ = len(self.reqs_ids)
        self.net.start_async(request_id=id_,
                             inputs={self.input_key: self._preprocess(img)})
        self.reqs_ids.append(id_)

    def grab_all_async(self):
        outputs = []
        for id_ in self.reqs_ids:
            self.net.requests[id_].wait(-1)
            output_list = [self.net.requests[id_].output_blobs[key].buffer for key in self.output_key]
            outputs.append(output_list)
        self.reqs_ids = []
        return outputs

    def get_input_shape(self):
        """Returns an input shape of the wrapped IE model"""
        return self.inputs_info[self.input_key].input_data.shape


def load_ie_model(ie, model_xml, device, plugin_dir, cpu_extension='', num_reqs=1):
    """Loads a model in the Inference Engine format"""
    # Plugin initialization for specified device and load extensions library if specified
    log.info("Initializing Inference Engine plugin for %s ", device)

    if cpu_extension and 'CPU' in device:
        ie.add_extension(cpu_extension, 'CPU')
    # Read IR
    log.info("Loading network")
    net = ie.read_network(model_xml, os.path.splitext(model_xml)[0] + ".bin")
    log.info("Preparing input blobs")
    input_blob = next(iter(net.input_info))
    out_blob = net.outputs.keys()
    net.batch_size = 1

    # Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=device, num_requests=num_reqs)
    model = IEModel(exec_net, net.input_info, input_blob, out_blob)
    return model


class IEClassifier:
    """Wrapper class for classification model"""
    def __init__(self, model_path, ie=IECore(), device='CPU', ext_path=''):
        self.net = load_ie_model(ie, model_path, device, None, ext_path)

    def get_detections(self, img):
        """Returns an index of predicted class"""
        out = self.net.forward(img)
        return np.argmax(out[0])
