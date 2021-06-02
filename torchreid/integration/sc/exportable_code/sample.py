"""
This script will take either a video file or capture footage from the first available camera on the
system and pass it through an image classification model for inference. Finally the frame will be shown along with the results
from the inference on the top left. Press Q to quit the program.
"""
import json
from argparse import ArgumentParser

import cv2 as cv
import numpy as np
from PIL import ImageDraw, ImageFont, Image
from torchvision import transforms

from sc_sdk.communication.mappers.mongodb_mapper import LabelToMongo
from sc_sdk.entities.label import Label

from ie_utils import IEClassifier


def draw_label_on_image(image: Image, label: Label):
    """
    Draws label string on the image on the top left corner
    :param image:
    :param label:
    """
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 48)
    except IOError:
        font = ImageFont.load_default()

    text_width, text_height = font.getsize(str(label.name))
    margin = np.ceil(0.1 * text_height)
    draw.rectangle(
        [(0, 0), (text_width + margin, text_height + margin)],
        fill=label.color.bgr_tuple)
    draw.text(
        (0, 0),
        str(label.name),
        fill='black',
        font=font)


def streamer(file_path: str):
    """
    Start by making an iterator that will be responsible for handling the data that will be processed. In this case the
    iterator is capable of either handling a video file or grabs frames live from the first available camera on the system.
    """

    if file_path != '':
        file_path = file_path.replace("\\", "/")
        # capture from video file
        capture = cv.VideoCapture(file_path)
    else:
        # capture from first available camera on the system
        capture = cv.VideoCapture(0)

    while True:
        frame_available, frame = capture.read()
        if not frame_available:
            break

        yield frame
    capture.release()


def classifier(filepath: str):
    """
    Starts a loop that does inference on available frames from passed streamer. If there are no frames available
    the loop will stop.
    """

    file_streamer = streamer(filepath)
    labelfname = "model/labels.json"
    modelfname = "model/inference_model.xml"

    # Create labels from the labels.json file stored in the model folder
    with open(labelfname) as label_file:
        label_data = json.load(label_file)
    labels = [LabelToMongo().backward(label) for label in label_data]

    inference_model = IEClassifier(modelfname)

    for frame in file_streamer:
        pil_img = transforms.ToPILImage()(frame)
        label_index = inference_model.forward(frame)
        label = labels[label_index]

        draw_label_on_image(pil_img, label)

        display_image = np.array(pil_img)
        cv.imshow("Classification", display_image)
        if ord("q") == cv.waitKey(30):
            break

    del file_streamer


def main():
    arguments_parser = ArgumentParser()
    arguments_parser.add_argument("--file", required=False,
                                  help="Specify a videofile location, if nothing is specified it will grab the webcam",
                                  default='')
    arguments = arguments_parser.parse_args()
    classifier(str(arguments.file))


if __name__ == '__main__':
    main()
