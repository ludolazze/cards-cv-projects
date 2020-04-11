from enum import Enum
import numpy as np
import cv2
from typing import List

class SetColour(Enum):
    NONE = 0
    RED = 1
    GREEN = 2
    PURPLE = 3

class SetShape(Enum):
    NONE = 0
    ROMBUS = 1
    ELLIPSE = 2
    SQUIGGLE = 3

class SetFilling(Enum):
    NONE = 0
    EMPTY = 1
    SHADED = 2
    FULL = 3

def set_colour_mapping(set_colour):
    if set_colour == SetColour.RED:
        return (0, 0, 255)
    elif set_colour == SetColour.GREEN:
        return (0, 255, 0)
    elif set_colour == SetColour.PURPLE:
        return (200, 0, 200)
    else:
        return None

def set_filling_text_mapping(set_shading):
    if set_shading == SetFilling.EMPTY:
        return "empty"
    elif set_shading == SetFilling.SHADED:
        return "shaded"
    elif set_shading == SetFilling.FULL:
        return "full"
    else:
        return None

def set_shape_text_mapping(set_shape):
    if set_shape == SetShape.ROMBUS:
        return "rombus"
    elif set_shape == SetShape.ELLIPSE:
        return "ellipse"
    elif set_shape == SetShape.SQUIGGLE:
        return "squiggle"
    else:
        return None

class CardFeatures():

    def __init__(self, colour = SetColour.NONE, number = 0, shape = SetShape.NONE, filling = SetFilling.NONE):
        self.colour = colour
        self.number = number
        self.shape = shape
        self.filling = filling


def extract_features(card_img_list : List[np.array]) -> List[CardFeatures]:

    output_features = []

    # assume all cards have same size
    height, width, _ = card_img_list[0].shape
    card_area = height * width

    for card in card_img_list:

        feature = CardFeatures()

        """ Extract shape mask """
        card_hsv = cv2.cvtColor(card, cv2.COLOR_BGR2HSV)

        lower_white = np.array([0, 0, 100], dtype=np.uint8)
        upper_white = np.array([179, 30, 255], dtype=np.uint8)

        # Threshold the HSV image to get only white colors
        white_mask = cv2.inRange(card_hsv, lower_white, upper_white)
        not_white_mask = 255 - white_mask
        blurred_not_white_mask = cv2.GaussianBlur(not_white_mask,(5,5),0)

        cv2.imshow("object mask", not_white_mask)

        """ Extract colour """
        valid_pixels = card_hsv[not_white_mask == 255]
        hue_values = valid_pixels[:, 0]
        hist, _ = np.histogram(hue_values, [0, 30, 90, 165, 255])
        colour_idx = np.argmax(hist)

        if colour_idx == 0 or colour_idx == 3:
            # print('Colour is red')
            feature.colour = SetColour.RED
        elif colour_idx == 1:
            # print('Colour is green')
            feature.colour = SetColour.GREEN
        else:
            # print('Colour is purple')
            feature.colour = SetColour.PURPLE


        """ Extract number """
        _, shape_contours, _ = cv2.findContours(blurred_not_white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # remove small contours
        shape_contours = list(filter(lambda x: cv2.contourArea(x) > (card_area // 50), shape_contours))
        feature.number = len(shape_contours)
        # print("Number is {}".format(feature.number))

        """ Extract shape """
        if (len(shape_contours) > 0):
            shape = shape_contours[0]
            peri = cv2.arcLength(shape, True)
            area = cv2.contourArea(shape)
            circularity = 4 * np.pi * area / peri / peri

            polygon_approx = cv2.approxPolyDP(shape, 0.015 * peri, True)

            if len(polygon_approx) == 4:
                # print("Shape is rombus")
                feature.shape = SetShape.ROMBUS
            elif circularity > 0.62:
                # print("Shape is ellipse")
                feature.shape = SetShape.ELLIPSE
            elif cv2.isContourConvex(shape):
                # print("Shape is squiggle")
                feature.shape = SetShape.SQUIGGLE
            else:
                feature.shape = SetShape.NONE
        else:
            feature.shape = SetShape.NONE

        """ Extract shade """
        filled_mask = cv2.drawContours(np.zeros_like(white_mask), shape_contours, -1, 255, -1)
        pixels_in_mask = not_white_mask[filled_mask == 255]
        num_pixels_in_mask = float(pixels_in_mask.size)
        if num_pixels_in_mask == 0:
            num_pixels_in_mask = 1

        pixel_density = np.sum(pixels_in_mask == 255) / num_pixels_in_mask
        # print("Density: {}".format(pixel_density))

        if pixel_density < 0.5:
            # print("Shading is emtpy")
            feature.filling = SetFilling.EMPTY
        elif 0.5 <= pixel_density < 0.8:
            # print("Shading is shaded")
            feature.filling = SetFilling.SHADED
        else:
            # print("Shading is full")
            feature.filling = SetFilling.FULL

        ### show contours
        # to_show = cv2.drawContours(card, shape_contours, -1, (0, 255, 0), 3)
        # cv2.imshow("Contour", to_show)
        # cv2.imshow("Mask", blurred_not_white_mask)
        # cv2.imshow("Filled Mask", filled_mask)
        # cv2.waitKey(0)

        ###

        output_features.append(feature)

    return output_features

def show_features(img, card_countours, features, homographies):

    to_show = img.copy()

    for idx in range(len(card_countours)):

        f = features[idx]
        # cv2.putText(im, 'Christmas', (10, 450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
        # cv2.imwrite(path + 'pillar_text.jpg', im)
        # h_inv = np.linalg.inv(homographies[idx])
        M  = cv2.moments(card_countours[idx])
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # text_shape = set_shape_text_mapping(f.shape)set_filling_text_mapping(f.filling)
        # text_filling = set_shape_text_mapping(f.shape) + " " + set_filling_text_mapping(f.filling)
        # text_number =  "X " + str(features[idx].number)
        to_show = cv2.drawContours(to_show, card_countours, idx, set_colour_mapping(features[idx].colour), 3)
        cv2.putText(to_show,  set_shape_text_mapping(f.shape), (cx - 40, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(to_show, set_filling_text_mapping(f.filling), (cx - 40, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(to_show, "X " + str(features[idx].number), (cx - 40, cy + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    return to_show
