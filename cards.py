import cv2
import numpy as np

class CardDetector():

    def __init__(self, background_thresh = 100, card_min_area_ratio = 500, card_max_area_ratio = 4, card_height_to_width_ratio = 1.55):
        self.background_thresh = background_thresh
        self.card_min_area_ratio = card_min_area_ratio
        self.card_max_area_ratio = card_max_area_ratio
        self.card_height_to_width_ratio = card_height_to_width_ratio

    def __call__(self):
        pass
    
    def detect_cards(self, img):
        """Returns contours for the detected cards"""

        _, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        img_area = img.shape[0] * img.shape[1]

        card_countours = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < img_area // self.card_min_area_ratio or area > img_area // self.card_max_area_ratio:
                continue

            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.015 * peri, True)
            if len(approx) == 4:
                card_countours.append(approx)

        return card_countours

    def project_cards(self, img, contours):

        card_img_width = img.shape[1] // 2
        card_img_height = int(card_img_width * self.card_height_to_width_ratio)

        # flat card points in anticlockwise, top left corner first
        card_projected_contour = np.zeros_like(contours[0])
        card_projected_contour[1, :] = np.array([0, card_img_height])
        card_projected_contour[2, :] = np.array([card_img_width, card_img_height])
        card_projected_contour[3, :] = np.array([card_img_width, 0])

        card_img_list = []

        for c in contours:

            first_edge_length = np.linalg.norm(c[0, :, :] - c[1, :, :])
            second_edge_length = np.linalg.norm(c[1, :, :] - c[2, :, :])

            if second_edge_length > first_edge_length:
                c = np.roll(c, 1,  axis = 0)

            h, _ = cv2.findHomography(c, card_projected_contour)
            card_img = cv2.warpPerspective(img, h, (card_img_width, card_img_height))
            card_img_list.append(card_img)
            cv2.imshow('warped', card_img)
            cv2.waitKey(0)


    def preprocess_image(self, image):
        """Returns a grayed, blurred, and adaptively thresholded camera image."""

        width, height, _ = image.shape
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)

        bkg_level = np.percentile(blur, 25)

        thresh_level = bkg_level + self.background_thresh

        _, thresh = cv2.threshold(blur, thresh_level, 255, cv2.THRESH_BINARY)
        
        return thresh




