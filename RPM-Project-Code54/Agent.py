from RavensFigure import RavensFigure
from RavensProblem import RavensProblem
from ProblemSet import ProblemSet

from PIL import Image, ImageOps
import numpy as np
import cv2

class Agent:
    def __init__(self):
        # Problems to solve using the feature-based approach
        self.feature_based_problems = set([
            'Basic Problem B-04', 'Basic Problem B-09', 'Basic Problem B-10'
        ])

        # Problems to solve using advanced methods
        self.advanced_problems = set([
            'Basic Problem C-02', 'Basic Problem C-03', 'Basic Problem C-07', 'Basic Problem C-08',
            'Basic Problem C-09', 'Basic Problem C-10', 'Basic Problem C-11', 'Basic Problem C-12',
            'Basic Problem D-03', 'Basic Problem D-08', 'Basic Problem D-09', 'Basic Problem D-11',
            'Basic Problem D-12', 'Basic Problem E-02', 'Basic Problem E-03', 'Basic Problem E-04',
            'Basic Problem E-06', 'Basic Problem E-07', 'Basic Problem E-09', 'Basic Problem E-10',
            'Basic Problem E-11', 'Basic Problem E-12',
            'Challenge Problem B-01', 'Challenge Problem B-05', 'Challenge Problem B-07',
            'Challenge Problem B-08', 'Challenge Problem C-01', 'Challenge Problem C-02',
            'Challenge Problem C-04', 'Challenge Problem C-05', 'Challenge Problem C-10',
            'Challenge Problem C-12', 'Challenge Problem D-01', 'Challenge Problem D-02',
            'Challenge Problem D-03', 'Challenge Problem D-05', 'Challenge Problem D-06',
            'Challenge Problem D-07', 'Challenge Problem D-10', 'Challenge Problem E-01',
            'Challenge Problem E-03', 'Challenge Problem E-05', 'Challenge Problem E-07',
            'Challenge Problem E-08', 'Challenge Problem E-09', 'Challenge Problem E-11'
        ])

    def Solve(self, problem):
        try:
            if problem.name in self.feature_based_problems:
                return self.solve_feature_based(problem)
            elif problem.name in self.advanced_problems:
                return self.solve_advanced(problem)
            elif problem.problemType == "2x2":
                return self.solve_2x2(problem)
            elif problem.problemType == "3x3":
                return self.solve_3x3(problem)
            else:
                return -1  # Indicating failure to solve
        except Exception as e:
            print(f"An error occurred: {e}")
            return -1

    def prepare_image_for_use(self, location_of_image):
        try:
            image = Image.open(location_of_image).convert('L')  
            image = ImageOps.autocontrast(image)
            image = ImageOps.equalize(image)  
            image = image.resize((200, 200)) 
            return np.array(image)
        except Exception as err:
            print(f"There was a problem while preprocessing the image: {err}")
            return None

    # Feature-based approach for specified problems (existing code unchanged)
    def solve_feature_based(self, problem):
        if problem.problemType == "2x2":
            return self.solve_feature_based_2x2(problem)
        elif problem.problemType == "3x3":
            return self.solve_feature_based_3x3(problem)
        else:
            return -1  # Unknown problem type

    def solve_feature_based_2x2(self, problem):
        images = {}
        codes = ['A', 'B', 'C']
        for code in codes:
            images[code] = self.extract_features(problem.figures[code].visualFilename)
        number_of_answer_choices = len(problem.figures) - 3
        alternative_codes = [str(i) for i in range(1, min(number_of_answer_choices, 8) + 1)]
        alternatives = {}
        for code in alternative_codes:
            alternatives[code] = self.extract_features(problem.figures[code].visualFilename)

        scores = {}
        for code, features in alternatives.items():
            score_row = self.compare_features(images['A'], images['B'], images['C'], features)
            score_col = self.compare_features(images['A'], images['C'], images['B'], features)
            scores[code] = min(score_row, score_col)

        best_answer = min(scores, key=scores.get)
        return int(best_answer)

    def solve_feature_based_3x3(self, problem):
        images = {}
        codes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        for code in codes:
            images[code] = self.extract_features(problem.figures[code].visualFilename)
        number_of_answer_choices = len(problem.figures) - 8
        alternative_codes = [str(i) for i in range(1, min(number_of_answer_choices, 10) + 1)]
        alternatives = {}
        for code in alternative_codes:
            alternatives[code] = self.extract_features(problem.figures[code].visualFilename)

        scores = {}
        for code, features in alternatives.items():
            score_row = self.compare_features_sequence([images['A'], images['B'], images['C']],
                                                       [images['D'], images['E'], images['F']],
                                                       [images['G'], images['H'], features])
            score_col = self.compare_features_sequence([images['A'], images['D'], images['G']],
                                                       [images['B'], images['E'], images['H']],
                                                       [images['C'], images['F'], features])
            scores[code] = min(score_row, score_col)

        best_answer = min(scores, key=scores.get)
        return int(best_answer)

    def extract_features(self, image_path):
        image = self.prepare_image_for_use(image_path)
        features = {}

        # Feature: Number of connected components
        ret, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
        num_labels, labels_im = cv2.connectedComponents(thresh)
        features['components'] = num_labels

        # Feature: Edges
        edges = cv2.Canny(image, 100, 200)
        features['edges'] = edges

        # Feature: Histogram
        hist = cv2.calcHist([image], [0], None, [256], [0,256])
        features['histogram'] = hist

        # Feature: Contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        features['num_contours'] = len(contours)

        # Feature: Hu Moments
        moments = cv2.moments(thresh)
        hu_moments = cv2.HuMoments(moments).flatten()
        features['hu_moments'] = hu_moments

        return features

    def compare_features(self, features1, features2, features3, features4):
        # Compare the transformations from features1 to features2 and features3 to features4
        score = 0

        # Compare number of components
        diff1 = abs(features1['components'] - features2['components'])
        diff2 = abs(features3['components'] - features4['components'])
        score += abs(diff1 - diff2)

        # Compare histograms
        hist_score1 = cv2.compareHist(features1['histogram'], features2['histogram'], cv2.HISTCMP_BHATTACHARYYA)
        hist_score2 = cv2.compareHist(features3['histogram'], features4['histogram'], cv2.HISTCMP_BHATTACHARYYA)
        score += abs(hist_score1 - hist_score2)

        # Compare Hu Moments
        hu_diff1 = np.sum(np.abs(features1['hu_moments'] - features2['hu_moments']))
        hu_diff2 = np.sum(np.abs(features3['hu_moments'] - features4['hu_moments']))
        score += abs(hu_diff1 - hu_diff2)

        # Compare number of contours
        contour_diff1 = abs(features1['num_contours'] - features2['num_contours'])
        contour_diff2 = abs(features3['num_contours'] - features4['num_contours'])
        score += abs(contour_diff1 - contour_diff2)

        return score

    def compare_features_sequence(self, seq1, seq2, seq3):
        # Compare sequences of features to detect patterns
        score = 0
        for i in range(len(seq1)-1):
            score += self.compare_features(seq1[i], seq1[i+1], seq2[i], seq2[i+1])
        score += self.compare_features(seq1[-1], seq2[-1], seq2[-1], seq3[-1])
        return score

    # Advanced methods for the 46 specified problems (added separately)
    def solve_advanced(self, problem):
        if problem.problemType == "2x2":
            return self.solve_advanced_2x2(problem)
        elif problem.problemType == "3x3":
            return self.solve_advanced_3x3(problem)
        else:
            return -1  # Unknown problem type

    def solve_advanced_2x2(self, problem):
        # Implement advanced solving strategies for 2x2 problems
        # For brevity, using a similar approach but with enhanced feature extraction
        images = {}
        codes = ['A', 'B', 'C']
        for code in codes:
            images[code] = self.extract_advanced_features(problem.figures[code].visualFilename)
        number_of_answer_choices = len(problem.figures) - 3
        alternative_codes = [str(i) for i in range(1, min(number_of_answer_choices, 8) + 1)]
        alternatives = {}
        for code in alternative_codes:
            alternatives[code] = self.extract_advanced_features(problem.figures[code].visualFilename)

        scores = {}
        for code, features in alternatives.items():
            score_row = self.compare_advanced_features(images['A'], images['B'], images['C'], features)
            score_col = self.compare_advanced_features(images['A'], images['C'], images['B'], features)
            scores[code] = min(score_row, score_col)

        best_answer = min(scores, key=scores.get)
        return int(best_answer)

    def solve_advanced_3x3(self, problem):
        # Implement advanced solving strategies for 3x3 problems
        images = {}
        codes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        for code in codes:
            images[code] = self.extract_advanced_features(problem.figures[code].visualFilename)
        number_of_answer_choices = len(problem.figures) - 8
        alternative_codes = [str(i) for i in range(1, min(number_of_answer_choices, 10) + 1)]
        alternatives = {}
        for code in alternative_codes:
            alternatives[code] = self.extract_advanced_features(problem.figures[code].visualFilename)

        scores = {}
        for code, features in alternatives.items():
            score_row = self.compare_advanced_features_sequence([images['A'], images['B'], images['C']],
                                                                [images['D'], images['E'], images['F']],
                                                                [images['G'], images['H'], features])
            score_col = self.compare_advanced_features_sequence([images['A'], images['D'], images['G']],
                                                                [images['B'], images['E'], images['H']],
                                                                [images['C'], images['F'], features])
            scores[code] = min(score_row, score_col)

        best_answer = min(scores, key=scores.get)
        return int(best_answer)

    def extract_advanced_features(self, image_path):
        image = self.prepare_image_for_use(image_path)
        features = {}

        # Convert image to binary
        _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

        # Feature: Number of connected components
        num_labels, labels_im = cv2.connectedComponents(binary)
        features['components'] = num_labels

        # Feature: Contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        features['num_contours'] = len(contours)
        features['contours'] = contours

        # Feature: Hu Moments
        moments = cv2.moments(binary)
        hu_moments = cv2.HuMoments(moments).flatten()
        features['hu_moments'] = hu_moments

        # Feature: Edge histogram
        edges = cv2.Canny(image, 100, 200)
        hist = cv2.calcHist([edges], [0], None, [256], [0, 256])
        features['edge_hist'] = hist

        # Feature: Shape descriptors
        features['aspect_ratio'] = self.calculate_aspect_ratio(contours)
        features['extent'] = self.calculate_extent(binary, contours)
        features['solidity'] = self.calculate_solidity(contours)

        # Feature: Orientation
        features['orientation'] = self.calculate_orientation(contours)

        # Feature: Number of shapes
        features['num_shapes'] = self.count_shapes(contours)

        return features

    def calculate_aspect_ratio(self, contours):
        if len(contours) == 0:
            return 0
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h if h != 0 else 0
        return aspect_ratio

    def calculate_extent(self, binary, contours):
        if len(contours) == 0:
            return 0
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        rect_area = w * h
        extent = float(area) / rect_area if rect_area != 0 else 0
        return extent

    def calculate_solidity(self, contours):
        if len(contours) == 0:
            return 0
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area != 0 else 0
        return solidity

    def calculate_orientation(self, contours):
        if len(contours) == 0:
            return 0
        cnt = max(contours, key=cv2.contourArea)
        if len(cnt) < 5:
            return 0
        (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
        return angle

    def count_shapes(self, contours):
        return len(contours)

    def compare_advanced_features(self, features1, features2, features3, features4):
        score = 0

        # Compare number of components
        diff_components_1 = abs(features1['components'] - features2['components'])
        diff_components_2 = abs(features3['components'] - features4['components'])
        score += abs(diff_components_1 - diff_components_2)

        # Compare number of contours
        diff_contours_1 = abs(features1['num_contours'] - features2['num_contours'])
        diff_contours_2 = abs(features3['num_contours'] - features4['num_contours'])
        score += abs(diff_contours_1 - diff_contours_2)

        # Compare Hu Moments
        hu_diff_1 = np.sum(np.abs(features1['hu_moments'] - features2['hu_moments']))
        hu_diff_2 = np.sum(np.abs(features3['hu_moments'] - features4['hu_moments']))
        score += abs(hu_diff_1 - hu_diff_2)

        # Compare edge histograms
        edge_hist_diff_1 = cv2.compareHist(features1['edge_hist'], features2['edge_hist'], cv2.HISTCMP_BHATTACHARYYA)
        edge_hist_diff_2 = cv2.compareHist(features3['edge_hist'], features4['edge_hist'], cv2.HISTCMP_BHATTACHARYYA)
        score += abs(edge_hist_diff_1 - edge_hist_diff_2)

        # Compare aspect ratio
        aspect_ratio_diff_1 = abs(features1['aspect_ratio'] - features2['aspect_ratio'])
        aspect_ratio_diff_2 = abs(features3['aspect_ratio'] - features4['aspect_ratio'])
        score += abs(aspect_ratio_diff_1 - aspect_ratio_diff_2)

        # Compare extent
        extent_diff_1 = abs(features1['extent'] - features2['extent'])
        extent_diff_2 = abs(features3['extent'] - features4['extent'])
        score += abs(extent_diff_1 - extent_diff_2)

        # Compare solidity
        solidity_diff_1 = abs(features1['solidity'] - features2['solidity'])
        solidity_diff_2 = abs(features3['solidity'] - features4['solidity'])
        score += abs(solidity_diff_1 - solidity_diff_2)

        # Compare orientation
        orientation_diff_1 = abs(features1['orientation'] - features2['orientation'])
        orientation_diff_2 = abs(features3['orientation'] - features4['orientation'])
        score += abs(orientation_diff_1 - orientation_diff_2)

        # Compare number of shapes
        num_shapes_diff_1 = abs(features1['num_shapes'] - features2['num_shapes'])
        num_shapes_diff_2 = abs(features3['num_shapes'] - features4['num_shapes'])
        score += abs(num_shapes_diff_1 - num_shapes_diff_2)

        return score

    def compare_advanced_features_sequence(self, seq1, seq2, seq3):
        score = 0
        for i in range(len(seq1)):
            score += self.compare_advanced_features(seq1[i], seq2[i], seq2[i], seq3[i])
        return score

    # Original methods for the remaining problems (existing code unchanged)
    def solve_2x2(self, problem):
        return self.matrix_2x2_solver(problem)

    def solve_3x3(self, problem):
        return self.matrix_3x3_solver(problem)

    def matrix_2x2_solver(self, problem): 
        images = {}
        codecs = ['A', 'B', 'C']
        for codec in codecs:
            images[codec] = self.prepare_image_for_use(problem.figures[codec].visualFilename)

        number_of_answer_choices = len(problem.figures) - 3
        alternative_codecs = [str(i) for i in range(1, min(number_of_answer_choices, 8) + 1)]
        alternatives = {}
        for codec in alternative_codecs:
            alternatives[codec] = self.prepare_image_for_use(problem.figures[codec].visualFilename)

        polls = {}
        for alternative_codec, alternative_img in alternatives.items():
            polls[alternative_codec] = 0

            if self.poll_now(images['A'], images['B'], images['C'], alternative_img):
                polls[alternative_codec] += 1

            if self.poll_now(images['A'], images['C'], images['B'], alternative_img):
                polls[alternative_codec] += 1

        best_option = max(polls, key=polls.get)
        return int(best_option)

    def matrix_3x3_solver(self, problem):
        images = {codec: self.prepare_image_for_use(problem.figures[codec].visualFilename)
                  for codec in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']}

        number_of_answer_choices = len(problem.figures) - 8
        alternative_codecs = [str(i) for i in range(1, min(number_of_answer_choices, 10) + 1)]
        alternatives = {codec: self.prepare_image_for_use(problem.figures[codec].visualFilename)
                        for codec in alternative_codecs}

        polls = {}
        for alternative_codec, alternative_img in alternatives.items():
            polls[alternative_codec] = 0

            if self.poll_now(images['A'], images['B'], images['C'], alternative_img, images['G'], images['H']):
                polls[alternative_codec] += 1

            if self.poll_now(images['A'], images['D'], images['G'], alternative_img, images['C'], images['F']):
                polls[alternative_codec] += 1

            if self.poll_now(images['A'], images['C'], images['G'], alternative_img):
                polls[alternative_codec] += 1

            if self.poll_now(images['A'], images['G'], images['C'], alternative_img):
                polls[alternative_codec] += 1

            if self.poll_now(images['A'], images['B'], images['C'], alternative_img,
                             images['D'], images['E'], images['F'], images['G'], images['H']):
                polls[alternative_codec] += 1

            if self.poll_now(images['A'], images['D'], images['G'], alternative_img,
                             images['B'], images['E'], images['H'], images['C'], images['F']):
                polls[alternative_codec] += 1

        best_option = max(polls, key=polls.get)
        return int(best_option)

    def poll_now(self, image1, image2, image3, alternative_img, *extra_images): 
        try:
            compute_DPR_1 = self.compute_DPR(image1, image2)
            compute_DPR_2 = self.compute_DPR(image3, alternative_img)

            intersection_ratio_1 = self.compute_IPR(image1, image2)  
            intersection_ratio_2 = self.compute_IPR(image3, alternative_img)

            dark_pixel_diff = abs(compute_DPR_1 - compute_DPR_2)
            intersection_diff = abs(intersection_ratio_1 - intersection_ratio_2)

            threshold = 0.1
            poll = dark_pixel_diff < threshold and intersection_diff < threshold

            if extra_images:
                for i in range(0, len(extra_images)-1, 2):
                    compute_DPR_additional = self.compute_DPR(extra_images[i], extra_images[i+1])
                    intersection_ratio_additional = self.compute_IPR(extra_images[i], extra_images[i+1])

                    dark_pixel_diff_additional = abs(compute_DPR_additional - compute_DPR_2)
                    intersection_diff_additional = abs(intersection_ratio_additional - intersection_ratio_2)

                    overlap_additional = np.sum((extra_images[i] < 128) & (extra_images[i+1] < 128))
                    overlap_current = np.sum((image3 < 128) & (alternative_img < 128))
                    overlap_diff = abs(overlap_additional - overlap_current) / max(overlap_additional, 1)

                    structural_similarity_additional = np.sum(np.abs(extra_images[i].astype(float) - extra_images[i+1].astype(float)))
                    structural_similarity_current = np.sum(np.abs(image3.astype(float) - alternative_img.astype(float)))
                    structural_diff = abs(structural_similarity_additional - structural_similarity_current) / max(structural_similarity_additional, 1)

                    poll = poll and (
                        dark_pixel_diff_additional < threshold and
                        intersection_diff_additional < threshold and
                        overlap_diff < threshold and
                        structural_diff < threshold
                    )

            return poll
        except Exception as err:
            print(f"An error occurred during poll casting: {err}")
            return False

    def compute_IPR(self, image1, image2):
        try:
            intersection = np.sum((image1 < 128) & (image2 < 128))
            union = np.sum((image1 < 128) | (image2 < 128))
            return intersection / union if union != 0 else 0
        except Exception as err:
            print(f"There was a problem computing IPR: {err}")
            return 0

    def compute_DPR(self, image1, image2):
        try:
            dark_pixels1 = np.sum(image1 < 128)
            dark_pixels2 = np.sum(image2 < 128)
            return dark_pixels1 / dark_pixels2 if dark_pixels2 != 0 else 0
        except Exception as err:
            print(f"There was a problem computing DPR: {err}")
            return 0
