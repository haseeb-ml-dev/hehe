import numpy as np

class Utils360:
    @staticmethod
    def wrap_x(x, width):
        """
        Wraps an X coordinate around the 360 image width.
        Example: If width is 1000, x=1005 becomes 5, x=-5 becomes 995.
        """
        return x % width

    @staticmethod
    def shortest_distance_x(x1, x2, width):
        """
        Calculates the shortest horizontal distance between two points
        on a 360 cylinder (considering the seam).
        """
        diff = abs(x1 - x2)
        return min(diff, width - diff)

    @staticmethod
    def calculate_360_distance(p1, p2, width):
        """
        Euclidean distance that respects the 360 seam.
        p1, p2: tuples of (x, y)
        """
        dx = Utils360.shortest_distance_x(p1[0], p2[0], width)
        dy = p1[1] - p2[1]
        return np.sqrt(dx**2 + dy**2)

    @staticmethod
    def get_iou_360(bbox1, bbox2, width):
        """
        Calculates Intersection over Union (IOU) handling 360 wrap-around.
        It checks the standard IOU and the 'wrapped' IOU (shifting one box).
        """
        # Standard IOU
        iou_standard = Utils360._bbox_iou(bbox1, bbox2)
        
        # Wrapped IOU (Shift bbox2 to the left and right by width)
        bbox2_left = [bbox2[0] - width, bbox2[1], bbox2[2] - width, bbox2[3]]
        bbox2_right = [bbox2[0] + width, bbox2[1], bbox2[2] + width, bbox2[3]]
        
        iou_left = Utils360._bbox_iou(bbox1, bbox2_left)
        iou_right = Utils360._bbox_iou(bbox1, bbox2_right)
        
        return max(iou_standard, iou_left, iou_right)

    @staticmethod
    def _bbox_iou(boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA) * max(0, yB - yA)

        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou