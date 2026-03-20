import numpy as np

class ResultFusion:
    @staticmethod
    def nms(proposals, iou_threshold=0.5):
        """非极大值抑制：解决多算法重复框问题"""
        if not proposals or len(proposals) == 0:
            return []

        # 转换为 numpy 数组
        boxes = np.array(proposals)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            
            # 计算 IoU
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            # 保留重叠度低于阈值的框
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]

        # 仅返回坐标，不返回分数
        final_boxes = []
        for i in keep:
            b = boxes[i]
            final_boxes.append((int(b[0]), int(b[1]), int(b[2]), int(b[3])))
            
        return final_boxes
