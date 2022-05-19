import torch


def intersection_over_union(boxes, targets):
    # shape = (N, 4)
    # (...,4) = (..., [x center, y center, height, weight])
    xc1 = boxes[..., 0:1]
    yc1 = boxes[..., 1:2]
    h1 = boxes[..., 2:3]
    w1 = boxes[..., 3:4]
    xc2 = targets[..., 0:1]
    yc2 = targets[..., 1:2]
    h2 = targets[..., 2:3]
    w2 = targets[..., 3:4]

    x1 = torch.max(xc1 - h1 / 2, xc2 - h2 / 2)
    y1 = torch.max(yc1 - w1 / 2, yc2 - w2 / 2)
    x2 = torch.min(xc1 + h1 / 2, xc2 + h2 / 2)
    y2 = torch.min(yc1 + w1 / 2, yc2 + w2 / 2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    union = h1 * w1 + h2 * w2 - intersection
    return intersection / union
