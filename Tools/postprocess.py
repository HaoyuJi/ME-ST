import numpy as np
import torch


__all__ = ["PostProcessor"]

def argrelmax(prob: np.ndarray, threshold: float = 0.7):
    """
    Calculate arguments of relative maxima.
    prob: np.array. boundary probability maps distributerd in [0, 1]
    prob shape is (T)
    ignore the peak whose value is under threshold

    Return:
        Index of peaks for each batch
    """
    # ignore the values under threshold
    prob[prob < threshold] = 0.0

    # calculate the relative maxima of boundary maps
    # treat the first frame as boundary
    peak = np.concatenate(
        [
            np.ones((1), dtype=np.bool),
            (prob[:-2] < prob[1:-1]) & (prob[2:] < prob[1:-1]),
            np.zeros((1), dtype=np.bool),
        ],
        axis=0,
    )

    peak_idx = np.where(peak)[0].tolist()

    return peak_idx


def decide_boundary_prob_with_similarity(x: torch.Tensor) -> torch.Tensor:
    """
    Decide action boundary probabilities based on adjacent frame similarities.
    Args:
        x: frame-wise video features (N, C, T)
    Return:
        boundary: action boundary probability (N, 1, T)
    """
    device = x.device

    # gaussian kernel.
    diff = x[0, :, 1:] - x[0, :, :-1]
    similarity = torch.exp(-torch.norm(diff, dim=0) / (2 * 1.0))

    # define action starting point as action boundary.
    start = torch.ones(1).float().to(device)
    boundary = torch.cat([start, similarity])
    boundary = boundary.view(1, 1, -1)
    return boundary


class PostProcessor(object):
    def __init__(
        self,
        name: str,
        boundary_th: int = 0.7,
        theta_t: int = 15,
        kernel_size: int = 15,
    ) -> None:
        self.func = {
            "refinement_with_boundary": self._refinement_with_boundary,
            "relabeling": self._relabeling,
            "smoothing": self._smoothing,
        }
        assert name in self.func

        self.name = name #'refinement_with_boundary'
        self.boundary_th = boundary_th #0.5
        self.theta_t = theta_t #15
        self.kernel_size = kernel_size #15


    def _is_probability(self, x: np.ndarray) -> bool:
        assert x.ndim == 3

        if x.shape[1] == 1: #若是回归结果
            # sigmoid
            if x.min() >= 0 and x.max() <= 1:
                return True
            else:
                return False #是否在范围内，没有就是没sigmoid
        else: #若是分类结果
            # softmax
            _sum = np.sum(x, axis=1).astype(np.float32)
            _ones = np.ones_like(_sum, dtype=np.float32) # 创建一个与 _sum 相同形状的数组，其中的元素全部为 1，数据类型为 32 位浮点数
            return np.allclose(_sum, _ones) #是否加起来是全1,不是就是没有softmax

    def _convert2probability(self, x: np.ndarray) -> np.ndarray:
        """
        Args: x (N, C, T)
        """
        assert x.ndim == 3

        if self._is_probability(x):
            return x
        else:
            if x.shape[1] == 1: #如果是回归
                # sigmoid
                prob = 1 / (1 + np.exp(-x))
            else: #如果是分类
                # softmax
                prob = np.exp(x) / np.sum(np.exp(x), axis=1) #softmax一下
            return prob.astype(np.float32)

    def _convert2label(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 2 or x.ndim == 3

        if x.ndim == 2:
            return x.astype(np.int64)
        else:
            if not self._is_probability(x):
                x = self._convert2probability(x)

            label = np.argmax(x, axis=1) #从onehot变成索引
            return label.astype(np.int64)

    def _refinement_with_boundary(
        self,
        outputs: np.array,
        boundaries: np.ndarray,
        masks: np.ndarray,
    ) -> np.ndarray:
        """
        Get segments which is defined as the span b/w two boundaries,
        and decide their classes by majority vote.
        Args:
            outputs: numpy array. shape (N, C, T)
                the model output for frame-level class prediction.
            boundaries: numpy array.  shape (N, 1, T)
                boundary prediction.
            masks: np.array. np.bool. shape (N, 1, T)
                valid length for each video
        Return:
            preds: np.array. shape (N, T)
                final class prediction considering boundaries.
        """

        preds = self._convert2label(outputs) #变成分类预测的索引（1,6000）
        boundaries = self._convert2probability(boundaries) #变成边界的预测（1,1,6000）

        for i, (output, pred, boundary, mask) in enumerate(
            zip(outputs, preds, boundaries, masks)
        ):
            boundary = boundary[mask] #拿到有效部分
            idx = argrelmax(boundary, threshold=self.boundary_th)

            # add the index of the last action ending
            T = pred.shape[0]
            idx.append(T) #结尾也算一个边界索引

            # majority vote
            for j in range(len(idx) - 1):
                count = np.bincount(pred[idx[j] : idx[j + 1]]) #计算每个边界段里的内容占比[0,0,0,0,3]表示第index=4的类占了全部的3帧
                modes = np.where(count == count.max())[0] #最大占比的类索引
                if len(modes) == 1: #如果只有一个最大占比
                    mode = modes #拿到最大索引
                else: #如果两个类最大占比相同
                    if outputs.ndim == 3:
                        # if more than one majority class exist
                        prob_sum_max = -100
                        for m in modes: # 计算每个可能的最大类别的概率总和，选最大的那个
                            prob_sum = output[m, idx[j] : idx[j + 1]].sum()
                            if prob_sum_max < prob_sum:
                                mode = m
                                prob_sum_max = prob_sum
                    else:
                        # decide first mode when more than one majority class
                        # have the same number during oracle experiment
                        mode = modes[0]

                preds[i, idx[j] : idx[j + 1]] = mode #填充这个边界段的预测

        return preds

    def _relabeling(self, outputs: np.ndarray, **kwargs: np.ndarray) -> np.ndarray:
        """
        Relabeling small action segments with their previous action segment
        Args:
            output: the results of action segmentation. (N, T) or (N, C, T)
            theta_t: the threshold of the size of action segments.
        Return:
            relabeled output. (N, T)
        """

        preds = self._convert2label(outputs)

        for i in range(preds.shape[0]):
            # shape (T,)
            last = preds[i][0]
            cnt = 1
            for j in range(1, preds.shape[1]):
                if last == preds[i][j]:
                    cnt += 1
                else:
                    if cnt > self.theta_t:
                        cnt = 1
                        last = preds[i][j]
                    else:
                        preds[i][j - cnt : j] = preds[i][j - cnt - 1]
                        cnt = 1
                        last = preds[i][j]

            if cnt <= self.theta_t:
                preds[i][j - cnt : j] = preds[i][j - cnt - 1]

        return preds

    def _smoothing(self, outputs: np.ndarray, **kwargs: np.ndarray) -> np.ndarray:
        """
        Smoothing action probabilities with gaussian filter.
        Args:
            outputs: frame-wise action probabilities. (N, C, T)
        Return:
            predictions: final prediction. (N, T)
        """

        outputs = self._convert2probability(outputs)
        outputs = self.filter(torch.Tensor(outputs)).numpy()

        preds = self._convert2label(outputs)
        return preds

    def __call__(self, outputs, **kwargs: np.ndarray) -> np.ndarray:
        preds = self.func[self.name](outputs, **kwargs)
        return preds #通过边界以及分类填充段
