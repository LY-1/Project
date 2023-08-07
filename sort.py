#from __future__ import print_function

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

from sklearn import preprocessing

import glob
import time
import argparse
from filterpy.kalman import KalmanFilter

np.random.seed(0)


def linear_assignment(cost_matrix):
    # 两种实现方式 仅返回的形式不同
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        # minimize cost x:每一行被分配到哪一列   y：每一列别分配到哪一行
        return np.array([[y[i], i] for i in x if i >= 0])
        # 返回使得cost最小的分配方式
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

def odis_batch_norm(bb_test, bb_gt):
  """
  odis_batch_norm(detections, trackers)，都是左上和右下点的坐标
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
  m = bb_test.shape[0]
  n = bb_gt.shape[0]
  if m == 0 or n == 0:
    o = []
    print(o)
    return (np.array(o))
  test = np.zeros((m, 4))
  gt = np.zeros((n, 4))
  try:   # 归一化
      test[:, 0] = bb_test[:, 0] / 512
      test[:, 1] = bb_test[:, 1] / 512
      test[:, 2] = bb_test[:, 2] / 512
      test[:, 3] = bb_test[:, 3] / 512
  except:
      a=1

  gt[:, 0] = bb_gt[:, 0] / 512
  gt[:, 1] = bb_gt[:, 1] / 512
  gt[:, 2] = bb_gt[:, 2] / 512
  gt[:, 3] = bb_gt[:, 3] / 512

  o = np.zeros((m, n))
  cd = (test[:, :2] + test[:, 2:4]) / 2    # 中心点坐标
  ct = (gt[:, :2] + gt[:, 2:4]) / 2
  for i in range(m):
    for j in range(n):
      o[i, j] = np.linalg.norm(cd[i] - ct[j])    # 根据中心点求范数，默认2范数，即欧氏距离，这里越小越好

  # min_max_scaler = preprocessing.MinMaxScaler()
  # print(o,m,n)
  # o =1 - min_max_scaler.fit_transform(o)

  o = 1 - o    # 这里越大越好

  return (o)

def odis_batch(bb_test, bb_gt):
  """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
  m = bb_test.shape[0]
  n = bb_gt.shape[0]
  if m == 0 or n == 0:
    o = []
    #print(o)
    return (np.array(o))

  o = np.zeros((m, n))
  cd = (bb_test[:, :2] + bb_test[:, 2:4]) / 2
  ct = (bb_gt[:, :2] + bb_gt[:, 2:4]) / 2
  for i in range(m):
    for j in range(n):
      o[i, j] = np.linalg.norm(cd[i] - ct[j])

  min_max_scaler = preprocessing.MinMaxScaler()
  #print(o,m,n)
  o =1 - min_max_scaler.fit_transform(o)

  return (o)

def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    批量计算IOU
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h  # union
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return (o)


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
    s:area
    r:w/h
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    s=w*h       r=w/h
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score == None):
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)        # 状态变量7维   观测值4维
        # x为状态变量 x y s r vx vy vs   z为观测值 x y s r
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        # F 为状态转移矩阵         7*7
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])
        # H为观测矩阵            4*7
        self.kf.R[2:, 2:] *= 10.
        # R为观测噪声的协方差        4*4
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        # P为状态协方差矩阵         7*7
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        # 状态转移协方差矩阵 or 预测噪声协方差矩阵        7*7
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        # 状态        7*1
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        #
        self.hits = 0
        #
        self.hit_streak = 0
        # 连续匹配到的次数
        self.age = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []       # 清空history
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))     # bbox是当前帧的检测结果

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):    # 如果超过超过一帧没有检测到，那么连续检测就会断开
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    匹配BBox与Kalman输出     将检测结果与跟踪结果进行关联
    """
    if (len(trackers) == 0):        # 若跟踪器为空 直接返回： 无跟上  若干个漏跟  无多余跟踪器
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    #iou_matrix = iou_batch(detections, trackers)        # IOU 行数：目标个数 列数：tracker个数
    iou_matrix = odis_batch_norm(detections, trackers)

    if min(iou_matrix.shape) > 0:       # 如果 目标，tracker有一个不为0
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:             # 出现歧义时，使用匈牙利算法 such as:one tacker with tow object one object with two tracker
            matched_indices = np.stack(np.where(a), axis=1)         # 获取坐标 [0]第i个目标  [1]第j个tracker
        else:
            matched_indices = linear_assignment(-iou_matrix)        # 通过匈牙利算法匹配tracker与BBox  保存到matched_indices中
    else:
        # 两列，左边是检测结果，右边是跟踪器
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)      # 没有匹配上的BBox  需要使用tracker跟踪
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)        # Kalman filter的输出没有与BBox匹配上  需要删除对应的tracker

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):    # 将IOU小于threshold的放入列表  无卵用
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))     # 匹配上的存入
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
    # 成功跟踪的物体   新增加的物体（还没跟）  消失的物体（删除tracker）


# def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
#     """
#     Assigns detections to tracked object (both represented as bounding boxes)
#
#     Returns 3 lists of matches, unmatched_detections and unmatched_trackers
#     匹配BBox与Kalman输出     将检测结果与跟踪结果进行关联
#     """
#     if (len(trackers) == 0):        # 若跟踪器为空 直接返回： 无跟上  若干个漏跟  无多余跟踪器
#         return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
#
#     #iou_matrix = iou_batch(detections, trackers)        # IOU 行数：目标个数 列数：tracker个数
#     iou_matrix = odis_batch(detections, trackers)
#
#     if min(iou_matrix.shape) > 0:       # 如果 目标，tracker有一个不为0
#         a = (iou_matrix > iou_threshold).astype(np.int32)
#         if a.sum(1).max() == 1 and a.sum(0).max() == 1:             # 出现歧义时，使用匈牙利算法 such as:one tacker with tow object one object with two tracker
#             matched_indices = np.stack(np.where(a), axis=1)         # 获取坐标 [0]第i个目标  [1]第j个tracker
#         else:
#             matched_indices = linear_assignment(-iou_matrix)        # 通过匈牙利算法匹配tracker与BBox  保存到matched_indices中
#     else:
#         matched_indices = np.empty(shape=(0, 2))
#
#     unmatched_detections = []
#     for d, det in enumerate(detections):
#         if (d not in matched_indices[:, 0]):
#             unmatched_detections.append(d)      # 没有匹配上的BBox  需要使用tracker跟踪
#     unmatched_trackers = []
#     for t, trk in enumerate(trackers):
#         if (t not in matched_indices[:, 1]):
#             unmatched_trackers.append(t)        # Kalman filter的输出没有与BBox匹配上  需要删除对应的tracker
#
#     # filter out matched with low IOU
#     matches = []
#     for m in matched_indices:
#         if (iou_matrix[m[0], m[1]] < iou_threshold):    # 将IOU小于threshold的放入列表  无卵用
#             unmatched_detections.append(m[0])
#             unmatched_trackers.append(m[1])
#         else:
#             matches.append(m.reshape(1, 2))     # 匹配上的存入
#     if (len(matches) == 0):
#         matches = np.empty((0, 2), dtype=int)
#     else:
#         matches = np.concatenate(matches, axis=0)
#
#     return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
#     # 成功跟踪的物体   新增加的物体（还没跟）  消失的物体（删除tracker）

class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age              # tracker存活周期
        self.min_hits = min_hits            # 目标出现多少帧开始初始化tracker
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
        dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided. because min_hints
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))        # 根据tracker个数创建二维矩阵，存放卡尔曼的预测结果
        to_del = []                                     # 待删除的tracker
        ret = []                                        # 待返回的结果
        for t, trk in enumerate(trks):                  # 第一次跳过
            pos = self.trackers[t].predict()[0]         # 获取tracker预测的bbox
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]    # 获取四个坐标    trk与trks的第t行共享内存
            if np.any(np.isnan(pos)):
                to_del.append(t)                        # 若预测的bbox为空  则删除tracker        没执行过
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))      # 删除预测为空的tracker  存放上一帧中目标在当前帧中预测的非空bbox
        for t in reversed(to_del):
            self.trackers.pop(t)            # 删除trackers中tracker     没执行过
        # 两列，左边是检测结果，右边是跟踪器
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)
        # unmatched_trks无用
        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])   # 拿相对应的检测框（内部还要考虑前一帧的预测结果）更新跟踪器

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:                # 新增目标
            trk = KalmanBoxTracker(dets[i, :])  # 新增目标的结果送入Kalman
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):     # 新的trackers
            d = trk.get_state()[0]              # 获取tracker状态,返回的是bbox的中心点坐标和长宽
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # time_since_update < 1 且 连续三帧匹配到 或者在头三帧，连续匹配是防止检测时的误检
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            if (trk.time_since_update > self.max_age):   # 为匹配到的跟踪器的time_since_update不会重置，超过一定帧数将会被删除
                self.trackers.pop(i)    # remove dead tracklet
        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SORT')
    parser.add_argument('--display', type=bool,default=True, help='Display online tracker output ')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='det')
    parser.add_argument("--max_age",
                        help="Maximum number of frames to keep alive a track without associated detections.",
                        type=int, default=1)
    parser.add_argument("--min_hits", help="Minimum number of associated detections before track is initialised.",
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()

    display = args.display
    #phase = args.phase
    total_time = 0.0        # 总时间
    total_frames = 0        # 总帧数

    colours = np.random.rand(32, 3)  # used only for display BBox颜色

    if (display):
        if not os.path.exists('det'):
            print('ERROR')
            exit()
        plt.ion()   # 交互模式
        fig = plt.figure()
        ax1 = fig.add_subplot(111, aspect='equal')      # 一个图

    if not os.path.exists('output'):
        os.makedirs('output')
    # pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt')
    pattern = os.path.join(args.seq_path, '*.txt')
    a = glob.glob(pattern)[11:]
    b = glob.glob(pattern)

    for seq_dets_fn in glob.glob(pattern)[11:]:
        KalmanBoxTracker.count=0
        mot_tracker = Sort(max_age=args.max_age,
                           min_hits=args.min_hits,
                           iou_threshold=args.iou_threshold)  # create instance of the SORT tracker
        seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')       # 读取anchor信息
        #seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]     # 获取文件名
        seq = os.path.split(seq_dets_fn)[1][:-4]  # 以最后一个/作为分隔符，返回列表，第一个是路径，第二个是文件名
        with open(os.path.join('output', '%s.txt' % (seq)), 'w') as out_file:
            print("Processing %s." % (seq))
            for frame in range(int(seq_dets[:, 0].max())):      # 获取总帧数
                frame += 1  # detection and frame numbers begin at 1
                dets = seq_dets[seq_dets[:, 0] == frame, 2:7]       # 取出当前帧的数据
                dets[:, 2:4] += dets[:, 0:2]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
                total_frames += 1

                if (display):
                    fn = os.path.join('test', str(seq[4:])+'_'+'%06d.png' % (frame))
                    im = io.imread(fn)
                    ax1.imshow(im)
                    plt.title(seq + ' Tracked Targets')

                start_time = time.time()
                trackers = mot_tracker.update(dets)     # 将当前帧的所有目标信息输入tracker 进行更新
                cycle_time = time.time() - start_time
                total_time += cycle_time

                for d in trackers:
                    print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]),
                          file=out_file)
                    if (display):
                        d = d.astype(np.int32)
                        ax1.add_patch(patches.Rectangle((d[0], d[1]), d[2] - d[0], d[3] - d[1], fill=False, lw=2,
                                                        ec=colours[d[4] % 32, :]))
                        ax1.text(d[0], d[1],str(d[4]),color=colours[d[4] % 32, :],fontsize=15)

                if (display):
                    fig.canvas.flush_events()
                    plt.draw()
                    ax1.cla()
                print(frame)

    print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))