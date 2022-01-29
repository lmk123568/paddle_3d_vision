import numpy as np

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def random_point_dropout(pc, max_dropout_ratio=0.875):
    dropout_ratio = np.random.random() * max_dropout_ratio  # 0~0.875
    drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]
    if len(drop_idx) > 0:
        pc[drop_idx, :] = pc[0, :]  # set to the first point
    return pc


def random_scale_point_cloud(data, scale_low=0.8, scale_high=1.25):
    scales = np.random.uniform(scale_low, scale_high)
    data *= scales
    return data


def shift_point_cloud(data, shift_range=0.1):
    shifts = np.random.uniform(-shift_range, shift_range, (3))
    data += shifts
    return data


def jitter_point_cloud(data: np.ndarray, sigma: float = 0.02, clip: float = 0.05):
    assert clip > 0
    jittered_data = np.clip(sigma * np.random.randn(*data.shape), -clip, clip)
    data = data + jittered_data
    return data


def random_rotate_point_cloud(data: np.ndarray):
    angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(angle)
    sinval = np.sin(angle)
    rotation_matrix = np.array(
        [[cosval, sinval, 0], [-sinval, cosval, 0], [0, 0, 1]], dtype=data.dtype
    )
    data = data @ rotation_matrix
    return data


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point