import torch

import torch_geometric
from torch_geometric.nn import MLP
import torch.nn.init as init


def knn_thresh(distance_matrix, k=20, distance_threshold=0.5):
  """Get KNN based on the pairwise distance.
  Args:
    distance_matrix: (batch_size, num_points, num_points)
    k: int
    distance_threshold: float

  Returns:
    nearest neighbors indices: (batch_size, num_points, k)
  """
  nearest_neighbours = torch.topk(distance_matrix, k=k, largest=False)

  # [0, 1, 2, ..., N]
  identity_indices = torch.arange(distance_matrix.shape[-2], device=distance_matrix.device)
  # Repeats every row k times
  # Results in batched:
  # 0 0 0 0 ... 0
  # 1 1 1 1 ... 1
  # ...
  # k-1 ....... k-1
  identity_indices = identity_indices.reshape(-1, 1).expand(distance_matrix.shape[0], -1, k)
  # Clip neighbours with distance more than the threshold, by replacing each edge
  # with distance > distance_threshold with a self-edge
  neighbours_within_thresh_mask = (nearest_neighbours.values <= distance_threshold)
  return nearest_neighbours.indices * neighbours_within_thresh_mask + identity_indices * ~neighbours_within_thresh_mask


def get_local_feature(point_cloud, nn_idx):
  """Construct edge feature for each point
  Args:
    point_cloud: (batch_size, num_points, 1, num_dims)
    nn_idx: (batch_size, num_points, k)

  Returns:
    edge features: (batch_size, num_points, k, num_dims)
  """
  point_cloud_shape = point_cloud.shape
  batch_size = point_cloud_shape[0]
  num_points = point_cloud_shape[1]
  num_dims = point_cloud_shape[2]

  idx_ = torch.arange(batch_size, device=point_cloud.device) * num_points
  idx_ = idx_.reshape(batch_size, 1, 1)

  point_cloud_flat = point_cloud.reshape(-1, num_dims)

  point_cloud_neighbors = point_cloud_flat[(nn_idx + idx_).reshape(point_cloud_flat.shape[0], -1)]
  return point_cloud_neighbors.reshape(batch_size, num_points, -1, num_dims)


class ASIS(torch.nn.Module):
  def __init__(self, feature_dim, inst_emb_dim, num_class, k, distance_threshold, norm_degree):
    super().__init__()
    self.k = k
    self.distance_threshold = distance_threshold
    self.norm_degree = norm_degree

    # Defaults to batch_norm and ReLU
    self.semantic_adaptation_fc = MLP([feature_dim, feature_dim], plain_last=False) # Original
    self.inst_fc = MLP([8, feature_dim], plain_last=False)  # For hacking

    # Default plain_last=True ignores batch_norm and ReLU for the last (and only) layer
    self.instance_embedding_fc = MLP([feature_dim, inst_emb_dim], dropout=0.5)
    self.classifier_fc = MLP([feature_dim, num_class], dropout=0.5)

  def forward(self, f_sem, f_ins, batch):
    f_sem_prime = self.semantic_adaptation_fc(f_sem) # Original
    # f_sem_prime = self.semantic_adaptation_fc_hc(f_sem) # for hacking
    f_ins = self.inst_fc(f_ins)
    f_sins = f_ins + f_sem_prime
    e_ins = self.instance_embedding_fc(f_sins)

    f_isem = self.instance_fusion(e_ins, f_sem, batch)
    p_sem = self.classifier_fc(f_isem)

    return p_sem, e_ins

  def instance_fusion(self, e_ins, f_sem, batch):
    '''Calculate the instance-fused semantic features f_isem by finding the k-nearest
    neighbours of each point in instance embedding space (e_ins) and then max-pooling
    their respective semantic features (f_sem).
    '''
    # Index col[i] is one of the k-nearest neighbours of point with index row[i]
    # HACK: cosine flag equal to True is treated to represent the L1 norm
    #       Ensure the underlying torch_cluster implementation of KNN is also hacked!
    cosine = self.norm_degree != 2
    row, col = torch_geometric.nn.pool.knn(x=e_ins, y=e_ins, k=self.k, batch_x=batch, batch_y=batch, cosine=cosine)

    # Clip neighbours with distance more than the threshold, by replacing each edge
    # with distance > distance_threshold with a self-edge
    pdist = torch.nn.PairwiseDistance(p=self.norm_degree, eps=0)
    neighbours_within_thresh_mask = pdist(e_ins[row], e_ins[col]) <= self.distance_threshold
    col = col * neighbours_within_thresh_mask + row * ~neighbours_within_thresh_mask

    # Max pools the semantic features of the neighbours (col) for each point (row)
    # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/pool/max_pool.html#max_pool_neighbor_x
    return torch_geometric.utils.scatter(f_sem[col], row, dim=0, dim_size=len(f_sem), reduce='max')

  def instance_fusion_loop(self, batched_e_ins, batched_f_sem, batch):
    '''Identical (up to floating point error) to `instance_fusion` but uses the torch
    version of the original tensorflow-based implementation of the nearest neighbour
    calculation and feature projection.

    Requires a lot of GPU memory for large point clouds due to the distance_matrix
    calculation.
    '''
    f_isem_list = []
    for e_ins, f_sem in zip(torch_geometric.utils.unbatch(batched_e_ins, batch),
                            torch_geometric.utils.unbatch(batched_f_sem, batch)):
      distance_matrix = torch.cdist(e_ins, e_ins, p=self.norm_degree)  # N x N
      neighbours_indices = knn_thresh(distance_matrix.unsqueeze(0), k=self.k,
                                      distance_threshold=self.distance_threshold).squeeze(0)  # N x k

      # Index col[i] is one of the k-nearest neighbours of point with index row[i]
      row = torch.arange(neighbours_indices.shape[0], device=neighbours_indices.device).reshape(-1, 1).expand_as(
        neighbours_indices).flatten()
      col = neighbours_indices.flatten()

      # Max pools the semantic features of the neighbours (col) for each point (row)
      # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/pool/max_pool.html#max_pool_neighbor_x
      f_isem = torch_geometric.utils.scatter(f_sem[col], row, dim=0, dim_size=len(f_sem), reduce='max')
      # Equivalent to the following as per the original tf implementation:
      # f_sem_nearest_neighbours = get_local_feature(f_sem.unsqueeze(0), nn_idx=neighbours_indices)
      # f_isem = torch.max(f_sem_nearest_neighbours, dim=-2).values.squeeze(0)
      f_isem_list.append(f_isem)
    return torch.cat(f_isem_list)

  def instance_fusion_loop2(self, batched_e_ins, batched_f_sem, batch):
    '''Identical to `instance_fusion_loop`, but first calculates the nearest neighbours
    for the whole batch and then does the feature projection in one go.
    '''
    index_offsets = torch.cumsum(torch_geometric.utils.degree(batch, dtype=torch.long), dim=0)
    index_offsets = torch.cat((torch.tensor([0], device=index_offsets.device), index_offsets))

    indices = []
    for e_ins, f_sem, index_offset in zip(torch_geometric.utils.unbatch(batched_e_ins, batch),
                                          torch_geometric.utils.unbatch(batched_f_sem, batch), index_offsets):
      distance_matrix = torch.cdist(e_ins, e_ins, p=self.norm_degree)  # N x N
      neighbours_indices = knn_thresh(distance_matrix.unsqueeze(0), k=self.k,
                                      distance_threshold=self.distance_threshold).squeeze(0)  # N x k
      neighbours_indices += index_offset
      indices.append(neighbours_indices)

    indices = torch.cat(indices)
    row = torch.arange(indices.shape[0], device=indices.device).reshape(-1, 1).expand_as(indices).flatten()
    col = indices.flatten()

    # Max pools the semantic features of the neighbours (col) for each point (row)
    # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/pool/max_pool.html#max_pool_neighbor_x
    return torch_geometric.utils.scatter(batched_f_sem[col], row, dim=0, dim_size=len(batched_f_sem), reduce='max')


def main():
  import random
  from torch_geometric.data import Data
  from torch_geometric.loader import DataLoader

  class MyDataset(torch_geometric.data.Dataset):
    def __init__(self, size, feature_dim, pcd_size, pcd_size_variance_ratio=0.1):
      self.size = size
      self.feature_dim = feature_dim
      self.pcd_size = pcd_size
      self.pcd_size_variance_ratio = pcd_size_variance_ratio
      super().__init__(root=None, transform=None, pre_transform=None, pre_filter=None)

    def len(self):
      return self.size

    def get(self, idx):
      delta = self.pcd_size_variance_ratio * self.pcd_size
      pcd_size = self.pcd_size + random.randint(-delta, delta)

      return Data(f_sem=torch.rand((pcd_size, self.feature_dim), dtype=torch.float),
                  f_ins=torch.rand((pcd_size, self.feature_dim), dtype=torch.float),
                  pos=torch.rand(pcd_size, 3, dtype=torch.float))

  batch_size = 4
  num_batches = 2
  dataset_size = num_batches * batch_size
  feature_dim = 128
  inst_emb_dim = 5
  num_class = 20
  pcd_size = 120000

  dataset = MyDataset(size=dataset_size, feature_dim=feature_dim, pcd_size=pcd_size, pcd_size_variance_ratio=0.2)
  dataloader = DataLoader(dataset=dataset, batch_size=batch_size)

  asis = ASIS(feature_dim=feature_dim, inst_emb_dim=inst_emb_dim, num_class=num_class, k=30, distance_threshold=0.5,
              norm_degree=2)

  for data in dataloader:
    if torch.cuda.is_available():
      asis.cuda()
      data.cuda()
    print(data)
    asis(data.f_sem, data.f_ins, data.batch)


if __name__ == "__main__":
  main()