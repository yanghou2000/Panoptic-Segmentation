# This is modified from https://github.com/Buding-97/SIJ-segmentation/blob/1ac750ae6213e720850c3f7022b07f32f193babf/utils/discriminative_loss.py#L10
# Modifications made based on the original repo:
# To make the loss suitable for Semantic-KITTI, stuff classes with instance id 0 in target and input are ignored in loss calculation
# In dist term, the orignal repo assums unique instance ids are bounded by the number of clusters (i.e. unique_instance_ids = [0, 1, 2, 3] for 4 cluster), 
# but in SemanticKITTI dataset, the unique ids look like this ([0, 1, 5, 230] for 4 clusters)
"""
This is the implementation of following paper:
https://arxiv.org/pdf/1708.02551.pdf
"""
from torch.autograd import Variable
import torch
import torch.nn as nn


class DiscriminativeLoss(nn.Module):
    '''Warning: currently this loss can only be applied to data with batch size one'''
    def __init__(self, delta_var=0.5, delta_dist=1.5,
                 norm=1, alpha=1.0, beta=1.0, gamma=0.001,
                 usegpu=True):
        super(DiscriminativeLoss, self).__init__()
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.norm = norm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.usegpu = usegpu
        assert self.norm in [1, 2]

    def forward(self, input, target):
        # with torch.no_grad():
        return self._discriminative_loss(input, target)

    def _discriminative_loss(self, input, target):
        c_means, mean_samples = self._cluster_means(input, target)
        if torch.equal(mean_samples, torch.tensor([False]).cuda()):
            return torch.tensor(0).cuda()
        l_var = self._variance_term(input, target, c_means)
        l_dist = self._distance_term(target, c_means, mean_samples)
        l_reg = self._regularization_term(c_means, mean_samples)
        # print('var_term', self.alpha * l_var, 'dis_term', self.beta * l_dist, 'reg_term', self.gamma * l_reg)
        return self.alpha * l_var + self.beta * l_dist + self.gamma * l_reg

    def _cluster_means(self, input, target):
        '''
        :param input: torch.size(B,5,4096)
        :param target: torch.size(B,4096)
        :return: means: torch.size(B,5,4096)
                mean_samples: torch.size(NeXNp)
        '''
        bs, n_feature, num_points = input.size()
        # print('input.size:', input.size())
        # add lists of unique cluster centers
        means, mean_samples = [], []
        for i in range(bs):
            # TODO: need to deal with target[i], as it only extracts one element in target
            num_cluster = torch.unique(target[i])
            # print('num_cluster:', num_cluster)
            mean = torch.zeros_like(input[i]) #torch.size(5,4096)
            for j in num_cluster:
                if j == 0:
                    continue
                # print('j', j)
                target_flage = target[i] == j #torch.size(4096) type(Boolean)
                # print(f'target_flage: {target_flage}')
                input_sample = input[i] * target_flage.unsqueeze(0).expand(n_feature,num_points) #torch.size(5,4096) feature of the j-th instance
                # print(f'input_sample: {input_sample}')
                mean_sample = input_sample.sum(1) / target_flage.sum() #torch.size(5)
                # print(f'mean_sample: {mean_sample}')
                m = target_flage.unsqueeze(0).expand(n_feature,num_points) * mean_sample.unsqueeze(1) #torch.size(5,4096)
                # print(f'm: {m}')
                mean += m
                # print('mean', mean)
                mean_samples.append(mean_sample)
            means.append(mean)
            # print('means', means)
        means = torch.stack(means)
        if mean_samples:
            mean_samples = torch.stack(mean_samples).transpose(0, 1)
        else:
            print('skipping loop')
            mean_samples = torch.tensor([False])
        # breakpoint()
        # mean_samples = torch.permute(mean_samples, (0, 2, 1)) # 1XNeXNp
        # print('means', means)
        return means.cuda(), mean_samples.cuda()

    def _variance_term(self, input, target, c_means):
        '''
        :param input: torch.size(B,5,4096)
        :param target: torch.size(B,4096)
        :param c_means: torch.size(B,5,4096)
        :return:
        '''
        var = (torch.clamp(torch.norm((input - c_means), self.norm, 1) - self.delta_var, min=0) ** 2)
        bs, n_feature, num_points = input.size()
        var_term = 0
        for i in range(bs):
            num_cluster = torch.unique(target[i])
            # print('num_cluster', num_cluster)
            for j in num_cluster:
                if j == 0:
                    continue
                target_flage = target[i] == j #torch.size(4096) type(Boolean)
                c_var = (var[i] * target_flage).sum()/target_flage.sum() # divided by number of points in instance j
                var_term += c_var
            var_term /= len(num_cluster)
        var_term /= bs
        # print('var_term', var_term)
        return var_term

    def _distance_term(self, target, c_means, mean_samples):
        '''
        :param c_means: torch.size(B,Ne,Np)
        :param mean_samples: torch.size(Ne,Np)
        :return:
        '''
        bs, n_features, num_points = c_means.size()
        dist_term = 0
        for i in range(bs):
            num_cluster = torch.unique(target[i]).long()
            # breakpoint()
            num_cluster = num_cluster[num_cluster != 0]  # Ignore label 0
            mean_cluster = mean_samples
            if mean_cluster.shape[1] <= 1:
                continue
            diff = mean_cluster.unsqueeze(1) - mean_cluster.unsqueeze(2)
            dist = torch.norm(diff, dim=0, p=self.norm)
            # breakpoint()
            margin = 2 * self.delta_dist * (1.0 - torch.eye(mean_cluster.shape[1]))
            if self.usegpu:
                margin = margin.cuda()
            c_dist = torch.sum(torch.clamp(margin - torch.norm(diff, self.norm, 0), min=0) ** 2)
            dist_term += c_dist / (mean_cluster.shape[1] * (mean_cluster.shape[1]-1))
            # dist_term /= len(num_cluster)
        dist_term /= bs
        return dist_term

    def _regularization_term(self, c_means, mean_samples):
        bs, n_features, num_points = c_means.size()
        _, num_cluster = mean_samples.size()
        reg_term = 0
        for i in range(bs):
            # n_features, n_clusters
            reg_term += torch.mean(torch.norm(mean_samples, self.norm, 0))
            reg_term /= num_cluster
        reg_term /= bs
        return reg_term

if __name__ == '__main__':
    # points = torch.randn(16, 5, 4096)
    # target1 = torch.randint(0,3,(8,4096))
    # target2 = torch.randint(7,8,(8,4096))
    # points = torch.randn(16, 6, 10).cuda()
    # target1 = torch.randint(0,3,(8,10)).cuda()
    # target2 = torch.randint(7,8,(8,10)).cuda()
    # # target3 = torch.randint(0,5,(16,4096)).cuda()
    # # print(points.shape,'line 115')
    # target = torch.cat((target1,target2),dim=0)
    # dis_meter = DiscriminativeLoss()
    # # criterion = nn.CrossEntropyLoss().cuda()
    # # loss_sem = criterion(points,target3)
    # print(target.shape)
    # loss_ins = dis_meter(points,target)
    # # print(loss_sem)
    # # print(target.type(),'line 122')
    # print(loss_ins,'line 123')
    discriminative_loss = DiscriminativeLoss(norm=2)
    inst_out = torch.tensor([[-0.3353,  0.5224, -0.2534, -0.3371, -1.4979],
        [ 0.4763,  1.3601, -0.3288, -0.6334,  0.6878],
        [ 1.4763,  0.3601, -0.3288, -1.6334,  0.6878]]).cuda()

    inst_label = torch.tensor([0, 1, 3]).long().cuda()
    inst_loss = discriminative_loss(inst_out.permute(1, 0).unsqueeze(0), inst_label.unsqueeze(0)) # input size: [NpXNe] -> [1XNeXNp] defined in discriminative loss, target size: [1XNp] -> [1X1xNp]
    print(inst_loss)