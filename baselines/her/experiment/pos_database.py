from sklearn.decomposition import PCA


class PosDatabase:
    def __init__(self, reward_lambda, num_axis, init_poslist, maxn_pos):
        self.poslist = init_poslist
        self.reward_lambda = reward_lambda
        self.num_axis = num_axis
        self.maxn_pos = maxn_pos

        self.pca = PCA(self.num_axis)

    def set_lambda(self, lambda_c=0.0):
        self.reward_lambda = lambda_c

    def get_lambda(self):
        return self.reward_lambda

    def add_pos(self, pos):
        self.poslist.append(pos)

    def set_poslist(self, poslist):
        self.poslist = poslist

    def get_poslist(self):
        if len(self.poslist) > self.maxn_pos:
            self._pop_pos()
        return self.poslist

    def _pop_pos(self):
        return self.poslist.pop(0)

    def calc_pca(self):
        self.pca.fit(self.poslist)

    def calc_transform(self, pos):
        t_pos = self.pca.transform(pos)
        return t_pos

    def calc_inverse(self, pos):
        i_pos = self.pca.inverse_transform(pos)
        return i_pos

    def get_variance_ratio(self):
        return self.pca.explained_variance_ratio_
