from sklearn.decomposition import PCA


class PosDatabase:
    def __init__(self, reward_lambda, num_axis, init_poslist):
        self.poslist = init_poslist
        self.reward_lambda = reward_lambda
        self.num_axis = num_axis

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
        return self.poslist

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
