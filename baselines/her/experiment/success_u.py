import sklearn
from sklearn.decomposition import PCA

_success_u = []
_lambda = 0.0
_axis = 3 
_pca = PCA(_axis)

def set_PCA_instance():
    global _pca, _axis
    _pca = PCA(_axis)

def set_PCA_axis(axis_c=3):
    global _axis
    _axis = axis_c

def set_lambda(lambda_c=0.0):
    global _lambda
    _lambda = lambda_c

def get_lambda():
    global _lambda
    return _lambda

def set_success_u(grasp_u = []):
    global _success_u
    _success_u = grasp_u

def get_success_u():
    global _success_u
    return _success_u

def calc_pca():
    global _pca
    _pca.fit(_success_u)

def calc_transform(pos):
    global _pca
    transformed_pos = _pca.transform(pos)
    return transformed_pos

def calc_inverse(pos):
    global _pca
    inverse_pos = _pca.inverse_transform(pos)
    return inverse_pos

def get_variance_ratio():
    global _pca
    return _pca.explained_variance_ratio_
