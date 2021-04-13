_success_u = []
_lambda = 0.0

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