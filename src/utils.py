import numpy as np

def generalized_proc(shape_mean, shape_x, w=None):
    ''' compute translation and rotation matrix 
    The problem can be convert to a 4-order linear system: 
        mat * (a_x, a_y, t_x, t_y).T = arr
    where the trainsformation is (a_x, -a_y; a_y, a_x) * (x, y).T + (t_x, t_y).T

    shape_mean: a reference shapes. (N_points, 2)
    shape_x:    a shape to be transformed. (N_points, 2)
    w:          a diagonal matrix of weights for each points. here it's shape is
                (N_points,)
                if w is None, then it's default value is 1
    '''
    if w == None:
        w = np.ones(shape_x.shape[0])
    assert shape_mean.shape == shape_x.shape

    # X_i = \sum_k w_k * x_ik
    X1, Y1 = w @ shape_mean
    X2, Y2 = w @ shape_x
    Z = np.dot(w, np.sum(shape_x**2, axis=1))
    W = np.sum(w)
    C1 = np.dot(w, np.sum(shape_mean * shape_x, axis=1))
    C2 = np.dot(w, (shape_mean[:,1] * shape_x[:,0] - shape_mean[:,0] * shape_x[:,1]))

    mat = np.array([[X2, -Y2, W, 0],
                    [Y2, X2, 0, W],
                    [Z, 0, X2, Y2],
                    [0, Z, -Y2, X2]])
    arr = np.array([X1, Y1, C1, C2])
    a_x, a_y, t_x, t_y = np.linalg.solve(mat, arr)
    rotate_mat = np.array([[a_x, -a_y], [a_y, a_x]])
    shift = np.array([t_x, t_y])

    aligned_x = shape_x @ rotate_mat.T + shift
    return rotate_mat, shift, aligned_x


