import numpy as np
import scipy as sci
import utils

class Shape_Model:
    """
    build shape model using PCA
    """
    def __init__(self, unaligned_shapes, ali_method='proc') -> None:
        self.ali_method = ali_method
        self.unaligned_shapes = unaligned_shapes
        self.aligned_shape = self.align(unaligned_shapes)
        self.x_mean = np.mean(self.aligned_shape, axis=1)
        self.n_shapes = unaligned_shapes.shape[1]
        
        self.eig_val, self.eig_vec = self.pca(self.aligned_shape)


    def align(self, unaligned_shapes, ali_method='proc'):
        n_shapes = unaligned_shapes.shape[1]
        shape_mean_raw = np.mean(unaligned_shapes, axis=1)
        # shape_mean = np.hstack(shape_mean_raw[::2, None], shape_mean_raw[1::2, None])
        shape_mean = shape_mean_raw.reshape(-1, 2)

        aligned_shape = np.zeros_like(unaligned_shapes)
        for i in range(n_shapes):
            shape_i = unaligned_shapes[:, i].reshape(-1, 2)

            if ali_method == 'proc':
                shape_i_aligned = utils.proc_align(shape_mean, shape_i)
            elif ali_method == 'maha':
                shape_i_aligned = utils.maha_align(shape_mean, shape_i)
            else:
                raise

            aligned_shape[:, i] = shape_i_aligned.flatten()
        
        return aligned_shape

    def pca(self, x):
        x_mean = np.mean(x, axis=1)
        x = x - x_mean
        # covariance matrix of shape [dim_x, dim_x]
        S = x @ x.T 
        # eigen decomposition
        # since S is a symmetric matrix , use eigh 
        eig_val, eig_vec = sci.linalg.eigh(S)
        # eig_val are in ascending order,
        eig_val = eig_val[::-1]
        eig_vec = np.fliplr(eig_vec)
        return eig_val, eig_vec


class One_level_gray_model:
    def __init__(self, ) -> None:
        pass