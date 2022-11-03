import numpy as np
import scipy as sci
import utils
import cv2


class Shape_Model:
    """
    build shape model using PCA
    """
    def __init__(self, unaligned_shapes, ali_method='proc') -> None:
        self.ali_method = ali_method
        self.unaligned_shapes = unaligned_shapes
        self.aligned_shapes = self.align(unaligned_shapes)
        self.x_mean = np.mean(self.aligned_shapes, axis=1)
        self.n_shapes = unaligned_shapes.shape[1]
        
        self.eig_val, self.eig_vec = self.pca(self.aligned_shapes)


    def align(self, unaligned_shapes, ali_method='proc'):
        n_shapes = unaligned_shapes.shape[1]
        shape_mean_raw = np.mean(unaligned_shapes, axis=1)
        # shape_mean = np.hstack(shape_mean_raw[::2, None], shape_mean_raw[1::2, None])
        shape_mean = shape_mean_raw.reshape(-1, 2)

        aligned_shape = np.zeros_like(unaligned_shapes)
        for i in range(n_shapes):
            shape_i = unaligned_shapes[:, i].reshape(-1, 2)

            if ali_method == 'proc':
                _, _, shape_i_aligned = utils.generalized_proc(shape_mean, shape_i)
            elif ali_method == 'maha':
                raise ValueError("maha metric Not implemented")
                shape_i_aligned = utils.generalized_proc(shape_mean, shape_i)
            else:
                raise

            aligned_shape[:, i] = shape_i_aligned.flatten()
        
        return aligned_shape

    def cut_b(self, b):
        n = b.shape[0]
        for i in range(n):
            eig_value_i = self.eig_val[i]
            b_max = 3*np.sqrt(eig_value_i)
            b[i] = max(-b_max, min(b[i], b_max))
        return b

    def pca(self, x):
        x_mean = np.mean(x, axis=1, keepdims=True)
        x = x - x_mean
        # covariance matrix of shape [dim_x, dim_x]
        S = x @ x.T / x.shape[1]
        # eigen decomposition
        # since S is a symmetric matrix , use eigh 
        from scipy.linalg import eigh
        eig_val, eig_vec = eigh(S)
        # eig_val are in ascending order,
        eig_val = eig_val[::-1]
        eig_vec = np.fliplr(eig_vec)
        return eig_val, eig_vec


class One_level_gray_model:
    def __init__(self, image_paths, filter_dict, landmarks, profile_size=5, C=5) -> None:
        '''landmarks: landmarks coordinates. shape (num_landmarks, 2, n_image)
        filter_dict: a dictionary with filter sigma and size. '''
        self.n_image = len(image_paths)
        self.image_paths = image_paths
        self.landmarks = landmarks
        assert self.n_image == landmarks.shape[2]

        self.n_landmarks = landmarks.shape[0]
        self.filter_size = filter_dict["window_size"]
        self.profile_size = profile_size
        self.C = C

        self.g_mean, self.g_S = self.build_model()


    def build_model(self):
        """
        Return the mean profile and covariance matrix for every landmarks in the training set
        Return:
        profile_mean: shape(profile_size**2, n_landmarks)
        lm_cov_inv:   the inverse of covariance matrix for every landmarks profile
                      shape: (profile_size**2, profile_size**2, n_landmarks)
        """
        landmarks_profile = np.zeros([self.profile_size**2, self.n_landmarks, self.n_image])
        for i in range(self.n_image):
            image = cv2.imread(self.image_paths[i])
            image = self.pre_process(image)
            image_edge = self.edge_feature(image)
            landmarks_profile[:,:,i] = self.get_profile(image_edge, self.landmarks[:,:,i])
        # mean value for landmarks profile
        # landmarks_profile = landmarks_profile.reshape(-1, self.n_landmarks, self.n_image)
        profile_mean = np.mean(landmarks_profile, axis=-1)

        lm_cov_inv = np.zeros([self.profile_size**2, self.profile_size**2, self.n_landmarks])
        for i in range(self.n_landmarks):
            stat = landmarks_profile[:,i,:]
            stat = landmarks_profile[:,i,:]
            cov_i = (stat - profile_mean[:,i:i+1]) @ (stat - profile_mean[:,i:i+1]).T # shape (profile_size**2, profile_size**2)

            cov_i_inv = np.linalg.inv(cov_i)
            lm_cov_inv[:,:,i] = cov_i_inv

        return profile_mean, lm_cov_inv

    def g_metric(self, g, n, normalize_first=True):
        '''compute how suitable the current profile g is for the nth landmark
        input: 
        g: shape(self.profile_size**2, 1)
        n: 0 <= n < self.n_landmarks'''
        if normalize_first:
            g = self.profile_normalize(g)
            
        if len(g.shape) == 1:
            g = g[:,None]
        assert g.shape[0] == self.profile_size**2

        metric = (g - self.g_mean[:,n:n+1]).T @ self.g_S[:,:,n] @ (g - self.g_mean[:, n:n+1])
        return metric

    def pre_process(self, image):
        # convert rgb to gray
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # compute sigma with window size
        smooth_image = cv2.GaussianBlur(image,(self.filter_size, self.filter_size), 0)
        smooth_image = np.float32(smooth_image) 
        return smooth_image

    def edge_feature(self, image):
        ddepth = cv2.CV_32F
        dx = cv2.Sobel(image, ddepth, 1, 0)
        dy = cv2.Sobel(image, ddepth, 0, 1)
        edge_strength = cv2.magnitude(dx, dy)
        return edge_strength

    def profile_normalize(self, profile):
        """normalize a profile. profile has shape (profile_size**2, ...)"""
        profile = profile / np.sum(np.abs(profile), axis=0, keepdims=True)
        profile = profile / (np.abs(profile) + self.C)
        return profile

    def get_profile(self, image, landmark):
        # landmark is all landmarks in this image. shape (n_landmarks, 2)
        # return is a profile matrix of shape (profile_size, profile_size, n_landmarks)
        ps_half = self.profile_size // 2
        profiles = np.zeros([self.profile_size, self.profile_size, self.n_landmarks])
        for i0, i in enumerate(range(-ps_half, ps_half+1)):
            for j0, j in enumerate(range(-ps_half, ps_half+1)):
                #! it seems that index should be transposed
                index_i = landmark[:,1] + i 
                index_j = landmark[:,0] + j 
                profile = image[index_i, index_j]
                # place
                profiles[i0, j0, :] = profile

        # normalize profile
        profiles = profiles.reshape(-1, self.n_landmarks)
        profiles = self.profile_normalize(profiles)

        return profiles


class Face_Detect:
    def __init__(self, shape_model, gray_model, search_size=5):
        self.shape_model = shape_model
        self.gray_model = gray_model
        self.x_mean = self.shape_model.x_mean.astype(np.int32)

        self.profile_size = self.gray_model.profile_size
        self.search_size = search_size

    def __call__(self, image, max_iter=20, n_pcs=5, callback=None):
        feature = self.initialize(image)
        im_modified = self.gray_model.pre_process(image)
        im_edge = self.gray_model.edge_feature(im_modified)

        for i in range(max_iter):
            if callback is not None:
                callback(feature, i)
            feature_new = self.update_all_landmark(im_edge, feature)
            feature = self.restrict_shift(feature_new, n_pcs=n_pcs)
            
        return feature


    def restrict_shift(self, feature, n_pcs=5, to_int=True):
        # feature, x_mean are all (n_landmarks*2,)
        # map to model space
        _, _, x_new = utils.generalized_proc(self.x_mean.reshape(-1, 2), feature.reshape(-1, 2))
        x_delta = x_new.flatten() - self.x_mean
        x_delta = x_delta.astype(np.float64)
        # project into P space
        P = self.shape_model.eig_vec[:, :n_pcs]
        b = P.T @ x_delta.reshape(-1, 1)
        # resize b
        b = self.shape_model.cut_b(b)
        x_delta_restrict = P @ b
        x_new_restrict = self.x_mean + x_delta_restrict.flatten()
        # map to picture space
        _, _, feature_x = utils.generalized_proc(feature.reshape(-1, 2), x_new_restrict.reshape(-1, 2))
        if to_int:
            feature_x = feature_x.astype(np.int32)
        return feature_x.flatten()
        

    def initialize(self, image):
        
        nose_pos = self.pick_nose(image)
        # The 44th landmark is for nose
        landmark = self.x_mean.reshape(-1, 2)
        landmark = landmark - landmark[67,:] + nose_pos
        return landmark.flatten()


    def pick_nose(self, image):
        mouse_pos = []
        def get_mouse(event, x, y, flag, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print(1)
                mouse_pos.append([x, y])

        # im1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.namedWindow("image")
        cv2.imshow("image", image)
        cv2.setMouseCallback("image", get_mouse)

        cv2.waitKey(10000)

        cv2.destroyAllWindows()
        return np.array([mouse_pos[0][0], mouse_pos[0][1]])

    def update_landmark(self, im_edge, lm_n, n):
        """
        update
        lm_n: the nth landmark. shape (2,)
        n: the index of lm_n
        return: dx, a tensor of shape (2,)"""

        search_half = self.search_size // 2
        profile_half = self.profile_size // 2

        pos_best = np.array((lm_n[1], lm_n[0]))
        g_min = np.Inf
        for i in range(-search_half, search_half+1):
            for j in range(-search_half, search_half+1):
                pos_try = np.array((lm_n[1]+i, lm_n[0]+j))
                
                profile = im_edge[pos_try[0]-profile_half:pos_try[0]+profile_half+1, 
                                pos_try[1]-profile_half:pos_try[1]+profile_half+1]

                profile = profile.reshape(-1,1)
                # test pos_try's distance to mean profile
                g_try = self.gray_model.g_metric(profile, n)
                if g_try < g_min:
                    pos_best = pos_try
                    g_min = g_try

        lm_n_new = np.array((pos_best[1], pos_best[0]))

        return lm_n_new


    def update_all_landmark(self, im_edge, x):
        '''x: shape (n_landmarks * 2,)'''
        landmark = x.reshape(-1,2)
        landmark_new = np.zeros_like(landmark)
        for i in range(landmark.shape[0]):
            landmark_new[i,:] = self.update_landmark(im_edge, landmark[i,:], i)
        x_new = landmark_new.flatten()
        x_delta = x_new - x

        return x_new