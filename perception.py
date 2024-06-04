from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
import cv2
from DER_func import DER_func
from DER_sim import DER_sim
import torch
from util import computeLengths, computeEdges, compute_m_u0, parallelTransportFrame
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from index import self_index, directed_index_fill
import open3d as o3d
from scipy.interpolate import CubicSpline

class perception(nn.Module):
    def __init__(self, device):
        super().__init__()
        np.random.seed(0)
        sam_checkpoint_path = "save_model/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        self.device = device
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path)
        sam.to(device=device)
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=8,
            pred_iou_thresh=0.99,
            stability_score_thresh=0.90,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=1,
        )

        n_vert = 13
        self.n_vert = n_vert
        device = device
        self.device = device
        self.DER_func = DER_func(n_vert=n_vert, n_edge=n_vert - 1, device=device)
        self.simulation = DER_sim(n_vert=n_vert, n_edge=n_vert-1, pbd_iter=10, device=device)
        self.simulation.load_state_dict(torch.load("save_model/thin_DLO_1020_1.pth"))
        # self.simulation.DER_func.bend_stiffness = nn.Parameter(1.5e-4 * torch.ones((1, n_vert - 1), device=device))
        # self.simulation.DER_func.twist_stiffness = nn.Parameter(1.-4 * torch.ones((1, n_vert - 1), device=device))

        self.init_direction = torch.tensor(((0., 0.6, 0.8), (0., .0, 1.))).to(device).unsqueeze(dim=0)
        self.clamped_index = torch.zeros(n_vert)
        self.clamped_selection = torch.tensor((0, 1,  -2, -1))
        self.clamped_index[self.clamped_selection] = torch.tensor((1.))
        rest_vert = (torch.tensor(((0.893471, -0.133465, 0.018059),
                                   (0.880771, -0.119666, 0.017733),
                                   (0.791946, -0.084258, 0.009944),
                                   (0.680462, -0.102366, 0.018528),
                                   (0.590795, -0.144219, 0.021808),
                                   (0.494905, -0.156384, 0.017816),
                                   (0.396916, -0.143114, 0.021549),
                                   (0.299291, -0.148755, 0.014955),
                                   (0.200583, -0.146497, 0.01727),
                                   (0.09586, -0.142385, 0.016456),
                                   (-0.000782, -0.147084, 0.016081),
                                   (-0.071514, -0.17382, 0.015446),
                                   (-0.094659, -0.186181, 0.012403)))).unsqueeze(dim=0).repeat(1, 1, 1).to(device)
        rest_vert = torch.cat((rest_vert[:, :, 0].unsqueeze(dim=-1), rest_vert[:, :, 2].unsqueeze(dim=-1), -rest_vert[:, :, 1].unsqueeze(dim=-1)), dim=-1)
        self.rest_vert = rest_vert
        vis_rest_vert = torch.Tensor.numpy(rest_vert.to('cpu'))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(X_obs, Y_obs, Z_obs, label='Obstacle', s=4, c='orange')
        ax.plot(vis_rest_vert[0, :, 0], vis_rest_vert[0, :, 1], vis_rest_vert[0, :, 2], label='pred')
        ax.set_xlim(-.5, 1.)
        ax.set_ylim(-1, .5)
        ax.set_zlim(0, 1.)
        plt.legend()
        plt.show()
        print(self.rest_vert)
        self.average_distance = (torch.mean(torch.linalg.norm(self.rest_vert[:, 1:] - self.rest_vert[:, :-1], dim=2), dim=1) * 1000.).cpu().numpy()
        self.rest_distance = (torch.linalg.norm(self.rest_vert[:, 1:] - self.rest_vert[:, :-1], dim=2) * 1000.).cpu().numpy()[0]

    def crop_image(self, image, box):
        return image[box[0]:box[1], box[2]:box[3]]

    def show_ann(self, ann, box, image, color_mask=None):
        # Show a single segmentation mask in the image
        if len(ann) == 0:
            return
        img = np.ones((image.shape[0], image.shape[1], 4))
        img[:, :, 3] = 0
        color = np.concatenate([[255, 0, 0], [0.6]])
        m = ~ann['segmentation']
        m[:2, :], m[-2:, :], m[:, :2], m[:, -2:] = False, False, False, False
        m = np.pad(m, pad_width=((box[0], image.shape[0] - box[0] - m.shape[0]), (box[2], image.shape[1] - box[2] - m.shape[1])), mode='constant', constant_values=False)
        if color_mask is not None:
            m = np.logical_and(m, color_mask)
        indices = np.where(m == True)
        img[m] = color
        return img, indices

    def segmentation_forward(self, image, crop_shape=[0.0, 1.0, 0.0, 1.0]):
        height, width, _ = image.shape
        box = [int(crop_shape[0] * height), int(crop_shape[1] * height), int(crop_shape[2] * width), int(crop_shape[3] * width)]
        cropped_image = self.crop_image(image, box)
        masks = self.mask_generator.generate(cropped_image)
        sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
        img, raw_indices = self.show_ann(sorted_anns[0], box, image)
        # plt.figure()
        # plt.imshow(image)
        # plt.imshow(img)
        # plt.show()
        return raw_indices

    def inverse_transformation_matrix(self, T):
        R = T[:3, :3]  # Extract rotation matrix
        t = T[:3, 3]  # Extract translation vector

        R_inv = R.T  # Compute transpose of rotation matrix
        t_inv = -np.dot(R_inv, t)  # Compute inverse translation vector

        # Construct inverse transformation matrix
        T_inv = np.eye(4)
        T_inv[:3, :3] = R_inv
        T_inv[:3, 3] = t_inv
        return T_inv

    def sphere_filter(self, noisy_points, mean, sigma):
        x = np.expand_dims(noisy_points[:, 0], 1)
        y = np.expand_dims(noisy_points[:, 1], 1)
        z = np.expand_dims(noisy_points[:, 2], 1)

        x_mean = np.expand_dims(mean[:, 0], 0)
        y_mean = np.expand_dims(mean[:, 1], 0)
        z_mean = np.expand_dims(mean[:, 2], 0)

        """ Gaussian function for 3D data. """
        return np.sqrt((x - x_mean) ** 2 + (y - y_mean) ** 2 + (z - z_mean) ** 2)

    def gmm(self, points, n_points=13):
        filtered_points = GaussianMixture(n_components=n_points, covariance_type="full").fit(points)
        return filtered_points

    def reconstruction_wo_pred(self, mocap_point, clamped_points, masked_indices, depth, ground_truth = None, sigma=1e-1):
        fx = 606.1124267
        fy = 605.8821411
        cx = 641.7578125
        cy = 365.6518859
        # indices = np.concatenate((np.expand_dims(masked_indices[0], 0), np.expand_dims(masked_indices[1], 0)), 0)
        # indices = indices.T
        # print(indices.shape)
        width = 720
        height = 1080

        # Generate all combinations of pixel coordinates
        x_coords, y_coords = np.meshgrid(range(width), range(height), indexing='xy')
        indices = np.column_stack((x_coords.ravel(), y_coords.ravel()))

        X = np.where(depth[indices[:, 0], indices[:, 1]] != 0, (indices[:, 1] - cx) * depth[indices[:, 0], indices[:, 1]] / fx, 0)
        Y = np.where(depth[indices[:, 0], indices[:, 1]] != 0, (indices[:, 0] - cy) * depth[indices[:, 0], indices[:, 1]] / fy, 0)
        Z = np.where(depth[indices[:, 0], indices[:, 1]] != 0, depth[indices[:, 0], indices[:, 1]], 0)
        X = X[X!=0]
        Y = Y[Y!=0]
        Z = Z[Z!=0]
        raw_point_cloud = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        extrinsic = np.array(((-6.77410600e-02,  3.05752883e-03,  9.97698251e-01, -2.79992631e+02),
                              (-6.25751883e-01, -7.78990759e-01, -4.00996084e-02,  4.58980076e+02),
                              (7.77075112e-01, -6.27027949e-01,  5.46829141e-02,  8.71977363e+02),
                              (0, 0, 0, 1)))
        extrinsic_inv = self.inverse_transformation_matrix(extrinsic)
        raw_point_cloud = np.hstack((raw_point_cloud, np.ones((raw_point_cloud.shape[0], 1))))
        raw_point_cloud_transformed = extrinsic_inv @ raw_point_cloud.T
        raw_point_cloud_transformed = raw_point_cloud_transformed / raw_point_cloud_transformed[3]
        raw_point_cloud_transformed = raw_point_cloud_transformed[:3, :]
        raw_point_cloud_transformed = raw_point_cloud_transformed.T
        raw_point_cloud_transformed = np.concatenate((np.expand_dims(raw_point_cloud_transformed[:,0], -1), -np.expand_dims(raw_point_cloud_transformed[:, 2], -1), np.expand_dims(raw_point_cloud_transformed[:, 1], -1)), 1)

        rest_edges = computeEdges(self.rest_vert)
        m_restEdgeL, m_restRegionL = computeLengths(rest_edges.clone())
        RegionL = torch.cat(((m_restEdgeL[:, 0]/2.).unsqueeze(dim=-1), (m_restEdgeL[:, 1:] + m_restEdgeL[:, :-1])/2., (m_restEdgeL[:, -1]/2.).unsqueeze(dim=-1)), dim=1)
        m_pmass = self.simulation.m_pmass.repeat(1, 1) * RegionL + torch.clamp(self.simulation.mocap_mass, min=1e-10)
        interpolated_points = np.linspace(clamped_points[0], clamped_points[-1], self.n_vert)
        for i in range((self.simulation.n_vert - 2)//2 + 1):
            if i != (self.simulation.n_vert - 2)//2:
                interpolated_points[i + 1, 2] = interpolated_points[i, 2] - self.rest_distance[i]
                interpolated_points[-(i + 2), 2] = interpolated_points[-(i + 1), 2] - self.rest_distance[-(i + 1)]
            else:
                interpolated_points[i + 1, 2] = interpolated_points[i, 2] - self.rest_distance[i + 1]
        interpolated_points[1] = clamped_points[1]
        interpolated_points[-2] = clamped_points[-2]
        interpolated_points = torch.from_numpy(interpolated_points).unsqueeze(dim=0).to(self.device)/1000.
        clusters_centers = self.simulation.applyInternalConstraintsIteration(interpolated_points, m_restEdgeL, m_pmass, self.clamped_index, iterative_times=1000)[0].detach().cpu().numpy() * 1000.
        init_pack = self.DER_sim_init(torch.from_numpy(clusters_centers).unsqueeze(dim=0).to(self.device).float()/1000., torch.from_numpy(clamped_points).unsqueeze(dim=0).to(self.device).float()/1000., ground_truth)
        pred, inference_list = init_pack[0][0].cpu().numpy() * 1000., init_pack[1]
        wasted_offset = 10
        offset = 390
        traj_list = []
        # for k, inference in enumerate(inference_list):
        #     inference_interval = 500
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111, projection='3d')
        #     ax.plot(mocap_point[:, 0] / 1000., mocap_point[:, 1] / 1000., mocap_point[:, 2] / 1000., label='Ground Truth')
        #     ax.plot(inference[:, 0] / 1000., inference[:, 1] / 1000., inference[:, 2] / 1000., label='pred')
        #     ax.set_xlim(-.5, 1.)
        #     ax.set_ylim(-1, .5)
        #     ax.set_zlim(0, 1.)
        #     plt.legend()
        #     i=0
        #     plt.savefig('tracking/vertices_video/guessed_%s.png' % (i * inference_interval + k + 0))
        #     image_projection = np.load(r'%s' % image_dir)[int((i * inference_interval + 0 + wasted_offset) * 0.3)]
        #     projected_image = self.projection(inference, inference, image_projection)
        #     cv2.imwrite('tracking/images_video/guessed_%s.png' % (i * inference_interval + k), cv2.cvtColor(cv2.cvtColor(projected_image, cv2.COLOR_RGB2BGR), cv2.COLOR_RGB2BGR))
        #     traj_list.append(inference / 1000.)
        #     traj_list.append(inference / 1000.)
        # print('saved')
        np.save("tracking/vertices_video/guessed_inference.npy", traj_list)
        sphere_blob = self.sphere_filter(raw_point_cloud_transformed, pred, sigma)
        filtered_point_cloud_transformed = raw_point_cloud_transformed[np.min(sphere_blob, axis=1) <= 50.]
        # test = self.gmm(filtered_point_cloud_transformed/1000., 10).means_
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(inference_list[-1][3:5, 0]/1000., inference_list[-1][3:5, 1]/1000., inference_list[-1][3:5, 2]/1000., s=20, marker='o', label='Predicted Occluded Points', depthshade=False, color='orange')
        ax.scatter(filtered_point_cloud_transformed[:, 0] / 1000., filtered_point_cloud_transformed[:, 1] / 1000., filtered_point_cloud_transformed[:, 2] / 1000., s=1, marker='o', color="blue", label='Observed Point Cloud')
        # ax.scatter(inference_list[-1][0:3, 0]/1000., inference_list[-1][0:3, 1]/1000., inference_list[-1][3:5, 2]/1000., s=40, marker='o', label='Predicted Occluded Points', color='orange')
        ax.set_xlim(-.5, 1.)
        ax.set_ylim(-1, .5)
        ax.set_zlim(0, 1.)
        # ax.set_title('Occluded Point Cloud')
        ax.set_xlabel("X(m)")
        ax.set_ylabel("Y(m)")
        ax.set_zlabel("Z(m)")
        plt.legend()
        plt.savefig("occluded.png", dpi=500)
        # plt.show()

        if filtered_point_cloud_transformed.all():
            "clustering"
            db = DBSCAN(eps=100., min_samples=10).fit(filtered_point_cloud_transformed)
            labels = db.labels_
            fig = plt.figure()

            ax = fig.add_subplot(111, projection='3d')
            unique_labels = set(labels)
            colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
            clusters_centers = []

            for k, col in zip(unique_labels, colors):
                if k == -1:
                    continue
                class_member_mask = (labels == k)
                xyz = filtered_point_cloud_transformed[class_member_mask]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz)
                voxel_size = 10  # size of the voxel: adjust this parameter as needed
                downsampled_pcd = pcd.voxel_down_sample(voxel_size)
                xyz = np.asarray(downsampled_pcd.points)

                ave_distances = []
                for i in range(1, self.DER_func.n_vert):
                    if i <= xyz.shape[0]:
                        filtered_points = self.gmm(xyz, i).means_
                        if i >= 2:
                            filtered_points = self_index(filtered_points)
                            ave_distance = np.mean(np.linalg.norm(filtered_points[:-1] - filtered_points[1:], axis=1))
                            ave_distances.append(ave_distance)

                cluster_n = np.absolute(ave_distances - self.average_distance).argmin() + 1
                clusters_center = self.gmm(xyz, cluster_n).means_
                clusters_centers.append(clusters_center)
            vis_clusters_centers = np.vstack(clusters_centers).copy()

            clusters_centers = directed_index_fill(np.vstack(clusters_centers), pred)
            clusters_centers = torch.from_numpy(clusters_centers).unsqueeze(dim=0).to(self.device)/1000.
            rest_edges = computeEdges(self.rest_vert)
            m_restEdgeL, m_restRegionL = computeLengths(rest_edges.clone())
            clusters_centers = self.simulation.applyInternalConstraintsIteration(clusters_centers, m_restEdgeL, m_pmass, self.clamped_index, iterative_times=1000)[0].detach().cpu().numpy() * 1000.
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    # Black used for noise.
                    col = [0, 0, 0, 1]
                class_member_mask = (labels == k)
                xyz = filtered_point_cloud_transformed[class_member_mask]
                ax.scatter(xyz[:, 0]/1000., xyz[:, 1]/1000., xyz[:, 2]/1000., c=tuple(col), marker='o', label='Group:' + str(k), s=1)
            ax.set_xlim(-.5, 1.)
            ax.set_ylim(-1, .5)
            ax.set_zlim(0, 1.)
            ax.set_xlabel("X(m)")
            ax.set_ylabel("Y(m)")
            ax.set_zlabel("Z(m)")
            # ax.scatter(vis_clusters_centers[:, 0]/1000., vis_clusters_centers[:, 1]/1000, vis_clusters_centers[:, 2]/1000., marker='o', color='red', s=30, label="Un-occluded Predicted Vertices")
            ax.set_title('Un-occluded Predicted Vertices')
            plt.legend()
            plt.show()
            return filtered_point_cloud_transformed, clusters_centers
                # ax.set_ylabel("Y(m)")
        else:
            return None, pred

    def reconstruction(self, pred, masked_indices, depth, sigma=1e-1):
        fx = 606.1124267
        fy = 605.8821411
        cx = 641.7578125
        cy = 365.6518859
        # indices = np.concatenate((np.expand_dims(masked_indices[0], 0), np.expand_dims(masked_indices[1], 0)), 0)
        # indices = indices.T
        width = 720
        height = 1080
        x_coords, y_coords = np.meshgrid(range(width), range(height), indexing='xy')
        indices = np.column_stack((x_coords.ravel(), y_coords.ravel()))

        X = np.where(depth[indices[:, 0], indices[:, 1]] != 0, (indices[:, 1] - cx) * depth[indices[:, 0], indices[:, 1]] / fx, 0)
        Y = np.where(depth[indices[:, 0], indices[:, 1]] != 0, (indices[:, 0] - cy) * depth[indices[:, 0], indices[:, 1]] / fy, 0)
        Z = np.where(depth[indices[:, 0], indices[:, 1]] != 0, depth[indices[:, 0], indices[:, 1]], 0)
        X = X[X!=0]
        Y = Y[Y!=0]
        Z = Z[Z!=0]
        raw_point_cloud = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        extrinsic = np.array(((-6.77410600e-02,  3.05752883e-03,  9.97698251e-01, -2.79992631e+02),
                              (-6.25751883e-01, -7.78990759e-01, -4.00996084e-02,  4.58980076e+02),
                              (7.77075112e-01, -6.27027949e-01,  5.46829141e-02,  8.71977363e+02),
                              (0, 0, 0, 1)))
        extrinsic_inv = self.inverse_transformation_matrix(extrinsic)
        raw_point_cloud = np.hstack((raw_point_cloud, np.ones((raw_point_cloud.shape[0], 1))))
        raw_point_cloud_transformed = extrinsic_inv @ raw_point_cloud.T
        raw_point_cloud_transformed = raw_point_cloud_transformed / raw_point_cloud_transformed[3]
        raw_point_cloud_transformed = raw_point_cloud_transformed[:3, :]
        raw_point_cloud_transformed = raw_point_cloud_transformed.T
        raw_point_cloud_transformed = np.concatenate((np.expand_dims(raw_point_cloud_transformed[:,0], -1), -np.expand_dims(raw_point_cloud_transformed[:, 2], -1), np.expand_dims(raw_point_cloud_transformed[:, 1], -1)), 1)

        gaussian_blob = self.sphere_filter(raw_point_cloud_transformed, pred, sigma)
        print(gaussian_blob.shape)# wasted_offset = 0
        # offset = 516 - 18
        filtered_point_cloud_transformed = raw_point_cloud_transformed[np.min(gaussian_blob, axis=1) <= 50.]
        print(filtered_point_cloud_transformed.shape)
        if filtered_point_cloud_transformed.all():
            print('here')
            "clustering"
            db = DBSCAN(eps=100., min_samples=10).fit(filtered_point_cloud_transformed)
            labels = db.labels_
            fig = plt.figure()

            ax = fig.add_subplot(111, projection='3d')
            unique_labels = set(labels)
            colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

            clusters_centers = []
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    continue
                class_member_mask = (labels == k)
                xyz = filtered_point_cloud_transformed[class_member_mask]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz)
                voxel_size = 10  # size of the voxel: adjust this parameter as needed
                downsampled_pcd = pcd.voxel_down_sample(voxel_size)
                xyz = np.asarray(downsampled_pcd.points)

                ave_distances = []
                for i in range(1, self.DER_func.n_vert):
                    if i <= xyz.shape[0]:
                        filtered_points = self.gmm(xyz, i).means_
                        filtered_points = self.gmm(xyz, i).means_
                        if i >= 2:
                            filtered_points = self_index(filtered_points)
                            ave_distance = np.mean(np.linalg.norm(filtered_points[:-1] - filtered_points[1:], axis=1))
                            ave_distances.append(ave_distance)

                cluster_n = np.absolute(ave_distances - self.average_distance).argmin() + 1
                clusters_center = self.gmm(xyz, cluster_n).means_
                clusters_centers.append(clusters_center)
            vis_clusters_centers = np.vstack(clusters_centers).copy()

            clusters_centers = directed_index_fill(np.vstack(clusters_centers), pred)
            clusters_centers = torch.from_numpy(clusters_centers).unsqueeze(dim=0).to(self.device)/1000.
            rest_edges = computeEdges(self.rest_vert)
            m_restEdgeL, m_restRegionL = computeLengths(rest_edges.clone())
            RegionL = torch.cat(((m_restEdgeL[:, 0] / 2.).unsqueeze(dim=-1), (m_restEdgeL[:, 1:] + m_restEdgeL[:, :-1]) / 2., (m_restEdgeL[:, -1] / 2.).unsqueeze(dim=-1)), dim=1)
            m_pmass = self.simulation.m_pmass.repeat(1, 1) * RegionL + torch.clamp(self.simulation.mocap_mass,min=1e-10)
            clusters_centers = self.simulation.applyInternalConstraintsIteration(clusters_centers, m_restEdgeL, m_pmass, self.clamped_index, iterative_times=1000)[0].detach().cpu().numpy() * 1000.
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    # Black used for noise.        print('here')

                    col = [0, 0, 0, 1]

                class_member_mask = (labels == k)

                # Plot the clustered points
                xyz = filtered_point_cloud_transformed[class_member_mask]
                ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=tuple(col), marker='o', label='Group:' + str(k), s=1)
            ax.set_xlabel("X(mm)")
            ax.set_ylabel("Y(mm)")
            ax.set_zlabel("Z(mm)")
            ax.scatter(vis_clusters_centers[:, 0], vis_clusters_centers[:, 1], vis_clusters_centers[:, 2], marker='o', color='black', s=20, label="GMM Centers")
            ax.set_title('No-Indexing GMM')
            plt.legend()
            plt.show()
            return filtered_point_cloud_transformed, clusters_centers
        else:
            return None, pred

    def projection(self, mocap_points, perc_points, color_image):
        color_image = color_image.copy()
        mocap_points = np.hstack((mocap_points, np.ones((mocap_points.shape[0], 1))))
        perc_points = np.hstack((perc_points, np.ones((perc_points.shape[0], 1))))
        mocap_points = mocap_points.T
        perc_points = perc_points.T
        transformation_matrix = np.array(((-6.77410600e-02,  3.05752883e-03,  9.97698251e-01, -2.79992631e+02),
                              (-6.25751883e-01, -7.78990759e-01, -4.00996084e-02,  4.58980076e+02),
                              (7.77075112e-01, -6.27027949e-01,  5.46829141e-02,  8.71977363e+02),
                              (0, 0, 0, 1)))

        fx = 606.1124267
        fy = 605.8821411
        cx = 641.7578125
        cy = 365.6518859
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        shape_matrix = np.hstack((np.eye(3), np.ones((3, 1))))
        orientation_matrix = np.array(((1, 0, 0, 0), (0, 0, 1, 0), (0, -1, 0, 0), (0, 0, 0, 1)))

        for i in range(mocap_points.shape[1]):
            imgpoint = intrinsic @ shape_matrix @ transformation_matrix @ orientation_matrix @ mocap_points[:, i]
            imgpoint = imgpoint.flatten()
            imgpoint = imgpoint / imgpoint[2]
            cv2.circle(color_image, (int(imgpoint[0]), int(imgpoint[1])), 5, (0, 0 ,255), -1)
            cv2.waitKey(0)

        for i in range(perc_points.shape[1]):
            imgpoint = intrinsic @ shape_matrix @ transformation_matrix @ orientation_matrix @ perc_points[:, i]
            imgpoint = imgpoint.flatten()
            imgpoint = imgpoint / imgpoint[2]
            cv2.circle(color_image, (int(imgpoint[0]), int(imgpoint[1])), 5, (255, 0, 0), -1)
            cv2.waitKey(0)
        return color_image

    def DER_sim_init(self, initial_vertices, inputs, eval_target_vertices=None):
        inference_list = []
        inference_list.append(initial_vertices[0].cpu().numpy() * 1000.)
        for sim_step in range(500):
            with torch.no_grad():
                if sim_step == 0:
                    theta_full = torch.zeros(1, self.n_vert - 1).to(self.device)
                    rest_edges = computeEdges(self.rest_vert)
                    m_u0 = self.DER_func.compute_m_u0(rest_edges[:, 0].float(), self.init_direction.repeat(1, 1, 1)[:, 0])
                    current_v = torch.zeros_like(initial_vertices)
                    init_pred_vert_0, current_v, theta_full = self.simulation(initial_vertices, current_v, self.init_direction.repeat(1, 1, 1), self.clamped_index, m_u0, inputs, self.clamped_selection, theta_full)
                    inference_list.append(init_pred_vert_0[0].cpu().numpy() * 1000.)

                if sim_step == 1:
                    previous_edge = computeEdges(initial_vertices)
                    current_edges = computeEdges(init_pred_vert_0)
                    m_u0 = self.DER_func.parallelTransportFrame(previous_edge[:, 0], current_edges[:, 0], m_u0)
                    pred_vert, current_v, theta_full = self.simulation(init_pred_vert_0, current_v, self.init_direction.repeat(1, 1, 1), self.clamped_index, m_u0, inputs, self.clamped_selection, theta_full)
                    vert = init_pred_vert_0.clone()
                    inference_list.append(pred_vert[0].cpu().numpy() * 1000.)

                if sim_step >= 2:
                    previous_vert = vert.clone()
                    vert = pred_vert.clone()
                    current_v = current_v.clone()
                    m_u0 = m_u0.clone()
                    previous_edge = computeEdges(previous_vert)
                    current_edges = computeEdges(vert)
                    m_u0 = self.DER_func.parallelTransportFrame(previous_edge[:, 0], current_edges[:, 0], m_u0)
                    pred_vert, current_v, theta_full = self.simulation(vert, current_v, self.init_direction.repeat(1, 1, 1), self.clamped_index, m_u0, inputs, self.clamped_selection, theta_full)
                    inference_list.append(pred_vert[0].cpu().numpy() * 1000.)
        return pred_vert, inference_list

    def DER_sim(self, mocap_points, initial_vertices, current_v, inputs, inference_interval, eval_target_vertices=None, start=False):
        inputs = torch.from_numpy(inputs).unsqueeze(dim=0).to(self.device).float() / 1000.
        mocap_points = torch.from_numpy(mocap_points).unsqueeze(dim=0).to(self.device).float() / 1000.
        dir_path = "calibration_test/"
        inference_list = []
        for sim_step in range(inference_interval):
            if sim_step >= 0:
                weights = 0.125
            else:
                weights = 0.
            with torch.no_grad():
                if sim_step == 0:
                    theta_full = torch.zeros(1, self.n_vert - 1).to(self.device)
                    rest_edges = computeEdges(self.rest_vert)
                    m_u0 = self.DER_func.compute_m_u0(rest_edges[:, 0].float(), self.init_direction.repeat(1, 1, 1)[:, 0])
                    if start:
                        current_v = torch.zeros_like(current_v)
                    else:
                        current_v = current_v
                    init_pred_vert_0, current_v, theta_full = self.simulation(initial_vertices, current_v, self.init_direction.repeat(1, 1, 1), self.clamped_index, m_u0, inputs[:, sim_step], self.clamped_selection, theta_full)
                    init_pred_vert_0 = init_pred_vert_0 * (1 - weights) + mocap_points[:, sim_step] * weights
                    inference_list.append(init_pred_vert_0[0].cpu().numpy() * 1000.)

                if sim_step == 1:
                    previous_edge = computeEdges(initial_vertices)
                    current_edges = computeEdges(init_pred_vert_0)
                    m_u0 = self.DER_func.parallelTransportFrame(previous_edge[:, 0], current_edges[:, 0], m_u0)
                    pred_vert, current_v, theta_full = self.simulation(init_pred_vert_0, current_v,
                                                                       self.init_direction.repeat(1, 1, 1),
                                                                       self.clamped_index, m_u0, inputs[:, sim_step],
                                                                       self.clamped_selection, theta_full)
                    vert = init_pred_vert_0.clone()
                    pred_vert = pred_vert * (1 - weights) + mocap_points[:, sim_step] * weights
                    inference_list.append(pred_vert[0].cpu().numpy() * 1000.)

                if sim_step >= 2:
                    previous_vert = vert.clone()
                    vert = pred_vert.clone()
                    current_v = current_v.clone()
                    m_u0 = m_u0.clone()
                    previous_edge = computeEdges(previous_vert)
                    current_edges = computeEdges(vert)
                    m_u0 = self.DER_func.parallelTransportFrame(previous_edge[:, 0], current_edges[:, 0], m_u0)
                    pred_vert, current_v, theta_full = self.simulation(vert, current_v, self.init_direction.repeat(1, 1, 1), self.clamped_index, m_u0, inputs[:, sim_step], self.clamped_selection, theta_full)
                    pred_vert = pred_vert * (1 - weights) + mocap_points[:, sim_step] * weights
                    inference_list.append(pred_vert[0].cpu().numpy() * 1000.)
        return inference_list, current_v

    def inference(self, mocap_dir, image_dir, depth_dir, inference_interval=3, perception_time=1000, inference_method = "DiffDER"):
        """small"""
        """rr"""
        wasted_offset = 10
        offset = 390
        """rh"""
        # wasted_offset = 0
        # offset = 516 - 18
        traj_list = []
        for i in range(perception_time):
            image = np.load(r'%s'%image_dir)[int((i * inference_interval + wasted_offset) * 0.3)]
            depth = np.load(r'%s'%depth_dir)[int((i * inference_interval + wasted_offset) * 0.3)]
            image = cv2.cvtColor(image[:, :, 0:3], cv2.COLOR_BGR2RGB)
            plt.figure()
            plt.imshow(depth/255.)
            plt.show()
            # indices = self.segmentation_forward(image)
            width = 720
            height = 1080

            if i == 0:
                mocap_point = np.transpose(np.array(pd.read_pickle(r'%s' % mocap_dir))[i*inference_interval+offset+wasted_offset]) * 1000.
                reconstruction, clusters_centers = self.reconstruction_wo_pred(mocap_point, mocap_point[self.clamped_selection], 0, depth, mocap_point, sigma=15.)

                fig = plt.figure(figsize=(10, 7))
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(mocap_point[:, 0] / 1000, mocap_point[:, 1] / 1000, mocap_point[:, 2] / 1000, s=40, c="black", depthshade=False, label='Ground Truth')
                # ax.scatter(reconstruction[:, 0]/ 1000, reconstruction[:, 1]/ 1000, reconstruction[:, 2]/ 1000, s=1, c="blue", marker='o', label='Observed Point Cloud')
                # ax.scatter(mocap_point[:, 0]/1000, mocap_point[:, 1]/1000, mocap_point[:, 2]/1000, s=20, c="red", marker='o', label='Mocap')
                ax.scatter(clusters_centers[3:5, 0]/1000, clusters_centers[3:5, 1]/1000, clusters_centers[3:5, 2]/1000, s=40, depthshade=False, c="orange", marker='o', label='Predicted Occluded Points')
                ax.scatter(clusters_centers[0:3, 0]/1000, clusters_centers[0:3, 1]/1000, clusters_centers[0:3, 2]/1000, s=40, depthshade=False, c="red", marker='o', label='Estimated Vertices Group1')
                ax.scatter(clusters_centers[5:, 0]/1000, clusters_centers[5:, 1]/1000, clusters_centers[5:, 2]/1000, s=40, depthshade=False, c="green", marker='o', label='Estimated Vertices Group2')

                for j in range(clusters_centers.shape[0]):
                    ax.text(clusters_centers[j, 0]/1000, clusters_centers[j, 1]/1000, clusters_centers[j, 2]/1000, f'{j}', fontsize=10)

                loss_func = torch.nn.L1Loss()
                test_loss = loss_func(torch.from_numpy(clusters_centers/1000.).unsqueeze(dim=0), torch.from_numpy(mocap_point/1000.).unsqueeze(dim=0))
                print(test_loss)
                ax.set_xlim(-.5, 1.)
                ax.set_ylim(-1, .5)
                ax.set_zlim(0, 1.)
                # ax.set_title('Indexed GMM')
                ax.set_xlabel("X(m)")
                ax.set_ylabel("Y(m)")
                ax.set_zlabel("Z(m)")
                plt.legend()
                plt.savefig("cluster.png", dpi=500)
                # plt.show()

                mocap_point = np.swapaxes(np.array(pd.read_pickle(r'%s' % mocap_dir))[i*inference_interval+offset+wasted_offset:(i+1)*inference_interval+offset+wasted_offset], axis1=1, axis2=2) * 1000.
                inference_list, current_v = self.DER_sim(mocap_point, torch.from_numpy(clusters_centers).float().unsqueeze(dim=0).to(self.device) / 1000., torch.zeros_like(torch.from_numpy(clusters_centers).float().unsqueeze(dim=0).to(self.device) / 1000.),mocap_point[:, self.clamped_selection], inference_interval, start=True)
                for k, inference in enumerate(inference_list):
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    # ax.scatter(X_obs, Y_obs, Z_obs, label='Obstacle', s=4, c='orange')
                    traj_list.append(inference/1000.)
                    traj_list.append(mocap_point[k]/1000.)
                    ax.plot(inference[:, 0]/1000., inference[:, 1]/1000., inference[:, 2]/1000., label='pred')
                    ax.plot(mocap_point[k, :, 0]/1000., mocap_point[k, :, 1]/1000., mocap_point[k, :, 2]/1000., label='gt')
                    ax.set_xlim(-.5, 1.)
                    ax.set_ylim(-1, .5)
                    ax.set_zlim(0, 1.)
                    plt.legend()
                    plt.savefig('tracking/vertices_video/%s.png' %(i*inference_interval + k))
                    image_projection = np.load(r'%s' % image_dir)[int((i*inference_interval + k + wasted_offset) * 0.3)]
                    projected_image = self.projection(mocap_point[k], inference, image_projection)
                    cv2.imwrite('tracking/images_video/%s.png' %(i*inference_interval + k), cv2.cvtColor(cv2.cvtColor(projected_image, cv2.COLOR_RGB2BGR),  cv2.COLOR_RGB2BGR))

            else:
                mocap_point = np.transpose(np.array(pd.read_pickle(r'%s' % mocap_dir))[i*inference_interval+offset+wasted_offset]) * 1000.
                pred = inference_list[-1]
                reconstruction, clusters_centers = self.reconstruction(pred, 0, depth, sigma=15.)
                fig = plt.figure(figsize=(10, 7))
                ax = fig.add_subplot(111, projection='3d')
                # ax.scatter(reconstruction[:, 0], reconstruction[:, 1], reconstruction[:, 2], s=1, c="green", marker='o', label='Camera Percpeiton')
                ax.scatter(mocap_point[:, 0]/1000, mocap_point[:, 1]/1000, mocap_point[:, 2]/1000, s=20, c="red", marker='o', label='Mocap')
                ax.scatter(clusters_centers[:, 0]/1000, clusters_centers[:, 1]/1000, clusters_centers[:, 2]/1000, s=20, c="black", marker='o', label='GMM Centers')

                for j in range(clusters_centers.shape[0]):
                    ax.text(clusters_centers[j, 0] / 1000, clusters_centers[j, 1] / 1000, clusters_centers[j, 2] / 1000, f'{j}', fontsize=15)

                loss_func = torch.nn.L1Loss()
                test_loss = loss_func(torch.from_numpy(clusters_centers / 1000.).unsqueeze(dim=0), torch.from_numpy(mocap_point / 1000.).unsqueeze(dim=0))
                print(test_loss)
                ax.set_xlim(-.5, 1.)
                ax.set_ylim(-1, .5)
                ax.set_zlim(0, 1.)
                ax.set_title('Indexed GMM')
                ax.set_xlabel("X(mm)")
                ax.set_ylabel("Y(mm)")
                ax.set_zlabel("Z(mm)")
                plt.legend()
                # plt.savefig('small_wire_dataset/tracking/image_video/%s.png'%i)
                mocap_point = np.swapaxes(np.array(pd.read_pickle(r'%s' % mocap_dir))[i*inference_interval+offset+wasted_offset:(i+1)*inference_interval+offset+wasted_offset], axis1=1, axis2=2) * 1000.
                inference_list, current_v = self.DER_sim(mocap_point, torch.from_numpy(clusters_centers).float().unsqueeze(dim=0).to(self.device)/1000., current_v, mocap_point[:, self.clamped_selection], inference_interval, start=False)
                for k, inference in enumerate(inference_list):
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    # ax.scatter(X_obs, Y_obs, Z_obs, label='Obstacle', s=4, c='orange')
                    traj_list.append(inference / 1000.)
                    traj_list.append(mocap_point[k] / 1000.)
                    ax.plot(inference[:, 0]/1000., inference[:, 1]/1000., inference[:, 2]/1000., label='pred')
                    ax.plot(mocap_point[k, :, 0]/1000., mocap_point[k, :, 1]/1000., mocap_point[k, :, 2]/1000., label='gt')
                    ax.set_xlim(-.5, 1.)
                    ax.set_ylim(-1, .5)
                    ax.set_zlim(0, 1.)
                    plt.legend()
                    plt.savefig('tracking/vertices_video/%s.png' %(i*inference_interval + k))
                    image_projection = np.load(r'%s' % image_dir)[int((i * inference_interval + k + wasted_offset) * 0.3)]
                    projected_image = self.projection(mocap_point[k], inference, image_projection)
                    cv2.imwrite('tracking/images_video/%s.png' %(i*inference_interval + k), cv2.cvtColor(cv2.cvtColor(projected_image, cv2.COLOR_RGB2BGR),  cv2.COLOR_RGB2BGR))
            print("saved")
            np.save("tracking/vertices_video/inference.npy", traj_list)

perception_test = perception("cpu")
mocap_dir = 'tracking/rr_thin_pkl.pkl'
image_dir = 'tracking/rr_thin_image.npy'
depth_dir = 'tracking/rr_thin_depth.npy'
inference_method = "DiffDER"
test = perception_test.inference(mocap_dir, image_dir, depth_dir)
# #
# image = cv2.cvtColor(image[:, :, 0:3], cv2.COLOR_BGR2RGB)
# indices = mask_test.segmentation_forward(image)
# reconstruction, clusters_centers = mask_test.reconstruction_wo_pred(mocap_points[0], mocap_points[-1], indices, depth, mocap_points, sigma=15.)
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')
# # ax.scatter(reconstruction[:, 0], reconstruction[:, 1], reconstruction[:, 2], s=1, c="green", marker='o', label='Camera Percpeiton')
# ax.scatter(mocap_points[:, 0], mocap_points[:, 1], mocap_points[:, 2], s=20, c="red", marker='o', label='Mocap')
# ax.scatter(clusters_centers[:, 0], clusters_centers[:, 1], clusters_centers[:, 2], s=20, c="black", marker='o', label='GMM Centers')
# for i in range(clusters_centers.shape[0]):
#     ax.text(clusters_centers[i, 0], clusters_centers[i, 1], clusters_centers[i, 2], f'{i}', fontsize=15)
#
# print(clusters_centers.shape)
# print(mocap_points.shape)
# loss_func = torch.nn.L1Loss()
# test = loss_func(torch.from_numpy(clusters_centers/1000.).unsqueeze(dim=0), torch.from_numpy(mocap_points/1000.).unsqueeze(dim=0))
# print(test)
# ax.set_title('Indexed GMM')
# ax.set_xlabel("X(mm)")
# ax.set_ylabel("Y(mm)")
# ax.set_zlabel("Z(mm)")
# plt.legend()
# plt.show()
#
# projection = mask_test.projection(mocap_points, clusters_centers, image)
# plt.figure()
# plt.imshow(projection)
# plt.show()

# from sklearn.cluster import DBSCAN
# from sklearn.datasets import make_blobs
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np
#
# # Generate 3D sample data
# centers = [[1, 1, 1], [-1, -1, -1], [-1, -1, -1.02], [1, -1, 1], [1, -1.1, 1]]
# X, _ = make_blobs(n_samples=750, centers=centers, cluster_std=0.4, random_state=0)
# X = StandardScaler().fit_transform(X)
# print(X.shape)
# # Compute DBSCAN
# db = DBSCAN(eps=0.3, min_samples=10).fit(X)
# labels = db.labels_
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# # Plotting
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# unique_labels = set(labels)
# colors = [plt.cm.Spectral(each)
#           for each in np.linspace(0, 1, len(unique_labels))]
#
# for k, col in zip(unique_labels, colors):
#     if k == -1:
#         # Black used for noise.
#         col = [0, 0, 0, 1]
#
#     class_member_mask = (labels == k)
#
#     # Plot the clustered points
#     xyz = X[class_member_mask]
#     ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=tuple(col), marker='o')
#
# ax.set_title('Estimated number of clusters: %d' % n_clusters_)
# plt.show()

# import numpy as np
# import open3d as o3d
#
# # Example: Create a NumPy array representing points in a point cloud
# # This is just a placeholder. Replace it with your actual point cloud data.
# points = np.random.rand(1000, 3)  # 1000 random points in 3D space
#
# # Convert the NumPy array to an Open3D point cloud
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)
#
# # Perform voxel grid downsampling
# voxel_size = 0.05  # size of the voxel: adjust this parameter as needed
# downsampled_pcd = pcd.voxel_down_sample(voxel_size)
#
# # Now you can proceed to perform clustering on 'downsampled_pcd'
# # ...
#
# # For visualization, you can convert it back to NumPy array (if needed)
# downsampled_points_np = np.asarray(downsampled_pcd.points)
#
# # (Optional) Visualize the downsampled point cloud
# o3d.visualization.draw_geometries([downsampled_pcd])