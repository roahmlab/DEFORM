import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
import os
import pandas as pd
from tqdm import tqdm
from DEFORM_func import DEFORM_func
from DEFORM_sim import DEFORM_sim
from util import computeLengths, computeEdges, compute_u0, parallelTransportFrame
import pickle
import random
import torch.nn as nn

random.seed(0)
torch.manual_seed(0)

class Train_DeformData(Dataset):
    def __init__(self, DLO_type, train_set_number, time_horizon, device):
        super(Train_DeformData, self).__init__()
        '''
        change the root dir based in your dir
        '''
        self.root_dir = "data_set/%s/train/" %DLO_type
        inputs_file_list = glob.glob(self.root_dir + "*")
        self.device = device
        self.inputs = []
        bar = tqdm(random.choices(inputs_file_list, k=train_set_number))
        length = time_horizon
        self.previous_vertices = []
        self.vertices = []
        self.target_vertices = []
        self.end_vertices = []
        self.mu_0 = []
        for rope_data in bar:
            rope_verts = pd.read_pickle(r'%s' % str(rope_data))
            mu_0_list = torch.zeros(len(rope_verts) - 1 - 1, 3).to(self.device)
            for i in range(len(rope_verts) - 1 - 1):
                if i == 0:
                    init_direction = torch.tensor(((0., 0.6, 0.8), (0., .0, 1.))).to(self.device).unsqueeze(dim=0)
                    vertices = torch.transpose(torch.tensor(np.array(rope_verts[i + 1: i + 1 + 1])).to(self.device),1, 2).float()
                    rest_edges = computeEdges(vertices)
                    m_u0 = compute_u0(rest_edges.float()[:, 0], init_direction.repeat(1, 1, 1)[:, 0])
                    mu_0_list[i] = m_u0

                else:
                    previous_vertices = torch.transpose(torch.tensor(np.array(rope_verts[i: i + 1])).to(self.device),1, 2).float()
                    current_vertices = torch.transpose(torch.tensor(np.array(rope_verts[i + 1: i + 1 + 1])).to(self.device),1, 2).float()
                    previous_edge = computeEdges(previous_vertices)
                    current_edges = computeEdges(current_vertices)
                    m_u0 = parallelTransportFrame(previous_edge[:, 0], current_edges[:, 0], m_u0.clone())
                    mu_0_list[i] = m_u0

            for i in range(len(rope_verts) - 1 - length):
                self.previous_vertices.append(rope_verts[i:i + length])
                self.vertices.append(rope_verts[i + 1: i + 1 + length])
                self.target_vertices.append(rope_verts[i + 2:i + 2 + length])
                self.mu_0.append(mu_0_list[i:i + length])

        self.previous_vertices = np.array(self.previous_vertices)
        self.previous_vertices[:, :, -1] = np.clip(self.previous_vertices[:, :, -1], a_min=2e-3 + 1e-6, a_max=10000.)

        self.vertices = np.array(self.vertices)
        self.vertices[:, :, -1] = np.clip(self.vertices[:, :, -1], a_min=2e-3 + 1e-6, a_max=10000.)

        self.target_vertices = np.array(self.target_vertices)
        self.target_vertices[:, :, -1] = np.clip(self.target_vertices[:, :, -1], a_min=2e-3 + 1e-6, a_max=10000.)

    def __len__(self):
        return len(self.vertices)

    def __getitem__(self, index):
        previous_vertices = torch.transpose(torch.tensor(np.array(self.previous_vertices[index])).to(self.device), 1, 2).float()
        vertices = torch.transpose(torch.tensor(np.array(self.vertices[index])).to(self.device), 1,2).float()
        target_vertices = torch.transpose(torch.tensor(np.array(self.target_vertices[index])).to(self.device), 1, 2).float()
        return previous_vertices.clone().detach(), vertices.clone().detach(), target_vertices.clone().detach(), self.mu_0[index].clone().detach()

class Eval_DeformData(Dataset):
    def __init__(self, DLO_type, eval_set_number, time_horizon, device):
        super(Eval_DeformData, self).__init__()
        self.root_dir = "data_set/%s/eval/" %DLO_type
        inputs_file_list = glob.glob(self.root_dir + "*")
        self.device = device
        bar = tqdm(random.choices(inputs_file_list, k=eval_set_number))
        length = time_horizon

        self.previous_vertices = []
        self.vertices = []
        self.target_vertices = []
        self.gt_m0 = []
        for rope_data in bar:
            rope_verts = pd.read_pickle(r'%s' % str(rope_data))
            self.previous_vertices.append(rope_verts[:0 + length])
            self.vertices.append(rope_verts[1:1 + length])
            self.target_vertices.append(rope_verts[2:2 + length])

        self.previous_vertices = np.array(self.previous_vertices)
        self.previous_vertices[:, :, 2] = np.clip(self.previous_vertices[:, :, 2], a_min=2e-3 + 1e-6, a_max=10000.)

        self.vertices = np.array(self.vertices)
        self.vertices[:, :, 2] = np.clip(self.vertices[:, :, 2], a_min=2e-3 + 1e-6, a_max=10000.)

        self.target_vertices = np.array(self.target_vertices)
        self.target_vertices[:, :, 2] = np.clip(self.target_vertices[:, :, 2], a_min=2e-3 + 1e-6, a_max=10000.)

    def __len__(self):
        return len(self.vertices)

    def __getitem__(self, index):
        previous_vertices = torch.transpose(torch.tensor(np.array(self.previous_vertices[index])).to(self.device), 1, 2).float()
        vertices = torch.transpose(torch.tensor(np.array(self.vertices[index])).to(self.device), 1, 2).float()
        target_vertices = torch.transpose(torch.tensor(np.array(self.target_vertices[index])).to(self.device),1, 2).float()
        return previous_vertices, vertices, target_vertices

def save_pickle(data, myfile):
    with open(myfile, "wb") as f:
        pickle.dump(data, f)

def train(DLO_type, train_set_number, eval_set_number, train_time_horizon, eval_time_horizon, batch, DEFORM_func, DEFORM_sim, device):
    '''
    Dataset Loading
    '''
    train_dataset = Train_DeformData(DLO_type, train_set_number, train_time_horizon, device)
    eval_dataset = Eval_DeformData(DLO_type, eval_set_number, eval_time_horizon, device)
    eval_data_len = len(eval_dataset)
    train_data_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, drop_last=True)
    '''
    pre set for DLO:
    n_vert: number of vertices
    n_edge: number of edges
    '''
    if DLO_type == "DLO1":
        n_vert = 13
        n_edge = n_vert - 1
        device = device
        DEFORM_func = DEFORM_func(n_vert=n_vert, n_edge=n_vert - 1, device=device)
        '''pbd itr: inextensibility enforcement loop. number > 5 should able to satisfy the condition'''
        DEFORM_sim = DEFORM_sim(n_vert=n_vert, n_edge=n_vert-1, pbd_iter=10, device=device)
        '''
        rest_vert: undeformed states. Dependent on wires. In simulation, it is typically initialized with a straight wire that is segemented equally.
        '''
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
        DEFORM_sim.rest_vert = nn.Parameter(rest_vert)
        '''
        stiffness of bending and twisting: dependent on wires. typically initialized with heuristic trials/grid search with the smallest eval loss.
        '''
        DEFORM_sim.DEFORM_func.bend_stiffness = nn.Parameter(5e-5 * torch.ones((1, n_edge), device=device))
        DEFORM_sim.DEFORM_func.twist_stiffness = nn.Parameter(2e-5 * torch.ones((1, n_edge), device=device))
        '''
        load trained model. comment following when train first time.
        '''
        # DEFORM_sim.load_state_dict(torch.load("save_model/DLO1_540.pth"))

    elif DLO_type == "DLO2":
        n_vert = 12
        n_edge = n_vert - 1
        device = device
        DEFORM_func = DEFORM_func(n_vert=n_vert, n_edge=n_vert - 1, device=device)
        DEFORM_sim = DEFORM_sim(n_vert=n_vert, n_edge=n_vert - 1, pbd_iter=10, device=device)
        rest_vert = (torch.tensor(((0.725862, -0.196132, 0.013556),
                                   (0.719875, -0.165722, 0.009538),
                                   (0.697891, -0.068908, 0.013519),
                                   (0.642622, 0.006184, 0.008588),
                                   (0.559875, 0.054215, 0.008419),
                                   (0.468611, 0.075446, 0.009509),
                                   (0.376396, 0.07341, 0.010467),
                                   (0.289067, 0.041016, 0.008857),
                                   (0.214187, -0.019351, 0.017508),
                                   (0.170766, -0.099437, 0.006587),
                                   (0.161013, -0.200349, 0.007841),
                                   (0.161086, -0.228518, 0.007807)))).unsqueeze(dim=0).repeat(1, 1, 1).to(device)
        rest_vert = torch.cat((rest_vert[:, :, 0].unsqueeze(dim=-1), rest_vert[:, :, 2].unsqueeze(dim=-1), -rest_vert[:, :, 1].unsqueeze(dim=-1)), dim=-1)
        DEFORM_sim.m_restEdgeL, DEFORM_sim.m_restRegionL = computeLengths(computeEdges(rest_vert.clone()))
        DEFORM_sim.rest_vert = nn.Parameter(rest_vert)
        DEFORM_sim.DEFORM_func.bend_stiffness = nn.Parameter(8e-4 * torch.ones((1, n_edge), device=device))
        DEFORM_sim.DEFORM_func.twist_stiffness = nn.Parameter(3e-5 * torch.ones((1, n_edge), device=device))

    elif DLO_type == "DLO3":
        n_vert = 12
        n_edge = n_vert - 1
        device = device
        DEFORM_func = DEFORM_func(n_vert=n_vert, n_edge=n_vert - 1, device=device)
        DEFORM_sim = DEFORM_sim(n_vert=n_vert, n_edge=n_vert - 1, pbd_iter=10, device=device)
        rest_vert = (torch.tensor(((0.704214, -0.046593, 0.020496),
                                   (0.712317, -0.078647, 0.025723),
                                   (0.727923, -0.180886, 0.032423),
                                   (0.702225, -0.273037, 0.031611),
                                   (0.634172, -0.347682, 0.027974),
                                   (0.53685, -0.373692, 0.035285),
                                   (0.430097, -0.379901, 0.029374),
                                   (0.337156, -0.366995, 0.030347),
                                   (0.258182, -0.311241, 0.021588),
                                   (0.2192, -0.209264, 0.022677),
                                   (0.199719, -0.120685, 0.019185),
                                   (0.190919, -0.082036, 0.018718)))).unsqueeze(dim=0).repeat(1, 1, 1).to(device)

        rest_vert = torch.cat((rest_vert[:, :, 0].unsqueeze(dim=-1), rest_vert[:, :, 2].unsqueeze(dim=-1), rest_vert[:, :, 1].unsqueeze(dim=-1)), dim=-1)
        DEFORM_sim.m_restEdgeL, DEFORM_sim.m_restRegionL = computeLengths(computeEdges(rest_vert.clone()))
        DEFORM_sim.rest_vert = nn.Parameter(rest_vert)
        DEFORM_sim.DEFORM_func.bend_stiffness = nn.Parameter(8e-4 * torch.ones((1, n_edge), device=device))
        DEFORM_sim.DEFORM_func.twist_stiffness = nn.Parameter(5e-5 * torch.ones((1, n_edge), device=device))
    else:
        raise ValueError("No matching DLO type")

    """clamped start edge and end edge"""
    clamped_index = torch.zeros(n_vert)
    clamped_selection = torch.tensor((0, 1, -2, -1))
    clamped_index[clamped_selection] = torch.tensor((1.))

    """learning setup"""
    loss_func = torch.nn.L1Loss()
    network_lr = 1e-4
    lr_scale = 0.1
    parameters_to_update = [
        {"params": DEFORM_sim.integration_ratio, "lr": 1e-5 * lr_scale},
        {"params": DEFORM_sim.velocity_ratio, "lr": 1e-5 * lr_scale},
        {"params": DEFORM_sim.rest_vert, "lr": 1e-5 * lr_scale},
        {"params": DEFORM_sim.mocap_mass, "lr": 1e-5 * lr_scale},
        {"params": DEFORM_sim.DEFORM_func.bend_stiffness, "lr": 1e-11 * lr_scale},
        {"params": DEFORM_sim.DEFORM_func.twist_stiffness, "lr": 1e-11 * lr_scale},
        {"params": DEFORM_sim.vert_conv1.parameters(), "lr": network_lr * lr_scale},
        {"params": DEFORM_sim.vert_conv2.parameters(), "lr": network_lr * lr_scale},
        {"params": DEFORM_sim.delta_vert_conv1.parameters(), "lr": network_lr * lr_scale},
        {"params": DEFORM_sim.delta_vert_conv2.parameters(), "lr": network_lr * lr_scale},
        {"params": DEFORM_sim.fc.parameters(), "lr": network_lr * lr_scale},
    ]
    # Create an optimizer with different learning rates
    optimizer = torch.optim.SGD(parameters_to_update)

    """record steps and losses"""
    epochs = []
    losses = []
    eval_epochs = []
    eval_losses = []

    train_epoch = 100
    save_steps = 0
    evaluate_period = 20
    save_period = 20
    update_steps = 0

    for epoch in range(train_epoch):
        bar = tqdm(train_data_loader)
        for data in bar:
            if save_steps % evaluate_period == 0:
                print("evaluating")
                eval_batch = eval_set_number
                part_eval = eval_set_number
                eval_set, test_set = torch.utils.data.random_split(eval_dataset, [part_eval, eval_data_len - part_eval])
                eval_data_loader = DataLoader(eval_set, batch_size=eval_batch, shuffle=True, drop_last=True)
                torch.save(DEFORM_sim.state_dict(),os.path.join("save_model/", "%s_%s.pth" % (DLO_type, str(update_steps))))
                eval_loss = 0
                eval_bar = tqdm(eval_data_loader)
                """evaluation"""
                with torch.no_grad():
                    eval_time = 0
                    for eval_data in eval_bar:
                        init_direction = torch.tensor(((0., 0.6, 0.8), (0., .0, 1.))).to(device).unsqueeze(dim=0)
                        eval_previous_vertices, eval_vertices, eval_target_vertices = eval_data
                        inputs = eval_target_vertices[:, :, clamped_selection]
                        """
                        initialize all theta = 0
                        """
                        theta_full = torch.zeros(eval_batch, n_vert - 1).to(device)
                        for traj_num in range(eval_target_vertices.size()[1]):
                            with torch.no_grad():
                                if traj_num == 0:
                                    rest_edges = computeEdges(eval_vertices[:, traj_num])
                                    m_u0 = DEFORM_func.compute_u0(rest_edges[:, 0].float(), init_direction.repeat(eval_batch, 1, 1)[:, 0])
                                    current_v = (eval_vertices[:, traj_num] - eval_previous_vertices[:, traj_num]).div(DEFORM_sim.dt)
                                    init_pred_vert_0, current_v, theta_full = DEFORM_sim(eval_vertices[:, traj_num], current_v, init_direction.repeat(eval_batch, 1, 1), clamped_index, m_u0, inputs[:, traj_num], clamped_selection, theta_full)
                                    traj_loss = loss_func(init_pred_vert_0, eval_target_vertices[:, traj_num].float())
                                    eval_loss += traj_loss

                                    """visualization"""
                                    # init_vis_vert = torch.Tensor.numpy(init_pred_vert_0.to('cpu'))
                                    # vis_gt_vert = torch.Tensor.numpy(eval_target_vertices[:, traj_num].to('cpu'))
                                    # fig = plt.figure()
                                    # ax = fig.add_subplot(111, projection='3d')
                                    # # ax.scatter(X_obs, Y_obs, Z_obs, label='Obstacle', s=4, c='orange')
                                    # ax.plot(init_vis_vert[0, :, 0], init_vis_vert[0, :, 1], init_vis_vert[0, :, 2], label='pred')
                                    # ax.plot(vis_gt_vert[0, :, 0], vis_gt_vert[0, :, 1], vis_gt_vert[0, :, 2], label='gt')
                                    # ax.set_xlim(-.5, 1.)
                                    # ax.set_ylim(-1, .5)
                                    # ax.set_zlim(0, 1.)
                                    # plt.legend()
                                    # plt.savefig(dir_path + '/%s.png' % (traj_num))

                                if traj_num == 1:
                                    previous_edge = computeEdges(eval_previous_vertices[:, traj_num])
                                    current_edges = computeEdges(init_pred_vert_0)
                                    m_u0 = DEFORM_func.parallelTransportFrame(previous_edge[:, 0], current_edges[:, 0], m_u0)
                                    pred_vert, current_v, theta_full = DEFORM_sim(init_pred_vert_0, current_v, init_direction.repeat(eval_batch, 1, 1), clamped_index, m_u0, inputs[:, traj_num], clamped_selection, theta_full)
                                    vert = init_pred_vert_0.clone()
                                    traj_loss = loss_func(pred_vert, eval_target_vertices[:, traj_num])
                                    eval_loss += traj_loss

                                    # vis_pred_vert = torch.Tensor.numpy(pred_vert.to('cpu'))
                                    # vis_gt_vert = torch.Tensor.numpy(eval_target_vertices[:, traj_num].to('cpu'))
                                    # fig = plt.figure()
                                    # ax = fig.add_subplot(111, projection='3d')
                                    # # ax.scatter(X_obs, Y_obs, Z_obs, label='Obstacle', s=4, c='orange')
                                    # ax.plot(vis_pred_vert[0, :, 0], vis_pred_vert[0, :, 1], vis_pred_vert[0, :, 2], label='pred')
                                    # ax.plot(vis_gt_vert[0, :, 0], vis_gt_vert[0, :, 1], vis_gt_vert[0, :, 2], label='gt')
                                    # ax.set_xlim(-.5, 1.)
                                    # ax.set_ylim(-1, .5)
                                    # ax.set_zlim(0, 1.)
                                    # plt.legend()
                                    # plt.savefig(dir_path + '/%s.png' % (traj_num))

                                if traj_num >= 2:
                                    previous_vert = vert.clone()
                                    vert = pred_vert.clone()
                                    current_v = current_v.clone()
                                    m_u0 = m_u0.clone()
                                    previous_edge = computeEdges(previous_vert)
                                    current_edges = computeEdges(vert)
                                    m_u0 = DEFORM_func.parallelTransportFrame(previous_edge[:, 0], current_edges[:, 0],m_u0)
                                    pred_vert, current_v, theta_full = DEFORM_sim(vert, current_v,init_direction.repeat(eval_batch, 1, 1),clamped_index, m_u0, inputs[:, traj_num], clamped_selection, theta_full)
                                    traj_loss = loss_func(pred_vert, eval_target_vertices[:, traj_num])
                                    eval_loss += traj_loss

                                    # vis_pred_vert = torch.Tensor.numpy(pred_vert.to('cpu'))
                                    # vis_gt_vert = torch.Tensor.numpy(eval_target_vertices[:, traj_num].to('cpu'))
                                    # fig = plt.figure()
                                    # ax = fig.add_subplot(111, projection='3d')
                                    # # ax.scatter(X_obs, Y_obs, Z_obs, label='Obstacle', s=4, c='orange')
                                    # ax.plot(vis_pred_vert[0, :, 0], vis_pred_vert[0, :, 1], vis_pred_vert[0, :, 2],label='pred')
                                    # ax.plot(vis_gt_vert[0, :, 0], vis_gt_vert[0, :, 1], vis_gt_vert[0, :, 2], label='gt')
                                    # ax.set_xlim(-.5, 1.)
                                    # ax.set_ylim(-1, .5)
                                    # ax.set_zlim(0, 1.)
                                    # plt.legend()
                                    # plt.savefig(dir_path + '/%s.png' % (traj_num))

                            eval_time += 1
                eval_losses.append(eval_loss.cpu().detach().numpy() / (eval_time_horizon * part_eval // eval_batch))
                eval_epochs.append(update_steps)
                save_pickle(eval_losses, "loss_record/eval_loss_%s.pkl" % (DLO_type))
                save_pickle(eval_epochs, "loss_record/eval_epoch_%s.pkl" % (DLO_type))
            """"""

            """training"""
            theta_full = torch.zeros(batch, n_vert - 1).to(device)
            traj_loss_record = 0

            if train_time_horizon == 1:
                previous_vertices, vertices, target_vertices, m_u0 = data
                inputs = target_vertices[:, :, clamped_selection]

                traj_num = 0
                optimizer.zero_grad()
                current_v = (vertices[:, traj_num] - previous_vertices[:, traj_num]).div(DEFORM_sim.dt)
                target_v = (target_vertices[:, traj_num] - vertices[:, traj_num]).div(DEFORM_sim.dt)
                pred_vertice, pred_v, theta_full = DEFORM_sim(vertices[:, traj_num], current_v, init_direction.repeat(batch, 1, 1), clamped_index, m_u0[:, traj_num], inputs[:, traj_num], clamped_selection, theta_full)
                traj_loss = loss_func(pred_vertice, target_vertices[:, traj_num])
                v_loss = loss_func(pred_v, target_v)
                (traj_loss + v_loss).backward(retain_graph=True)
                optimizer.step()

                save_steps += 1
                update_steps += 1

                losses.append(traj_loss.cpu().detach().numpy() / train_time_horizon)
                epochs.append(update_steps)
                if save_steps % save_period == 0:
                    save_pickle(losses, "loss_record/train_loss_%s.pkl" %DLO_type)
                    save_pickle(epochs, "loss_record/train_epoch_%s.pkl" %DLO_type)

            if train_time_horizon > 1:
                previous_vertices, vertices, target_vertices, m_u0 = data
                if train_time_horizon == 2:
                    inputs = target_vertices[:, :, clamped_selection]
                    optimizer.zero_grad()
                    loss = 0
                    for traj_num in range(2):
                        if traj_num == 0:
                            current_v = (vertices[:, traj_num] - previous_vertices[:, traj_num]).div(DEFORM_sim.dt)
                            target_v = (target_vertices[:, traj_num] - vertices[:, traj_num]).div(DEFORM_sim.dt)
                            pred_vertice, current_v, theta_full = DEFORM_sim(vertices[:, traj_num], current_v, init_direction.repeat(batch, 1, 1), clamped_index, m_u0[:, traj_num], inputs[:, traj_num], clamped_selection, theta_full)
                            traj_loss = loss_func(pred_vertice, target_vertices[:, traj_num])
                            v_loss = loss_func(current_v, target_v)
                            loss += traj_loss + v_loss

                        if traj_num == 1:
                            previous_edge = computeEdges(previous_vertices[:, traj_num])
                            current_edges = computeEdges(pred_vertice)
                            m_u0 = DEFORM_func.parallelTransportFrame(previous_edge[:, 0], current_edges[:, 0],m_u0[:, traj_num])
                            target_v = (target_vertices[:, traj_num] - vertices[:, traj_num]).div(DEFORM_sim.dt)
                            pred_vertice, current_v, theta_full = DEFORM_sim(pred_vertice.clone(), current_v.clone(), init_direction.repeat(batch, 1, 1), clamped_index, m_u0, inputs[:, traj_num],
                                clamped_selection, theta_full)
                            traj_loss = loss_func(pred_vertice, target_vertices[:, traj_num])
                            v_loss = loss_func(current_v, target_v)
                            loss += traj_loss + v_loss
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    save_steps += 1
                    update_steps += 1
                    losses.append(traj_loss.cpu().detach().numpy() / train_time_horizon)
                    epochs.append(update_steps)
                    if save_steps % save_period == 0:
                        save_pickle(losses, "loss_record/train_loss_%s.pkl" % DLO_type)
                        save_pickle(epochs, "loss_record/train_epoch_%s.pkl" % DLO_type)

                else:
                    inputs = target_vertices[:, :, clamped_selection]
                    optimizer.zero_grad()
                    loss = 0
                    for traj_num in range(train_time_horizon):
                        if traj_num == 0:
                            current_v = (vertices[:, traj_num] - previous_vertices[:, traj_num]).div(DEFORM_sim.dt)
                            target_v = (target_vertices[:, traj_num] - vertices[:, traj_num]).div(DEFORM_sim.dt)
                            pred_vert, current_v, theta_full = DEFORM_sim(vertices[:, traj_num], current_v, init_direction.repeat(batch, 1, 1), clamped_index, m_u0[:, traj_num], inputs[:, traj_num], clamped_selection, theta_full)
                            traj_loss = loss_func(pred_vert, target_vertices[:, traj_num])
                            v_loss = loss_func(current_v, target_v)
                            loss += traj_loss + v_loss
                            traj_loss_record += traj_loss

                        if traj_num == 1:
                            previous_edge = computeEdges(previous_vertices[:, traj_num])
                            current_edges = computeEdges(pred_vert)
                            vert = pred_vert.clone()
                            m_u0 = DEFORM_func.parallelTransportFrame(previous_edge[:, 0], current_edges[:, 0], m_u0[:, traj_num])
                            target_v = (target_vertices[:, traj_num] - vertices[:, traj_num]).div(DEFORM_sim.dt)
                            pred_vert, current_v, theta_full = DEFORM_sim(pred_vert.clone(), current_v.clone(), init_direction.repeat(batch, 1, 1), clamped_index, m_u0, inputs[:, traj_num], clamped_selection, theta_full)
                            traj_loss = loss_func(pred_vert, target_vertices[:, traj_num])
                            v_loss = loss_func(current_v, target_v)
                            loss += traj_loss + v_loss
                            traj_loss_record += traj_loss

                        if traj_num >= 2:
                            previous_vert = vert.clone()
                            vert = pred_vert.clone()
                            current_v = current_v.clone()
                            m_u0 = m_u0.clone()
                            previous_edge = computeEdges(previous_vert)
                            current_edges = computeEdges(vert)
                            m_u0 = DEFORM_func.parallelTransportFrame(previous_edge[:, 0], current_edges[:, 0], m_u0)
                            target_v = (target_vertices[:, traj_num] - vertices[:, traj_num]).div(DEFORM_sim.dt)
                            pred_vert, current_v, theta_full = DEFORM_sim(vert.clone(), current_v.clone(), init_direction.repeat(batch, 1, 1), clamped_index, m_u0, inputs[:, traj_num], clamped_selection, theta_full)
                            traj_loss = loss_func(pred_vert, target_vertices[:, traj_num])
                            v_loss = loss_func(current_v, target_v)
                            traj_loss_record += traj_loss
                            loss += traj_loss + v_loss

                    loss.backward(retain_graph=True)
                    optimizer.step()
                    save_steps += 1
                    update_steps += 1
                    losses.append(traj_loss_record.cpu().detach().numpy() / train_time_horizon)
                    epochs.append(update_steps)
                    if save_steps % save_period == 0:
                        save_pickle(losses, "loss_record/train_loss_%s.pkl" % DLO_type)
                        save_pickle(epochs, "loss_record/train_epoch_%s.pkl" % DLO_type)

if __name__ == "__main__":
    '''
    DLO_type: DLO type name, related to training dataset folder, saved model name and loss record. For loss record, try to explore using tensor board
    eval/train set number = number of pickle file
    eval/train time horizon: in this case, FPS = 100 hz. change self.dt in DEFROM_sim but test it stability first 
    batch: training batch. eval batch default = eval set number
    device: CUDA/CPU switchable
    '''
    train(DLO_type="DLO1", train_set_number=56, eval_set_number=14, train_time_horizon=100, eval_time_horizon=500, batch=32, DEFORM_func=DEFORM_func, DEFORM_sim=DEFORM_sim, device="cpu")
    # training log:
    # DEFORM_sim.DEFORM_func.bend_stiffness = nn.Parameter(5e-5 * torch.ones((1, n_edge), device=device))
    # DEFORM_sim.DEFORM_func.twist_stiffness = nn.Parameter(2e-5 * torch.ones((1, n_edge), device=device))
    # new edge index
    # learning weight: 0.5
    # loss:
