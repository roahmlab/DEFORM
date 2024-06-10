import torch
import torch.nn as nn
import torch.nn.functional as F
from DEFORM_func import DEFORM_func
from util import rotation_matrix
from util import computeW, computeLengths, computeEdges, computeLength_only
import theseus as th
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


class DEFORM_sim(nn.Module):
    def __init__(self, n_vert, n_edge, pbd_iter, device):
        super().__init__()
        """
        Parameters:
        n_vert: number of vertices
        n_edge: number of edges 
        pbd_iter: iteration of enforcing inextensibility with momentum conservation
        device: cpu/cuda
        """
        self.n_vert = n_vert
        self.n_edge = n_edge
        self.device = device

        self.DEFORM_func = DEFORM_func(n_vert, n_edge, device=device)
        self.integration_ratio = nn.Parameter(torch.tensor(1., device=device))
        self.velocity_ratio = nn.Parameter(torch.tensor(0., device=device))
        self.force_scale = torch.tensor((5.)).to(device)

        """
        Residual Learning with GCN
        To do: consider using other smaller size neural network, e.g., lstm
        """
        hidden_size = 32
        self.vert_conv1 = GCNConv(3, hidden_size).to(device)
        self.vert_conv2 = GCNConv(hidden_size, hidden_size).to(device)

        self.delta_vert_conv1 = GCNConv(3, hidden_size).to(device)
        self.delta_vert_conv2 = GCNConv(hidden_size, hidden_size).to(device)
        self.fc = nn.Sequential(nn.Linear(self.n_vert * (hidden_size * 2) + 4 * 3, hidden_size * 3),
                                nn.ReLU(),
                                nn.Linear((hidden_size * 3), (self.n_vert-4) * 3)).to(device)  # Final output size per time step

        """undeformed configuration"""
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
        self.m_restEdgeL, self.m_restRegionL = computeLengths(computeEdges(rest_vert.clone()))
        self.rest_vert = nn.Parameter(rest_vert)
        self.m_pmass = torch.ones(1, self.n_vert).to(self.device) * 0.02
        self.mocap_mass = nn.Parameter(torch.ones(1, self.n_vert).to(self.device) * 1e-7)
        self.gravity = torch.tensor((0, 0, -9.81), device=device)
        self.dt = 1e-2
        self.pbd_iter = pbd_iter

        """vectorized force accumulation"""
        self.w_masks = torch.zeros(1, n_vert, n_edge, 1).to(device)
        self.m_masks = torch.zeros(1, n_vert, n_edge, 1, 1).to(device)
        self.plusGKB_masks = torch.zeros(1, n_vert, n_edge, 1, 1).to(device)
        self.eqGKB_masks = torch.zeros(1, n_vert, n_edge, 1, 1).to(device)
        self.minusGKB_masks = torch.zeros(1, n_vert, n_edge, 1, 1).to(device)
        self.plusGH_masks_1 = torch.zeros(1, n_vert, n_edge, 1).to(device)
        self.eqGH_masks_1 = torch.zeros(1, n_vert, n_edge, 1).to(device)
        self.minusGH_masks_1 = torch.zeros(1, n_vert, n_edge, 1).to(device)
        self.plusGH_masks_2 = torch.zeros(1, n_vert, n_edge, 1).to(device)
        self.eqGH_masks_2 = torch.zeros(1, n_vert, n_edge, 1).to(device)
        self.minusGH_masks_2 = torch.zeros(1, n_vert, n_edge, 1).to(device)
        self.plusGH_masks_n = torch.zeros(1, n_vert, n_edge, 1).to(device)
        self.eqGH_masks_n = torch.zeros(1, n_vert, n_edge, 1).to(device)
        self.minusGH_masks_n = torch.zeros(1, n_vert, n_edge, 1).to(device)
        n = n_edge - 1
        for i in range(n_vert):
            if i == 0 or i == n_vert - 1:
                continue
            for k in range(max(i - 1, 1), n_edge):
                j_1 = k - 1
                j_2 = k
                self.w_masks[:, i, k, :] = 1
                if k < i + 2:
                    self.m_masks[:, i, k] = 1
                    if k == i - 1:
                        self.plusGKB_masks[:, i, k] = 1
                    elif k == i:
                        self.eqGKB_masks[:, i, k] = 1
                    elif k == i + 1:
                        self.minusGKB_masks[:, i, k] = 1

                if j_1 >= (i - 1) and i > 1 and (i - 1) < n_edge:
                    self.plusGH_masks_1[:, i, k] = 1
                if j_1 >= i and i < n_edge:
                    self.eqGH_masks_1[:, i, k] = 1
                if j_1 >= (i + 1) and (i + 1) < n_edge:
                    self.minusGH_masks_1[:, i, k] = 1
                if j_2 >= (i - 1) and i > 1 and (i - 1) < n_edge:
                    self.plusGH_masks_2[:, i, k] = 1
                if j_2 >= i and i < n_edge:
                    self.eqGH_masks_2[:, i, k] = 1
                if j_2 >= (i + 1) and (i + 1) < n_edge:
                    self.minusGH_masks_2[:, i, k] = 1

            if n >= (i - 1) and i > 1 and (i - 1) < n_edge:
                self.plusGH_masks_n[:, i, n] = 1
            if n >= i and i < n_edge:
                self.eqGH_masks_n[:, i, n] = 1
            if n >= (i + 1) and (i + 1) < n_edge:
                self.minusGH_masks_n[:, i, n] = 1

        """for inference: """
        self.m_restWprev = 0
        self.m_restWnext = 0
        self.learned_pmass = 0


    def Rod_Init(self, batch, init_direction, m_restEdgeL, clamped_index):
        """Just in case if training and evaluation's batches are different"""
        """prepare edges, length of edges and length of Voronoi region"""
        """if not clone, then refer: https://medium.com/@mrityu.jha/understanding-the-grad-of-autograd-fc8d266fd6cf"""
        rest_vert = self.applyInternalConstraintsIteration((self.rest_vert.clone()).repeat(batch, 1, 1), m_restEdgeL, self.m_pmass, clamped_index)
        m_edges = computeEdges(rest_vert)
        RegionL = torch.cat(((m_restEdgeL[:, 0]/2.).unsqueeze(dim=-1), (m_restEdgeL[:, 1:] + m_restEdgeL[:, :-1])/2., (m_restEdgeL[:, -1]/2.).unsqueeze(dim=-1)), dim=1)
        """Initialize bishop frame at the first edge and compute bishop frame for the rest of edges"""
        m_u0 = self.DEFORM_func.compute_u0(m_edges[:, 0], init_direction[:, 0])
        m_m1, m_m2, m_kb = self.DEFORM_func.computeBishopFrame(m_u0, m_edges, m_restEdgeL)
        """Initialize material curvature with theta=0 (bishop frame) as reference)"""
        m_restWprev, m_restWnext = self.DEFORM_func.computeMaterialCurvature(m_kb, m_m1, m_m2)
        """Initialize mass for each vertices"""
        return m_restWprev, m_restWnext, self.m_pmass.repeat(m_edges.size()[0], 1) * RegionL + torch.clamp(self.mocap_mass, min=1e-10)

    def Internal_Force(self, m_edges, clamped_index, m_restEdgeL, m_restRegionL, m_kb, m_restWprev, m_restWnext, theta_full, m_m1, m_m2):
        """accumulate internal forces, original version"""
        batch = m_kb.size()[0]
        o_forces = torch.zeros(batch, self.n_vert, 3).to(self.device)
        m_theta = theta_full
        """Calculate gradient of the curvature binormal"""
        minusGKB, plusGKB, eqGKB = self.DEFORM_func.computeGradientKB(m_kb, m_edges, m_restEdgeL)
        """Calculate gradient of the holonomy"""
        minusGH, plusGH, eqGH = self.DEFORM_func.computeGradientHolonomyTerms(m_kb, m_restEdgeL)
        """ignore this for now? add after clamping"""
        J = rotation_matrix(torch.pi/2. * torch.ones(batch)).to(self.device)
        n = self.n_edge - 1
        dEdtheta = self.DEFORM_func.computedEdtheta(n, m_kb, m_m1[:, n], m_m2[:, n], m_theta, J * torch.clamp(self.DEFORM_func.bend_stiffness[:, n].unsqueeze(-1), self.DEFORM_func.stiff_threshold), m_restWprev, m_restWnext, m_restRegionL)
        for i in range(self.n_vert):
            if clamped_index[i]:
                continue
            for k in range(max(i - 1, 1), self.n_edge):
                """prepare material frame curvature and its gradient for j = k - 1"""
                b_wkj = computeW(m_kb[:, k].unsqueeze(dim=1), m_m1[:, k - 1].unsqueeze(dim=1), m_m2[:, k - 1].unsqueeze(dim=1))
                # computedEdtheta                """b_w checked"""
                GW, GH = self.DEFORM_func.computeGradientCurvature(i, k, k-1, m_m1, m_m2, minusGKB, plusGKB, eqGKB, minusGH, plusGH, eqGH, b_wkj, J)
                """gw checked"""
                """force for j = k - 1"""
                term = torch.bmm(torch.transpose(GW, 2, 1),  torch.clamp(self.DEFORM_func.bend_stiffness[:, k-1].unsqueeze(-1), self.DEFORM_func.stiff_threshold) * (b_wkj.view(-1, 2) - m_restWprev[:, k]).unsqueeze(dim=2))
                """prepare material frame curvature for j = k"""
                b_wkj = computeW(m_kb[:, k].unsqueeze(dim=1), m_m1[:, k].unsqueeze(dim=1), m_m2[:, k].unsqueeze(dim=1))
                GW, _ = self.DEFORM_func.computeGradientCurvature(i, k, k, m_m1, m_m2, minusGKB, plusGKB, eqGKB, minusGH, plusGH, eqGH, b_wkj, J)
                # test_temsor[:, i, k] = GW.clone()
                """force for j = k"""
                term += torch.bmm(torch.transpose(GW, 2, 1),  torch.clamp(self.DEFORM_func.bend_stiffness[:, k].unsqueeze(-1), self.DEFORM_func.stiff_threshold) * (b_wkj.view(-1, 2) - m_restWnext[:, k]).unsqueeze(dim=2))
                """normalize Voronoi region length"""
                o_forces[:, i] -= term.view(batch, 3) / m_restRegionL[:, k].unsqueeze(dim=1)
            GH = self.DEFORM_func.computeGradientHolonomy(i, n, minusGH, plusGH, eqGH)
            o_forces[:, i] += dEdtheta.unsqueeze(dim=1) * GH
        o_forces = torch.where(torch.norm(o_forces, dim=2).unsqueeze(dim=-1) >= self.force_scale.repeat(1, self.n_vert).unsqueeze(dim=-1), F.normalize(o_forces, dim=2) * self.force_scale.unsqueeze(dim=-1), o_forces)
        return o_forces

    def Internal_Force_Vectorize(self, m_edges, clamped_index, m_restEdgeL, m_restRegionL, m_kb, m_restWprev, m_restWnext, theta_full, m_m1, m_m2):
        """accumulate internal forces"""
        batch = m_kb.size()[0]
        m_theta = theta_full
        """Calculate gradient of the curvature binormal"""
        minusGKB, plusGKB, eqGKB = self.DEFORM_func.computeGradientKB(m_kb, m_edges, m_restEdgeL)
        """Calculate gradient of the holonomy"""
        minusGH, plusGH, eqGH = self.DEFORM_func.computeGradientHolonomyTerms(m_kb, m_restEdgeL)
        """ignore this for now? add after clamping"""
        J = rotation_matrix(torch.pi/2. * torch.ones(batch)).to(self.device)
        n = self.n_edge - 1
        dEdtheta = self.DEFORM_func.computedEdtheta(n, m_kb, m_m1[:, n], m_m2[:, n], m_theta, J * torch.clamp(self.DEFORM_func.bend_stiffness[:, n].unsqueeze(-1), self.DEFORM_func.stiff_threshold), m_restWprev, m_restWnext, m_restRegionL)
        """vectorize w, i-1, i, done"""
        b_w1 = self.w_masks * computeW(m_kb, torch.cat((torch.zeros(batch, 1, 3).to(self.device), m_m1[:, :-1]), dim=1), torch.cat((torch.zeros(batch, 1, 3).to(self.device), m_m2[:, :-1]), dim=1)).unsqueeze(dim=1).repeat(1, self.n_vert, 1, 1)
        b_w2 = self.w_masks * computeW(m_kb, torch.cat((torch.zeros(batch, 1, 3).to(self.device), m_m1[:, 1:]), dim=1), torch.cat((torch.zeros(batch, 1, 3).to(self.device), m_m2[:, 1:]), dim=1)).unsqueeze(dim=1).repeat(1, self.n_vert, 1, 1)
        """computeGradientHolonomy Vectorize"""
        """first edge, GW: first"""
        b_m1 = torch.cat((torch.zeros(batch, self.n_vert, 1, 2, 3).to(self.device), torch.cat((m_m2.unsqueeze(dim=1).unsqueeze(dim=-2).repeat(1, self.n_vert, 1, 1, 1), -m_m1.unsqueeze(dim=1).unsqueeze(dim=-2).repeat(1, self.n_vert, 1, 1, 1)), -2)[:, :, :-1]), dim=2) * self.m_masks
        O_GWplus1 = torch.bmm(b_m1.view(-1, 2, 3), (plusGKB.unsqueeze(dim=1).repeat(1, self.n_vert, 1, 1, 1)).view(-1, 3, 3)).view(batch, self.n_vert, self.n_edge, 2, 3) * self.plusGKB_masks
        O_GWeq1 = torch.bmm(b_m1.view(-1, 2, 3), (eqGKB.unsqueeze(dim=1).repeat(1, self.n_vert, 1, 1, 1)).view(-1, 3, 3)).view(batch, self.n_vert, self.n_edge, 2, 3) * self.eqGKB_masks
        O_GWminus1 = torch.bmm(b_m1.view(-1, 2, 3), (minusGKB.unsqueeze(dim=1).repeat(1, self.n_vert, 1, 1, 1)).view(-1, 3, 3)).view(batch, self.n_vert, self.n_edge, 2, 3) * self.minusGKB_masks
        O_GW1 = O_GWplus1 + O_GWeq1 + O_GWminus1

        """second edge, GW: first"""
        b_m2 = torch.cat((m_m2.unsqueeze(dim=1).unsqueeze(dim=-2).repeat(1, self.n_vert, 1, 1, 1), -m_m1.unsqueeze(dim=1).unsqueeze(dim=-2).repeat(1, self.n_vert, 1, 1, 1)), -2) * self.m_masks
        O_GWplus2 = torch.bmm(b_m2.view(-1, 2, 3), (plusGKB.unsqueeze(dim=1).repeat(1, self.n_vert, 1, 1, 1)).view(-1, 3, 3)).view(batch, self.n_vert, self.n_edge, 2, 3) * self.plusGKB_masks
        O_GWeq2 = torch.bmm(b_m2.view(-1, 2, 3), (eqGKB.unsqueeze(dim=1).repeat(1, self.n_vert, 1, 1, 1)).view(-1, 3, 3)).view(batch, self.n_vert, self.n_edge, 2, 3) * self.eqGKB_masks
        O_GWminus2= torch.bmm(b_m2.view(-1, 2, 3), (minusGKB.unsqueeze(dim=1).repeat(1, self.n_vert, 1, 1, 1)).view(-1, 3, 3)).view(batch, self.n_vert, self.n_edge, 2, 3) * self.minusGKB_masks
        O_GW2 = O_GWplus2 + O_GWeq2 + O_GWminus2

        b_plusGH = torch.cat((torch.zeros(batch, 1, 3).to(self.device), plusGH), dim=1).unsqueeze(-2).repeat(1, 1,self.n_edge,1)
        b_eqGH = torch.cat((eqGH, torch.zeros(batch, 1, 3).to(self.device)), dim=1).unsqueeze(-2).repeat(1, 1,self.n_edge,1)
        b_minusGH = torch.cat((minusGH[:, 1:], torch.zeros(batch, 2, 3).to(self.device)), dim=1).unsqueeze(-2).repeat(1, 1,self.n_edge,1)

        b_GH1 = b_plusGH * self.plusGH_masks_1 + b_eqGH * self.eqGH_masks_1 + b_minusGH * self.minusGH_masks_1
        b_GH2 = b_plusGH * self.plusGH_masks_2 + b_eqGH * self.eqGH_masks_2 + b_minusGH * self.minusGH_masks_2

        b_GHn = (b_plusGH * self.plusGH_masks_n + b_eqGH * self.eqGH_masks_n + b_minusGH * self.minusGH_masks_n)
        """first edge, GW: second"""
        O_GW1 = O_GW1 - torch.bmm((J.unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, self.n_vert, self.n_edge, 1, 1)).view(-1, 2, 2), torch.einsum('bijc,bijd->bijcd', b_w1, b_GH1).view(-1, 2, 3)).view(batch, self.n_vert, self.n_edge, 2, 3)
        """second edge, GW: second"""
        O_GW2 = O_GW2 - torch.bmm((J.unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, self.n_vert, self.n_edge, 1, 1)).view(-1, 2, 2), torch.einsum('bijc,bijd->bijcd', b_w2, b_GH2).view(-1, 2, 3)).view(batch, self.n_vert, self.n_edge, 2, 3)

        """final force accumulation"""
        b_m_restRegionL = m_restRegionL.unsqueeze(dim=1).unsqueeze(dim=-1).repeat(1, self.n_vert, 1, 3) * self.w_masks
        b_bend_stiffness1 = torch.cat((torch.zeros(1, 1).to(self.device), self.DEFORM_func.bend_stiffness[:, :-1]), dim=1).unsqueeze(dim=1).unsqueeze(dim=-1).repeat(1, self.n_vert, 1, 1)
        b_m_restWprev = m_restWprev.unsqueeze(dim=1).repeat(1, self.n_vert, 1, 1) * self.w_masks
        term1 = torch.bmm(torch.transpose(O_GW1.view(-1, 2, 3), 2, 1),  (torch.clamp(b_bend_stiffness1, self.DEFORM_func.stiff_threshold) * (b_w1 - b_m_restWprev)).view(-1, 2, 1)).view(batch, self.n_vert, self.n_edge, 3)
        b_bend_stiffness2 = torch.cat((torch.zeros(1, 1).to(self.device), self.DEFORM_func.bend_stiffness[:, 1:]),dim=1).unsqueeze(dim=1).unsqueeze(dim=-1).repeat(1, self.n_vert, 1, 1)
        b_m_restWnext = m_restWnext.unsqueeze(dim=1).repeat(1, self.n_vert, 1, 1) * self.w_masks
        term2 = torch.bmm(torch.transpose(O_GW2.view(-1, 2, 3), 2, 1), (torch.clamp(b_bend_stiffness2, self.DEFORM_func.stiff_threshold) * (b_w2 - b_m_restWnext)).view(-1, 2, 1)).view(batch, self.n_vert, self.n_edge, 3)
        o_forces = torch.div(-(term1 + term2), b_m_restRegionL.where(b_m_restRegionL != 0, torch.tensor(1.).to(self.device)))
        o_forces[b_m_restRegionL == 0] = 0.
        o_forces = torch.sum(o_forces, -2)
        o_forces += b_GHn[:, :, -1] * dEdtheta.unsqueeze(dim=1).unsqueeze(dim=1)
        o_forces = torch.where(torch.norm(o_forces, dim=2).unsqueeze(dim=-1) >= self.force_scale.repeat(1, self.n_vert).unsqueeze(dim=-1), F.normalize(o_forces, dim=2) * self.force_scale.unsqueeze(dim=-1), o_forces)

        """set force = 0 for clamped part, introduce prescription later"""
        o_forces = o_forces * (1 - clamped_index.to(self.device)).unsqueeze(dim=0).unsqueeze(dim=-1)
        return o_forces

    def External_Force(self, m_pmass, m_pvel, o_forces, clamped_index):
        """accumulate external forces: gravity/velocity compensation"""
        batch = m_pmass.size()[0]
        """gravity"""
        o_forces = o_forces + self.gravity.view(1, 1, 3).repeat(batch, self.n_vert, 1) * m_pmass.unsqueeze(dim=2) * (1 - (clamped_index.unsqueeze(dim=0).unsqueeze(dim=-1)).to(self.device))
        """velocity compensation"""
        o_forces = o_forces + (self.velocity_ratio * torch.linalg.norm(m_pvel, dim=2).unsqueeze(dim=2)) * m_pvel * (1 - (clamped_index.unsqueeze(dim=0).unsqueeze(dim=-1)).to(self.device))
        return o_forces

    def Integrate_Centerline(self, vertices, v_vertices, forces, m_pmass):
        """semi-implicit Euler"""
        v_vertices = v_vertices + forces/m_pmass.unsqueeze(dim=2) * self.dt
        vertices = vertices + v_vertices * self.dt * self.integration_ratio
        return vertices, v_vertices * self.dt * self.integration_ratio

    """Inextensibility Enforcement"""
    def applyInternalConstraintsIteration(self, updated_vertices, m_restEdgeL, m_pmass, clamped_index, iterative_times=10, mode="pytorch"):
        """

        :param updated_vertices:
        :param m_restEdgeL:
        :param m_pmass:
        :param clamped_index:
        :param iterative_times:
        :param mode:
        :return:
        """
        if mode == "pytorch":
            for _ in range(iterative_times):
                for i in range(self.n_vert - 1):
                    # i = self.n_vert - 1 - i
                    updated_edges = updated_vertices[:, i + 1] - updated_vertices[:, i]
                    l = 1 - 2 * m_restEdgeL[:, i] * m_restEdgeL[:, i] / (m_restEdgeL[:, i] * m_restEdgeL[:, i] + (updated_edges * updated_edges).sum(dim=1))
                    if clamped_index[i]:
                        l1 = torch.zeros(1).to(self.device)
                        l2 = -l
                    elif clamped_index[i + 1]:
                        l1 = l
                        l2 = torch.zeros(1).to(self.device)
                    else:
                        l1 = l * m_pmass[:, i + 1] / (m_pmass[:, i] + m_pmass[:, i + 1])
                        l2 = -l * m_pmass[:, i] / (m_pmass[:, i] + m_pmass[:, i + 1])
                    updated_vertices[:, i] = updated_vertices[:, i] + l1.unsqueeze(dim=1) * updated_edges
                    updated_vertices[:, i + 1] = updated_vertices[:, i + 1] + l2.unsqueeze(dim=1) * updated_edges
            return updated_vertices

        if mode == "numpy":
            batch = updated_vertices.size()[0]
            updated_vertices = updated_vertices[0].cpu().numpy()
            m_restEdgeL = m_restEdgeL[0].cpu().numpy()
            m_pmass = m_pmass[0].cpu().numpy()
            clamped_index = clamped_index.cpu().numpy()
            m_restEdgeL_square = m_restEdgeL * m_restEdgeL
            for _ in range(iterative_times):
                for i in range(self.n_vert - 1):
                    updated_edges = updated_vertices[i + 1] - updated_vertices[i]
                    l = 1 - 2 * m_restEdgeL_square[i] / (m_restEdgeL_square[i] + (updated_edges * updated_edges).sum())
                    if clamped_index[i]:
                        updated_vertices[i + 1] = updated_vertices[i + 1] - l * updated_edges
                    elif clamped_index[i + 1]:
                        updated_vertices[i] = updated_vertices[i] + l * updated_edges
                    else:
                        updated_vertices[i] = updated_vertices[i] + l * m_pmass[i + 1] / (m_pmass[i] + m_pmass[i + 1]) * updated_edges
                        updated_vertices[i + 1] = updated_vertices[i + 1] - l * m_pmass[i] / (m_pmass[i] + m_pmass[i + 1]) * updated_edges
            updated_vertices = torch.from_numpy(updated_vertices).unsqueeze(0).repeat(batch, 1, 1).to(self.device)
            test_m_restEdgeL, test_m_restRegionL = computeLengths(computeEdges(updated_vertices.clone()))
            return updated_vertices

    def forward(self, current_vert, current_v, init_direction, clamped_index, m_u0, input, clamped_selection, theta_full, mode="train"):
        learning_weight = 2.
        if mode == "train":
            previous_vert = current_vert.clone()
            current_x = current_vert.clone()
            batch = current_vert.size()[0]
            m_restEdgeL, m_restRegionL = self.m_restEdgeL.repeat(batch, 1), self.m_restRegionL.repeat(batch, 1)
            m_restWprev, m_restWnext, m_pmass = self.Rod_Init(batch, init_direction, m_restEdgeL, clamped_index)
            control_theta = torch.zeros(batch, 2, 1).to(self.device)
            current_edges = computeEdges(current_vert)
            theta_full, material_m1, material_m2, m_kb = self.DEFORM_func.updateCurrentState(current_vert, m_u0, m_restEdgeL, m_restWprev, m_restWnext, m_restRegionL, control_theta, theta_full)

            Internal_force = self.Internal_Force_Vectorize(current_edges, clamped_index, m_restEdgeL, m_restRegionL, m_kb, m_restWprev, m_restWnext, theta_full, material_m1, material_m2)
            Total_force = self.External_Force(m_pmass, current_v.clone(), Internal_force, clamped_index)
            """update"""
            current_vert, delta_vert = self.Integrate_Centerline(current_vert, current_v, Total_force, m_pmass)
            current_vert[:, clamped_selection] = input

            """learning"""
            edge_index = torch.combinations(torch.arange(self.n_vert), r=2).t().contiguous().to(self.device)
            # edges = []
            #
            # for i in range(self.n_vert):
            #     # Connect to the next three nodes if possible
            #     for j in range(1, 4):
            #         if i + j < self.n_vert:  # Check if the target node index is within bounds
            #             edges.append([i, i + j])
            #             edges.append([i + j, i])

            # edge_index = torch.transpose(torch.tensor(edges, dtype=torch.long), dim0=0, dim1=1).to(self.device)
            # print(edge_index)
            graph_data = Data(x=current_x, edge_index=edge_index)
            graph_delta_data = Data(x=delta_vert, edge_index=edge_index)
            x, edge_index = graph_data.x, graph_data.edge_index
            delta_x, edge_index = graph_delta_data.x, graph_delta_data.edge_index
            # print(delta_x.size())
            x = self.vert_conv1(x, edge_index)
            delta_x = self.delta_vert_conv1(delta_x, edge_index)

            encoded_x = self.vert_conv2(x, edge_index)
            # print(encoded_x.size())
            delta_x = self.delta_vert_conv2(delta_x, edge_index)

            input = (input - current_x[:, (0, 1, -2, -1)])
            encoded_x_input = torch.cat((encoded_x.view(current_x.size()[0], -1), delta_x.view(current_x.size()[0], -1), input.view(current_x.size()[0], -1)), dim=-1)

            x_dot = self.fc(encoded_x_input).view(current_x.size()[0], self.n_vert-4, 3)
            """"""

            """residual"""
            current_vert[:, 2:-2] = current_vert[:, 2:-2] + x_dot * self.dt * learning_weight
            current_vert = self.applyInternalConstraintsIteration(current_vert, m_restEdgeL, m_pmass, clamped_index)
            # current_vert[:, 2:-2] = current_vert[:, 2:-2] + x_dot * self.dt * learning_weight
            # current_vert = self.non_linear_opt_edge(torch.cat((current_vert[:, 0].unsqueeze(dim=1), current_vert[:, -1].unsqueeze(dim=1)), dim=1), current_vert[:, 1:-1], m_restEdgeL).view(-1, self.n_vert, 3)
            current_v = (current_vert - previous_vert) / self.dt
            return current_vert, current_v, theta_full

        if mode == "evaluation":
            previous_vert = current_vert.clone()
            current_x = current_vert.clone()
            current_edges = computeEdges(current_vert)
            batch = current_vert.size()[0]
            m_restEdgeL, m_restRegionL = self.m_restEdgeL.repeat(batch, 1), self.m_restRegionL.repeat(batch, 1)
            control_theta = torch.zeros(batch, 2, 1).to(self.device)
            theta_full, material_m1, material_m2, m_kb = self.DEFORM_func.updateCurrentState(current_vert, m_u0, m_restEdgeL, self.m_restWprev, self.m_restWnext, m_restRegionL, control_theta, theta_full)
            Internal_force = self.Internal_Force_Vectorize(current_edges, clamped_index, m_restEdgeL, m_restRegionL, m_kb, self.m_restWprev, self.m_restWnext, theta_full, material_m1, material_m2)
            Total_force = self.External_Force(self.learned_pmass, current_v.clone(), Internal_force, clamped_index)
            """update"""
            current_vert, delta_vert = self.Integrate_Centerline(current_vert, current_v, Total_force, self.learned_pmass)
            current_vert[:, clamped_selection] = input

            """learning"""
            edge_index = torch.combinations(torch.arange(self.n_vert), r=2).t().contiguous().to(self.device)
            # edges = []
            #
            # for i in range(self.n_vert):
            #     # Connect to the next three nodes if possible
            #     for j in range(1, 4):
            #         if i + j < self.n_vert:  # Check if the target node index is within bounds
            #             edges.append([i, i + j])
            #             edges.append([i + j, i])
            #
            # edge_index = torch.transpose(torch.tensor(edges, dtype=torch.long), dim0=0, dim1=1)

            graph_data = Data(x=current_x, edge_index=edge_index)
            graph_delta_data = Data(x=delta_vert, edge_index=edge_index)

            x, edge_index = graph_data.x, graph_data.edge_index
            delta_x, edge_index = graph_delta_data.x, graph_delta_data.edge_index

            x = self.vert_conv1(x, edge_index)
            delta_x = self.delta_vert_conv1(delta_x, edge_index)

            encoded_x = self.vert_conv2(x, edge_index)
            delta_x = self.delta_vert_conv2(delta_x, edge_index)

            input = (input - current_x[:, (0, 1, -2, -1)])
            encoded_x_input = torch.cat((encoded_x.view(current_x.size()[0], -1), delta_x.view(current_x.size()[0], -1), input.view(current_x.size()[0], -1)), dim=-1)

            x_dot = self.fc(encoded_x_input).view(current_x.size()[0], self.n_vert - 4, 3)
            """"""

            """residual"""
            # current_vert[:, 2:-2] = current_vert[:, 2:-2] + x_dot * self.dt * learning_weight
            current_vert[:, 2:-2] = current_vert[:, 2:-2] + x_dot * self.dt * learning_weight
            current_vert = self.applyInternalConstraintsIteration(current_vert, m_restEdgeL, self.learned_pmass, clamped_index)
            # current_vert[:, 2:-2] = current_vert[:, 2:-2] + x_dot * self.dt * learning_weight
            # current_vert = self.non_linear_opt_edge(torch.cat((current_vert[:, 0].unsqueeze(dim=1), current_vert[:, -1].unsqueeze(dim=1)), dim=1), current_vert[:, 1:-1], m_restEdgeL).view(-1, self.n_vert, 3).view(-1, self.n_vert, 3)
            current_v = (current_vert - previous_vert) / self.dt
            return current_vert, current_v, theta_full

        if mode == "evaluation_numpy":
            previous_vert = current_vert.clone()
            current_x = current_vert.clone()
            current_edges = computeEdges(current_vert)
            batch = current_vert.size()[0]
            m_restEdgeL, m_restRegionL = self.m_restEdgeL.repeat(batch, 1), self.m_restRegionL.repeat(batch, 1)
            control_theta = torch.zeros(batch, 2, 1).to(self.device)
            theta_full, material_m1, material_m2, m_kb = self.DEFORM_func.updateCurrentState_numpy(current_vert, m_u0, m_restEdgeL, self.m_restWprev, self.m_restWnext, m_restRegionL, control_theta, theta_full)
            Internal_force = self.Internal_Force_Vectorize(current_edges, clamped_index, m_restEdgeL, m_restRegionL, m_kb, self.m_restWprev, self.m_restWnext, theta_full, material_m1, material_m2)
            Total_force = self.External_Force(self.learned_pmass, current_v.clone(), Internal_force, clamped_index)
            """update"""
            current_vert, delta_vert = self.Integrate_Centerline(current_vert, current_v, Total_force, self.learned_pmass)
            current_vert[:, clamped_selection] = input

            """learning"""
            edge_index = torch.combinations(torch.arange(self.n_vert), r=2).t().contiguous().to(self.device)

            # edges = []
            #
            # for i in range(self.n_vert):
            #     # Connect to the next three nodes if possible
            #     for j in range(1, 4):
            #         if i + j < self.n_vert:  # Check if the target node index is within bounds
            #             edges.append([i, i + j])
            #             edges.append([i + j, i])
            #
            # edge_index = torch.transpose(torch.tensor(edges, dtype=torch.long), dim0=0, dim1=1)

            graph_data = Data(x=current_x, edge_index=edge_index)
            graph_delta_data = Data(x=delta_vert, edge_index=edge_index)

            x, edge_index = graph_data.x, graph_data.edge_index
            delta_x, edge_index = graph_delta_data.x, graph_delta_data.edge_index

            x = self.vert_conv1(x, edge_index)
            delta_x = self.delta_vert_conv1(delta_x, edge_index)

            encoded_x = self.vert_conv2(x, edge_index)
            delta_x = self.delta_vert_conv2(delta_x, edge_index)

            input = (input - current_x[:, (0, 1, -2, -1)])
            encoded_x_input = torch.cat((encoded_x.view(current_x.size()[0], -1), delta_x.view(current_x.size()[0], -1), input.view(current_x.size()[0], -1)), dim=-1)

            x_dot = self.fc(encoded_x_input).view(current_x.size()[0], self.n_vert - 4, 3)
            """"""

            """residual"""
            current_vert[:, 2:-2] = current_vert[:, 2:-2] + x_dot * self.dt * learning_weight
            current_vert = self.applyInternalConstraintsIteration(current_vert, m_restEdgeL, self.learned_pmass, clamped_index, mode="numpy")
            # current_vert[:, 2:-2] = current_vert[:, 2:-2] + x_dot * self.dt * learning_weight
            # current_vert = self.non_linear_opt_edge(torch.cat((current_vert[:, 0].unsqueeze(dim=1), current_vert[:, -1].unsqueeze(dim=1)), dim=1), current_vert[:, 1:-1], m_restEdgeL).view(-1, self.n_vert, 3)
            current_v = (current_vert - previous_vert) / self.dt
            return current_vert, current_v, theta_full
