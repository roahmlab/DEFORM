import torch
import torch.nn as nn
import torch.nn.functional as F
from util import extractSinandCos, computeEdges, computeKB, quaternion_q, computeW, skew_symmetric, scalar_func, quaternion_rotation, quaternion_rotation_parallel, computeKB_numpy, extractSinandCos_numpy, quaternion_rotation_numpy, computeW_numpy
import theseus as th
import numpy as np

class DEFORM_func(nn.Module):
    def __init__(self, n_vert, n_edge, device):
        super().__init__()
        """
        Parameters:
        n_vert: number of vertices
        n_edge: number of edges 
        device: cpu/cuda
        bend_stiffness: bending learnable material properties
        twist_stiffness: twisting learnable material properties
        err: error threshold
        stiff_threshold: lower threshold of stiffness
        bend_stiffness_last_unsq/bend_stiffness_next_unsq: for batch-wise operation
        """
        self.n_vert = n_vert
        self.n_edge = n_edge
        self.device = device
        self.bend_stiffness = nn.Parameter(5e-5 * torch.ones((1, n_edge), device=device))
        self.twist_stiffness = nn.Parameter(1e-5 * torch.ones((1, n_edge), device=device))
        self.err = torch.tensor(1e-6).to(device)
        self.stiff_threshold = torch.tensor(1e-10).to(device)
        self.bend_stiffness_last_unsq = self.bend_stiffness[:, -1].unsqueeze(dim=-1)
        self.bend_stiffness_next_unsq = self.bend_stiffness[:, 1:].unsqueeze(dim=-1)

    def compute_u0(self, e0, init_direct):
        """
        Initialize first edge's trigonometric bishop frame

        Parameters:
        e0: first edge
        init_direct: a randomly chosen 3D vector (not aligned with e0)

        return:
        m_u0: first edge's bishop frame's u axis that is perpendicular to e0
        """
        batch = e0.size()[0]
        N_0 = torch.cross(e0, init_direct.view(batch, -1))
        u0 = F.normalize(torch.cross(N_0, e0), dim=1)
        return u0

    def computeBishopFrame(self, u0, edges, restEdgeL):
        """
        compute the rest of bishop frame as reference frame based on the first bishop reference frame
        Bishop: twist-minimized reference frame. I am editing a document to clarify its derivation.

        Parameters:
        e0: first edge
        edges: edge vectors of wire segmentation
        restEdgeL: edge length of undeformed wire

        returns:
        b_u: Bishop frame u
        b_v: Bishop frame v
        kb: discrete curvature binormal
        """
        batch = edges.size()[0]
        kb = computeKB(edges, restEdgeL) # DER paper: discrete curvature binormal: DER paper eq 1
        b_u = u0.unsqueeze(dim=1)
        b_v = F.normalize(torch.cross(edges[:, 0], b_u[:, 0], dim=1)).unsqueeze(dim=1)
        magnitude = (kb * kb).sum(dim=2)
        sinPhi, cosPhi = extractSinandCos(magnitude)
        q = quaternion_q(cosPhi, sinPhi.unsqueeze(dim=2) * F.normalize(kb, dim=2))
        for i in range(1, self.n_edge):
            b_u = torch.cat((b_u, torch.where(torch.ones(batch, 1).to(self.device) - cosPhi[:, i].unsqueeze(dim=1) <= self.err * torch.ones(batch, 1).to(self.device), b_u[:, i - 1], quaternion_rotation(b_u, edges, q, i)[0][:, 0, :]).unsqueeze(dim=1)), dim=1)
            b_v = torch.cat((b_v, torch.where(torch.ones(batch, 1).to(self.device) - cosPhi[:, i].unsqueeze(dim=1) <= self.err * torch.ones(batch, 1).to(self.device), b_v[:, i - 1], quaternion_rotation(b_u, edges, q, i)[1][:, 0, :]).unsqueeze(dim=1)), dim=1)
        return b_u, b_v, kb

    def computeMaterialCurvature(self, kb, m1, m2):
        """
        Compute material curvature in material frame. DER paper eq 2

        Parameters:
        kb: discrete curvature binormal
        m1, m2: material frame

        returns: material curvature
        """
        batch, n_edge = kb.size()[0], kb.size()[1]
        m_W1 = torch.zeros(batch, n_edge, 2).to(self.device)
        m_W2 = torch.zeros(batch, n_edge, 2).to(self.device)

        m_W1[:, 1:] = computeW(kb[:, 1:], m1[:, :-1], m2[:, :-1])
        m_W2[:, 1:] = computeW(kb[:, 1:], m1[:, 1:], m2[:, 1:])
        return m_W1, m_W2

    def parallelTransportFrame(self, previous_e0, current_e0, b_u0):
        """
        update first edge's bishop frame after one time update to assure orthonormality of bishop frame triad: DER paper: section 4.2.2 Bishop Frame Update

        Parameters:
        previous_e0: previous first edge vector
        current_e0: current first edge vector
        b_u0: previous bishop frame u

        returns: current first edge's bishop frame u
        """

        batch = previous_e0.size()[0]
        axis = 2 * torch.cross(previous_e0, current_e0, dim=1) / (previous_e0.norm(dim=1) * current_e0.norm(dim=1) + (previous_e0 * current_e0).sum(dim=1)).unsqueeze(dim=1)
        magnitude = (axis * axis).sum(dim=1)
        sinPhi, cosPhi = extractSinandCos(magnitude)
        b_u0 = torch.where(torch.ones(batch, 1).to(self.device) - cosPhi.unsqueeze(dim=1) <= self.err * torch.ones(batch, 1).to(self.device), F.normalize(torch.cross(torch.cross(current_e0, b_u0, dim=1), current_e0), dim=1), quaternion_rotation_parallel(cosPhi, sinPhi, axis, b_u0))
        return b_u0

    def computeMaterialFrame(self, m_theta, b_u, b_v):
        """
        Material Frame Construction: relative rotation with magnitude of m_theta rotation from Bishop Frame. DER paper section 4.1

        Parameters:
        m_theta: scalar function that measures the rotation about the tangent of the material frame relative to the Bishop frame
        b_u: bishop frame u
        b_v: bishop frame v

        return m1 m2: material frame
        """
        cosQ = torch.cos(m_theta).unsqueeze(dim=2)
        sinQ = torch.sin(m_theta).unsqueeze(dim=2)
        m1 = cosQ * b_u + sinQ * b_v
        m2 = -sinQ * b_u + cosQ * b_v
        return m1, m2

    def non_linear_opt_theta_full(self, kb, b_u, b_v, m_restW1, m_restW2, restRegionL, end_theta, inner_theta):
        """
        minimize elastic energy (twisting + bending) for calculating inner theta: DER paper eq.4
        with theseus non-linear optimizer. The reason to use theseus is because of its differentiablity of
        calculating non-linear optimization

        Parameters:
        kb: discrete curvature binormal
        b_u b_v: material frame
        m_restW1 m_restW2: undeformed material curvature
        restRegionL: Voronoi region length. DER paper 4.2
        end_theta: prescribed twisting theta
        inner_theta: twisting theta as decision variables

        return: theta asterisk
        """

        """setup"""
        batch = kb.size()[0]
        objective = th.Objective()

        kb_Variable = th.Variable(kb, name="kb")
        b_u_Variable = th.Variable(b_u, name="b_u")
        b_v_Variable = th.Variable(b_v, name="b_v")
        m_restW1_Variable = th.Variable(m_restW1, name="m_restW1")
        m_restW2_Variable = th.Variable(m_restW2, name="m_restW2")
        restRegionL_Variable = th.Variable(restRegionL, name="restRegionL")
        Energy_target_Variable = th.Variable(torch.zeros((batch, 1), device=self.device), name="E_target") #target energy is 0
        theta_inner_Variable = th.Vector(tensor=inner_theta, name="theta")
        theta_control_Variable = th.Variable(end_theta, name="controlled_theta")

        cost_function = th.AutoDiffCostFunction(
            [theta_inner_Variable], self.err_function, Energy_target_Variable.tensor.shape[1],
            aux_vars=[kb_Variable, b_u_Variable, b_v_Variable, m_restW1_Variable, m_restW2_Variable,
                      restRegionL_Variable, Energy_target_Variable, theta_control_Variable],
        )

        objective.add(cost_function)

        optimizer = th.LevenbergMarquardt(
            objective,
            linear_solver_cls=th.CholeskyDenseSolver,
            linearization_cls=th.DenseLinearization,
            linear_solver_kwargs={'check_singular': False},
            vectorize=True,
            max_iterations=100,
            step_size=0.05,
            abs_err_tolerance=1e-6,
            rel_err_tolerance=1e-10,
            adaptive_damping=True,
        )

        non_linear_opt_layer_init = th.TheseusLayer(optimizer)
        non_linear_opt_layer_init.to(self.device)
        """execute optimization"""
        solution, info = non_linear_opt_layer_init.forward(
            input_tensors={"kb": kb.clone(), "b_u": b_u.clone(), "b_v": b_v.clone(), "m_restW1": m_restW1.clone(),
                           "m_restW2": m_restW2.clone(), "restRegionL": restRegionL.clone(),
                           "theta": inner_theta.clone(), "controlled_theta": end_theta.clone()},
        )

        return torch.concatenate((end_theta[:, 0], solution["theta"], end_theta[:, -1]), dim=1)

    def Model_Energy(self, kb, b_u, b_v, m_restW1, m_restW2, restRegionL, inner_theta, end_theta):
        """
        calculate elastic energy of DLO for optimization with decision variable inner_theta. Boundary twisting condition is prescribed with end_theta
        b_u, b_v: bishop frame u & v
        m_restW1 m_restW2: undeformed material curvature
        restRegionL: Voronoi region length. DER paper 4.2
        inner_theta: decision variable to minimize elastic energy
        end_theta: boundary twisting angle

        return: summation of elastic energy
        """
        theta_full = torch.concatenate((end_theta[:, 0], inner_theta, end_theta[:, -1]), dim=1)
        m1, m2 = self.computeMaterialFrame(theta_full, b_u, b_v) # compute material frame
        restRegionL_unsq = restRegionL[:, 1:].unsqueeze(dim=-1)

        """project discrete curvature binormal onto material frame"""
        o_W = computeW(kb[:, 1:], m1[:, 1:], m2[:, 1:])

        """difference between current material curvature bi-normal and undeformed curvature bi-normal"""
        diff_prev = o_W - m_restW1[:, 1:]
        diff_next = o_W - m_restW2[:, 1:]
        # print(diff_prev[0])
        # print(diff_next[0])
        """bending energy"""
        O_E = (0.5 * (torch.clamp(self.bend_stiffness_last_unsq, self.stiff_threshold) * diff_prev * diff_prev + torch.clamp(self.bend_stiffness_next_unsq, self.stiff_threshold) * diff_next * diff_next) / restRegionL_unsq).sum(dim=(1, 2))

        """twisting energy"""
        m = theta_full[:, 1:] - theta_full[:, :-1]
        O_E += 0.5 * (torch.clamp(self.twist_stiffness[:, 1:], self.stiff_threshold) * m * m / restRegionL[:, 1:]).sum(dim=1)
        return O_E.unsqueeze(dim=1)

    def err_function(self, optim_vars, aux_vars):
        """
        object function for elastic energy minimization for theseus: https://arxiv.org/pdf/2207.09442

        return: elastic energy
        """
        m_kb, m1, m2, m_restWprev, m_restWnext, restRegionL, Energy_target, end_theta = aux_vars
        return self.Model_Energy(m_kb.tensor, m1.tensor, m2.tensor, m_restWprev.tensor, m_restWnext.tensor, restRegionL.tensor, optim_vars[0].tensor, end_theta.tensor) - Energy_target.tensor

    def updateCurrentState(self, current_vert, b_u0, restEdgeL, m_restW1, m_restW2, restRegionL, end_theta, theta_full):
        """
        compute material frame based on the asterisk theta (minimize the summation of elastic energy)

        Parameters:
        current_vert: current vertices
        b_u0: current first edge' Bishop frame's u
        restEdgeL: undeformed edge lengths
        m_restW1 m_restW2: undeformed material curvature
        restRegionL: Voronoi region length. DER paper 4.2
        end_theta: prescribed twisting theta
        inner_theta: twisting theta as decision variables

        returns:
        theta_full: theta asterisk
        m1, m2: material frame m1, m2
        kb: discrete curvature binormal

        todos:
        what if the angle is larger than 2pi?
        """

        current_edges = computeEdges(current_vert)
        b_u, b_v, kb = self.computeBishopFrame(b_u0, current_edges, restEdgeL)
        theta_full = self.non_linear_opt_theta_full(kb, b_u, b_v, m_restW1, m_restW2, restRegionL, end_theta, theta_full[:, 1:-1])
        m1, m2 = self.computeMaterialFrame(theta_full, b_u, b_v)
        return theta_full, m1, m2, kb

    def computeGradientKB(self, kb, edges, restEdgeL):
        """
        calculate gradient of the curvature binormal

        Parameters:
        kb: discrete curvature binormal
        edges: edges of vertices
        restEdgeL: undeformed length

        return: gradient of curvature binormal
        """
        batch = edges.size()[0]
        n_egde = edges.size()[1]
        o_minusGKB = torch.zeros(batch, n_egde, 3, 3).to(self.device)
        o_plusGKB = torch.zeros(batch, n_egde, 3, 3).to(self.device)

        edgeMatrix = skew_symmetric(edges)
        scalar_factor = scalar_func(edges, restEdgeL).view(batch, n_egde - 1, 1, 1)

        o_minusGKB[:, 1:] = (2 * edgeMatrix[:, :-1] + torch.einsum('bki,bkj->bkij', kb[:, 1:], edges[:, :-1]))/scalar_factor
        o_plusGKB[:, 1:] = (2 * edgeMatrix[:, 1:] - torch.einsum('bki,bkj->bkij', kb[:, 1:], edges[:, 1:]))/scalar_factor
        o_eqGKB = -(o_minusGKB + o_plusGKB)
        return o_minusGKB, o_plusGKB, o_eqGKB

    def computeGradientHolonomyTerms(self, kb, restEdgeL):
        """
        holonomy gradient computation: DER eq.9

        Parameters:
        kb: discrete curvature binormal
        restEdgeL: undeformed length

        return: gradient holonomy terms
        """
        batch = kb.size()[0]
        n_egde = kb.size()[1]
        o_minusGH = torch.zeros(batch, n_egde, 3).to(self.device)
        o_plusGH = torch.zeros(batch, n_egde, 3).to(self.device)
        o_minusGH[:, 1:] = 0.5 * kb[:, 1:]/restEdgeL[:, :-1].unsqueeze(dim=2)
        o_plusGH[:, 1:] = -0.5 * kb[:, 1:]/restEdgeL[:, 1:].unsqueeze(dim=2)
        o_eqGH = -(o_minusGH + o_plusGH)
        return o_minusGH, o_plusGH, o_eqGH

    def computeGradientHolonomy(self, i, j, minusGH, plusGH, eqGH):
        """
        distribute gradient of holonomy according based on i and j. for DER eq: 11

        Parameters:
        i, j = n
        minusGH, plusGH, eqGH: gradient of holonomy from computeGradientHolonomyTerms

        return gradient of holonomy
        """

        batch = minusGH.size()[0]
        o_Gh = torch.zeros(batch, 3).to(self.device)
        if j >= (i - 1) and i > 1 and (i - 1) < plusGH.size()[1]:
            o_Gh += plusGH[:, i - 1]

        if j >= i and i < eqGH.size()[1]:
            o_Gh += eqGH[:, i]

        if j >= (i + 1) and (i + 1) < minusGH.size()[1]:
            o_Gh += minusGH[:, i + 1]

        return o_Gh

    def computeGradientCurvature(self, i, k, j, m_m1, m_m2, minusGKB, plusGKB, eqGKB, minusGH, plusGH, eqGH, wkj, J):
        """
        compute gradient of material frame curvature and gradient of holonomy

        Parameters:
        i, k, j from DER equation 11.
        rest parameters from previous functions follow notations of eq.11

        return: gradient of material frame curvature and gradient of holonomy
        """

        batch = minusGH.size()[0]
        o_GW = torch.zeros(batch, 2, 3).to(self.device)
        if k < i + 2:
            o_GW[:, 0, 0] = m_m2[:, j, 0]
            o_GW[:, 0, 1] = m_m2[:, j, 1]
            o_GW[:, 0, 2] = m_m2[:, j, 2]

            o_GW[:, 1, 0] = -m_m1[:, j, 0]
            o_GW[:, 1, 1] = -m_m1[:, j, 1]
            o_GW[:, 1, 2] = -m_m1[:, j, 2]
            if k == (i - 1):
                o_GW = torch.bmm(o_GW, plusGKB[:, k])

            elif k == i:
                o_GW = torch.bmm(o_GW, eqGKB[:, k])

            elif k == i + 1:
                o_GW = torch.bmm(o_GW, minusGKB[:, k])

        GH = self.computeGradientHolonomy(i, j, minusGH, plusGH, eqGH)
        o_GW -= torch.bmm(J, torch.einsum('bi,bj->bij', wkj.view(batch, 2), GH))
        return o_GW, GH

    def computedEdtheta(self, j, m_kb, m1j, m2j, theta, JB, m_restWprev, m_restWnext, restRegionL):
        """
        compute gradient of energy with respect to theta: in this case it is just dE/dtheta^n
        parameters: correspondingly DER paper section 7.1. General case parameters

        return dEdtheta
        """
        dEdtheta = 0
        if j > 0:
            wij = computeW(m_kb[:, j].unsqueeze(dim=1), m1j.unsqueeze(dim=1), m2j.unsqueeze(dim=1)).view(-1, 2)
            term = (wij * torch.bmm(JB,  (wij - m_restWnext[:, j]).unsqueeze(dim=2)).view(-1, 2)).sum(dim=1)
            term += 2 * torch.clamp(self.twist_stiffness[:, j], self.stiff_threshold) * (theta[:, j] - theta[:, j - 1])
            term /= restRegionL[:, j]
            dEdtheta += term

        if j < self.n_edge - 1:
            wij = computeW(m_kb[:, j+1].unsqueeze(dim=1), m1j.unsqueeze(dim=1), m2j.unsqueeze(dim=1)).view(-1, 2)
            term = (wij * torch.bmm(JB,  (wij - m_restWprev[:, j+1]).unsqueeze(dim=2)).view(-1, 2)).sum(dim=1)
            term -= 2 * torch.clamp(self.twist_stiffness[:, j+1], self.stiff_threshold) * (theta[:, j+1] - theta[:, j])
            term /= restRegionL[:, j + 1]
            dEdtheta += term
        return dEdtheta

    def updateCurrentState_numpy(self, current_vert, m_u0, restEdgeL, m_restWprev, m_restWnext, restRegionL, end_theta, theta_full):
        "python version of updateCurrentState"
        current_edges = computeEdges(current_vert)
        m_m1, m_m2, m_kb = self.computeBishopFrame_numpy(m_u0, current_edges, restEdgeL)
        theta_full = self.non_linear_opt_theta_full(m_kb, m_m1, m_m2, m_restWprev, m_restWnext, restRegionL, end_theta, theta_full[:, 1:-1])
        material_m1, material_m2 = self.computeMaterialFrame(theta_full, m_m1, m_m2)
        return theta_full, material_m1, material_m2, m_kb

    def computeBishopFrame_numpy(self, u0, edges, restEdgeL):
        "python version of computeBishopFrame"
        batch = edges.size()[0]
        u0 = u0[0].cpu().numpy()
        edges = edges[0].cpu().numpy()
        restEdgeL = restEdgeL[0].cpu().numpy()
        o_kb = computeKB_numpy(edges, restEdgeL)
        o_u = np.expand_dims(u0, axis=0)
        o_v_norm = np.linalg.norm(np.cross(edges[0], o_u[0]))
        o_v = np.expand_dims(np.cross(edges[0], o_u[0])/o_v_norm, axis=0)
        magnitude = (o_kb * o_kb).sum(axis=1)
        sinPhi, cosPhi = extractSinandCos_numpy(magnitude)
        o_kb_norm = np.expand_dims(np.linalg.norm(o_kb, axis=1), axis=1)
        q = np.concatenate((np.expand_dims(sinPhi, axis=1) * np.where(o_kb_norm !=0, o_kb/o_kb_norm, o_kb), np.expand_dims(cosPhi, axis=1)), axis=1)
        for i in range(1, self.n_edge):
            uv = quaternion_rotation_numpy(o_u, edges, q, i)
            o_u = np.concatenate((o_u, np.where(1 - np.expand_dims(cosPhi[i], axis=0) <= 1e-6, o_u[i - 1], uv[0])), axis=0)
            o_v = np.concatenate((o_v, np.where(1 - np.expand_dims(cosPhi[i], axis=0) <= 1e-6, o_v[i - 1], uv[1])), axis=0)

        """output dim = batch x (n_vert - 1) = n_edge x 3 for both"""
        return torch.from_numpy(o_u).unsqueeze(dim=0).repeat(batch, 1, 1).float(), torch.from_numpy(o_v).unsqueeze(dim=0).repeat(batch, 1, 1).float(), torch.from_numpy(o_kb).unsqueeze(dim=0).repeat(batch, 1, 1).float()


