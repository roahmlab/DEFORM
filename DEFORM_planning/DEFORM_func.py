import torch
import torch.nn as nn
import torch.nn.functional as F
from util import extractSinandCos, computeEdges, computeKB, quaternion_q, computeW, skew_symmetric, scalar_func, quaternion_rotation, quaternion_rotation_parallel, computeKB_numpy, extractSinandCos_numpy, quaternion_rotation_numpy, computeW_numpy
import theseus as th
import numpy as np

class DEFORM_func(nn.Module):
    def __init__(self, n_vert, n_edge, device):
        super().__init__()
        self.n_vert = n_vert
        self.n_edge = n_edge
        self.device = device
        self.force_scale = torch.tensor((5.)).to(device)
        self.bend_stiffness = nn.Parameter(5e-5 * torch.ones((1, n_edge), device=device))
        self.twist_stiffness = nn.Parameter(1e-5 * torch.ones((1, n_edge), device=device))
        self.err = torch.tensor(1e-6).to(device)
        self.stiff_threshold = torch.tensor(1e-10).to(device)
        self.bend_stiffness_last_unsq = self.bend_stiffness[:, -1].unsqueeze(dim=-1)
        self.bend_stiffness_next_unsq = self.bend_stiffness[:, 1:].unsqueeze(dim=-1)
        """for minimization energy"""

    def compute_m_u0(self, e0, init_direct):

        """Initialize first  trigonometric bishop frame"""
        batch = e0.size()[0]
        N_0 = torch.cross(e0, init_direct.view(batch, -1))
        m_u0 = F.normalize(torch.cross(N_0, e0), dim=1)
        """output dim = batch x 3 for first edge"""
        return m_u0

    # @torch.compile()
    def computeBishopFrame(self, u0, edges, restEdgeL):
        """compute the rest of bishop frame as reference frame based on the first bishop reference frame"""
        batch = edges.size()[0]
        o_kb = computeKB(edges, restEdgeL)
        o_u = u0.unsqueeze(dim=1)
        o_v = F.normalize(torch.cross(edges[:, 0], o_u[:, 0], dim=1)).unsqueeze(dim=1)
        magnitude = (o_kb * o_kb).sum(dim=2)
        sinPhi, cosPhi = extractSinandCos(magnitude)
        q = quaternion_q(cosPhi, sinPhi.unsqueeze(dim=2) * F.normalize(o_kb, dim=2))
        for i in range(1, self.n_edge):
            o_u = torch.cat((o_u, torch.where(torch.ones(batch, 1).to(self.device) - cosPhi[:, i].unsqueeze(dim=1) <= self.err * torch.ones(batch, 1).to(self.device), o_u[:, i - 1], quaternion_rotation(o_u, edges, q, i)[0][:, 0, :]).unsqueeze(dim=1)), dim=1)
            o_v = torch.cat((o_v, torch.where(torch.ones(batch, 1).to(self.device) - cosPhi[:, i].unsqueeze(dim=1) <= self.err * torch.ones(batch, 1).to(self.device), o_v[:, i - 1], quaternion_rotation(o_u, edges, q, i)[1][:, 0, :]).unsqueeze(dim=1)), dim=1)
        """output dim = batch x (n_vert - 1) = n_edge x 3 for both"""
        return o_u, o_v, o_kb

    # @torch.compile()
    def computeMaterialCurvature(self, kb, m1, m2):
        """input dim = batch x (n_vert - 1) = n_edge x 3 for both m1 and m2"""
        """compute the center20line curvature vector expressed in the material frame coordinates"""
        """transfer m1 & m2 (from bishop) with kb"""

        batch, n_edge = kb.size()[0], kb.size()[1]
        o_Wprev = torch.zeros(batch, n_edge, 2).to(self.device)
        o_Wnext = torch.zeros(batch, n_edge, 2).to(self.device)

        o_Wprev[:, 1:] = computeW(kb[:, 1:], m1[:, :-1], m2[:, :-1])
        o_Wnext[:, 1:] = computeW(kb[:, 1:], m1[:, 1:], m2[:, 1:])
        """output dim = batch x (n_vert - 1) = n_edge x 1 for both"""
        return o_Wprev, o_Wnext

    def parallelTransportFrame(self, e0, e1, io_u):
        """compute the rest of parallelTransportFrame: related to holonomy, not fully understand"""
        batch = e0.size()[0]
        axis = 2 * torch.cross(e0, e1, dim=1) / (e0.norm(dim=1) * e1.norm(dim=1) + (e0 * e1).sum(dim=1)).unsqueeze(dim=1)
        magnitude = (axis * axis).sum(dim=1)
        sinPhi, cosPhi = extractSinandCos(magnitude)
        io_u = torch.where(torch.ones(batch, 1).to(self.device) - cosPhi.unsqueeze(dim=1) <= self.err * torch.ones(batch, 1).to(self.device), F.normalize(torch.cross(torch.cross(e1, io_u, dim=1), e1), dim=1), quaternion_rotation_parallel(cosPhi, sinPhi, axis, io_u))
        return io_u

    def computeMaterialFrame(self, m_theta, io_m1, io_m2):
        """compute the material frame based on theta (as a function to calculate)"""
        cosQ = torch.cos(m_theta).unsqueeze(dim=2)
        sinQ = torch.sin(m_theta).unsqueeze(dim=2)
        io_m1 = cosQ * io_m1 + sinQ * io_m2
        io_m2 = -sinQ * io_m1 + cosQ * io_m2
        return io_m1, io_m2

    def non_linear_opt_theta_full(self, m_kb, m1, m2, m_restWprev, m_restWnext, restRegionL, end_theta, inner_theta):
        batch = m_kb.size()[0]
        objective = th.Objective()

        m_kb_Variable = th.Variable(m_kb, name="m_kb")
        m1_Variable = th.Variable(m1, name="m1")
        m2_Variable = th.Variable(m2, name="m2")
        m_restWprev_Variable = th.Variable(m_restWprev, name="m_restWprev")
        m_restWnext_Variable = th.Variable(m_restWnext, name="m_restWnext")
        restRegionL_Variable = th.Variable(restRegionL, name="restRegionL")
        Energy_target_Variable = th.Variable(torch.zeros((batch, 1), device=self.device), name="E_target")
        theta_inner_Variable = th.Vector(tensor=inner_theta, name="theta")
        theta_control_Variable = th.Variable(end_theta, name="controlled_theta")

        cost_function = th.AutoDiffCostFunction(
            [theta_inner_Variable], self.err_function, Energy_target_Variable.tensor.shape[1],
            aux_vars=[m_kb_Variable, m1_Variable, m2_Variable, m_restWprev_Variable, m_restWnext_Variable,
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
        solution, info = non_linear_opt_layer_init.forward(
            input_tensors={"m_kb": m_kb.clone(), "m1": m1.clone(), "m2": m2.clone(), "m_restWprev": m_restWprev.clone(),
                           "m_restWnext": m_restWnext.clone(), "restRegionL": restRegionL.clone(),
                           "theta": inner_theta.clone(), "controlled_theta": end_theta.clone()},
        )

        return torch.concatenate((end_theta[:, 0], solution["theta"], end_theta[:, -1]), dim=1)

    def updateCurrentState(self, current_vert, m_u0, clamped_index, restEdgeL, m_restWprev, m_restWnext, restRegionL, end_theta, theta_full):
        """compute the material frame based on energy optimized m_theta"""
        """clamping problem!!!"""
        current_edges = computeEdges(current_vert)
        """problem with optimization"""
        """m_m1, m_m2 will be precalculated"""
        m_m1, m_m2, m_kb = self.computeBishopFrame(m_u0, current_edges, restEdgeL)
        theta_full = self.non_linear_opt_theta_full(m_kb, m_m1, m_m2, m_restWprev, m_restWnext, restRegionL, end_theta, theta_full[:, 1:-1])
        """what if the angle is larger than 2pi?"""
        """what if the angle is larger than 2pi?"""
        material_m1, material_m2 = self.computeMaterialFrame(theta_full, m_m1, m_m2)
        return theta_full, material_m1, material_m2, m_kb

    def Model_Energy(self, m_kb, m1, m2, m_restWprev, m_restWnext, restRegionL, inner_theta, end_theta):
        """compute the energy of the material frame to minimize energy for
        rotation about the tangent of the material frame relative to the Bishop frame"""
        """bending energy"""
        theta_full = torch.concatenate((end_theta[:, 0], inner_theta, end_theta[:, -1]), dim=1)
        m1, m2 = self.computeMaterialFrame(theta_full, m1, m2)
        # Pre-calculate repeated terms
        restRegionL_unsq = restRegionL[:, 1:].unsqueeze(dim=-1)

        o_W = computeW(m_kb[:, 1:], m1[:, 1:], m2[:, 1:])
        # First part of O_E calculation
        diff_prev = o_W - m_restWprev[:, 1:]
        diff_next = o_W - m_restWnext[:, 1:]
        O_E = (0.5 * (torch.clamp(self.bend_stiffness_last_unsq, self.stiff_threshold) * diff_prev * diff_prev + torch.clamp(self.bend_stiffness_next_unsq, self.stiff_threshold) * diff_next * diff_next) / restRegionL_unsq).sum(dim=(1, 2))

        # Twisting energy
        m = theta_full[:, 1:] - theta_full[:, :-1]
        O_E += 0.5 * (torch.clamp(self.twist_stiffness[:, 1:], self.stiff_threshold) * m * m / restRegionL[:, 1:]).sum(dim=1)
        return O_E.unsqueeze(dim=1)

    def err_function(self, optim_vars, aux_vars):
        m_kb, m1, m2, m_restWprev, m_restWnext, restRegionL, Energy_target, end_theta = aux_vars
        return self.Model_Energy(m_kb.tensor, m1.tensor, m2.tensor, m_restWprev.tensor, m_restWnext.tensor, restRegionL.tensor, optim_vars[0].tensor, end_theta.tensor) - Energy_target.tensor

    def computeGradientKB(self, kb, edges, m_restEdgeL):
        batch = edges.size()[0]
        n_egde = edges.size()[1]
        o_minusGKB = torch.zeros(batch, n_egde, 3, 3).to(self.device)
        o_plusGKB = torch.zeros(batch, n_egde, 3, 3).to(self.device)

        edgeMatrix = skew_symmetric(edges)
        scalar_factor = scalar_func(edges, m_restEdgeL).view(batch, n_egde - 1, 1, 1)
        # assert torch.isfinite(scalar_factor.all()), "infinite problem: computeGradientKB"
        # assert not torch.isnan(scalar_factor).any(), "nan problem: computeGradientKB"

        o_minusGKB[:, 1:] = (2 * edgeMatrix[:, :-1] + torch.einsum('bki,bkj->bkij', kb[:, 1:], edges[:, :-1]))/scalar_factor
        o_plusGKB[:, 1:] = (2 * edgeMatrix[:, 1:] - torch.einsum('bki,bkj->bkij', kb[:, 1:], edges[:, 1:]))/scalar_factor
        o_eqGKB = -(o_minusGKB + o_plusGKB)
        return o_minusGKB, o_plusGKB, o_eqGKB

    def computeGradientHolonomyTerms(self, kb, restEdgeL):
        batch = kb.size()[0]
        n_egde = kb.size()[1]
        o_minusGH = torch.zeros(batch, n_egde, 3).to(self.device)
        o_plusGH = torch.zeros(batch, n_egde, 3).to(self.device)
        o_minusGH[:, 1:] = 0.5 * kb[:, 1:]/restEdgeL[:, :-1].unsqueeze(dim=2)
        o_plusGH[:, 1:] = -0.5 * kb[:, 1:]/restEdgeL[:, 1:].unsqueeze(dim=2)
        o_eqGH = -(o_minusGH + o_plusGH)
        return o_minusGH, o_plusGH, o_eqGH

    def computeGradientHolonomy(self, i, j, minusGH, plusGH, eqGH):
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

    def computedEdQj(self, j, m_kb, m1j, m2j, theta, JB, m_restWprev, m_restWnext, restRegionL):
        o_dEQj = 0

        if j > 0:
            wij = computeW(m_kb[:, j].unsqueeze(dim=1), m1j.unsqueeze(dim=1), m2j.unsqueeze(dim=1)).view(-1, 2)
            term = (wij * torch.bmm(JB,  (wij - m_restWnext[:, j]).unsqueeze(dim=2)).view(-1, 2)).sum(dim=1)
            term += 2 * torch.clamp(self.twist_stiffness[:, j], self.stiff_threshold) * (theta[:, j] - theta[:, j - 1])
            term /= restRegionL[:, j]
            o_dEQj += term

        if j < self.n_edge - 1:
            wij = computeW(m_kb[:, j+1].unsqueeze(dim=1), m1j.unsqueeze(dim=1), m2j.unsqueeze(dim=1)).view(-1, 2)
            term = (wij * torch.bmm(JB,  (wij - m_restWprev[:, j+1]).unsqueeze(dim=2)).view(-1, 2)).sum(dim=1)
            term -= 2 * torch.clamp(self.twist_stiffness[:, j+1], self.stiff_threshold) * (theta[:, j+1] - theta[:, j])
            term /= restRegionL[:, j + 1]
            o_dEQj += term
        return o_dEQj

    def updateCurrentState_numpy(self, current_vert, m_u0, clamped_index, restEdgeL, m_restWprev, m_restWnext, restRegionL, end_theta, theta_full):
        """compute the material frame based on energy optimized m_theta"""
        """clamping problem!!!"""
        current_edges = computeEdges(current_vert)
        """problem with optimization"""
        """m_m1, m_m2 will be precalculated"""
        m_m1, m_m2, m_kb = self.computeBishopFrame_numpy(m_u0, current_edges, restEdgeL)
        # test_m_kb = m_kb[0].cpu().numpy()
        # test_m_m1 = m_m1[0].cpu().numpy()
        # test_m_m2 = m_m2[0].cpu().numpy()
        # test_m_restWprev = m_restWprev[0].cpu().numpy()
        # test_m_restWnext = m_restWnext[0].cpu().numpy()
        # test_restRegionL = restRegionL[0].cpu().numpy()
        # test_theta_full = theta_full[0, 1:-1].cpu().numpy()
        # test_end_theta = end_theta[0].cpu().numpy()
        # start = time.time()
        # solution = opt.minimize(lambda theta_full: self.Model_Energy_numpy(test_m_kb, test_m_m1, test_m_m2, test_m_restWprev, test_m_restWnext, test_restRegionL, theta_full, test_end_theta), test_theta_full)
        # print(time.time()-start)
        # print("scipy")
        # print(solution.x
        theta_full = self.non_linear_opt_theta_full(m_kb, m_m1, m_m2, m_restWprev, m_restWnext, restRegionL, end_theta, theta_full[:, 1:-1])
        """what if the angle is larger than 2pi?"""
        """what if the angle is larger than 2pi?"""
        material_m1, material_m2 = self.computeMaterialFrame(theta_full, m_m1, m_m2)
        return theta_full, material_m1, material_m2, m_kb

    def computeBishopFrame_numpy(self, u0, edges, restEdgeL):
        """compute the rest of bishop frame as reference frame based on the first bishop reference frame"""
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


