import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.transform import Rotation as R

def computeEdges(vertices):
    # edge-link vectors
    # vertices dim = batch x n_vert x 3
    edges = vertices[:, 1:] - vertices[:, :-1]
    # assert torch.isfinite(edges.all()), "infinite problem: computeEdges"
    # assert not torch.isnan(edges).any(), "nan problem: computeEdges"
    # output dim = batch x (n_vert - 1) = n_edge x 3
    return edges

def computeLengths(edges):
    # input dim = batch x (n_vert - 1) = n_edge x 3
    # compute the length of each link: EdgeL
    # compute the sum of two adjacent links: RegionL, used in energy computation
    batch = edges.size()[0]
    EdgeL = torch.norm(edges, dim=2)
    RegionL = torch.zeros(batch, 1, device=edges.device)
    RegionL = torch.cat((RegionL, (EdgeL[:, 1:] + EdgeL[:, :-1])), dim=1)
    # output dim = batch x (n_vert - 1) = n_edge x 1 for botha
    return EdgeL, RegionL

def computeLength_only(vertices):
    EdgeL = torch.norm(vertices[:, 1:] - vertices[:, :-1], dim=2)
    return EdgeL

def sqrt_safe(value):
    # safe square root for rotation angle calculation
    return torch.sqrt(torch.clamp(value, 1e-10))

def sqrt_safe_numpy(value):
    # safe square root for rotation angle calculation
    return np.sqrt(np.clip(value, a_min=1e-10, a_max=1e10))
def extractSinandCos(magnitude):
    # extract phi: the turning angle between two consecutive edges
    constant = 4.0
    o_sinPhi = sqrt_safe(magnitude/(constant+magnitude))
    o_cosPhi = sqrt_safe(constant/(constant+magnitude))
    return o_sinPhi, o_cosPhi

def extractSinandCos_numpy(magnitude):
    # extract phi: the turning angle between two consecutive edges
    constant = 4.0
    o_sinPhi = sqrt_safe_numpy(magnitude/(constant+magnitude))
    o_cosPhi = sqrt_safe_numpy(constant/(constant+magnitude))
    return o_sinPhi, o_cosPhi

def computeKB(edges, m_restEdgeL):
    # discrete curvature binormal at a vertex (an integrated quantity)
    o_kb = torch.zeros_like(edges)
    o_kb[:, 1:] = torch.clamp(2 * torch.cross(edges[:, :-1], edges[:, 1:], dim=2) / (m_restEdgeL[:, :-1] * m_restEdgeL[:, 1:] + (edges[:, :-1] * edges[:, 1:]).sum(dim=2)).unsqueeze(dim=2), min=-20, max=20)
    return o_kb

def computeKB_numpy(edges, m_restEdgeL):
    # discrete curvature binormal at a vertex (an integrated quantity)
    o_kb = np.zeros_like(edges)
    o_kb[1:] = np.clip(2 * np.cross(edges[:-1], edges[1:], axis=1) / np.expand_dims(m_restEdgeL[:-1] * m_restEdgeL[1:] + (edges[:-1] * edges[1:]).sum(axis=1), axis=1), a_min=-20, a_max=20)
    return o_kb


def quaternion_q(theta, kb):
    # form quaternion coordinates for rotation
    # output dim = batch x selected_edge x 3
    return torch.cat((theta.unsqueeze(dim=2), kb), dim=2)

def quaternion_p(theta, kb):
    return torch.cat((theta, kb), dim=1)


def computeW(kb, m1, m2):
    o_wij = torch.cat(((kb * m2).sum(dim=2).unsqueeze(dim=2), -(kb * m1).sum(dim=2).unsqueeze(dim=2)), dim=2)
    return o_wij

def computeW_numpy(kb, m1, m2):
    o_wij = np.concatenate((np.expand_dims((kb * m2).sum(axis=1), axis=-1), np.expand_dims((-(kb * m1).sum(axis=1)), axis=-1)), axis=1)
    return o_wij
def skew_symmetric(edges):
    """Create a batch of skew-symmetric matrices given a batch of vectors v."""
    batch = edges.size()[0]
    n_edges = edges.size()[1]
    matrix = torch.zeros(batch, n_edges, 3, 3, dtype=edges.dtype, device=edges.device)
    matrix[:, :, 0, 1] = -edges[:, :, 2]
    matrix[:, :, 0, 2] = edges[:, :, 1]
    matrix[:, :, 1, 0] = edges[:, :, 2]
    matrix[:, :, 1, 2] = -edges[:, :, 0]
    matrix[:, :, 2, 0] = -edges[:, :, 1]
    matrix[:, :, 2, 1] = edges[:, :, 0]
    return matrix

def scalar_func(edges, restEdgeL):
    return restEdgeL[:, :-1] * restEdgeL[:, 1:] + (edges[:, :-1] * edges[:, 1:]).sum(dim=2)

def rotation_matrix(theta):
    batch = theta.size()[0]
    rot_sin = torch.sin(theta)
    rot_cos = torch.cos(theta)
    transform_basis = torch.zeros(batch, 2, 2)
    transform_basis[:, 0, 0] = rot_cos
    transform_basis[:, 0, 1] = -rot_sin
    transform_basis[:, 1, 0] = rot_sin
    transform_basis[:, 1, 1] = rot_cos
    return transform_basis

def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions.
    q1 and q2 are tensors of shape (batch_size, 4).
    """
    w1 = q1[:, 0]
    x1 = q1[:, 1]
    y1 = q1[:, 2]
    z1 = q1[:, 3]
    w2 = q2[:, 0]
    x2 = q2[:, 1]
    y2 = q2[:, 2]
    z2 = q2[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack((w, x, y, z), -1)

def quaternion_conjugate(q):
    """
    Compute the conjugate of a quaternion.
    q is a tensor of shape (..., 4).
    """
    q_conj = q.clone()  # make a copy of the input tensor
    q_conj[..., 1:] *= -1  # negate the vector part of the quaternion
    return q_conj

def quaternion_rotation(o_u, edges, q, i):
    batch = o_u.size()[0]
    p = quaternion_p(torch.zeros(batch, 1).to(o_u.device), o_u[:, i - 1])
    quat_p = quaternion_multiply(quaternion_multiply(q[:, i], p), quaternion_conjugate(q[:, i]))
    u = F.normalize(quat_p[:, 1:4], dim=1)
    v = F.normalize(torch.cross(edges[:, i], u), dim=1)
    return u.unsqueeze(dim=1), v.unsqueeze(dim=1)

def quaternion_rotation_numpy(o_u, edges, q, i):
    rotation = R.from_quat(q[i])
    quat_p = rotation.apply(o_u[i-1])
    u = quat_p/np.linalg.norm(quat_p)
    v = np.cross(edges[i], u)/np.linalg.norm(np.cross(edges[i], u))
    return np.expand_dims(u, axis=0), np.expand_dims(v, axis=0)
def quaternion_rotation_parallel(cosPhi, sinPhi, axis, io_u):
    batch = cosPhi.size()[0]
    q = quaternion_p(cosPhi.view(-1, 1), sinPhi.view(-1, 1) * F.normalize(axis, dim=1))
    p = quaternion_p(torch.zeros(batch, 1).to(io_u.device), io_u)
    quat_p = quaternion_multiply(quaternion_multiply(q, p), quaternion_conjugate(q))
    io_u = F.normalize(quat_p[:, 1:4], dim=1)
    return io_u

def compute_m_u0(e0, init_direct):

    """Initialize first  trigonometric bishop frame"""
    batch = e0.size()[0]
    N_0 = torch.cross(e0, init_direct.view(batch, -1))
    m_u0 = F.normalize(torch.cross(N_0, e0), dim=1)
    """output dim = batch x 3 for first edge"""
    return m_u0

def parallelTransportFrame(e0, e1, io_u):
    """compute the rest of parallelTransportFrame: related to holonomy, not fully understand"""
    batch = e0.size()[0]
    err = torch.tensor(1e-6).to(io_u.device)
    axis = 2 * torch.cross(e0, e1, dim=1) / (e0.norm(dim=1) * e1.norm(dim=1) + (e0 * e1).sum(dim=1)).unsqueeze(dim=1)
    magnitude = (axis * axis).sum(dim=1)
    sinPhi, cosPhi = extractSinandCos(magnitude)
    io_u = torch.where(torch.ones(batch, 1).to(io_u.device) - cosPhi.unsqueeze(dim=1) <= err * torch.ones(batch, 1).to(io_u.device), F.normalize(torch.cross(torch.cross(e1, io_u, dim=1), e1), dim=1), quaternion_rotation_parallel(cosPhi, sinPhi, axis, io_u))
    return io_u
