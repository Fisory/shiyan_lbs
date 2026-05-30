#!/usr/bin/env python3
"""
实验：SMPL 线性混合蒙皮（LBS）
手写完整 LBS 流程，逐阶段可视化，并与官方前向结果比对。
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import smplx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio.v2 as imageio

os.makedirs('outputs', exist_ok=True)

# 把 SMPL_NEUTRAL.pkl 放在 models/ 下
SMPL_MODEL_PATH = 'models/SMPL_NEUTRAL.pkl'
DEVICE = torch.device('cpu')

# 固定随机种子，保证各阶段使用同一组子采样面片
RNG = np.random.default_rng(42)

# =========================================================
# 手写 LBS 工具函数
# =========================================================

def batch_rodrigues(theta: torch.Tensor) -> torch.Tensor:
    """
    轴角 -> 旋转矩阵（Rodrigues 公式）
    theta: (N, 3) -> (N, 3, 3)
    """
    angle = torch.norm(theta + 1e-8, p=2, dim=1, keepdim=True)   # (N, 1)
    nv    = theta / angle                                           # 单位旋转轴
    c     = torch.cos(angle).unsqueeze(-1)                         # (N, 1, 1)
    s     = torch.sin(angle).unsqueeze(-1)

    x, y, z = nv[:, 0], nv[:, 1], nv[:, 2]
    zeros   = torch.zeros_like(x)
    # 反对称矩阵 K
    K = torch.stack([zeros, -z, y,
                     z,  zeros, -x,
                     -y,  x,  zeros], dim=1).view(-1, 3, 3)
    I = torch.eye(3, device=theta.device).unsqueeze(0)
    return I + s * K + (1.0 - c) * torch.bmm(K, K)


def blend_shapes(betas: torch.Tensor, shapedirs: torch.Tensor) -> torch.Tensor:
    """
    形状混合：将 beta 系数加权到各形状主方向上
    betas:     (B, n_betas)
    shapedirs: (V, 3, n_betas)
    -> (B, V, 3)
    """
    return torch.einsum('bl,mkl->bmk', betas, shapedirs)


def vertices2joints(J_regressor: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    """
    稀疏线性回归：从顶点坐标估计关节位置
    J_regressor: (J, V)
    vertices:    (B, V, 3)
    -> (B, J, 3)
    """
    return torch.einsum('jv,bvd->bjd', J_regressor, vertices)


def _transform_mat(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    拼装 4×4 齐次变换矩阵
    R: (N, 3, 3)  t: (N, 3, 1) -> (N, 4, 4)
    """
    return torch.cat([
        F.pad(R, [0, 0, 0, 1]),           # (N, 4, 3)
        F.pad(t, [0, 0, 0, 1], value=1),  # (N, 4, 1)
    ], dim=2)


def batch_rigid_transform(rot_mats: torch.Tensor,
                           joints: torch.Tensor,
                           parents: torch.Tensor):
    """
    沿运动学树累乘局部变换，得到各关节的全局蒙皮矩阵。

    rot_mats: (B, J, 3, 3)  每个关节的局部旋转
    joints:   (B, J, 3)     rest-pose 下的关节世界坐标
    parents:  (J,)          父关节索引，根节点为 -1

    Returns:
        J_transformed: (B, J, 3)    运动后的关节世界坐标
        A:             (B, J, 4, 4) 相对 rest-pose 的蒙皮变换
    """
    B, J = rot_mats.shape[:2]
    joints_4 = joints.unsqueeze(-1)          # (B, J, 3, 1)

    # 各关节相对父关节的偏移（在父坐标系下）
    rel_j = joints_4.clone()
    rel_j[:, 1:] -= joints_4[:, parents[1:].long()]

    # 局部 4×4 变换
    local_T = _transform_mat(
        rot_mats.view(-1, 3, 3),
        rel_j.view(-1, 3, 1)
    ).view(B, J, 4, 4)

    # 沿运动学树累乘
    chain = [local_T[:, 0]]
    for i in range(1, J):
        chain.append(torch.bmm(chain[parents[i].item()], local_T[:, i]))
    global_T = torch.stack(chain, dim=1)     # (B, J, 4, 4)

    J_transformed = global_T[:, :, :3, 3]   # 变换后的关节位置

    # A_k = G_k * [[I, -j_k^0], [0, 1]]
    # 效果：在 rest-pose 下 A_k 恰好为恒等变换
    joints_h = F.pad(joints_4, [0, 0, 0, 1])  # (B, J, 4, 1)
    A = global_T - F.pad(
        torch.matmul(global_T, joints_h),
        [3, 0, 0, 0, 0, 0, 0, 0]
    )
    return J_transformed, A


def lbs_forward(betas, pose, v_template, shapedirs, posedirs,
                J_regressor, parents, lbs_weights):
    """
    手写完整 LBS 前向传播。

    pose:     (B, J*3) 轴角
    posedirs: (P, V*3) 其中 P = (J-1)*9，smplx 已做转置

    Returns:
        verts, J_transformed, v_shaped, v_posed, J_rest
    """
    B      = betas.shape[0]
    device = betas.device

    # ---------- 1. 形状校正 ----------
    v_shaped = v_template.unsqueeze(0) + blend_shapes(betas, shapedirs)   # (B, V, 3)

    # ---------- 2. 关节回归 ----------
    J_rest = vertices2joints(J_regressor, v_shaped)                       # (B, K, 3)

    # ---------- 3. 姿态校正 ----------
    n_j      = pose.shape[1] // 3
    rot_mats = batch_rodrigues(pose.view(-1, 3)).view(B, n_j, 3, 3)
    ident    = torch.eye(3, device=device)
    # 只取非根关节的旋转矩阵残差
    pose_feature = (rot_mats[:, 1:] - ident).view(B, -1)                 # (B, P)
    pose_offsets = torch.matmul(pose_feature, posedirs).view(B, -1, 3)   # (B, V, 3)
    v_posed = v_shaped + pose_offsets

    # ---------- 4. LBS ----------
    J_transformed, A = batch_rigid_transform(rot_mats, J_rest, parents)  # A: (B, K, 4, 4)

    K  = A.shape[1]
    W  = lbs_weights.unsqueeze(0).expand(B, -1, -1)                      # (B, V, K)
    T  = torch.matmul(W, A.view(B, K, 16)).view(B, -1, 4, 4)            # (B, V, 4, 4)

    ones   = torch.ones(B, v_posed.shape[1], 1, device=device)
    v_homo = torch.cat([v_posed, ones], dim=-1).unsqueeze(-1)            # (B, V, 4, 1)
    verts  = torch.matmul(T, v_homo).squeeze(-1)[:, :, :3]              # (B, V, 3)

    return verts, J_transformed, v_shaped, v_posed, J_rest


# =========================================================
# 渲染工具
# =========================================================

def _make_poly3d(verts_np, faces_subset, facecolors, alpha=0.88):
    tri = verts_np[faces_subset]
    return Poly3DCollection(tri, alpha=alpha, facecolor=facecolors, edgecolor='none', zsort='average')


def render_mesh(verts_np, faces_np, face_idx,
                facecolors=None, joint_positions=None,
                title="", figsize=(5, 7), elev=10, azim=65):
    """
    matplotlib 3D 渲染网格，返回 (H, W, 3) uint8 数组。
    face_idx: 预选的面片子集索引（固定随机种子，保证各图一致）
    """
    if facecolors is None:
        facecolors = np.full((len(face_idx), 3), [0.72, 0.72, 0.82])

    fig = plt.figure(figsize=figsize, dpi=110)
    ax  = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(_make_poly3d(verts_np, faces_np[face_idx], facecolors))

    pad = 0.06
    ax.set_xlim(verts_np[:, 0].min() - pad, verts_np[:, 0].max() + pad)
    ax.set_ylim(verts_np[:, 1].min() - pad, verts_np[:, 1].max() + pad)
    ax.set_zlim(verts_np[:, 2].min() - pad, verts_np[:, 2].max() + pad)

    if joint_positions is not None:
        jp = np.array(joint_positions)
        ax.scatter(jp[:, 0], jp[:, 1], jp[:, 2],
                   c='#FF4444', s=28, zorder=6, depthshade=False)

    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_axis_off()
    fig.tight_layout(pad=0.4)
    fig.canvas.draw()
    buf  = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img  = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img


def vertex_color_by_scalar(scalar_np, cmap='hot'):
    """顶点标量 -> 面片 facecolors（取三顶点均值后映射）"""
    return scalar_np  # 延迟到 render_mesh 调用处做 per-face 映射


def face_colors_from_vertex(scalar_np, faces_subset, cmap='hot'):
    fc_val = np.mean(scalar_np[faces_subset], axis=1)
    return plt.get_cmap(cmap)(fc_val)[:, :3]


# =========================================================
# 主程序
# =========================================================

def main():
    # ------------------------------------------------
    # 加载模型，打印基础信息
    # ------------------------------------------------
    print("=" * 50)
    print("加载 SMPL 模型 ...")
    model = smplx.create(
        SMPL_MODEL_PATH,
        model_type='smpl',
        gender='neutral',
        batch_size=1
    ).to(DEVICE)

    v_template  = model.v_template.float()
    shapedirs   = model.shapedirs.float()
    posedirs    = model.posedirs.float()
    J_regressor = model.J_regressor.float()
    lbs_weights = model.lbs_weights.float()
    parents     = model.parents.long()
    faces_np    = model.faces.astype(np.int64)

    n_verts  = v_template.shape[0]
    n_faces  = len(faces_np)
    n_joints = J_regressor.shape[0]
    n_betas  = shapedirs.shape[2]

    print(f"  顶点数:      {n_verts}")
    print(f"  面片数:      {n_faces}")
    print(f"  关节数:      {n_joints}")
    print(f"  betas 维度:  {n_betas}")

    # 固定子采样面片（加快渲染，全部面片约 14k，matplotlib 3D 渲染会很慢）
    N_DRAW = min(7000, n_faces)
    face_idx = RNG.choice(n_faces, N_DRAW, replace=False)

    verts_t = v_template.cpu().numpy()

    # ================================================
    # 阶段 A：模板网格 + 蒙皮权重热力图
    # ================================================
    print("\n[A] 模板网格与蒙皮权重 ...")

    JOINT_VIZ = 18  # 左手腕
    weight_map = lbs_weights[:, JOINT_VIZ].cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 7),
                              subplot_kw={'projection': '3d'})

    # 左图：模板网格（无着色）
    ax = axes[0]
    fc_default = np.full((N_DRAW, 3), [0.72, 0.72, 0.82])
    ax.add_collection3d(_make_poly3d(verts_t, faces_np[face_idx], fc_default))
    ax.set_xlim(verts_t[:, 0].min(), verts_t[:, 0].max())
    ax.set_ylim(verts_t[:, 1].min(), verts_t[:, 1].max())
    ax.set_zlim(verts_t[:, 2].min(), verts_t[:, 2].max())
    ax.view_init(elev=10, azim=65); ax.set_axis_off()
    ax.set_title(r'Template Mesh $\bar{T}$  (T-pose)', fontsize=12)

    # 右图：关节 JOINT_VIZ 的权重热力图
    ax = axes[1]
    fc_w = face_colors_from_vertex(weight_map, faces_np[face_idx], 'hot')
    ax.add_collection3d(_make_poly3d(verts_t, faces_np[face_idx], fc_w))
    ax.set_xlim(verts_t[:, 0].min(), verts_t[:, 0].max())
    ax.set_ylim(verts_t[:, 1].min(), verts_t[:, 1].max())
    ax.set_zlim(verts_t[:, 2].min(), verts_t[:, 2].max())
    ax.view_init(elev=10, azim=65); ax.set_axis_off()
    ax.set_title(f'Joint {JOINT_VIZ} (L-Wrist) Skinning Weights', fontsize=12)

    sm = plt.cm.ScalarMappable(cmap='hot', norm=plt.Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, ax=axes[1], shrink=0.45, aspect=20, label='weight')
    fig.suptitle('(a) Template Mesh & Skinning Weights', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig('outputs/stage_a_template_weights.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> outputs/stage_a_template_weights.png")

    # 可选：全关节主导权重分布图
    dom_joint  = lbs_weights.cpu().numpy().argmax(axis=1)  # 每个顶点主导关节
    dom_weight = lbs_weights.cpu().numpy().max(axis=1)     # 主导权重值

    cmap_j = plt.get_cmap('tab20', n_joints)
    dom_j_face  = dom_joint[faces_np[face_idx]].mean(axis=1).round().astype(int)
    dom_w_face  = dom_weight[faces_np[face_idx]].mean(axis=1)
    fc_dom = cmap_j(dom_j_face)[:, :3] * dom_w_face[:, None] + \
             (1.0 - dom_w_face[:, None]) * 0.45

    fig = plt.figure(figsize=(6, 8))
    ax  = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(_make_poly3d(verts_t, faces_np[face_idx], fc_dom))
    ax.set_xlim(verts_t[:, 0].min(), verts_t[:, 0].max())
    ax.set_ylim(verts_t[:, 1].min(), verts_t[:, 1].max())
    ax.set_zlim(verts_t[:, 2].min(), verts_t[:, 2].max())
    ax.view_init(elev=10, azim=65); ax.set_axis_off()
    ax.set_title('All-Joint Dominant Weight Distribution', fontsize=12)
    plt.tight_layout()
    plt.savefig('outputs/all_joint_weights.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> outputs/all_joint_weights.png")

    # ================================================
    # 阶段 B：形状校正 + 关节回归
    # ================================================
    print("\n[B] 形状校正与关节回归 ...")

    betas = torch.zeros(1, n_betas, device=DEVICE)
    betas[0, 0] =  2.0   # 体型偏胖
    betas[0, 1] = -1.5   # 身高偏高
    betas[0, 3] =  1.0   # 肩宽微调

    v_shaped_b = v_template.unsqueeze(0) + blend_shapes(betas, shapedirs)
    J_b        = vertices2joints(J_regressor, v_shaped_b)

    vs_np = v_shaped_b[0].cpu().numpy()
    jb_np = J_b[0].cpu().numpy()

    fc_b = np.full((N_DRAW, 3), [0.65, 0.78, 0.92])
    fig  = plt.figure(figsize=(5, 7))
    ax   = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(_make_poly3d(vs_np, faces_np[face_idx], fc_b))
    ax.scatter(jb_np[:, 0], jb_np[:, 1], jb_np[:, 2],
               c='red', s=28, zorder=6, depthshade=False, label='Regressed joints')
    ax.set_xlim(vs_np[:, 0].min() - 0.05, vs_np[:, 0].max() + 0.05)
    ax.set_ylim(vs_np[:, 1].min() - 0.05, vs_np[:, 1].max() + 0.05)
    ax.set_zlim(vs_np[:, 2].min() - 0.05, vs_np[:, 2].max() + 0.05)
    ax.view_init(elev=10, azim=65); ax.set_axis_off()
    ax.set_title('(b) $v_{shaped} = \\bar{T} + B_S(\\beta)$\n+ regressed joints $J(\\beta)$',
                 fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.savefig('outputs/stage_b_shaped_joints.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> outputs/stage_b_shaped_joints.png")

    # ================================================
    # 阶段 C：姿态校正偏移可视化
    # ================================================
    print("\n[C] 姿态校正 B_P(θ) ...")

    pose = torch.zeros(1, n_joints * 3, device=DEVICE)
    pose[0, 3 * 16 + 2] =  1.3   # 左肘弯曲 ~75°
    pose[0, 3 * 13 + 2] = -1.3   # 右肘对称
    pose[0, 3 *  3 + 0] =  0.25  # spine1 前倾
    pose[0, 3 *  9 + 0] = -0.15  # spine3 后倾（保持平衡）

    n_j      = pose.shape[1] // 3
    rot_mats = batch_rodrigues(pose.view(-1, 3)).view(1, n_j, 3, 3)
    ident    = torch.eye(3, device=DEVICE)
    p_feat   = (rot_mats[:, 1:] - ident).view(1, -1)
    p_offs   = torch.matmul(p_feat, posedirs).view(1, -1, 3)

    v_shaped_c = v_template.unsqueeze(0) + blend_shapes(betas, shapedirs)
    v_posed_c  = v_shaped_c + p_offs

    offs_mag      = torch.norm(p_offs[0], dim=-1).cpu().numpy()
    offs_mag_norm = offs_mag / (offs_mag.max() + 1e-8)

    vpc_np = v_posed_c[0].cpu().numpy()
    fc_c   = face_colors_from_vertex(offs_mag_norm, faces_np[face_idx], 'plasma')

    fig = plt.figure(figsize=(5, 7))
    ax  = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(_make_poly3d(vpc_np, faces_np[face_idx], fc_c))
    ax.set_xlim(vpc_np[:, 0].min() - 0.05, vpc_np[:, 0].max() + 0.05)
    ax.set_ylim(vpc_np[:, 1].min() - 0.05, vpc_np[:, 1].max() + 0.05)
    ax.set_zlim(vpc_np[:, 2].min() - 0.05, vpc_np[:, 2].max() + 0.05)
    sm2 = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(0, 1))
    sm2.set_array([])
    fig.colorbar(sm2, ax=ax, shrink=0.4, aspect=18, label='|pose offset| (norm.)')
    ax.view_init(elev=10, azim=65); ax.set_axis_off()
    ax.set_title('(c) Pose Corrective $B_P(\\theta)$\ncolor = offset magnitude',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig('outputs/stage_c_pose_offsets.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> outputs/stage_c_pose_offsets.png")

    # ================================================
    # 阶段 D：完整 LBS 结果
    # ================================================
    print("\n[D] 完整 LBS 结果 ...")

    verts_d, J_trans_d, v_shaped_d, v_posed_d, J_rest_d = lbs_forward(
        betas, pose,
        v_template, shapedirs, posedirs,
        J_regressor, parents, lbs_weights
    )

    vd_np  = verts_d[0].cpu().numpy()
    jd_np  = J_trans_d[0].cpu().numpy()
    fc_d   = np.full((N_DRAW, 3), [0.62, 0.82, 0.62])

    fig = plt.figure(figsize=(5, 7))
    ax  = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(_make_poly3d(vd_np, faces_np[face_idx], fc_d))
    ax.scatter(jd_np[:, 0], jd_np[:, 1], jd_np[:, 2],
               c='red', s=25, zorder=6, depthshade=False)
    ax.set_xlim(vd_np[:, 0].min() - 0.05, vd_np[:, 0].max() + 0.05)
    ax.set_ylim(vd_np[:, 1].min() - 0.05, vd_np[:, 1].max() + 0.05)
    ax.set_zlim(vd_np[:, 2].min() - 0.05, vd_np[:, 2].max() + 0.05)
    ax.view_init(elev=10, azim=65); ax.set_axis_off()
    ax.set_title('(d) Final LBS Result\n'
                 r"$v'_i = \sum_k w_{ik} G_k v_i^{posed}$", fontsize=11)
    plt.tight_layout()
    plt.savefig('outputs/stage_d_lbs_result.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> outputs/stage_d_lbs_result.png")

    # ================================================
    # 对比图（2×2 grid）
    # ================================================
    print("\n[E] 生成四阶段对比图 ...")

    panel_paths = [
        'outputs/stage_a_template_weights.png',
        'outputs/stage_b_shaped_joints.png',
        'outputs/stage_c_pose_offsets.png',
        'outputs/stage_d_lbs_result.png',
    ]
    panel_titles = [
        '(a) Template + Weights',
        '(b) Shape + Joints',
        '(c) Pose Offsets',
        '(d) Final Skinned Mesh',
    ]

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    for ax, path, label in zip(axes, panel_paths, panel_titles):
        ax.imshow(plt.imread(path))
        ax.set_title(label, fontsize=13, pad=7)
        ax.axis('off')

    fig.suptitle('SMPL LBS Pipeline — Four Stages', fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig('outputs/comparison_grid.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> outputs/comparison_grid.png")

    # ================================================
    # 一致性验证：手写 LBS vs 官方前向
    # ================================================
    print("\n[F] 手写 LBS 与官方前向结果比对 ...")

    global_orient = pose[:, :3].clone()
    body_pose     = pose[:, 3:].clone()

    with torch.no_grad():
        out = model(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            return_verts=True
        )
    official_verts = out.vertices.float()

    diff    = (verts_d - official_verts).abs()
    mae     = diff.mean().item()
    max_err = diff.max().item()

    print(f"  平均绝对误差 (MAE):  {mae:.6f} m")
    print(f"  最大绝对误差 (Max):  {max_err:.6f} m")

    with open('outputs/summary.txt', 'w', encoding='utf-8') as f:
        f.write("========== SMPL 模型基础信息 ==========\n")
        f.write(f"顶点数:      {n_verts}\n")
        f.write(f"面片数:      {n_faces}\n")
        f.write(f"关节数:      {n_joints}\n")
        f.write(f"betas 维度:  {n_betas}\n\n")
        f.write("========== 手写 LBS 与官方前向误差 ==========\n")
        f.write(f"平均绝对误差 (MAE): {mae:.6f} m\n")
        f.write(f"最大绝对误差 (Max): {max_err:.6f} m\n\n")
        f.write("测试参数:\n")
        f.write(f"  betas[0:4]: {betas[0, :4].tolist()}\n")
        f.write(f"  左肘 (joint 16, z): {pose[0, 3*16+2].item():.2f} rad\n")
        f.write(f"  右肘 (joint 13, z): {pose[0, 3*13+2].item():.2f} rad\n")
        f.write(f"  spine1 (joint 3, x): {pose[0, 3*3+0].item():.2f} rad\n")
    print("  -> outputs/summary.txt")

    # ================================================
    # 选做：姿态动画（双肘弯曲 & 张开）
    # ================================================
    print("\n[选做] 生成姿态动画 ...")

    betas_anim = betas.clone()
    n_frames   = 36
    frames     = []

    for i in range(n_frames):
        t     = i / (n_frames - 1)
        angle = np.pi * np.sin(t * np.pi)  # 0 -> π -> 0，模拟一次完整弯曲

        pa = torch.zeros(1, n_joints * 3, device=DEVICE)
        pa[0, 3 * 16 + 2] =  angle          # 左肘
        pa[0, 3 * 13 + 2] = -angle          # 右肘（镜像）
        pa[0, 3 *  3 + 0] =  angle * 0.12   # 躯干随动

        va, _, _, _, _ = lbs_forward(
            betas_anim, pa,
            v_template, shapedirs, posedirs,
            J_regressor, parents, lbs_weights
        )
        va_np  = va[0].cpu().numpy()
        fc_ani = np.full((N_DRAW, 3), [0.72, 0.72, 0.85])

        fig = plt.figure(figsize=(3.5, 5), dpi=90)
        ax  = fig.add_subplot(111, projection='3d')
        ax.add_collection3d(_make_poly3d(va_np, faces_np[face_idx], fc_ani))
        ax.set_xlim(va_np[:, 0].min() - 0.05, va_np[:, 0].max() + 0.05)
        ax.set_ylim(va_np[:, 1].min() - 0.05, va_np[:, 1].max() + 0.05)
        ax.set_zlim(va_np[:, 2].min() - 0.05, va_np[:, 2].max() + 0.05)
        ax.view_init(elev=8, azim=70); ax.set_axis_off()
        ax.set_title(f'Elbow bend: {np.degrees(angle):.0f}°', fontsize=10)
        fig.tight_layout(pad=0.3)
        fig.canvas.draw()
        buf   = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,)).copy()
        plt.close(fig)
        frames.append(frame)

    imageio.mimsave('outputs/pose_animation.gif', frames, fps=18, loop=0)
    print("  -> outputs/pose_animation.gif")

    print("\n" + "=" * 50)
    print("全部完成，输出保存在 outputs/ 目录。")


if __name__ == '__main__':
    main()
