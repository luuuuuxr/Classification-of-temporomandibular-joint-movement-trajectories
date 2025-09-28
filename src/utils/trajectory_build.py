import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm


def save_expanded_data_to_excel(expanded_data, labels, file_info, base_save_path='Expanded_Data'):
    """
    将扩充后的数据保存到 Excel 文件中，并保持原始文件夹结构。

    :param expanded_data: 扩充后的数据 (NumPy 数组)
    :param labels: 文件夹标签，用于重建文件夹结构
    :param file_info: 文件信息 (文件名和 sheet 索引)
    :param base_save_path: 保存的基本路径
    """
    for i, data in enumerate(expanded_data):
        # 获取文件夹名称、文件名和 sheet 索引
        original_folder = str(labels[i])
        excel_name, sheet_index = file_info[i]

        # 创建保存路径，保持原始文件夹结构
        save_folder = os.path.join(base_save_path, original_folder)
        os.makedirs(save_folder, exist_ok=True)

        # 创建保存的 Excel 文件路径
        save_path = os.path.join(save_folder, f'{excel_name}_expanded.xlsx')

        # 如果文件已经存在，则追加新的 sheet
        if os.path.exists(save_path):
            with pd.ExcelWriter(save_path, mode='a', engine='openpyxl') as writer:
                pd.DataFrame(data).to_excel(writer, sheet_name=f'Sheet{sheet_index}_expanded', index=False,
                                            header=False)
        else:
            with pd.ExcelWriter(save_path, mode='w', engine='openpyxl') as writer:
                pd.DataFrame(data).to_excel(writer, sheet_name=f'Sheet{sheet_index}_expanded', index=False,
                                            header=False)

    print(f'Saved expanded trajectory to {base_save_path}')


# 计算每个点到起始点的欧氏距离
def calculate_euclidean_distance(trajectory):
    start_point = trajectory[0]  # 使用轨迹的第一个点作为起始点
    distances = np.linalg.norm(trajectory - start_point, axis=1)  # 计算每个点到起始点的距离
    return distances


def split_into_phases(expanded_trajectory, num_phases=6):
    """
    根据运动轨迹分为开口和闭口过程，并进一步细分。适用于颞下颌关节运动轨迹。
    """
    distances = np.sqrt(np.sum(expanded_trajectory ** 2, axis=1))
    max_open_idx = np.argmax(distances)

    # 将轨迹划分为张口和闭口过程
    open_trajectory = expanded_trajectory[:max_open_idx + 1]
    close_trajectory = expanded_trajectory[max_open_idx:]

    # 计算张口和闭口阶段的长度
    open_length = len(open_trajectory)
    close_length = len(close_trajectory)

    # 张口阶段的比例
    open_init_length = np.argmin(np.abs(distances[:max_open_idx] - distances[max_open_idx] * 0.15))
    open_end_length = np.argmin(np.abs(distances[:max_open_idx] - distances[max_open_idx] * 0.85))
    open_mid_phase = open_trajectory[open_init_length:open_end_length]

    # 闭口阶段的比例
    close_init_length = np.argmin(np.abs(distances[max_open_idx:] - distances[max_open_idx] * 0.85))
    close_end_length = np.argmin(np.abs(distances[max_open_idx:] - distances[max_open_idx] * 0.15))
    close_mid_phase = close_trajectory[close_init_length:close_end_length]

    # 返回六个阶段
    phases = [
        open_trajectory[:open_init_length],  # 张口初始阶段
        open_mid_phase,  # 张口过程阶段
        open_trajectory[open_end_length:],  # 张口结束阶段
        close_trajectory[:close_init_length],  # 闭口初始阶段
        close_mid_phase,  # 闭口过程阶段
        close_trajectory[close_end_length:]  # 闭口结束阶段
    ]

    return phases


def plot_phases_3d(phases_data, labels, file_info, base_save_path='3D_Trajectories_Phases', num_phases=6):
    # 使用颜色映射来区分每个阶段的颜色
    colors = cm.viridis(np.linspace(0, 1, num_phases))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i, phases in enumerate(phases_data):  # phases_data 现在包含已经分割好的阶段数据
        # 获取文件夹名称、Excel文件名和sheet索引
        original_folder = str(labels[i])
        excel_name, sheet_index = file_info[i]

        # 创建保存路径，与原始文件夹格式相同
        save_folder = os.path.join(base_save_path, original_folder)
        os.makedirs(save_folder, exist_ok=True)

        # 创建3D图，并绘制每个阶段
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制每个阶段并使用不同颜色区分
        for phase_index, phase in enumerate(phases):
            if len(phase) > 0:  # 确保阶段不为空
                rps_x, rps_y, rps_z = phase[:, 0], phase[:, 1], phase[:, 2]
                ax.plot3D(rps_x, rps_y, rps_z, color=colors[phase_index], label=f'Phase {phase_index + 1}')

        # 图像标题和坐标轴设置
        plt.title(f'{excel_name} - Sheet {sheet_index} - Combined 3D Phases')
        ax.xaxis.line.set_color('black')
        ax.yaxis.line.set_color('black')
        ax.zaxis.line.set_color('black')

        # 设置网格线
        ax.grid(True, color='#505050', linestyle='-', linewidth=1)
        ax.set_xlim(-30, 23)
        ax.set_ylim(-20, 33)
        ax.set_zlim(-38, 10)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.legend()
        ax.view_init(elev=5, azim=-150)

        # 保存3D图像到指定路径
        save_path_3d = os.path.join(save_folder, f'{excel_name}_Sheet{sheet_index}_Combined_Phases.png')
        plt.savefig(save_path_3d)
        plt.close(fig)
    print('All phases images are saved!')
