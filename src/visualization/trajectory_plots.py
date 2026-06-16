# -*- coding: utf-8 -*-
"""
轨迹可视化模块：plot_2d_trajectory / plot_3d_trajectory
"""

import os

import numpy as np
from matplotlib import pyplot as plt


def plot_2d_trajectory(data, labels, file_info, base_save_path):
    for i, sequence in enumerate(data):
        original_folder = str(labels[i])
        excel_name, sheet_index = file_info[i]
        save_folder = os.path.join(base_save_path, original_folder)
        os.makedirs(save_folder, exist_ok=True)

        rps_x = sequence[:, 0]
        rps_y = sequence[:, 1]
        rps_z = sequence[:, 2]
        indices = np.arange(len(rps_x))

        fig, ax = plt.subplots()
        ax.plot(indices, rps_x, label='X')
        ax.plot(indices, rps_y, label='Y')
        ax.plot(indices, rps_z, label='Z')
        plt.title(f'{excel_name} - Sheet {sheet_index} - 2D Coordinates')
        ax.set_xlabel('Point Index')
        ax.set_ylabel('Coordinate Value')
        ax.grid(True, color='#505050', linestyle='-', linewidth=0.5)
        ax.legend()
        ax.set_ylim(-45, 50)
        ax.set_xlim(0, 200)

        save_path = os.path.join(save_folder,
                                 f'{excel_name}_Sheet{sheet_index}_2D.png')
        plt.savefig(save_path)
        plt.close(fig)
    print(f'All images are saved to {base_save_path}')


def plot_3d_trajectory(data, labels, file_info, base_save_path='3D_Trajectories'):
    for i, sequence in enumerate(data):
        original_folder = str(labels[i])
        excel_name, sheet_index = file_info[i]
        save_folder = os.path.join(base_save_path, original_folder)
        os.makedirs(save_folder, exist_ok=True)

        rps_x, rps_y, rps_z = sequence[:, 0], sequence[:, 1], sequence[:, 2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot3D(rps_x, rps_y, rps_z)
        plt.title(f'{excel_name} - Sheet {sheet_index} - 3D Trajectory')
        ax.xaxis.line.set_color('black')
        ax.yaxis.line.set_color('black')
        ax.zaxis.line.set_color('black')
        ax.grid(True, color='#505050', linestyle='-', linewidth=1)
        ax.set_xlim(-30, 23)
        ax.set_ylim(-20, 33)
        ax.set_zlim(-38, 10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.legend(['Trajectory'])
        ax.view_init(elev=5, azim=-150)

        save_path_3d = os.path.join(save_folder,
                                    f'{excel_name}_Sheet{sheet_index}.png')
        plt.savefig(save_path_3d)
        plt.close(fig)
    print(f'All images are saved to {base_save_path}')
