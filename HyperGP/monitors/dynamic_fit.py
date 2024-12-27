import math
import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import os

def curves(funcs, labels, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.grid(ls='--')
    for i in range(len(funcs)):
        ax.plot(funcs[i][0], funcs[i][1], funcs[i][2], linewidth=3.0, color=funcs[i][3], label=funcs[i][4])
    fig.savefig(save_path)

class Curve:
    def __init__(self, xrange, yrange):
        self.dim = 1 if np.array(xrange).ndim == 1 else len(xrange)
        x_size = 1
        while x_size ** 2 < self.dim:
            x_size += 1
        y_size = x_size
        while x_size * y_size >= self.dim:
            y_size -= 1
        y_size += 1
        self.size = (x_size, y_size)

        self.fig = plt.figure(0, figsize=(x_size * 10, y_size * 6))
        # plt.ion()
        if not os.path.exists('pic'):
            os.makedirs('pic')
        self.xrange = xrange
        self.yrange = yrange

    def pic_show(self, x_, y_, points, frame, fitness):
        y = []
        x = []
        for i in range(self.dim):
            Z = zip(x_[i], y_)
            Z = sorted(Z)
            (x_i, y_i) = zip(*Z)
            x.append(x_i)
            y.append(y_i)
        self.fig.clf()
        self.ax = self.fig.subplots(self.size[1], self.size[0])
        if self.dim == 1:
            self.ax = [self.ax]
        for i in range(self.dim):
            x_idx = int(i % self.size[0])
            y_idx = int(i / self.size[0])
            if self.size[1] == 1:
                ax_ = self.ax[x_idx]
            else:
                ax_ = self.ax[y_idx][x_idx]
            ax_.grid(ls='--')
            ax_.set_xlim(self.xrange[i][0], self.xrange[i][1])
            ax_.set_ylim(self.yrange[0], self.yrange[1])
            ax_.set_xlabel("x%d"%i)
            ax_.set_ylabel("y")
            ax_.plot(x[i], y[i],linewidth=3.0, color='orange', label='PGP, RMSE: {}'.format(fitness))
            ax_.legend()
            for j in range(len(points[1])):
                ax_.plot(points[0][i][j], points[1][j], 'o', color='red',linewidth=1.0)
        self.fig.tight_layout()
        self.fig.savefig('./pic/%d.png'%frame)
        # self.fig.show()
        # plt.pause(0.5)


class DCurve:
    def __init__(self, xrange, yrange, points=[[], []]):
        x_dim = 1 if np.array(xrange).ndim == 1 else len(xrange)
        self.dim = x_dim

        x_size = 1
        while x_size ** 2 < self.dim:
            x_size += 1
        y_size = x_size
        while x_size * y_size >= self.dim:
            y_size -= 1
        y_size += 1
        self.size = (x_size, y_size)

        self.fig = plt.figure(1, figsize=(x_size * 10, y_size * 6))
        self.ax = self.fig.subplots(self.size[1], self.size[0])
        if x_dim == 1:
            self.ax = [self.ax]
        if points is not None:
            self.points = {'0': points}
        else:
            self.points = {}
        self.fig.tight_layout()
        self.ani = []
        self.iter = []
        self.fitness = []

        for i in range(x_dim):
            x_idx = int(i % self.size[0])
            y_idx = int(i / self.size[0])
            if self.size[1] == 1:
                ax_ = self.ax[x_idx]
            else:
                ax_ = self.ax[y_idx][x_idx]
            ax_.set_xlim(xrange[i][0], xrange[i][1])
            ax_.set_ylim(yrange[0], yrange[1])
            ax_.grid(ls='--')
            ax_.set_xlabel("x%d"%i)
            ax_.set_ylabel("y")
            for j in range(len(points[1])):
                ax_.plot(points[0][i][j], points[1][j], 'o', color='red',linewidth=1.0)
            self.ani.append(ax_.plot([], [],linewidth=3.0, color='orange', label='PGP')[0])
            ax_.legend()
        self.data = []

    def dynamic_curve(self, file_name):
        plt.ioff()
        ani = animation.FuncAnimation(fig=self.fig, func=self.update, frames=len(self.data), interval=300)
        ani.save(file_name + '.gif')
        # plt.show()

    def append_data(self, x_, y_, iter, fit):
        self.iter.append(iter)
        y = [[] for i in range(len(x_))]
        x = [[] for i in range(len(x_))]
        for i in range(self.dim):
            Z = zip(x_[i], y_)
            Z = sorted(Z)
            (x[i], y[i]) = zip(*Z)
        self.data.append((x, y))
        self.fitness.append(round(fit, 4))

    def add_points(self, points):#第几幅图加点
        frame = len(self.data)
        if not self.points.get(frame):
            self.points[frame] = points
        else:
            self.points[frame].extend(points)
    def update(self, frame):

        for i in range(self.dim):

            x_idx = int(i % self.size[0])
            y_idx = int(i / self.size[0])
            if self.size[1] == 1:
                ax_ = self.ax[x_idx]
            else:
                ax_ = self.ax[y_idx][x_idx]
            self.ani[i].set_data(np.array(self.data[frame][0][i]), np.array(self.data[frame][1][i]))
            if self.points.get(frame):
                points = self.points[frame]
                for j in range(len(points[1])):
                    ax_.plot(points[0][i][j], points[1][j], 'o', color='red')
                if self.points.get(0) and frame > 0:
                    points = self.points[0]
                    for j in range(len(points[1])):
                        ax_.plot(points[0][i][j], points[1][j], 'o', color='green')
            self.ani[i].set_label('iteration: {}, RMSE: {}'.format(self.iter[frame], self.fitness[frame], 4))
            ax_.legend()
        return [].extend(self.ani)

# dim = 4
# dataset_x = [[random.randint(-10, 10) for i in range(10)] for j in range(dim)]
# dataset_y = [random.randint(-100, 100) for i in range(10)]
# tmp = Curve([(-10, 10), (-10, 10), (-10, 10), (-100, 100)], (-100, 100))
# tmp_1 = DCurve([(-10, 10), (-10, 10), (-10, 10), (-100, 100)], (-100, 100))
# tmp_1.add_points([dataset_x, dataset_y], 0)
# for i in range(10):
#     x = np.array(range(10))
#     x_1 = np.array([random.randint(-10, 10) for i in range(10)])
#     x_2 = np.array(range(- 7, 3))
#     x_3 = np.array([random.randint(-100, 100) for i in range(10)])
#     y = x**2 * math.sin(i) + i + x_1 + x_2 ** 3 + x_3 * 4
#     tmp.pic_show([x, x_1, x_2, x_3], y, [dataset_x, dataset_y], i)
#     tmp_1.append_data([x, x_1, x_2, x_3], y)
#     image_path = "./pic/%d.png"%i
#     with open(image_path, 'rb') as f:
#         image_data = f.read()
#     print(image_data)
#
# tmp_1.dynamic_curve()
