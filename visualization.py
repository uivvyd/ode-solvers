from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch
import numpy as np


def remove_ticks(ax):
    ax.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False,
        left=False,
        labelleft=False
    )


def remove_xticks(ax):
    ax.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False,
        left=True,
        labelleft=True
    )


def remove_frame(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


def compare_solvers_samples(samples, labels, title='', padding=2):
    """
    Visualize samples by different solvers, one solver per row
    :param samples: list of tensors BxCxHxW, len(samples) == len(labels)
    :param labels: list of solvers names
    :param title: str
    :param nrow: int, number of images in one row
    :param padding: int, padding for make_grid
    """
    nrow = samples[0].shape[0]
    _samples = torch.concat(samples)
    samples = (_samples * 127.5 + 128).clip(0, 255).to(torch.uint8).cpu()
    img_grid = make_grid(samples, nrow=nrow, padding=padding, pad_value=255)
    ncol = len(labels)
    fig, ax = plt.subplots(1, figsize=(nrow, ncol))

    h = samples.shape[2]
    remove_xticks(ax)
    remove_frame(ax)
    ax.tick_params(left=False)
    ax.set_yticks(np.arange(h // 2, ncol * (h + padding), h + padding), labels)

    ax.set_title(title, fontsize=14)
    ax.imshow(img_grid.permute(1, 2, 0))
    plt.show()


def compare_methods_x_steps(samples_x_steps, met_labels, steps_labels, title='', padding=2):
    """
    Visualize samples by different solvers with different num_steps, batch_size == 1
    :param samples_x_steps: list, samples_x_steps[i][j] == samples of met_labels[i] method
                                                                with steps_labels[j] steps
    """
    nrow = len(steps_labels)
    ncol = len(met_labels)

    _samples = []
    for method_samples in samples_x_steps:
        _samples.append(torch.concat(method_samples))
    _samples = torch.concat(_samples)

    samples = (_samples * 127.5 + 128).clip(0, 255).to(torch.uint8).cpu()
    img_grid = make_grid(samples, nrow=nrow, padding=padding, pad_value=255)
    fig, ax = plt.subplots(1, figsize=(nrow, ncol))

    h, w = samples.shape[2], samples.shape[3]
    remove_frame(ax)
    ax.tick_params(left=False, labeltop='on', bottom=False, labelbottom=False)
    ax.set_yticks(np.arange(h // 2, ncol * (h + padding), h + padding), met_labels)
    ax.set_xticks(np.arange(w // 2, nrow * (w + padding), w + padding), steps_labels)
    ax.set_title(title, fontsize=14)
    ax.imshow(img_grid.permute(1, 2, 0))
    plt.show()


def compare_histories(histories, met_labels, title='', padding=2, skip_freq=1):
    """
    Visualize steps of denoising by different solvers, batch_size == 1
    :param samples_x_steps: list, histories[i] == history of denoising by met_labels[i] method
    :param skip_freq: int, which steps to skip
    """
    steps_labels = list(range(1, len(histories[0]), skip_freq))
    nrow = len(steps_labels)
    ncol = len(histories)

    _histories = []
    for history in histories:
        _histories.append(torch.concat(history[1::skip_freq])) # excluding x_T == noise
    _histories = torch.concat(_histories)

    histories = (_histories * 127.5 + 128).clip(0, 255).to(torch.uint8).cpu()
    img_grid = make_grid(histories, nrow=nrow, padding=padding, pad_value=255)
    fig, ax = plt.subplots(1, figsize=(nrow, ncol))

    h, w = histories.shape[2], histories.shape[3]
    remove_frame(ax)
    ax.tick_params(left=False, labeltop='on', bottom=False, labelbottom=False)
    ax.set_yticks(np.arange(h // 2, ncol * (h + padding), h + padding), met_labels)
    ax.set_xticks(np.arange(w // 2, nrow * (w + padding), w + padding), steps_labels)
    ax.set_title(title, fontsize=14)
    ax.imshow(img_grid.permute(1, 2, 0))
    plt.show()
