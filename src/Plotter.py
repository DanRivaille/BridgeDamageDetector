import os

import matplotlib.pyplot as plt
import numpy as np

from src.CommonPath import CommonPath


class Plotter:
    """
    A class for creating various plots and providing options for showing or saving them.
    """
    @staticmethod
    def plot_signal(signal: np.ndarray, size: tuple = (3, 15), save: bool = True, filename: str = "") -> None:
        """
        Plots a signal based on a given array.
        @param signal An array representing The signal to be plotted.
        @param size The size of the plot (height, width). Default is (3, 15).
        @param save A save flag to indicate that the plot will be saved to a file if it is True. Default is True.
        @param filename The filename where the plot will be saved if the save flag is True. Default is "".
        """
        fig, axs = plt.subplots()
        fig.set_figheight(size[0])
        fig.set_figwidth(size[1])
        axs.set_title("Signal")
        axs.plot(signal, color='C0')
        axs.set_xlabel("Time")
        axs.set_ylabel("Amplitude")

        Plotter.__save_or_show_plot(save, filename)

    @staticmethod
    def plot_training_curves(train_error: list,
                             validation_error: list,
                             size: tuple = (3, 15),
                             save: bool = True,
                             filename: str = "") -> None:
        """
        Plots the training curves of a model with its training and validation errors.
        @param train_error List of training errors of the model over epochs.
        @param validation_error List of validation errors of the model over epochs.
        @param size The size of the plot (height, width). Default is (3, 15).
        @param save A save flag to indicate that the plot will be saved to a file if it is True. Default is True.
        @param filename The filename where the plot will be saved if the save flag is True. Default is "".
        """
        fig, axs = plt.subplots()
        fig.set_figheight(size[0])
        fig.set_figwidth(size[1])
        axs.plot(train_error, label='Train error')
        axs.plot(validation_error, label='Validation error')
        axs.set_xlabel("Epochs")
        axs.set_ylabel("Error (MSE)")
        plt.legend()

        Plotter.__save_or_show_plot(save, filename)

    @staticmethod
    def __save_or_show_plot(save: bool, filename: str):
        """
        Saves or shows the plot in a file based on the save flag.
        @param save A save flag to indicate that the plot will be saved to a file if it is True.
        @param filename The filename where the plot will be saved if the save flag is True.
        """
        if save:
            plt.savefig(os.path.join(CommonPath.ROOT_FOLDER.value, filename), bbox_inches='tight')
        else:
            plt.show()
