# *
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of PyHessian library.
#
# PyHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHessian.  If not, see <http://www.gnu.org/licenses/>.
# *

import torch
import lightning
import more_itertools
import numpy as np
import matplotlib.pyplot as plt

from .utils import (
    group_product,
    group_add,
    normalization,
    get_params_grad,
    hessian_vector_product,
    orthnormal,
)


class hessian:
    """
    The class used to compute :
        i) the top 1 (n) eigenvalue(s) of the neural network
        ii) the trace of the entire neural network
        iii) the estimated eigenvalue density
    """

    def __init__(
        self,
        model: lightning.LightningModule,
        dataloader: torch.utils.data.DataLoader,
        cuda: bool = True,
    ):
        """
        model: the model that needs Hessain information
        data: a single batch of data, including inputs and its corresponding labels
        dataloader: the data loader including bunch of batches of data
        """
        # Make the original PyHessian code play nicely with Lightning
        # Using lightning.Fabric will auto-manage the devices on which the computation is performed on.
        accel = "cuda" if cuda else "cpu"
        self.fabric = lightning.Fabric(accelerator=accel, devices=1)
        self.fabric.launch()

        self.model = self.fabric.setup_module(model)
        self.model.eval()  # Set model to eval mode
        self.data = self.fabric.setup_dataloaders(dataloader)

        self.is_single_batch = len(self.data) == 1

        # pre-processing for single batch case to simplify the computation.
        if self.is_single_batch:
            # if we only compute the Hessian information for a single batch data, we can re-use the gradients.
            batch = more_itertools.one(self.data)
            loss = self.model.training_step(batch, 0)
            loss.backward(create_graph=True)

        # this step is used to extract the parameters from the model
        params, gradsH = get_params_grad(self.model)
        self.params = params
        self.gradsH = gradsH  # gradient used for Hessian computation

    @property
    def device(self) -> torch.device:
        return self.model.device

    def dataloader_hv_product(self, v):
        device = self.device
        num_data = 0  # count the number of datum points in the dataloader

        THv = [
            torch.zeros(p.size()).to(device) for p in self.params
        ]  # accumulate result
        for i, batch in enumerate(self.data):
            self.model.zero_grad()
            tmp_num_data = self.data.batch_size  # won't work for uneven batch sizes

            loss = self.model.training_step(batch, i)
            loss.backward(create_graph=True)
            params, gradsH = get_params_grad(self.model)
            self.model.zero_grad()
            Hv = torch.autograd.grad(
                gradsH, params, grad_outputs=v, only_inputs=True, retain_graph=False
            )
            THv = [THv1 + Hv1 * float(tmp_num_data) + 0.0 for THv1, Hv1 in zip(THv, Hv)]
            num_data += float(tmp_num_data)

        THv = [THv1 / float(num_data) for THv1 in THv]
        eigenvalue = group_product(THv, v).cpu().item()
        return eigenvalue, THv

    def eigenvalues(self, maxIter=100, tol=1e-3, top_n=1):
        """
        compute the top_n eigenvalues using power iteration method
        maxIter: maximum iterations used to compute each single eigenvalue
        tol: the relative tolerance between two consecutive eigenvalue computations from power iteration
        top_n: top top_n eigenvalues will be computed
        """

        assert top_n >= 1

        device = self.device

        eigenvalues = []
        eigenvectors = []

        computed_dim = 0

        while computed_dim < top_n:
            eigenvalue = None
            v = [
                torch.randn(p.size()).to(device) for p in self.params
            ]  # generate random vector
            v = normalization(v)  # normalize the vector

            for i in range(maxIter):
                v = orthnormal(v, eigenvectors)
                self.model.zero_grad()

                if self.is_single_batch:
                    Hv = hessian_vector_product(self.gradsH, self.params, v)
                    tmp_eigenvalue = group_product(Hv, v).cpu().item()
                else:
                    tmp_eigenvalue, Hv = self.dataloader_hv_product(v)

                v = normalization(Hv)

                if eigenvalue == None:
                    eigenvalue = tmp_eigenvalue
                else:
                    if (
                        abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) + 1e-6)
                        < tol
                    ):
                        break
                    else:
                        eigenvalue = tmp_eigenvalue
            eigenvalues.append(eigenvalue)
            eigenvectors.append(v)
            computed_dim += 1

        return eigenvalues, eigenvectors

    def trace(self, maxIter=100, tol=1e-3):
        """
        compute the trace of hessian using Hutchinson's method
        maxIter: maximum iterations used to compute trace
        tol: the relative tolerance
        """

        device = self.device
        trace_vhv = []
        trace = 0.0

        for i in range(maxIter):
            self.model.zero_grad()
            v = [torch.randint_like(p, high=2, device=device) for p in self.params]
            # generate Rademacher random variables
            for v_i in v:
                v_i[v_i == 0] = -1

            if self.is_single_batch:
                Hv = hessian_vector_product(self.gradsH, self.params, v)
            else:
                _, Hv = self.dataloader_hv_product(v)
            trace_vhv.append(group_product(Hv, v).cpu().item())
            if abs(np.mean(trace_vhv) - trace) / (abs(trace) + 1e-6) < tol:
                return trace_vhv
            else:
                trace = np.mean(trace_vhv)

        return trace_vhv

    def density(self, iter=100, n_v=1):
        """
        compute estimated eigenvalue density using stochastic lanczos algorithm (SLQ)
        iter: number of iterations used to compute trace
        n_v: number of SLQ runs
        """
        eigen_list_full = []
        weight_list_full = []

        with torch.device(self.device):
            for _ in range(n_v):
                v = [torch.randint_like(p, high=2) for p in self.params]
                # generate Rademacher random variables
                for v_i in v:
                    v_i[v_i == 0] = -1
                v = normalization(v)

                # standard lanczos algorithm initlization
                v_list = [v]
                w_list = []
                alpha_list = []
                beta_list = []
                ############### Lanczos
                for i in range(iter):
                    self.model.zero_grad()
                    w_prime = [torch.zeros(p.size()) for p in self.params]
                    if i == 0:
                        if self.is_single_batch:
                            w_prime = hessian_vector_product(
                                self.gradsH, self.params, v
                            )
                        else:
                            _, w_prime = self.dataloader_hv_product(v)
                        alpha = group_product(w_prime, v)
                        alpha_list.append(alpha.cpu().item())
                        w = group_add(w_prime, v, alpha=-alpha)
                        w_list.append(w)
                    else:
                        beta = torch.sqrt(group_product(w, w))
                        beta_list.append(beta.cpu().item())
                        if beta_list[-1] != 0.0:
                            # We should re-orth it
                            v = orthnormal(w, v_list)
                            v_list.append(v)
                        else:
                            # generate a new vector
                            w = [torch.randn(p.size()) for p in self.params]
                            v = orthnormal(w, v_list)
                            v_list.append(v)
                        if self.is_single_batch:
                            w_prime = hessian_vector_product(
                                self.gradsH, self.params, v
                            )
                        else:
                            _, w_prime = self.dataloader_hv_product(v)
                        alpha = group_product(w_prime, v)
                        alpha_list.append(alpha.cpu().item())
                        w_tmp = group_add(w_prime, v, alpha=-alpha)
                        w = group_add(w_tmp, v_list[-2], alpha=-beta)

                T = torch.zeros(iter, iter)
                for i in range(len(alpha_list)):
                    T[i, i] = alpha_list[i]
                    if i < len(alpha_list) - 1:
                        T[i + 1, i] = beta_list[i]
                        T[i, i + 1] = beta_list[i]
            a_, b_ = torch.linalg.eig(T)

            eigen_list = a_.real
            weight_list = b_.real**2

            eigen_list_full.append(list(eigen_list.cpu().numpy()))
            weight_list_full.append(list(weight_list.cpu().numpy()))

        return eigen_list_full, weight_list_full


def get_esd_plot(eigenvalues, weights):
    density, grids = density_generate(eigenvalues, weights)
    plt.semilogy(grids, density + 1.0e-7)
    plt.ylabel("Density (Log Scale)", fontsize=14, labelpad=10)
    plt.xlabel("Eigenvlaue", fontsize=14, labelpad=10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.axis([np.min(eigenvalues) - 1, np.max(eigenvalues) + 1, None, None])
    plt.tight_layout()
    plt.savefig("example.pdf")


def density_generate(
    eigenvalues, weights, num_bins=10000, sigma_squared=1e-5, overhead=0.01
):
    eigenvalues = np.array(eigenvalues)
    weights = np.array(weights)

    lambda_max = np.mean(np.max(eigenvalues, axis=1), axis=0) + overhead
    lambda_min = np.mean(np.min(eigenvalues, axis=1), axis=0) - overhead

    grids = np.linspace(lambda_min, lambda_max, num=num_bins)
    sigma = sigma_squared * max(1, (lambda_max - lambda_min))

    num_runs = eigenvalues.shape[0]
    density_output = np.zeros((num_runs, num_bins))

    for i in range(num_runs):
        for j in range(num_bins):
            x = grids[j]
            tmp_result = gaussian(eigenvalues[i, :], x, sigma)
            density_output[i, j] = np.sum(tmp_result * weights[i, :])
    density = np.mean(density_output, axis=0)
    normalization = np.sum(density) * (grids[1] - grids[0])
    density = density / normalization
    return density, grids


def gaussian(x, x0, sigma_squared):
    return np.exp(-((x0 - x) ** 2) / (2.0 * sigma_squared)) / np.sqrt(
        2 * np.pi * sigma_squared
    )
