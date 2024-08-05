#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 15:49:55 2023

@author: balada
"""

import os
import json
import random
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
torch.backends.cudnn.allow_tf32 = False

import warnings


from os import path
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Callable, Type
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot


class GrowingNeuralGas:
    """
    A PyTorch-based implementation of the Growing Neural Gas Algorithm (GNG)
    The GNG follows the paradigm of Hebbian competitive learning and tries to 
    fit a learned graph to represent the topology of a data distribution as 
    good as possible.


    Attributes
    ----------
    nodes : torch.Tensor
        Representation of the nodes of the GNG
    adj_M : torch.Tensor
        Adjacency matrix of the GNG
    error : torch.Tensor
        Cumulated error of the nodes
    age : torch.Tensor
        Cumulated age of the edges


    Methods
    -------
    get_config:
        Returns the GNG configuration as python dictionary.
    print_config:
        Prints the string representation (config) of the GNG
    get_neighbourhood(n_idx: torch.Tensor):
        Returns the indices of all direct neighbours of n_idx
    fit(data: torch.Tensor, steps: int | None = None, epochs: int | None = None,
        save_progress: bool = False, save_path: str | None = None, progress_plot_title: str | None = None)
        Fits the GNG to the given dataset.
    """

    def __init__(self,
                 n_features: int,
                 initial_gng_size: int = 25,
                 max_gng_size: int = 500,
                 batch_size: int = 128,
                 random_seed: int | None = None,
                 device: str = "cpu",
                 gng_dist_func: Callable = torch.cdist, # l2 distance; any func like dist_func(batch, self.nodes) -> batch X self.nodes showing the distance
                 gng_node_min_distance_for_update: float = 0,
                 gng_num_closest_neurons: int = 2,
                 gng_move_closest_neurons_discount: float = 0.2,
                 gng_move_closest_neurons_hop2_discount: float = 0.03,
                 gng_error_decay: float = 0.995,
                 gng_error_discount: float = 1.,
                 gng_after_split_error_decay: float = 0.5,
                 gng_max_edge_age: int = 50,
                 gng_max_node_error: float = np.inf,
                 gng_steps_before_max_node_error_is_removed: int = 0,
                 gng_edge_entanglement_restriction: bool = True,
                 early_stopping: tuple[int, int] | tuple[None, None] = (None, None),
                 early_stopping_min_improvement: float = 1e-6,
                 early_stopping_monitor: str = "mean_dist",
                 early_stopping_goal: str = "minimize",
                 add_node_every_n_steps: int = 100,
                 check_edges_every_n_steps: int = 100,
                 remove_high_error_nodes_every_n_steps: int = 1000,
                 eval_func: Callable | None = None,
                 gng_batch_size_norm: str = "BOTH",
                 torch_num_threads: None | int = 12,
                 dtype: Type | str = torch.float32,
                 **kwargs
                 ):
        """
        Constructs all the necessary attributes for the GNG object.

        Parameters
        ----------
            n_features: int
                Number of features per sample.
            initial_gng_size: int
                Initial number of nodes in the GNG (default: ``25``).
            max_gng_size: int
                Maximum number of nodes in the GNG (default: ``500``).
            batch_size: int
                Batch size (default: ``128``).
            random_seed: int, optional
                Random seed (default: ``None``).
            device: str, optional
                PyTorch device used for all computations (default: ``cpu``).
            gng_dist_func: Callable
                Can be any Callable that takes $f(A, B)$ where A is of 
                :math:`B \times P \times M` and B is of :math:`B \times R \times M`.
                (default: ``torch.cdist``). 
            gng_node_min_distance_for_update: float
                Minimum distance for update (default: ``0.``).
            gng_num_closest_neurons: int
                Number of closest neurons that will be taken into account for 
                GNG fitting (default: ``2``).
            gng_move_closest_neurons_discount: float
                Discount factor to reduce the adjustment of the clostest neuron
                (default: ``0.2``).
            gng_move_closest_neurons_hop2_discount: float
                Discount factor to reduce the adjustment of the second clostest 
                neuron (default: ``0.03``).
            gng_error_decay: float
                Error decay factor which is applied after each iteration. 
                Affects all nodes (default: ``0.995``).
            gng_error_discount: float
                Error discount factor which is applied during error computation.
                Only affects the nodes for which the error is being updated 
                (default: ``1.``).
            gng_after_split_error_decay: float
                Error discount factor which is applied after a new node is inserted.
                Only affects the nodes that are neighbours of the new node (default: ``0.5``).
            gng_max_edge_age: int
                Maximum age of edges (default: ``50``).
            gng_max_node_error: float
                Maximum error of nodes (default: ``np.inf``).
            gng_steps_before_max_node_error_is_removed: int
                Number of steps (iterations) to wait before nodes are removed 
                from the GNG due to their node error (default: ``0``).
            gng_edge_entanglement_restriction: bool
                Whether to use or not to use additional restrictions for new 
                edges to limit the entanglement of the nodes (default: ``True``).
            early_stopping: tuple[int, int] | tuple[None, None] | None, optional
                Early stopping criteria. Expects a number of epochs without improvement 
                to wait and a minimum number of epochs to run the fitting
                (default: ``(None, None)``).
            early_stopping_min_improvement: float
                Minimum improvement after an epoch (default: ``1e-6``).
            early_stopping_monitor: str
                Early stopping monitor variable. Expects the name of the metric 
                to be used for the early stopping.
                (default: ``mean_dist``).
            early_stopping_goal: str
                Defines whether the monitor variable should be minimized or maximized
                (default: ``minimize``).
            add_node_every_n_steps: int
                Number of steps after which is checked whether a new node needs
                to be inserted (default: ``100``).
            remove_high_error_nodes_every_n_steps: int
                Number of steps after which is checked whether high error nodes
                need to be removed (default: ``1000``).
            eval_func: list
                A list of callables taking the nodes and adj_M tensor and returning
                a dict with performance mearsurements (default: ``None``).
            gng_batch_size_norm: str
                Type of the batch size normalization method. One of BOTH, CYCLE,
                WITHOUT or AGE (default: ``BOTH``).
            torch_num_threads: int, optional
                Number of CPU cores PyTorch will use for the computations 
                (default: ``all cores available``).
            dtype: str, Type
                Dtype to use for PyTorch tensors (default: ``torch.float32``).
        """
        # Sanity checks
        assert initial_gng_size > 0
        assert initial_gng_size <= max_gng_size
        assert batch_size > 0
        assert device in ["cpu", "cuda"]
        assert gng_node_min_distance_for_update >= 0
        assert 0 <= gng_move_closest_neurons_discount <= 1
        assert 0 <= gng_move_closest_neurons_hop2_discount <= 1
        assert 0 <= gng_error_decay <= 1
        assert 0 <= gng_error_discount <= 1
        assert 0 <= gng_after_split_error_decay <= 1
        assert gng_max_edge_age >= 0
        assert gng_max_node_error >= 0
        assert gng_steps_before_max_node_error_is_removed >= 0
        assert early_stopping_min_improvement >= 0
        assert add_node_every_n_steps >= 0
        assert check_edges_every_n_steps >= 0
        assert remove_high_error_nodes_every_n_steps >= 0
        assert gng_batch_size_norm in ["BOTH", "CYCLE", "WITHOUT", "AGE"]
        assert early_stopping_goal in ["minimize","maximize"]

        if isinstance(dtype, str):
            if dtype.lower() in ["float", "float32"]:
                dtype = torch.float32
            elif dtype.lower() in ["double", "float64"]:
                dtype = torch.double
            else:
                raise ValueError(f'Unknown dtype {dtype}')
                assert dtype == torch.float32 or dtype == torch.double or dtype == torch.float64, f'Unsupported dtype {dtype}'
        self.dtype = dtype
        
        # General settings
        self.dtype = dtype
        self.n_features = n_features
        if eval_func is not None and not isinstance(eval_func, list):
            eval_func = [eval_func]
        self.eval_func = eval_func

        # Early stopping
        if "acc" in early_stopping_monitor.lower() and early_stopping_goal == "minimize":
            raise ValueError("Monitor is ACC while goal is minimize?")

        self.early_stopping_epochs = early_stopping[0] if early_stopping is not None else None
        self.early_stopping_min_epochs = early_stopping[1] if early_stopping is not None else None
        self.early_stopping_min_improvement = early_stopping_min_improvement
        self.early_stopping_monitor = early_stopping_monitor
        self.early_stopping_goal = early_stopping_goal

        # GNG settings
        self.batch_size = batch_size
        self.initial_gng_size = initial_gng_size
        self.max_gng_size = max_gng_size
        self.gng_batch_size_norm = gng_batch_size_norm

        self.gng_node_min_distance_for_update = gng_node_min_distance_for_update
        self.gng_num_closest_neurons = gng_num_closest_neurons
        self.gng_move_closest_neurons_discount = gng_move_closest_neurons_discount
        self.gng_move_closest_neurons_hop2_discount = gng_move_closest_neurons_discount * gng_move_closest_neurons_hop2_discount # GNG_MOVE_CLOSEST_NEURONS_K2_DISCOUNT

        self.gng_error_decay = gng_error_decay
        self.gng_error_discount = gng_error_discount
        self.gng_after_split_error_decay = gng_after_split_error_decay

        self.gng_max_edge_age = gng_max_edge_age
        self.gng_max_node_error = gng_max_node_error
        self.gng_steps_before_max_node_error_is_removed = gng_steps_before_max_node_error_is_removed

        self.gng_edge_entanglement_restriction = gng_edge_entanglement_restriction
        if self.gng_edge_entanglement_restriction:
            assert self.gng_num_closest_neurons == 2, "Edge entanglement restrictions are not implemented for gng_num_closest_neurons != 2"

        if self.gng_batch_size_norm in ["BOTH", "CYCLE"]:
            self.add_node_every_n_steps = max(add_node_every_n_steps // max(np.sqrt(batch_size), 1), 1)
            self.check_edges_every_n_steps = max(check_edges_every_n_steps // max(np.sqrt(batch_size), 1), 1)
            self.remove_high_error_nodes_every_n_steps = max(remove_high_error_nodes_every_n_steps // max(np.sqrt(batch_size), 1), 1)
        else:
            self.add_node_every_n_steps = add_node_every_n_steps
            self.check_edges_every_n_steps = check_edges_every_n_steps
            self.remove_high_error_nodes_every_n_steps = remove_high_error_nodes_every_n_steps

        # Compute settings
        if device == "cuda":
            self.cdist_compute_mode = 'donot_use_mm_for_euclid_dist'
        else:
            self.cdist_compute_mode = 'use_mm_for_euclid_dist_if_necessary'
        self.gng_dist_func = gng_dist_func
        self.device = device

        if torch_num_threads is None:
            self.torch_num_threads = multiprocessing.cpu_count()
        else:
            self.torch_num_threads = torch_num_threads

        # Set seeds
        if random_seed is not None:
            self.random_seed = random_seed
            torch.manual_seed(self.random_seed)
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)

        # Runtime settings
        self.step = 1
        self.gng_is_initialised = False
        self.gng_is_not_converged = True
        self.nodes = None
        self.adj_M = None

        self.best_metric = None
        self.best_dist = np.inf
        self.best_error = None
        self.best_nodes = None
        self.best_adj_M = None
        self.best_age = None

        self.progress_plot = None


    def get_config(self):
        """
        Returns the GNG configuration as python dictionary
        
        Returns
        -------
        dict
            GNG configuration        
        """
        _exclude_runtime_vars = ["nodes",
                                 "age",
                                 "adj_M",
                                 "error",
                                 "gng_dist_func",
                                 "progress_plot",
                                 "eval_func"]
        return {k:str(self.__dict__[k]) for k in self.__dict__ if k not in _exclude_runtime_vars}

    def __repr__(self):
        """
        Returns the representation of the GNG as JSON dump
        
        Returns
        -------
        str
            String representation        
        """
        return json.dumps(self.get_config(), indent=4)

    def print_config(self):
        """
        Prints the string representation of the GNG

        Returns
        -------
        None        
        """
        print(self.__repr__())

    def _to(self,
            device: str):
        """
        Moves the GNG to CPU or CUDA device
        
        Returns
        -------
        None
        """
        self.nodes = self.nodes.to(device)
        self.adj_M = self.adj_M.to(device)
        self.error = self.error.to(device)
        self.age = self.age.to(device)

    def _init_gng(self,
                  data: torch.Tensor):
        """
        Initializes all required GNG tensors. Only called internaly in the fit function.
        
        Parameters
        ----------
        data: torch.Tensor 
            Batch of data samples
        
        Returns
        -------
        None
        
        """
        if len(data[0]) != self.n_features:
            raise ValueError("Datas features dimension does not match GNGs feature dimension")

        # sample 3 times as much data points from the dataset as initial nodes in the gng
        nodes = np.random.choice(range(len(data)), size=self.initial_gng_size*3, replace=False)
        # in groups of 3 data points, take the mean and use the mean as one intial GNG node
        nodes = data[nodes].reshape((self.initial_gng_size, -1, self.n_features)).mean(axis=1)
        self.nodes = nodes.clone().to(self.dtype)

        # zero init the adjencency matrix, age and error tensor
        adj_M = np.zeros((self.initial_gng_size, self.initial_gng_size), dtype=bool)
        self.adj_M = torch.Tensor(adj_M).bool()
        self.error = torch.zeros((self.initial_gng_size,)).to(self.dtype)
        self.age = torch.zeros_like(self.adj_M).to(self.dtype)

        self._to(self.device)
        self.gng_is_initialised = True
        self.first_drop = True

    def _update_adjM(self,
                    order):
        """
        Updates the adjacency matrix of the GNG

        Original work by Bernd Fritzke suggests to connect corresponding nodes
        without additional conditions:
        Fritzke, Bernd. "A growing neural gas network learns topologies." 
        Advances in neural information processing systems 7 (1994).

        However, recent work shows, that additional conditions can significantly
        improve the disentanglement of the embedded GNG nodes. 
        E.g. https://doi.org/10.1016/j.bdr.2021.100254

        Parameters
        ----------           
            order (torch.Tensor):Result of the distance function. Holds a NxS tensor, where 
            N corresponds to the samples in the batch and s to both, the closest
            and second-closest neurons in the GNG.

        Returns
        -------
        None

        """
        if self.gng_edge_entanglement_restriction and self.adj_M.any():
            # according to https://doi.org/10.1016/j.bdr.2021.100254
            # the following 5 additional conditions greatly enhance the
            # disentanglement of the embedded nodes

            # connect two nodes, A and B, only if all conditions are true:
            # 1. condition: A and B have at least one neighbor in common; a bridge
            mask = torch.logical_and(self.adj_M[order][:,0], self.adj_M[order][:,1])
            has_bridge = mask.any(dim=1)

            filtered_mask = has_bridge # Only create an edge if A and B have a common neighbour (bridge)

            if has_bridge.any():
                # 2. condition: A and B have no more than two brdiges
                two_or_less_bridges = mask.sum(dim=1) <= 2
                filtered_mask *= two_or_less_bridges
    
                # 3. condition: in case of 2 bridges: no connection between bridges
                two_bridges = mask.sum(dim=1) == 2
                if two_bridges.any():
                    bridges = torch.where(mask[two_bridges])
                    assert len(bridges[0]) == 2*len(torch.unique(bridges[0])) ### REMOVE after testing
                    bridges = bridges[1].reshape(-1,2)
                    two_bridges[two_bridges.clone()] = self.adj_M[bridges[:,0], bridges[:,1]]
                    filtered_mask *= ~two_bridges

                #4. condition: in case of 1 bridge: A and B should not have more than 1 common neighbour
                one_bridge = mask.sum(dim=1) == 1
                if one_bridge.any():
                    bridges = torch.where(mask[one_bridge])
                    assert len(bridges[0]) == len(torch.unique(bridges[0])) ### REMOVE after testing
                    bridges = bridges[1].unsqueeze(-1).repeat(1,2).unsqueeze(-1)
                    common_bridge_neighbours = torch.concat((order[one_bridge].unsqueeze(-1), bridges), dim=-1)
                    common_bridge_neighbours = self.adj_M[common_bridge_neighbours]
                    common_bridge_neighbours = common_bridge_neighbours[:,:,0]*common_bridge_neighbours[:,:,1]
                    common_bridge_neighbours = (common_bridge_neighbours.sum(dim=-1) >= 2).any(dim=-1)
                    one_bridge[one_bridge.clone()] = common_bridge_neighbours
                    filtered_mask *= ~one_bridge

            #5. condition: second closest neuron belongs to a "small network"
            # Not implemented due to computational complexity

            # filter edges
            filtered_order = order[filtered_mask]

            # set new edges
            self.adj_M[filtered_order[:,0],filtered_order[:,1]] = True
            self.adj_M[filtered_order[:,1],filtered_order[:,0]] = True

        else:
            # set new edges without checking any condition
            self.adj_M[order[:,0],order[:,1]] = True
            self.adj_M[order[:,1],order[:,0]] = True

    def get_neighbourhood(self,
                          n_idx: torch.Tensor):
        """
        Returns the indices of all direct neighbours of n_idx.
        
        Parameters
        ----------
        n_idx: torch.Tensor
            Node for which the neighborhood is to be returned
        
        Returns
        -------
        torch.Tensor
            Indices-tensor of all direct neighbours of n_idx
        
        """
        return torch.where(self.adj_M[n_idx])

    def _update_age(self,
                   winner: torch.Tensor,
                   second_winner: torch.Tensor,
                   neighbourhood: torch.Tensor):
        """
        Updates edge ages according to the corresponding winner (closest) and
        second winner (second closest) GNG neuron.
        
        Parameters
        ----------
        winner: torch.Tensor
            NxS tensor, where N corresponds to the samples in the batch and s the closest GNG neuron
        second_winner: torch.Tensor
            NxS tensor, where N corresponds to the samples in the batch and s the second-closest GNG neuron
        neighbourhood: torch.Tensor
            Indices-tensor of all direct neighbours of N
        
        Returns
        -------
        None
        """
        # step 1: increment the age of all edges connected to the winner
        src = winner[neighbourhood[0][neighbourhood[1] == 0]]
        dst = neighbourhood[2][neighbourhood[1] == 0]

        mask = torch.zeros_like(self.age).bool()
        mask[src, dst] = True
        mask[dst, src] = True

        if self.gng_batch_size_norm in ["BOTH", "AGE"]:
            self.age += (mask / np.sqrt(self.batch_size))
        else:
            self.age += mask

        # step 2: set age between winner and second-winner to zero
        mask = torch.zeros_like(self.age).bool()
        mask[winner, second_winner] = True
        mask[second_winner, winner] = True
        self.age[mask] = 0

    def _drop_old_edges(self):
        """
        Drops all edges with an age above the threshold and changes the adjacency matrix
        accordingly.
        
        Returns
        -------
        None
        """
        # identify edges with age > gng_max_edge_age
        _edges_to_be_deleted = torch.where(self.age >= self.gng_max_edge_age)
        if len(_edges_to_be_deleted[0]) > 0:
            self.adj_M[_edges_to_be_deleted] = False
            self.age[_edges_to_be_deleted] = 0

    def _drop_nodes_without_edge(self):
        """
        Drops all nodes without any edge and changes the gng tensors (nodes, error, age) and
        the adjacency matrix accordingly.

        Returns
        -------
        None        
        """
        # identify nodes without edges and drop all
        _nodes_to_be_deleted = torch.where(self.adj_M.sum(dim=0) == 0)[0]

        if self.first_drop:
            if len(_nodes_to_be_deleted) > 0:
                print(f'Dropped {len(_nodes_to_be_deleted):d} untouched nodes which were created during initialization, but never have been the nearest neighbour to any data sample.')
            self.first_drop = False

        if len(_nodes_to_be_deleted) > 0:
            mask = torch.isin(torch.arange(len(self.nodes),device=self.device), _nodes_to_be_deleted)
            self.nodes = self.nodes[~mask]
            self.error = self.error[~mask]
            self.adj_M = self.adj_M[~mask][:,~mask]
            self.age = self.age[~mask][:,~mask]

    def _drop_nodes(self):
        """
        Drops all nodes that exceed the maximal node error and changes the 
        gng tensors (nodes, error, age) and the adjacency matrix accordingly.

        Returns
        -------
        None        
        """
        # drop all nodes exceed the maximum error
        mask = self.error > self.gng_max_node_error
        ids_to_drop = torch.where(mask)[0]

        # delete all exisitng edges
        self.adj_M[ids_to_drop] = False
        self.adj_M[:,ids_to_drop] = False

        self.nodes = self.nodes[~mask]
        self.error = self.error[~mask]
        self.adj_M = self.adj_M[~mask][:,~mask]
        self.age = self.age[~mask][:,~mask]

    def _add_new_node_to_adj_M(self,
                              baseA: torch.Tensor,
                              baseB: torch.Tensor):
        """
        Adds a new node between baseA and baseB to the GNGs adjacency matrix.
        Therefore, baseA and baseB will be the only neighbours of the new node

        Parameters
        ----------
        baseA: torch.Tensor
            Node ID
        baseB: torch.Tensor
            Node ID

        Returns
        -------
        None        
        """
        _new = torch.zeros((1,len(self.adj_M)), device=self.device).bool()
        _new[0,baseA] = True
        _new[0,baseB] = True
        self.adj_M = torch.concat((self.adj_M, _new))

        _new = torch.zeros((len(self.adj_M), 1), device=self.device).bool()
        _new[baseA,0] = True
        _new[baseB,0] = True
        _new[-1,0] = False
        self.adj_M = torch.concat((self.adj_M, _new), dim=1)

    def _add_new_node_between(self,
                             nA: torch.Tensor,
                             nB: torch.Tensor):
        """
        Inserts a new node between nA and nB

        Parameters
        ----------
        nA: torch.Tensor
            Node ID
        nB: torch.Tensor 
            Node ID

        Returns
        -------
        None        
        """
        # Add a new node in between of nA and nB (linear interpolation)
        assert nA != nB
        new_node = .5 * (self.nodes[nA] + self.nodes[nB])
    
        self.nodes = torch.concat((self.nodes, new_node.unsqueeze(0)))
        self.error[nA] *= self.gng_after_split_error_decay
        self.error[nB] *= self.gng_after_split_error_decay
    
        self.adj_M[nA][nB] = False
        self.adj_M[nB][nA] = False
    
        self._add_new_node_to_adj_M(nA, nB)

        self.age = torch.concat((self.age, torch.zeros(1,len(self.age),device=self.device)))
        self.age = torch.concat((self.age, torch.zeros(len(self.age),1,device=self.device)), dim=1)
    
        self.error = torch.concat((self.error, torch.zeros((1,), device=self.device)))

    def _drop_no_change_sample(self,
                              batch: torch.Tensor,
                              _dist: torch.Tensor,
                              _order: torch.Tensor):
        """
        Drops all nodes from the batch that do not exceed the minimal change 
        threshold.

        Parameters
        ----------
        batch: torch.Tensor
            Batch tensor
        _dist: torch.Tensor
            Nx2 tensor representing the numerical distance to each sample and 
            its both closest GNG nodes. 
        _order: torch.Tensor
            Contains an Nx2 tensor, where N are the samples in the batch with 
            respect to their two closest node IDs

        Returns
        -------
        torch.Tensor 
            filtered batch tensor
        torch.Tensor 
            filtered _dist tensor
        torch.Tensor
            filtered _order tensor
        
        """
        _mask = (torch.gather(_dist, 1, _order[:,0].view((-1,1))) > self.gng_node_min_distance_for_update).squeeze()
        return batch[_mask], _dist[_mask], _order[_mask]

    def _on_train_end(self):
        """
        Always called after training is completed. Restores the best GNG, e.g.
        after early stopping.

        Returns
        -------
        None        
        """
        # Setting GNG back to the best GNG
        print('Restoring GNG with best performance')
        self.error = self.best_error
        self.nodes = self.best_nodes
        self.adj_M = self.best_adj_M
        self.age = self.best_age


    def _update_progress(self,
                        metrics: dict,
                        save_path: str | None = None,
                        fig_title: str | None = None):
        """
        Updates the progress plot after every epoch. 

        Parameters
        ----------
        metrics: dict
            Dictionary holding all performance measurements per epoch. Each 
            measurement corresponds to one key in the dict.
        epoch: int
            Finished epoch
        save_path: str, optional
            Save path
        fig_title: str, optional
            Figure title

        Returns
        -------
        None        
        """
        def set_title(fig, **kwargs):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if fig_title is not None:
                    fig.suptitle(f'{fig_title}', fontsize=16)
                fig.tight_layout()
                if save_path is not None:
                    fig.savefig(path.join(save_path, "metrics_plot.jpg"))
                plt.show()
                plt.close(fig) # <<--- check if this fixes the memory leak == memory leak fixed!
        if self.progress_plot is None:
            self.progress_plot = PlotLosses(outputs=[MatplotlibPlot(after_plots=set_title)])

        self.progress_plot.update(metrics)
        self.progress_plot.send()


    def _last_epoch_is_better(self,
                              metrics: dict):
        """
        Checks if the last epoch is the new best epoch

        Parameters
        ----------
        metrics: (dict): 
            Dictionary holding all performance measurements per epoch. Each 
            measurement corresponds to one key in the dict.

        Returns
        -------
        bool
            True if the current epoch is the new best epoch
        """
        if self.best_metric is None:
            return True

        if self.early_stopping_monitor not in metrics:
            raise KeyError(f"Early Stopping monitor variable \"{self.early_stopping_monitor}\" was not found in metrics dict: {metrics}")

        if self.early_stopping_goal == "minimize":
            if (metrics[self.early_stopping_monitor] + self.early_stopping_min_improvement) < self.best_metric:
                return True
            else:
                return False
        elif self.early_stopping_goal == "maximize":
            if (metrics[self.early_stopping_monitor] - self.early_stopping_min_improvement) > self.best_metric:
                return True
            else:
                return False

    def fit(self,
            data: torch.Tensor,
            steps: int | None = None,
            epochs: int | None = None,
            save_progress: bool = False,
            save_path: str | None = None,
            progress_plot_title: str | None = None,
            verbose: bool = False):
        """
        Fits the GNG to the given dataset. 

        Parameters
        ----------
        data (torch.Tensor): Data
        steps (int, optional): Stop training after $steps$ iterations (batches).
        epoch (int, optional): Stop training after $epochs$ epochs.
        save_progress (bool): Save progress plot of the training
        save_path (str, optional): Save path
        progress_plot_title (str, optional): Figure title of progress plot        
        verbose (bool, optional): Adjusts verbosity

        Returns
        -------
        pandas.Series
            Pandas series hold all relevant metrics of the best epoch
        """

        if save_progress and save_path is None:
            raise ValueError("If the progress is to be saved, a save_path must be specified, but got None")

        if self.device == "cpu":
            torch.set_num_threads(self.torch_num_threads)

        if not self.gng_is_initialised:
            self._init_gng(data)
            print('Init GNG with config:')
        else:
            if self.nodes is None or len(self.nodes) <= 0:
                print('Corrupt node data. Recreate GNG.')
                return
            else:
                if verbose:
                    print('GNG is already initialised. Continuing training with config:')
                else:
                    print('GNG is already initialised.')
        if verbose:
            self.print_config()

        data_loader = DataLoader(data, batch_size=self.batch_size, shuffle=True)
        n_features = torch.Tensor([data.shape[-1]]).to(self.device)
        n_features_norm = torch.sqrt(n_features.clone()).to(self.device)

        stop_training = False
        patience = 0
        best_epoch = -1
        epoch_duration = []

        for e in range(epochs):
            print('______________________________________')
            print(f'Starting epoch {e}:')
            avg_loss = {'mean_dist':[], 'mean_node_error':[], 'mean_node_age':[], 'mean_edge_count':[]}
            converged_flag = True

            epoch_start = datetime.now()
            progress = tqdm(data_loader)
            for j,b in enumerate(progress):
                # Load batch
                b = b.to(self.device).to(self.dtype)

                ###############################################################
                ## Compute distances between data points and gng
                # GNG and CUDAs cdist implementation: It seems that a lack of precision in the float
                # calculation leads to iterativ, partially random and undeterministic results.
                # There are two solutions for this:
                #   - Stick with fp32 and set "donot_use_mm_for_euclid_dist = True"
                #   - Use with fp64 (or both)
                # Howerver, using fp64 obviously lead to performance limitations

                # Furthermore, dividng by `` n_features_norm = torch.sqrt(n_features)``
                # improves distance stability in particular for high dimensional datasets
                _dist = self.gng_dist_func(b, self.nodes, compute_mode=self.cdist_compute_mode) / n_features_norm
                ###############################################################

                # find closest neurons and select the closest
                _order = torch.argsort(_dist)[:,:self.gng_num_closest_neurons]

                # compute the mean distance of the whole batch (just for evaluation purpose)
                _gng_batch_dist = torch.gather(_dist, 1, _order[:,0].view((-1,1))).mean()

                # drop nodes with a distance below the minimal threshold
                if self.gng_node_min_distance_for_update > 0:
                    b, _dist, _order = self._drop_no_change_sample(b, _dist, _order)
                    if len(b) == 0:
                        continue
                # if at least one node needs to be adapted, the gng is not converged
                converged_flag = False

                # print progress in tqdm-bar
                progress.set_description_str(f'Mean distance: {_gng_batch_dist:0.6f}')

                # update the adjacency matrix
                self._update_adjM(_order)

                # get the neighbourhood for each of the nodes
                _neighbourhood = self.get_neighbourhood(_order)

                # update the age of the edges
                self._update_age(_order[:,0], _order[:,1], _neighbourhood)

                ###############################################################
                ## Compute node shift for hop-1 neighbourhood
                _delta_shape = (len(self.nodes), self.n_features)
                _delta = self.gng_move_closest_neurons_discount * (b.unsqueeze(1).repeat(1,2,1) - self.nodes[_order])
                _delta = torch.zeros(_delta_shape, dtype=self.dtype, device=self.device).scatter_reduce_(dim=0,
                                                                                                         index=_order.reshape((-1,1)).repeat((1,_delta.shape[-1])),
                                                                                                         src=_delta.view((-1, _delta.shape[-1])),
                                                                                                         reduce="mean",
                                                                                                         include_self=False)

                ## Compute node shift for hop-2 neighbourhood
                _delta_hop2 = self.gng_move_closest_neurons_hop2_discount * (b[_neighbourhood[0]] - self.nodes[_neighbourhood[2]])
                _delta_hop2 = torch.zeros(_delta_shape, dtype=self.dtype, device=self.device).scatter_reduce_(dim=0,
                                                                                                              index=_neighbourhood[2].unsqueeze(1).repeat(1,_delta.shape[-1]),
                                                                                                              src=_delta_hop2,
                                                                                                              reduce="mean",
                                                                                                              include_self=False)
                _delta += _delta_hop2


                # Update nodes
                self.nodes += _delta

                ###############################################################
                ## Compute node error (error ist just computed for the closest and hop-2 nodes)
                _error_delta = torch.gather(_dist, 1, _order)
                _error_delta = torch.zeros((len(self.nodes),), dtype=self.dtype, device=self.device).scatter_reduce_(dim=0,
                                                                                                                     index=_order.flatten(),
                                                                                                                     src=_error_delta.flatten(),
                                                                                                                     reduce="mean",
                                                                                                                     include_self=False)

                # Compute error of hop-2 nodes
                _dist_hop2 = _dist[_neighbourhood[0],_neighbourhood[2]]
                _error_delta_hop2 = torch.zeros((len(self.nodes),), dtype=self.dtype, device=self.device).scatter_reduce_(dim=0,
                                                                                                                          index=_neighbourhood[2].flatten(),
                                                                                                                          src=_dist_hop2.flatten(),
                                                                                                                          reduce="mean",
                                                                                                                          include_self=False)

                # Update error
                self.error += self.gng_error_discount * (_error_delta + .5*_error_delta_hop2)

                ###############################################################
                ## Drop all nodes with an error > MAX_ERROR
                if self.step % self.remove_high_error_nodes_every_n_steps == 0 and self.step > self.gng_steps_before_max_node_error_is_removed:
                    if self.gng_max_node_error < np.inf:
                        self._drop_nodes()
                        self._drop_nodes_without_edge()

                ###############################################################
                ## - Check if edges need to be deleted due to their age
                ## - Delete nodes without any edge
                if self.step % self.check_edges_every_n_steps == 0:
                    self._drop_old_edges()
                    self._drop_nodes_without_edge()

                if len(self.nodes) <= 0: #"GNG is empty, no nodes left."
                    stop_training = True
                    break
                ###############################################################
                ## Add new node if gng is smaller than max_gng_size
                if self.step % self.add_node_every_n_steps == 0 and len(self.nodes) < self.max_gng_size:
                    # find node with highest error
                    _new_node_A = self.error.argmax()
                    # and its neighbour with the highest error
                    _new_node_B = torch.where(self.adj_M[_new_node_A])[0]
                    _new_node_B = _new_node_B[self.error[_new_node_B].argmax()]
                    self._add_new_node_between(_new_node_A, _new_node_B)

                self.error *= self.gng_error_decay
                self.step += 1


                avg_loss['mean_dist'].append(_gng_batch_dist.cpu().numpy())
                avg_loss['mean_node_error'].append(self.error.mean().cpu().numpy())
                avg_loss['mean_node_age'].append(self.age[self.adj_M].mean().cpu().numpy())
                avg_loss['mean_edge_count'].append(self.adj_M.sum().cpu().numpy()/2)

                if steps is not None and self.step >= steps:
                    stop_training = True
                    break

            ##### Post epoch --------------------------------------------------
            ## Update progress bar
            epoch_duration.append((datetime.now() - epoch_start).total_seconds())

            if converged_flag:
                print(f'Training converged due to the minimal distance for an update: gng_node_min_distance_for_update: {self.gng_node_min_distance_for_update}')
                stop_training = True

            ## Run eval callbacks
            if self.eval_func is not None and len(self.nodes) > 0:
                eval_results = [e(self.nodes, self.adj_M) for e in self.eval_func]
                metrics = {}
                for er in eval_results:
                    if not set(er.keys()).isdisjoint(metrics.keys()):
                        print(f'WARNING: OVERWRITING FOLLOWING METRIC KEYS: {[k for k in er if k in metrics]}')
                    metrics.update(er)
            else:
                metrics = {}

            metrics.update({'mean_dist': np.mean(avg_loss['mean_dist']),
                            'mean_node_error': np.mean(avg_loss['mean_node_error']),
                            'mean_node_age': np.mean(avg_loss['mean_node_age']),
                            'mean_edge_count': np.mean(avg_loss['mean_edge_count']),
                            '#nodes': len(self.nodes),
                            'epoch duration': epoch_duration[-1]})


            ## Check early stopping criteria
            if len(self.nodes) > 0 and self._last_epoch_is_better(metrics):
                print(f'ES >> Found new best improved from {self.best_metric} to {metrics[self.early_stopping_monitor]}')
                self.best_metric =  metrics[self.early_stopping_monitor]
                self.best_dist = metrics["mean_dist"]
                self.best_error = self.error.clone()
                self.best_nodes = self.nodes.clone()
                self.best_adj_M = self.adj_M.clone()
                self.best_age = self.age.clone()
                best_epoch = e
                patience = 0
            elif len(self.nodes) > 0 and self.early_stopping_min_epochs is not None:
                if e > self.early_stopping_min_epochs:
                    if patience > self.early_stopping_epochs:
                        print(f'ES >> Stopping training due to {self.early_stopping_epochs} epochs without improvement')
                        stop_training = True
                patience += 1

            ## Update & save progress plot
            if save_progress:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                self._update_progress(metrics,
                                     save_path=save_path if save_path is not None else None,
                                     fig_title=progress_plot_title)

            ## Check fit stopping criteria
            if (epochs is not None and e >= epochs) or stop_training:
                print(f'Finished training after {e} epochs at step {self.step} - summary:\n#Nodes: {len(self.nodes)}\n#Steps: {self.step}\n#Edges: {int(self.adj_M.sum()/2):d}')
                print(f'Mean dist: {metrics["mean_dist"]:0.2f}\nMean error: {metrics["mean_node_error"]:0.2f}\nMean age: {metrics["mean_node_age"]:0.2f}\n\n')
                break
            else:
                print(f'Epoch {e} - summary:\n#Nodes: {len(self.nodes)}\n#Steps: {self.step}\n#Edges: {int(self.adj_M.sum()/2):d}\n')
                print(f'Mean dist: {metrics["mean_dist"]:0.2f}\nMean error: {metrics["mean_node_error"]:0.2f}\nMean age: {metrics["mean_node_age"]:0.2f}')
                if len(self.nodes) == 0:
                    print('Growing Neural Gas collapsed - retry with different parameters, e.g. higher "gng_max_node_error".')


        ##### Post training ---------------------------------------------------
        ## Update & save progress plot
        if self.best_nodes is not None and len(self.best_nodes) > 0 and save_path is not None:
            if self.eval_func is not None:
                # Apply all eval callbacks
                best_eval_results = [e(self.best_nodes, self.best_adj_M) for e in self.eval_func]
                best_eval_metrics = {}
                for er in best_eval_results:
                    if not set(er.keys()).isdisjoint(best_eval_metrics.keys()):
                        print(f'WARNING: OVERWRITING FOLLOWING METRIC KEYS: {[k for k in er if k in best_eval_results]}')
                    best_eval_metrics.update(er)
            else:
                best_eval_metrics = {}
            result = pd.Series(dict(**{"best_mean_dist": self.best_dist,
                                       "best_mean_error": self.best_error.mean().cpu().numpy(),
                                       "best_mean_age": self.best_age[self.best_adj_M].mean().cpu().numpy(),
                                       "num_nodes": len(self.best_nodes),
                                       "edges": int(self.best_adj_M.sum()/2),
                                       "best_epoch": best_epoch,
                                       "mean epoch duration": np.array(epoch_duration)}, **best_eval_metrics))

            if save_progress and save_path is not None:
                result.to_csv(path.join(save_path, 'result.csv'), header=False)

        ## Set GNG back to the best GNG
        self._on_train_end()
        return result