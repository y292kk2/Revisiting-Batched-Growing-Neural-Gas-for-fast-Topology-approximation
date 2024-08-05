#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 16:57:09 2024

@author: balada
"""


import numpy as np

from pyGNG import GrowingNeuralGas
import networkx as nx
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

def plot_graph(nodes, adj_M, fsuptitle=None, data=None, batch_size=128, plot_sping_layout=False, spring_it=100000, spring_k=100, select_connected_components=None):
    assert len(data.shape) >= 3
    assert len(nodes.shape) == 2

    # Predict nearest neighbor image
    dist_n2n = torch.cdist(nodes, nodes).cpu().numpy()

    if data is not None:
        dist_n2d = []
        for n in tqdm(nodes):
            dist_n2d.append(torch.cdist(n.float().unsqueeze(0), data.view(data.shape[0], -1)).argmin(axis=1))
        dist_n2d = torch.concat(dist_n2d).cpu()
        node_images = data.cpu()[dist_n2d]


    # Translate GNG to NetworkX Graph
    nxg = nx.Graph()
    weights = dist_n2n[np.triu(adj_M.bool(), k=1)]
    weights = weights - weights.min()
    weights = weights / weights.max()
    for e1,e2,w in zip(*np.where(np.triu(adj_M, k=1)), weights):
        nxg.add_edge(e1, e2, weight=w)

    if plot_sping_layout:
        # Plot a nx.SpringLayout for each individual connected component
        cc = [(i,c) for i,c in enumerate(nx.connected_components(nxg))]

        if select_connected_components is not None:
            cc = [c for i,c in enumerate(cc) if i in select_connected_components]

        cols = int(np.ceil(np.sqrt(len(cc))))
        rows = (cols-1) if (cols-1)*cols >= len(cc) else cols

        fig, axs = plt.subplots(rows, cols, figsize=(10*cols, 10*rows), dpi=320)

        if isinstance(axs, np.ndarray):
            axs = axs.flatten()
        else: axs = [axs]

        cc += [(None,None)]*(len(axs)-len(cc))

        for (i,subgraph), ax in tqdm(zip(cc, axs)):
            if subgraph is None:
                ax.axis("off")
                continue
            nxg_sub = nxg.subgraph(subgraph)
            ax.set_title(f'{i}')
            pos = nx.spring_layout(nxg_sub, weight="weight", seed=42, iterations=spring_it, k=spring_k)

            trans=ax.transData.transform
            trans2=fig.transFigure.inverted().transform
        
            piesize=0.02 # image size
            p2=piesize/2.0
        
            if data is None:
                nx.draw(nxg_sub, pos=pos, ax=ax, node_size=50, width=0.5)
            else:
                nx.draw_networkx_edges(nxg_sub,pos,ax=ax)
                for n in nxg_sub:
                    xx,yy=trans(pos[n]) # figure coordinates
                    xa,ya=trans2((xx,yy)) # axes coordinates
                    a = plt.axes([xa-p2,ya-p2, piesize, piesize])
                    a.set_aspect('equal')
                    a.imshow(node_images[n])
                    a.axis('off')
                ax.axis('off')

        fig.suptitle(fsuptitle)
        return fig


def run(dataset,
        initial_gng_size,
        max_gng_size,
        gng_max_edge_age,
        epochs,
        gng_max_node_error=np.inf,
        random_seed=42,
        batch_size=128,
        early_stopping=(30,20),
        device="cpu",
        save_path=None,
        save_progress=False,
        **kwargs):
    import torch
    from torchvision import datasets
    from torchvision.transforms import ToTensor
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_blobs
    torch.backends.cudnn.allow_tf32 = False

    ###########################################################################
    ###########################################################################
    ## Load Dataset
    ##
    if dataset == "MNIST":
        ## MNIST (raw features)
        val_data = datasets.MNIST(root = 'data',
                                  train = False,
                                  transform = ToTensor(),
                                  download = True)
        val_labels = val_data.targets
        val_data = val_data.data.reshape((10000,28*28)).float()
    
        train_data = datasets.MNIST(root = 'data',
                              train = True,
                              transform = ToTensor(),
                              download = True)
        train_labels = train_data.targets
        train_data = torch.flatten(torch.Tensor(train_data.data), start_dim=1).float()
    ###########################################################################
    elif dataset == "F_MNIST":
        # F_MNIST (raw features)
        val_data = datasets.FashionMNIST(root = 'data',
                                  train = False,
                                  transform = ToTensor(),
                                  download = True)
        val_labels = val_data.targets
        val_data = torch.flatten(torch.Tensor(val_data.data), start_dim=1).float()
    
        train_data = datasets.FashionMNIST(root = 'data',
                              train = True,
                              transform = ToTensor(),
                              download = True)
        train_labels = train_data.targets
        train_data = torch.flatten(torch.Tensor(train_data.data), start_dim=1).float()
    ###########################################################################
    elif dataset == "GOOGLENEWS":
        # GoogleNews Word2Vec vectors
        train_data = np.load("data/googlenews/GN_vectors.npy")
        train_data = torch.Tensor(train_data).float()
    ###########################################################################
    elif dataset == "BLOBS":
        # Blobs
        train_data, train_labels = make_blobs(n_samples=10000, n_features=2, centers=4, cluster_std=[1.0, 2.5, 0.5, 1.5], random_state=42)
        train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)
        train_data = torch.Tensor(train_data).float()
        val_data = torch.Tensor(val_data).float()
        train_labels = torch.Tensor(train_labels)
        val_labels = torch.Tensor(val_labels)


    train_data = train_data - train_data.min()
    train_data = train_data / train_data.max()
    assert train_data.min() == 0 and train_data.max() == 1

    if val_data is not None:
        val_data = val_data - val_data.min()
        val_data = val_data / val_data.max()
        assert val_data.min() == 0 and val_data.max() == 1


    gng = GrowingNeuralGas(n_features = train_data.shape[1],
                           initial_gng_size = initial_gng_size,
                           max_gng_size = max_gng_size,
                           batch_size = batch_size,
                           random_seed = random_seed,
                           device = "cuda",
                           gng_max_edge_age = gng_max_edge_age,
                           gng_max_node_error = gng_max_node_error,
                           early_stopping = early_stopping,
                           **kwargs)

    gng.fit(data=train_data,
            epochs=epochs,
            save_progress=save_progress,
            save_path=save_path,
            progress_plot_title=f'{dataset} - BS{batch_size}')

    return gng, (train_data, val_data)


if __name__ == "__main__":
    dataset="F_MNIST"    
    batch_size=128

    gng, data =  run(dataset=dataset,
                     epochs=100,
                     initial_gng_size=25,
                     max_gng_size=100,
                     gng_max_edge_age=20,
                     gng_max_node_error=50,
                     random_seed=42,
                     batch_size=batch_size,
                     save_progress=True,
                     save_path="./trials/",
                     early_stopping=(30,20),
                     device="cpu",
                     torch_num_threads=12,
                     gng_steps_before_max_node_error_is_removed=300)

    img_shape = (28,28) if dataset in ["MNIST", "F_MNIST"] else None

    fig_train = plot_graph(nodes=gng.nodes.cpu().clone(),
                           adj_M=gng.adj_M.cpu().clone(),
                           fsuptitle=f'TRAIN - {dataset} - BS{batch_size}',
                           data=data[0].cpu().view((-1,)+img_shape),
                           plot_sping_layout=True,
                           spring_it=50)

    fig_val = plot_graph(nodes=gng.nodes.cpu().clone(),
                         adj_M=gng.adj_M.cpu().clone(),
                         fsuptitle=f'VAL - {dataset} - BS{batch_size}',
                         data=data[1].cpu().view((-1,)+img_shape),
                         plot_sping_layout=True,
                         spring_it=50,
                         spring_k=1)

    fig_train.savefig('/results/gng_spring_plot_train.jpg', dpi=320, format="jpg")
    fig_val.savefig('/results/gng_spring_plot_val.jpg', dpi=320, format="jpg")
    np.save('/results/gng_nodes', gng.nodes.cpu().numpy())
    np.save('/results/gng_adj_M', gng.adj_M.cpu().numpy())
