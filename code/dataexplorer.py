from datautils import load_corr_clinical
from fileutils import data_path
import os
from vae import load_latents
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_latent_type(df, latent='mu'):
    mus = df[df.columns[df.columns.to_series().str.contains(latent)]]
    mu_min, mu_max = mus.min().min(), mus.max().max()
    print(mu_min, mu_max)
    x_axis = np.linspace(mu_min, mu_max, num=20)
    n_mus = len(mus.columns)
    mu_hist = mus.agg(lambda x : np.histogram(x, bins=x_axis, density=True), axis=0)
    mu_hist = mu_hist.iloc[[0]]
    mu_hist_list = []
    for col in mu_hist.columns:
        mu_hist_list.append(mu_hist[col][0])
    mu_hist = np.array(mu_hist_list)

    ##print(mu_hist)

    fig, ax1 = plt.subplots(1,1,figsize=(16,12), dpi=100)
    ax1.set_xticks(x_axis[::2])
    ax1.set_yticks(np.linspace(0,n_mus,5))
    ax1.set_ylabel("Latent {} Variable".format(latent))
    ax1.set_xlabel("Value of {}".format(latent))

    im = ax1.imshow(mu_hist, interpolation='nearest', origin='lower', aspect='auto')

    plt.show()
    
type_list = ['RS', 'T1', 'DTI']

for type_name in type_list:
    df = load_latents('control_{}'.format(type_name), fname_tag='both')

    plot_latent_type(df)
    plot_latent_type(df, latent='logvar')


