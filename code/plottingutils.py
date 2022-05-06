import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as scistats
from utils import Stats
import os
import numpy as np
import pandas as pd


def plot_loss(net_name, plot_title=None, stat_label_dict=None, save_name=None):
    '''
    For plotting and saving the loss function
    
    params:
        net_name: str
            The name of the network
        plot_title: str (default = None)
            If not None, the title to give the plot
            Otherwise, uses the name of the network
        stat_label_dict: dict (default = None)
            If not None, renames the desired stats (keys)
            to the desired labels(values) in the plot legend
            Otherwise, uses the default stat names for the labels
        save_name: str (default = None)
            If not None name used when saving the plots
            Otherwise uses the net name
    '''
    fig, ax1 = plt.subplots(1,1,figsize=(16,12), dpi=200)
    stat_path = os.path.join("vae_nets", net_name)
    st = Stats(path=stat_path)
    if stat_label_dict is None:
        stat_label_dict = {k : k for k in st.stats.keys}
    stat_list_dict = {}
    for stat_label_orig, stat_label_new in stat_label_dict.items():
        stat_vals = st[stat_label_orig]
        x = list(stat_vals.keys())
        y = list(stat_vals.values())
        if type(y[0]) in [list, tuple]:
            ys = list(zip(*y))
            stat_labels = stat_label_new
            for stat_label_new, y in zip(stat_labels, ys):
                ax1.plot(x, y, label=stat_label_new)
        else:
            ax1.plot(x, y, label=stat_label_new)
    ax1.set_ylabel('Loss Divided by # Features')
    ax1.set_xlabel('Epoch')
    if plot_title is not None:
        ax1.set_title(plot_title)
    else:
        ax1.set_title("{} Loss".format(net_name))
    ax1.legend()
    if save_name is None:
        save_name = net_name
    plots_dir = os.path.join("plots", "loss")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    plt.savefig(os.path.join(plots_dir, "{}.png".format(save_name)))
    plt.show()


def plot_latent_individual(df, type_name, **kwargs):
    '''
    For plotting and saving individual latent distributions
    
    params:
        choices: list[int] (default = None)
            If not None, individuals of the selected indices
            are selected from the dataframe
            Otherwise, they are selected at random
        n_choices: int (default = 5)
            If indices is None, the number of individuals to select
            Otherwise, this argument is not used
        diagnosis: str (default = None)
            If not None, chooses the matching diagnosis type
            ('Control' or 'PTSD')
        sharex: bool (default = False)
            Whether the x axis of plots should be shared
        sharey: bool (default = False)
            Whether the y axis of plots should be shared
        run_id: int (default = 0)
            An indicator used for saving more than one plot
        save_name: str (default = None)
            If not None name used when saving the plots
            Otherwise uses brain scan type and diagnosis     
    '''
    choices = kwargs.get("choices", None)
    n_choices = kwargs.get("n_choices", 5)
    diagnosis = kwargs.get("diagnosis", None)
    sharex = kwargs.get("sharex", False)
    sharey = kwargs.get("sharey", False)
    run_id = kwargs.get("run_id", 0)
    save_name = kwargs.get("save_name", None)
    
    diagnosis_map = {'control':0, 'ptsd':1}
    
    if diagnosis is not None:
        df = df[df['Diagnosis'] == diagnosis_map[diagnosis.lower()]]
    
    if not choices:
        choices = np.random.randint(len(df.index), size=n_choices)
        
    mus = df[df.columns[df.columns.to_series().str.contains('mu')]]
    mus = mus.iloc[choices]
    logvars = df[df.columns[df.columns.to_series().str.contains('logvar')]]
    logvars = logvars.iloc[choices]
    patient_info = df['Site'] + "_" + df['SubjectID']
    patient_info = patient_info.iloc[choices]
    variances = logvars.apply(np.exp)
    
    diag_types = ["Control", "PTSD"]
    
    fig, axs = plt.subplots(len(choices), sharex=sharex, sharey=sharey, figsize=(16,12), dpi=300)
    if not diagnosis:
        fig.suptitle("{} Latents".format(type_name))
    else:
        fig.suptitle("{} {} Latents".format(diagnosis, type_name))
        
    fig.tight_layout()
    
    for i, ax in enumerate(axs):
        mu_curr = mus.iloc[i]
        variance_curr = variances.iloc[i]
        patient = patient_info.iloc[i]
        for mu, var in zip(mu_curr.tolist(), variance_curr.tolist()):
            std_dev = np.sqrt(var)
            x = np.linspace(mu-3*std_dev, mu+3*std_dev, 100)
            y = scistats.norm.pdf(x, mu, std_dev)
            ax.set_ylabel(patient, fontsize=10)
            ax.plot(x, y)
    
    plots_dir = os.path.join("plots", "latents")
    
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        
    if save_name is None:
        if diagnosis is not None:
            save_name = "{}_{}_latent".format(type_name, diagnosis.lower())
        else:
            save_name = "{}_{}_latent".format(type_name)
    save_name = "{}_{}.png".format(save_name, run_id)
    
    plt.savefig(os.path.join(plots_dir, save_name), bbox_inches='tight')
    plt.show()


def plot_latent_average(df, type_name, **kwargs):
    '''
    For plotting and saving averaged latent distributions over desired column
    
    params:
        choices: list[int] (default = None)
            Indices of unique elements in column of interest to be selected
        diagnosis: str  (default = None)
            If not None, chooses the matching diagnosis type
            ('Control' or 'PTSD')
        avg_col: str (default = 'Site')
            The column name over which to average
            ('Site', 'Age', 'Sex', or 'Diagnosis')
        sharex: bool (default = False)
            Whether the x axis of plots should be shared
        sharey: bool (default = False)
            Whether the y axis of plots should be shared
        x_range: tuple[int] (default = (-10, 10))
            The left and right bounds for the x axis
        run_id: int (default = 0)
            An indicator used for saving more than one plot
        save_name: str (default = None)
            If not None name used when saving the plots
            Otherwise uses brain scan type and diagnosis     
    '''
    choices = kwargs.get("choices", None)
    diagnosis = kwargs.get("diagnosis", None)
    avg_col = kwargs.get("avg_col", 'Site')
    sharex = kwargs.get("sharex", False)
    sharey = kwargs.get("sharey", False)
    x_range = kwargs.get("x_range", (-10, 10))
    run_id = kwargs.get("run_id", 0)
    save_name = kwargs.get("save_name", None)
    
    diagnosis_map = {'control':0, 'ptsd':1}
    
    if diagnosis is not None:
        df = df[df['Diagnosis'] == diagnosis_map[diagnosis.lower()]]
        
    df_mean = df.groupby(by=[avg_col]).mean()
        
    mus = df_mean[df_mean.columns[df_mean.columns.to_series().str.contains('mu')]]
    if choices:
        mus = mus.iloc[choices]
    logvars = df_mean[df_mean.columns[df_mean.columns.to_series().str.contains('logvar')]]
    if choices:
        logvars = logvars.iloc[choices]
    patient_info = df_mean.index
    if choices:
        patient_info = patient_info[choices]
    variances = logvars.apply(np.exp)
    
    if choices is None:
        n_plots = len(df_mean.index)
    else:
        n_plots = len(choices)
        
    fig, axs = plt.subplots(n_plots, sharex=sharex, sharey=sharey, figsize=(16,12), dpi=300)
    if not diagnosis:
        fig.suptitle("{} Latents".format(type_name))
    else:
        fig.suptitle("{} {} Latents".format(diagnosis, type_name))
        
    fig.tight_layout()
    
    for i, ax in enumerate(axs):
        mu_curr = mus.iloc[i]
        variance_curr = variances.iloc[i]
        patient = patient_info[i]
        for mu, var in zip(mu_curr.tolist(), variance_curr.tolist()):
            std_dev = np.sqrt(var)
            x = np.linspace(x_range[0], x_range[1], 250)
            y = scistats.norm.pdf(x, mu, std_dev)
            ax.plot(x,y)
            ax.set_ylabel(patient, fontsize=10)
    
    plots_dir = os.path.join("plots", "latents")
    
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        
    if save_name is None:
        if diagnosis is not None:
            save_name = "{}_{}_latent_{}_avg".format(type_name, diagnosis.lower(), avg_col.lower())
        else:
            save_name = "{}_latent_{}_avg".format(type_name, avg_col.lower())
    save_name = "{}_{}.png".format(save_name, run_id)
    
    plt.savefig(os.path.join(plots_dir, save_name), bbox_inches='tight')
    plt.show()