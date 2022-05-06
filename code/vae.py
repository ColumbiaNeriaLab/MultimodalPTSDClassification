import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import warnings
import re

from datautils import *
from utils import Log, Stats


# Network naming stuff
network_ext = '.nn'
network_name_fstr = '{}_{}' + network_ext

models_path = 'vae_nets'


def get_model_path(model_name):
    '''
    Returns the model path for the specified model name
    
    params:
        model_name: str
            The name of the model to get the path for
    
    return:
        str
            The path to the model folder
    '''
    return os.path.join(models_path, model_name)
    

def make_model_path(model_name):
    '''
    Creates the model folder for the specified model name
    
    params:
        model_name: str
            The name of the model to get the path for
    
    return:
        str
            The path to the model folder
    '''
    
    path = get_model_path(model_name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


class NetworkMetadata:
    '''
    For saving metadata about the network so it can easily be loaded
    
    attribs:
        model_name: str
            The name of the model to save metadata on
        input_size: int
            The size of the input layer
        hidden_size: int
            The size of the hidden layer
        latent_size: int
            The size of the latent encoded layer
        activations: tuple
            The tuple of activation functions for the network
            
    methods:
        save()
            Saves the metadata to a file
        load()
            Loads the metadata from a file
    
    static methods:
        exists(model_name)
            Returns whether the model exists or not
    '''
    
    filename = 'netinfo.mtd'
    
    def __init__(self, model_name, input_size=None, hidden_size=None, latent_size=None, activations=None):
        self.model_name = model_name
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.activations = activations
        
    def set_data(input_size, hidden_size, latent_size, activations):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.activations = activations
    
    def save(self):
        model_path = get_model_path(self.model_name)
        metadata_path = os.path.join(model_path, NetworkMetadata.filename)
        with open(metadata_path, 'w+') as f:
            f.write('model_name: {}\n'.format(self.model_name))
            f.write('input_size: {}\n'.format(self.input_size))
            f.write('hidden_size: {}\n'.format(self.hidden_size))
            f.write('latent_size: {}\n'.format(self.latent_size))
            f.write('activations: {}\n'.format(', '.join(self.activations)))
    
    def load(self):
        model_path = get_model_path(self.model_name)
        metadata_path = os.path.join(model_path, NetworkMetadata.filename)
        with open(metadata_path) as f:
            for line in f.readlines():
                line = line.replace('\n', '')
                linesplit = line.split(': ')
                line_name = linesplit[0]
                line_data = linesplit[1]
                if line_name == 'input_size':
                    self.input_size = int(line_data)
                elif line_name == 'hidden_size':
                    self.hidden_size = int(line_data)
                elif line_name == 'latent_size':
                    self.latent_size = int(line_data)
                elif line_name == 'activations':
                    self.activations = line_data.split(', ')
        
    @staticmethod
    def exists(model_name):
        model_path = get_model_path(model_name)
        metadata_path = os.path.join(model_path, NetworkMetadata.filename)
        return os.path.exists(metadata_path)


def save_network(model, model_name, idx, log=None):
    '''
    Saves the specified model
    
    params:
        model: VAE
            The model to save
        model_name: str
            The name of the model to save to
        idx: int
            The current epoch at which this model is training
        log: Log (default = None)
            The Log to which to log the saving operation
    '''
    
    path = get_model_path(model_name)
    if not os.path.exists(path):
        raise Exception("Path for network '{}' does not exist, please call make_model_path first!".format(path))
    save_fpath = os.path.join(path, network_name_fstr.format(model_name, idx))
    torch.save(model.state_dict(), save_fpath)
    if not NetworkMetadata.exists(model_name):
        metadata = NetworkMetadata(model_name, model.input_size, model.hidden_size, model.latent_size, model.activations_str)
        metadata.save()
    if log:
        log.print_and_log("Saving model to ", save_fpath)
    else:
        print("Saving model to ", save_fpath)


def get_file_index(file):
    '''
    Used to get the indices of files that have a number at
    the end. (Used for getting epoch of model checkpoints)
    
    params:
        file: str
            The name of the file to get the index from
    
    return:
        int
            The index of the file
    '''
    filename, _ =  os.path.splitext(file)
    idx = re.findall('[0-9]+', filename)[-1]
    return int(idx)


def load_network(model_name, model=None, idx=None, log=None):
    '''
    Loads the model of the specified name's state into the specified model
    
    params:
        model_name: str
            The name of the model to load the state from
        model: VAE (default = None)
            If not None model to load the state to
            Otherwise, load model from metadata
        idx: int (default = None)
            The epoch from which to load the model
        log: Log (default = None)
            The Log to which to log the loading operation
    '''
        
    path = os.path.join("vae_nets", model_name)
    if not os.path.exists(path):
        raise Exception("Model by model_name '{}' does not exist!".format(model_name))
    files = [f for f in os.listdir(path) if model_name in f and '.nn' in f]
    if not files:
        raise Exception("Model directory for '{}' exists, but no saved networks found inside the directory!".format(model_name))
    if idx is None:
        files_sorted = sorted(files, key=lambda f : get_file_index(f), reverse=True)
        model_file = files_sorted[0]
    else:
        file_target = network_name_fstr.format(model_name, idx)
        if file_target in files:
            model_file = file_target
        else:
            raise Exception("Model directory for '{}' exists, but no network at epoch {} found inside the directory!".format(model_name, idx))
    
    model_path = os.path.join(path, model_file)
    
    if model is None:
        if NetworkMetadata.exists(model_name):
            metadata = NetworkMetadata(model_name)
            metadata.load()
            model = VAE(metadata.input_size, metadata.hidden_size, metadata.latent_size, metadata.activations)
    
    if model is None:
        raise Exception("No model was provided to load state into and metadata does not exist for model.")
    
    model.load_state_dict(torch.load(model_path))
    
    if log:
        log.print_and_log("Successfully loaded model {}.".format(os.path.abspath(model_path)))
    else:
        print("Successfully loaded model {}.".format(os.path.abspath(model_path)))
    
    return model, get_file_index(model_file)
    

def overwrite_network(model_name, epoch=None):
    '''
    Clears the model path for the specified model
    
    params:
        model_name: str
            The name of the model to overwrite
        epoch: int (default = None)
            If an epoch is not None, then only the checkpoints
            after that checkpoint are deleted
            Otherwise, the whole model is deleted
    
    return:
        bool
            Whether the model path exists or not
    '''
    
    path = get_model_path(model_name)
    if not os.path.exists(path):
        return False
    if not epoch:
        exts_to_check = [network_ext, '.mtd', '.stt', '.csv']
        for f in os.listdir(path):
            fpath = os.path.join(path, f)
            _, fext = os.path.splitext(fpath)
            if fext in exts_to_check:
                os.remove(fpath)
        return True
    else:
        exts_to_check = [network_ext]
        for f in os.listdir(path):
            fpath = os.path.join(path, f)
            _, fext = os.path.splitext(fpath)
            if fext in exts_to_check and get_file_index(f) > epoch:
                os.remove(fpath)
        return True
        
        
def rename_network(net_name_old, net_name_new):
    import warnings
    model_path_old = get_model_path(net_name_old)
    model_path_new = get_model_path(net_name_new)
    if not os.path.exists(model_path_new):
        os.rename(model_path_old, model_path_new)
    else:
        warnings.warn("Model: {} already exists. Can't rename {} to {}".format(net_name_new, net_name_old, net_name_new))
        return
    net_meta = NetworkMetadata(net_name_new)
    net_meta.load()
    net_meta.model_name = net_name_new
    net_meta.save()
    for f in os.listdir(model_path_new):
        fname, ext = os.path.splitext(f)
        if(net_name_old in fname):
            f_old = os.path.join(model_path_new, "{}{}".format(fname, ext))
            f_new = os.path.join(model_path_new, "{}{}".format(fname.replace(net_name_old, net_name_new), ext))
            os.rename(f_old, f_new)
            
            
class VAE(nn.Module):
    '''
    Variational autoencoder implementation
    
    attribs:
        input_size: int
            The size of the input and output layers
        hidden_size: int (default = 400)
            The size of the hidden layer
        latent_size: int (default = 20)
            The size of the latent encoded layer
        fc1: Linear
            The weights and biases between input and hidden layer 1
        fc21: Linear
            The weights and biases between hidden layer 1 and means of the latent layer (hidden layer 2)
        fc22: Linear
            The weights and biases between hidden layer 1 and log variance of the latent layer (hidden layer 2)
        fc3: Linear
            The weights and biases between the latent layer (hidden layer 2) and hidden layer 3
        fc4: Linear
            The weights and biases between hidden layer 3 and output
        activations: tuple
            The list of activation functions of each of the layers
            
    methods:
        encode(x)
            Encodes the input to the latent means and log variances
        reparameterize(mu, logvar):
            Reparameterizes the encoded latent means and log variances to allow for learning
        decode(z):
            Decodes the latent variable z back to the output
        forward(x):
            The forward pass of the network
    '''
    def __init__(self, input_size, hidden_size=400, latent_size=20, activations=('tanh', 'selu', 'tanh', 'tanh')):
        super(VAE, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc21 = nn.Linear(self.hidden_size, self.latent_size)
        self.fc22 = nn.Linear(self.hidden_size, self.latent_size)
        self.fc3 = nn.Linear(self.latent_size, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size, self.input_size)
        self.activations_str = activations
        activation_list = []
        for activation in activations:
            if activation == 'selu':
                activation_list.append(torch.selu)
            elif activation == 'relu':
                activation_list.append(torch.relu)
            elif activation == 'sigmoid':
                activation_list.append(torch.sigmoid)
            elif activation == 'tanh':
                activation_list.append(torch.tanh)
            else:
                activation_list.append(lambda x : x)
        self.activations = tuple(activation_list)

    def encode(self, x):
        h1 = self.activations[0](self.fc1(x))
        return self.activations[1](self.fc21(h1)), self.activations[1](self.fc22(h1))

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = self.activations[2](self.fc3(z))
        return self.activations[3](self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar
        

# Reconstruction loss function
reconstruction_function = nn.MSELoss(reduction='none')


def loss_function(recon_x, x, mu, logvar, terms=('MSE', 'KLD'), beta=1): 
    '''
    Total Loss =  MSE(recon_x, x) + beta*KLD(mu, logvar)
    
    params:
        recon_x
            The output data
        x
            The input data
        mu
            The latent means generated by the VAE
        logvar
            The latent log variances generated by the VAE
            
        beta (optional)
            The weight of the KLD term
            
    return:
        Loss
            The loss function
    '''
    
    loss = 0
    
    if 'MSE' in terms:
        # MSE = 1/n*sum(x_recon - x)^2
        # Need to sum over feature size, and mean over batch size
        SE = reconstruction_function(recon_x, x)
        MSE = torch.mean(torch.sum(SE, dim=1), dim = 0)
        loss += MSE
        ##print('MSE =', MSE.data.item())
    
    if 'KLD' in terms:
        # KLD = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # Note: logvar = log(sigma^2)
        # Need to sum over feature size, and mean over batch size
        KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = 1), dim = 0)
        loss += beta * KLD
        ##print('KLD =', KLD.data.item())
    
    return loss
    

def train_net(net_name, dataset, **kwargs):
    '''
    The network training loop
    
    params:
        net_name: str
            The name of the network to train
        dataset: PatientDataSet
            The dataset to train on
    
    **kwargs:
        activations: tuple (default = ('tanh', 'selu', 'tanh', 'tanh'))
            The list of names of the activation functions for each layer
        hidden_size: int
            The size of the hidden layer
        latent_size: int
            The size of the latent encoded layer
        loss_terms: tuple (default = ('MSE', 'KLD'))
            The terms to include in the loss function
        beta: float (default = 1)
            The weight of the KLD term
        warm_start: int (default = 100)
            If not None and > 0, the number of epochs to perform a warm start on beta
            Otherwise, no warm start is done
        num_epochs: int (default = 100)
            The number of epochs to train for
        until: int (default = None)
            If not None, the max epoch to stop at if the network has trained previously
        batch_size: int (default = 128)
            The batch size for the number of datapoints to feed to the network each step
        shuffle: bool (default = False)
            Whether to shuffle the data or not
        learning_rate: float (default = 1e-3)
            The learning rate for the network training
        clip_val: float (default = None)
            If not None, the value at which to clip weights
        l2: float (default = 1e-2)
            The factor for l2 regularization
        load_net: bool (default = False)
            Whether to load a pre-existing network
        load_epoch: int (default = None)
            If None and load_net, then loads latest network
            If not None and load_net, then loads network from that epoch
        overwrite: bool (default = False)
            Whether to overwrite the existing network data
        save_net_freq: int (default = 10)
            How frequently to save the network
        validation: (default = None)
            If not None: 
                if PatientDataSet then performs validation on that dataset
                if list then performs validation on each dataset in the list
        print_net_freq: int (default = 1)
            How frequently to print the loss
        should_log: bool (default = True)
            Whether to log output to a log file
        overwrite_log: bool (default = False)
            Whether to overwrite the existing log file
        load_stats: bool (default = True)
            Whether to load the existing saved stats on the network
        stats_list: list (default = ["epoch", "loss", "loss_per_feature"])
            The stats to track
            
    '''
    activations = kwargs.get('activations', ('tanh', 'selu', 'tanh', 'tanh'))
    hidden_size = kwargs.get('hidden_size', 400)
    latent_size = kwargs.get('latent_size', 20)
    loss_terms = kwargs.get('loss_terms', ('MSE', 'KLD'))
    _beta = kwargs.get('beta', 1)
    warm_start = kwargs.get('warm_start', 100)
    num_epochs = kwargs.get('num_epochs', 100)
    until = kwargs.get('until', None)
    batch_size = kwargs.get('batch_size', 128)
    shuffle = kwargs.get('shuffle', True)
    learning_rate = kwargs.get('learning_rate', 1e-3)
    clip_val = kwargs.get('clip_val', None)
    l2 = kwargs.get('l2', 1e-2)
    load_net = kwargs.get('load_net', False)
    net_epoch = kwargs.get('net_epoch', None)
    overwrite = kwargs.get('overwrite', False)
    save_net_freq = kwargs.get('save_net_freq', 10)
    _validation = kwargs.get('validation', None)
    print_net_freq = kwargs.get('print_net_freq', 1)
    should_log = kwargs.get('should_log', True)
    overwrite_log = kwargs.get('overwrite_log', False)
    load_stats = kwargs.get('load_stats', True)
    stats_list = kwargs.get('stats_list', ["epoch", "loss", "loss_per_feature"])
    
    model_path = make_model_path(net_name)
    
    log = Log(log_name=net_name+'_training_log.txt', should_log=should_log, log_path=get_model_path(net_name), overwrite=overwrite_log)
    
    log.print_and_log("Training Network '{}'".format(net_name))
    
    idx_offset = 0
    
    if not load_net or not NetworkMetadata.exists(net_name):
        model = VAE(input_size=len(dataset.data_columns), hidden_size=hidden_size, latent_size=latent_size, activations=activations)
    
    # Loading the model if desired
    # Also handles loading stats
    if load_net:
        if not NetworkMetadata.exists(net_name):
            warnings.warn('Requested to load network, but no metadata exists. Loading model state into newly generated model.')
            _, idx_offset = load_network(net_name, model, idx=net_epoch, log=log)
        else:
            model, idx_offset = load_network(net_name, None, idx=net_epoch, log=log)
        idx_offset += 1
        
        if load_stats:
            stats = Stats(load=True, path=model_path)
        else:
            stats = Stats(load=False, path=model_path)
    else:
        stats = Stats(load=False, path=model_path)
    
    if overwrite:
        if load_net:
            overwrite_network(net_name, idx_offset)
        else:
            overwrite_network(net_name)
    
    if torch.cuda.is_available():
        model.cuda()
    
    if l2:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    for epoch in range(num_epochs):
        if until and (epoch + idx_offset >= until):
            break
        log.print_and_log("Train Epoch:", epoch + idx_offset)
        model.train()
        train_loss = 0
        for batch_idx, sample_batch in enumerate(dataloader):
            data_noisy = sample_batch['data_noisy']
            data = sample_batch['data']
            data_noisy = Variable(data_noisy)
            if torch.cuda.is_available():
                data_noisy = data_noisy.cuda()
                data = data.cuda()
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data_noisy)
            # The KLD weight for warm start
            if (warm_start is not None) and (warm_start > 0):
                beta = min((epoch+idx_offset)/warm_start*_beta, _beta)
            else:
                beta = _beta
            loss = loss_function(recon_batch, data, mu, logvar, terms=loss_terms, beta=beta)
            loss.backward()
            curr_loss = loss.data.item()
            train_loss += curr_loss
            # For clipping the loss
            if clip_val:
                clip_grad_norm_(model.parameters(), clip_val)
            optimizer.step()
            
            per_feature_loss = curr_loss / model.input_size
            
            if print_net_freq and print_net_freq > 0:
                if batch_idx % print_net_freq == 0:
                    log.print_and_log('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss Per Feature {:.6f}'.format(
                        epoch + idx_offset,
                        batch_idx * len(data_noisy),
                        len(dataloader.dataset), 100. * batch_idx / len(dataloader),
                        curr_loss, per_feature_loss))
            else:
                log.print_and_log('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss Per Feature: {:.6f}'.format(
                epoch + idx_offset,
                len(dataloader.dataset), 100. * batch_idx / len(dataloader),
                curr_loss, per_feature_loss))
        
        avg_train_loss = train_loss / (batch_idx + 1)
        avg_per_feature_loss = avg_train_loss / model.input_size
        log.print_and_log('====> Epoch: {} Average Loss: {:.6f}\tAverage Loss Per Feature {:.6f}'.format(
            epoch + idx_offset, avg_train_loss, avg_per_feature_loss))
        
        if 'epoch' in stats_list:
            stats.track_stat('epoch', epoch, epoch)
        
        if 'loss' in stats_list:
            stats.track_stat('loss', epoch, avg_train_loss)
            
        if 'loss_per_feature' in stats_list:
            stats.track_stat('loss_per_feature', epoch, avg_per_feature_loss)
        
        if save_net_freq and save_net_freq > 0:
            if (epoch + idx_offset) % save_net_freq == 0:
                save_network(model, net_name, epoch + idx_offset, log)
                stats.save()
                
        if _validation:
            if isinstance(_validation, (list, tuple)):
                val_loss = []
                val_loss_per_feature = []
                for val_ds in _validation:
                    val_loss_curr, val_loss_per_feature_curr = validation(net_name, val_ds, model=model, log=log)
                    val_loss.append(val_loss_curr)
                    val_loss_per_feature.append(val_loss_per_feature_curr)
            
            else:
                val_loss, val_loss_per_feature = validation(net_name, _validation, model=model, log=log)
            
            if 'validation_loss' in stats_list:
                stats.track_stat('validation_loss', epoch, val_loss)
            
            if 'validation_loss_per_feature' in stats_list:
                stats.track_stat('validation_loss_per_feature', epoch, val_loss_per_feature)
            
            if save_net_freq and save_net_freq > 0:
                if (epoch + idx_offset) % save_net_freq == 0:
                    stats.save()
        
        log.save()
    
    # Only save the final epoch if our network wasn't loaded past the until
    # epoch. Otherwise, there is no point in saving.
    if (until and (idx_offset < until)) or not until:
        save_network(model, net_name, epoch + idx_offset, log)
        stats.save()
    
    log.save()
    
    return epoch + idx_offset


def validation(net_name, dataset, **kwargs):
    '''
    For validating a network
    
    params:
        net_name: str
            The name of the network to validate
        dataset: PatientDataSet
            The dataset to validate with
    
    **kwargs:
        loss_terms: tuple (default = ('MSE', 'KLD'))
            The terms to include in the loss function
        batch_size: int (default = 64)
            The batch size for the number of datapoints to feed to the network each step
        shuffle: bool (default = False)
            Whether to shuffle the data or not
        use_noisy: bool (default = True)
            Whether to use the noisy version of the data or not
        net_epoch: bool (default = None)
            If not None, loads the net at the specified epoch
            Otherwise loads the net at the latest epoch
        model: VAE (default = None)
            If not None, uses the given model instead of loading the network
        log: Log (default = None)
            If not None, the log to perform logging with
        should_log: bool (default = True)
            Whether to log the validation
        overwrite_log: bool (default = False)
            Whether to overwrite the current log file or not
    '''
    loss_terms = kwargs.get('loss_terms', ('MSE', 'KLD'))
    batch_size = kwargs.get('batch_size', 64)
    shuffle = kwargs.get('shuffle', False)
    use_noisy = kwargs.get('use_noisy', True)
    net_epoch = kwargs.get('net_epoch', None)
    model = kwargs.get('model', None)
    log = kwargs.get('log', None)
    should_log = kwargs.get('should_log', True)
    overwrite_log = kwargs.get('overwrite_log', False)
    
    if not log:
        log = Log(log_name=net_name+'_validation_log.txt', should_log=should_log, log_path=get_model_path(net_name), overwrite=overwrite_log)
    
    log.print_and_log("Performing validation on model '{}'".format(net_name))
    
    if not model:
        model, _ = load_network(net_name, None, idx=net_epoch)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    test_loss = 0
    for batch_idx, sample_batch in enumerate(dataloader):
        data = sample_batch['data']
        if use_noisy:
            data_noisy = sample_batch['data_noisy']
            data_input = Variable(data_noisy)
        else:
            data_input = Variable(data)
        
        if torch.cuda.is_available():
            data_input = data_input.cuda()
            data = data.cuda()
            
        recon_batch, mu, logvar = model(data_input)
        
        loss = loss_function(recon_batch, data, mu, logvar, terms=loss_terms)
        
        curr_loss = loss.data.item()
        
        test_loss += curr_loss
        
        log.print_and_log('[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss Per Feature {:.6f}'.format(
                batch_idx * len(data_input),
                len(dataloader.dataset), 100. * batch_idx / len(dataloader),
                curr_loss, curr_loss / model.input_size))
        
        log.save()
    
    avg_loss = test_loss / (batch_idx + 1)
    avg_per_feature_loss = avg_loss / model.input_size
    
    log.print_and_log('Average Loss: {:.6f}\tAverage Loss Per Feature {:.6f}'.format(avg_loss, avg_per_feature_loss))
    
    log.save()
    
    return avg_loss, avg_per_feature_loss
    

def extract_latents(net_name, dataset, **kwargs):
    '''
    For extracting and storing latent representations of the given dataset using
    the given network
    
    params:
        net_name: str
            The name of the network to get latents from
        dataset:
            The dataset to apply the network to
    
    **kwargs:
        batch_size: int (default = 64)
            The size of batches to feed to the network to generate latents
        use_noisy: bool (default = False)
            Whether to use the noisy version of the data as input
        return_df: bool (default = True)
            Whether to return the DataFrame of the generated latent representations
        fname_tag: str (default = None)
            A tag to add to the filename to denote additional information about these latents
        net_epoch: int (default = None)
            If not None, loads the network at the given epoch
            Otherwise, loads the network at the latest epoch
        file_path: str (default = None)
            If not None, saves the latents to this path
            Otherwise, saves them to the default path
    '''
    
    batch_size = kwargs.get('batch_size', 64)
    use_noisy = kwargs.get('use_noisy', False)
    return_df = kwargs.get('return_df', True)
    fname_tag = kwargs.get('fname_tag', None)
    net_epoch = kwargs.get('net_epoch', None)
    latent_fpath = kwargs.get('file_path', None)
    
    model, _ = load_network(net_name, None, idx=net_epoch)

    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    latent_df = None
    
    print('Extracting latents...')
    
    for batch_idx, sample_batch in enumerate(dataloader):
        
        if use_noisy:
            data_noisy = sample_batch['data_noisy']
            data_input = Variable(data_noisy)
        else:
            data = sample_batch['data']
            data_input = Variable(data)
            
        if torch.cuda.is_available():
            data_input = data_input.cuda()
            
        print('[{}/{} ({:.0f}%)]'.format(
                batch_idx * len(data_input),
                len(dataloader.dataset), 100. * batch_idx / len(dataloader)))
    
        mu, logvar = model.encode(data_input)
        
        mu = mu.cpu().detach().numpy()
        logvar = logvar.cpu().detach().numpy()
        
        latent_arr = np.concatenate((mu, logvar), axis=1)
        
        latent_colnames = ['mu_{}'.format(i) for i in range(mu.shape[1])] + ['logvar_{}'.format(i) for i in range(logvar.shape[1])]

        curr_info_df = pd.DataFrame(sample_batch['info'])
        curr_clinical_df = pd.DataFrame(sample_batch['clinical'])
        curr_df = pd.concat([curr_info_df, curr_clinical_df], axis=1)
        
        if 'SubjectID' in curr_df.columns:
            curr_df['SubjectID'] = curr_df['SubjectID'].astype(str)
        if 'Sex' in curr_df.columns:
            curr_df['Sex'] = curr_df['Sex'].astype(str)
        if 'Age' in curr_df.columns:
            curr_df['Age'] = curr_df['Age'].astype(int)
        curr_df['Diagnosis'] = sample_batch['diagnosis'].cpu().detach().numpy()[:,0]
        temp_df = pd.DataFrame(data=latent_arr, columns=latent_colnames)
        curr_df[latent_colnames] = temp_df
        
        if latent_df is None:
            latent_df = curr_df
        else:
            latent_df = pd.concat([latent_df, curr_df], ignore_index=True)
        
    print('done.')
    
    if latent_fpath is None:
        model_path = get_model_path(net_name)

        latent_fpath = os.path.join(model_path, net_name)

        if fname_tag:
            latent_fpath += '_{}'.format(fname_tag)
        if net_epoch:
            latent_fpath += '_{}'.format(net_epoch)

        latent_fpath += '_latents.csv'

    latent_df.to_csv(latent_fpath, index=False)

    print("Saving latents to ", os.path.abspath(latent_fpath))
    
    if return_df:
        return latent_df
    
    else:
        return None
    

def load_latents(net_name, fname_tag=None, net_epoch=None):
    '''
    For loading the generated latent representation data
    
    params:
        net_name: str
            The name of the network
        fname_tag: str (default = None)
            A tag to add to the filename to denote additional information about these latents
        net_epoch: int (default = None)
            If not None, loads the network at the given epoch
            Otherwise, loads the network at the latest epoch
    '''
    
    model_path = get_model_path(net_name)
    
    latent_fpath = os.path.join(model_path, net_name)
    
    if fname_tag:
        latent_fpath += '_{}'.format(fname_tag)
    if net_epoch:
        latent_fpath += '_{}'.format(net_epoch)
    
    latent_fpath += '_latents.csv'
    
    print("Reading latents from ", os.path.abspath(latent_fpath))
    
    df = pd.read_csv(latent_fpath)
    
    return df