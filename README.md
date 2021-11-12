# MultimodalPTSDClassification
Supplementary code provided for the paper "Multimodal Imaging-based Classification of Posttraumatic Stress Disorder Using Data-Driven Computational Approaches: A multisite Big Data Study from ENIGMA-PTSD Consortium"

## Instructions
Dataset format for Resting State data must be the following:
1. **Site**: The site name
2. **SubjectID**: The alphanumerical id of the subject
3. **Diagnosis**: "Control" or "PTSD"
4. **Age**: Numerical age
5. **Sex**: "M" or "F"
6. **ROIx-ROIy**: All the resting state ROIs ordered in upper triangular matrix form where x and y represent the index of each ROI

Site | SubjectID | Diagnosis | Age | Sex | ROI1-ROI1 | ROI1-ROI2 | ...
--- | --- | --- | --- |--- |--- |--- |---
Columbia | CO123 | PTSD | 32 | M | 1 | 0.3543 | ... 
Stanford | S456 | Control | 76 | F | 1 | 0.5453 | ... 

[main.ipynb](code/main.ipynb): Contains all the central code for quickly running the VAE.

To use:
1. Replace the string stored in **RS_csv_path** with the path to your csv file.
2. Run the dataset generation code. If you want to train on PTSD or both instead of control you can change **patient_type** to 'ptsd' or 'control_ptsd' in the first call to generate_datasets in the section **Dataset Generation For Training On Control**. If you want to train including Age and Sex, you must set non_data_cols = [] instead of \['Age', 'Sex'\].
3. Set the desired hyperparameters for size of hidden layer, number of latents, beta (weight of KLD term), l2, and learning rate. These are in list form so that multiple models with each combination of hyperparameters can be run. In general the default values were determined by a sparse gridsearch.
4. Unless otherwise needed, leave the activations as is. If changing the activation of a layer is desired, the layers are arranged as follows: (input, encoder, decoder, output). The options for layer types is 'tanh', 'relu', 'selu', or 'sigmoid', and defaults to linear layer for all other choices.
5. Then you can specify the number of epochs to train for.
6. Finally, set the net_name variable. Our chosen format was 
   \[diagnosis_to_train_on\]_\[dataset_type\]_l\[number_of_latents\]_h\[number_of_hidden\]_b\[beta]
7. Then you can run the cell to train.
8. Once the model starts training, a folder called **vae_nets** is generated if it does not exist yet. Inside, a folder is made matching the name you chose for the network. Here, the model metadata, checkpoints, log information, stats information, and the latents will be stored.
9. The latents will be extracted at the end of training 
10. Next, you can easily plot the loss and view the distribution of latents for individuals or averaged across some cross section of the population. The plots get stored in a newly created **plots** folder.

The format of the resulting latents is as follows:

The filename is \[net_name\]_\[optional_tag\]_latents.csv

The output should look like:

Site | SubjectID | Diagnosis | mu_0 | mu_1 | ... | logvar_1 | logvar_2 | ...
--- | --- | --- | --- |--- |--- |--- |--- |---
Columbia | CO123 | 1 | -0.74453 | 1.8544 | ... | -0.3565 | -0.93231 | ...
Stanford | S456 | 0 | 2.456 | -0.432425 | ... | 0.12345 | -0.8543232 | ...

*Note that here Diagnosis 0 = Control and 1 = PTSD*

## Additional files

[utils.py](code/utils.py): Contains custom code for logging and maintaining stats

[datautils.py](code/datautils.py): Contains code for data cleaning and dataset generation

[vae.py](code/vae.py): Contains all the code for creating the VAE, training and validating the VAE, and extracting latents

[plottingutils.py](code/plottingutils.py): Contains functions for plotting loss and latents
