# Import of libraries
import sys
sys.path.append('./src/')
from functions import *
from models import *

import random
from argparse import ArgumentParser

import matplotlib.pyplot as plt

import argparse 

from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()

# Input parameters
parser.add_argument('--biosystem', default='PROTEIN', type=str)
#parser.add_argument('--input_directory', default='./example_input/' , type=str)
parser.add_argument('--input_directory', default='/home/acrnjar/Desktop/TEMP/MLdatasets/VariablePep/' , type=str)
parser.add_argument('--topology_0', default='peptide_lab0.prmtop', type=str) # Parameter and topology file
parser.add_argument('--trajectory_0', default='all_conformations_lab0.mdcrd', type=str) # Trajectory file
parser.add_argument('--topology_1', default='peptide_lab1.prmtop', type=str) # Parameter and topology file
parser.add_argument('--trajectory_1', default='all_conformations_lab1.mdcrd', type=str) # Trajectory file
parser.add_argument('--scaling_factor', default=1., type=float) # Useful to normalize data for training if wanted

# Model parameters
parser.add_argument("--no_train", default=False, type=bool)
parser.add_argument("--load_for_train", default=False, type=bool)
parser.add_argument('--batch_size', default=10, type=int) 
parser.add_argument('--n_epochs', default=200, type=int) 
parser.add_argument('--learning_rate', default=0.0001, type=float) 
parser.add_argument('--n_steps', default=1000, type=int) # Original paper: 1000
parser.add_argument('--min_beta', default=10**-4, type=float) # Original paper: 10 ** -4 
parser.add_argument('--max_beta', default=0.02, type=float) # Original paper:  0.02
parser.add_argument('--time_emb_dim', default=100, type=int) # Time embedding dimension
parser.add_argument('--label_emb_dim', default=100, type=int) # Label embedding dimension
parser.add_argument("--input_shape", default='batch_3Nat', type=str)
parser.add_argument('--seed', default=0, type=int) # Random seed

# Classes settings                                                                                                                                                                
parser.add_argument('--N_classes', default=2, type=int) # Number of classes                                                                                                        
parser.add_argument('--desired_labels', default=[0,1], type=list) # Classes to be considered for output   

# Output parameters
parser.add_argument('--output_directory', default='./example_output/', type=str)
parser.add_argument('--num_samples', default=1, type=int)


if __name__ == "__main__":   
    args = parser.parse_args()
    print(f"{args=}")

    print("AC: starting main")
    # Setting reproducibility
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    store_path = "ddpm_inpcrd.pt"

    prmfs = [ args.input_directory + args.topology_0 , args.input_directory + args.topology_1 ]
    trajfs = [ args.input_directory + args.trajectory_0 , args.input_directory + args.trajectory_1 ]
    # Define MDAnalysis universe and related parameters
    frames_f=[]
    for i_l,label in enumerate(list(range(len(prmfs)))):
        univ = mda.Universe(prmfs[i_l], trajfs[i_l])
        nframes = len(univ.trajectory)
        N_at = len(univ.select_atoms('all'))
        print(f"{label=} {N_at=}")
        box_s = max_size(prmfs[i_l],trajfs[i_l],'all',1.1) # Calculate largest coordinate for generation
        print(f"{label=} {box_s=}")
        print(f"{nframes=}")
        frames_f.append(nframes-(nframes%args.batch_size))
    if args.biosystem=='PROTEIN':
        backbone='name CA C N'
    if args.biosystem=='DNA':
        backbone='name P'
    print(f"{frames_f=}")
    dataset = generate_training_data(prm_top_files=prmfs, traj_files=trajfs, frames_i=[0,0], frames_f=frames_f, backbone=backbone, output_dir=args.output_directory, scaling_factor=args.scaling_factor)  

    # Load data                                                              
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(f"{device=}")

    # Defining model
    n_steps, min_beta, max_beta = 1000, 10 ** -4, 0.02  # Originally used by the authors
    if args.input_shape=='batch_3Nat':
        ddpm = inpcrd_DDPM_cond(inpcrd_Seq_Conditioned(max_size=box_s,n_steps=args.n_steps,time_emb_dim=args.time_emb_dim,Natoms=N_at,dims=3, label_emb_dim=args.label_emb_dim), n_steps=args.n_steps, min_beta=args.min_beta, max_beta=args.max_beta, device=device) 

    if not args.no_train:
        if args.load_for_train:
            ddpm.load_state_dict(torch.load(store_path, map_location=device)) 
        training_loop(ddpm, loader, args.n_epochs, optim=Adam(ddpm.parameters(), args.learning_rate), device=device, output_directory=args.output_directory, store_path=store_path)

    print("Loading the trained model")
    best_model=ddpm

    best_model.load_state_dict(torch.load(store_path, map_location=device)) 
    best_model.eval()
    print("Model loaded")
    print("Generating new conformations")
    for i_l,label in enumerate(list(range(len(prmfs)))):
        generate_new_inpcrds(ddpm, prmfs[i_l], output_directory=args.output_directory, n_samples=args.num_samples, device=device, ats=N_at, dims=3, what_label=torch.tensor(args.num_samples*[i_l]).to(device), scaling_factor=args.scaling_factor) 
        

