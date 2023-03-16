# Import of libraries
import random
import numpy as np
from argparse import ArgumentParser

from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import parmed

import MDAnalysis as mda
from MDAnalysis.analysis import align
from numpy.linalg import norm

import argparse 

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()

# Input parameters
parser.add_argument('--biosystem', default='PROTEIN', type=str)
parser.add_argument('--input_directory', default='./example_input/' , type=str)
parser.add_argument('--topology_0', default='peptide_lab0.prmtop', type=str) # Parameter and topology file
parser.add_argument('--trajectory_0', default='all_conformations_lab0.mdcrd', type=str) # Trajectory file
parser.add_argument('--topology_1', default='peptide_lab1.prmtop', type=str) # Parameter and topology file
parser.add_argument('--trajectory_1', default='all_conformations_lab1.mdcrd', type=str) # Trajectory file
#parser.add_argument('--scaling_factor', default=1., type=float) # Trajectory file  #

# Model parameters
parser.add_argument("--no_train", default=False, type=bool)
parser.add_argument("--load_for_train", default=False, type=bool)
parser.add_argument('--batch_size', default=10, type=int) #orig: 128
parser.add_argument('--n_epochs', default=200, type=int) #orig: 20
parser.add_argument('--learning_rate', default=0.0001, type=float) #orig: 0.001
parser.add_argument('--n_steps', default=1000, type=int) #orig: 1000
parser.add_argument('--min_beta', default=10**-4, type=float) #orig: 10 ** -4 
parser.add_argument('--max_beta', default=0.02, type=float) #orig:  0.02
parser.add_argument('--time_emb_dim', default=100, type=int) #orig: 100
parser.add_argument("--input_shape", default='batch_3Nat', type=str)

# Classes settings                                                                                                                                                                
parser.add_argument('--dist_cut', default=10.0, type=float) # distance cut-off for class defintion                                                                                 
parser.add_argument('--N_classes', default=2, type=int) # Number of classes                                                                                                        
parser.add_argument('--desired_labels', default=[0,1], type=list) # Classes to be considered for output   

# Output parameters
#parser.add_argument('--output_directory', default='./example_output/', type=str)
parser.add_argument('--output_directory', default='./', type=str)
parser.add_argument('--num_samples', default=5, type=int)


########################### 

def generate_training_data(prm_top_files, traj_files, frames_i, frames_f, backbone, output_dir):  #HERE
    """
    Generate training dataset.
    """
    input_dats = []
    counts = {0:0,1:0}
    for i_l,label in enumerate(list(range(len(prm_top_files)))):
        u = mda.Universe(prm_top_files[i_l], traj_files[i_l])

        ref_u = u
        ref_u.trajectory[0]
        ref_pos = ref_u.select_atoms(backbone).positions - ref_u.atoms.center_of_mass() #backbone works with both proteins and DNA
    
        for ts in u.trajectory[frames_i[i_l]:frames_f[i_l]:1]:

            counts[label]+=1
            
            # Align the current frame to the first one
            prot_pos = u.select_atoms(backbone).positions - u.atoms.center_of_mass()
            R_matrix, R_rmsd = align.rotation_matrix(prot_pos,ref_pos)
            u.atoms.translate(-u.select_atoms(backbone).center_of_mass())
            u.atoms.rotate(R_matrix)
            u.atoms.translate(ref_u.select_atoms(backbone).center_of_mass())
            
            sel = u.select_atoms('all')
            
            # Define observable for labeling data
            pos_dstz_at1 = u.select_atoms('resid 1 and name CA').center_of_mass()
            pos_dstz_at2 = u.select_atoms('resid 6 and name CA').center_of_mass()
            dist_dstz = norm(pos_dstz_at1-pos_dstz_at2)

            positions=sel.positions
            pos_resh=positions.reshape(positions.shape[0]*3)
            input_dats.append((torch.tensor(pos_resh/args.scaling_factor),label)) 
            
            if (ts.frame == 0):
                write_inpcrd(sel.positions/args.scaling_factor,outname=output_dir+'initial_lab'+str(label)+'.inpcrd')
    input_dataset=input_dats
    print("{} frames with label 0, {} frames with label 1.".format(counts[0],counts[1]))
    return input_dataset #,at_list


def max_size(prm_top_file,trajectory_file,selection='all',factor=1.0):
    """                                                                                                                                                                             
    Maximum value that any coordinate can have. This requires the system to be centered on the origin.                                                                              
    """
    universe=mda.Universe(prm_top_file,trajectory_file)
    all_maxm=[]
    for ts in universe.trajectory[::1]:
        pos=universe.select_atoms(selection).positions
        maxm=0.0
        for at in pos:
            for coord in at:
                if (maxm<np.sqrt(coord*coord)): maxm=np.sqrt(coord*coord)
        all_maxm.append(maxm)
    # A factor can be multiplied in order to allow some fluctuations on the protein surface                                                                                         
    return(factor*max(all_maxm))

def bonds_deviation(prm_top_file,inpcrd_file):
    """
    Root mean square deviation of bonds with respect to their equilibrium value (defined by the force field).
    """
    myparams=parmed.amber.readparm.AmberParm(prm_top_file,xyz=inpcrd_file)
    bonds=parmed.tools.actions.printBonds(myparams,'!(:WAT,Na+,Cl-)') 
    dev2s=[]
    for line in str(bonds).split('\n'):
        if ('Atom' not in line and len(line.split())!=0):
            Req=float(line.split()[10])
            Distance=float(line.split()[8])
            #dev2s.append((Req-Distance)**2)
            dev2s.append(abs(Req-Distance))
    #return np.sqrt(np.mean(dev2s))
    return max(dev2s)

def angles_deviation(prm_top_file,inpcrd_file):
    """
    Root mean square deviation of angles with respect to their equilibrium value (defined by the force field). 
    """
    myparams=parmed.amber.readparm.AmberParm(prm_top_file,xyz=inpcrd_file)
    angles=parmed.tools.actions.printAngles(myparams,'!(:WAT,Na+,Cl-)') 
    dev2s=[]
    for line in str(angles).split('\n'):
        if ('Atom' not in line and len(line.split())!=0):
            Theta_eq=float(line.split()[13])
            Angle=float(line.split()[14])
            difference=abs(Angle-Theta_eq)
            while difference>180.: difference-=180.
            dev2s.append(difference)
    #return np.sqrt(np.mean(dev2s))
    return max(dev2s)


def write_inpcrd(inp_tensor,outname='out.inpcrd'):
    """                                                                                                                                                                            
    Given input coordinates tensor, writes .inpcrd file.                                                                                                                           
    """
    inp_tensor=list(np.array(inp_tensor))
    outf=open(outname,'w')
    outf.write('default_name\n'+str(len(inp_tensor))+'\n')
    for at in range(len(inp_tensor)):
        outf.write(" {:11.7f} {:11.7f} {:11.7f}".format(float(inp_tensor[at][0]),float(inp_tensor[at][1]),float(inp_tensor[at][2])))
        if (at%2!=0):
            outf.write('\n')
    outf.close()

def generate_new_inpcrds(ddpm, prm_top_file, n_samples=1, device=None, ats=71, dims=3, what_label=0):
    """Given a DDPM model, a number of samples to be generated and a device, returns some newly generated samples"""

    frames_per_gif=10 
    frame_idxs = np.linspace(0, ddpm.n_steps, frames_per_gif).astype(np.uint) 
    frames = [] 

    atoms=ats
    ndims=dims
    atsdims=ats*dims
    
    with torch.no_grad(): 
        if device is None: 
            device = ddpm.device 

        # Starting from random noise 
        x = torch.randn(n_samples, atsdims).to(device) 
            
        for idx, t in enumerate(list(range(ddpm.n_steps))[::-1]): 
            # Estimating noise to be removed 
            time_tensor = (torch.ones(n_samples, 1) * t).to(device).long() 
            eta_theta = ddpm.backward(x, time_tensor, what_label) 

            alpha_t = ddpm.alphas[t] 
            alpha_t_bar = ddpm.alpha_bars[t] 

            # Partial denoising  
            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta) 

            if t > 0: 
                z = torch.randn(n_samples, atsdims).to(device)  
                
                # Option 1: sigma_t squared = beta_t 
                beta_t = ddpm.betas[t] 
                sigma_t = beta_t.sqrt() 

                # Option 2: sigma_t squared = beta_tilda_t
                # prev_alpha_t_bar = ddpm.alpha_bars[t-1] if t > 0 else ddpm.alphas[0]
                # beta_tilda_t = ((1 - prev_alpha_t_bar)/(1 - alpha_t_bar)) * beta_t
                # sigma_t = beta_tilda_t.sqrt()

                # Adding some more noise like in Langevin Dynamics fashion
                x = x + sigma_t * z 

            if idx in frame_idxs or t == 0:
                if idx==(frame_idxs[-1]-1):
                    acc=0.
                    ac_counts=0
                for i,x0 in enumerate(x):
                    outname=args.output_directory+'output_frame-'+str(idx)+'_sample-'+str(i)+'_lab'+str(what_label[0].item())+'.inpcrd' 
                    write_inpcrd((x0*args.scaling_factor).cpu().detach().numpy().reshape(ats,dims),outname=outname)
                    b_dev=bonds_deviation(prm_top_file,outname)
                    #a_dev=angles_deviation(prm_top_file,outname)
                    if idx==(frame_idxs[-1]-1):
                        ac_counts+=1
                        if b_dev<0.5:
                            acc+=1.
                    #print("DONE:",outname)
                if idx==(frame_idxs[-1]-1):
                    acc/=ac_counts
            if idx==(frame_idxs[-1]-1): print(f"{acc=}")
    
#############################

class inpcrd_DDPM_cond(nn.Module):
    def __init__(self, network, n_steps=200, min_beta=10 ** -4, max_beta=0.02, device=None):
        super(inpcrd_DDPM_cond, self).__init__()
        self.n_steps = n_steps
        self.device = device
        self.network = network.to(device)
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device)  # Number of steps is typically in the order of thousands
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(device)

    def forward(self, x0, t, label, eta=None):
        n, atsdims=x0.shape
        a_bar = self.alpha_bars[t]

        if eta is None:
            eta = torch.randn(n, atsdims).to(self.device)
            
        noisy = a_bar.sqrt().reshape(n, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1) * eta 

        return noisy

    def backward(self, x, t, label):
        return self.network(x, t, label)

###

def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:,::2] = torch.sin(t * wk[:,::2])
    embedding[:,1::2] = torch.cos(t * wk[:,::2])

    return embedding

###

class inpcrd_Seq_Conditioned(nn.Module): 
    def __init__(self, max_size, n_steps=1000, time_emb_dim=100, Natoms=71, dims=3, NClasses=2, label_emb_dim=100): ##
        super(inpcrd_Seq_Conditioned, self).__init__() ##
        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim) ##
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim) ##
        self.time_embed.requires_grad_(False) ##
        #
        self.label_embedding = nn.Embedding(NClasses, label_emb_dim)
        #
        N1=Natoms*dims
        N2=10*N1
        #                                                                                        
        self.te_N1 = self._make_te(time_emb_dim, N1) 
        self.te_N2 = self._make_te(time_emb_dim, N2) 
        self.te_N2a = self._make_te(time_emb_dim, N2) 
        #
        self.le_N1 = self._make_te(label_emb_dim, N1) 
        self.le_N2 = self._make_te(label_emb_dim, N2) 
        self.le_N2a = self._make_te(label_emb_dim, N2)
        #
        self.lin_N1N2=nn.Linear(N1,N2)
        self.lin_N2N2=nn.Linear(N2,N2)
        self.lin_N2N1=nn.Linear(N2,N1)
        #
        self.LReLU_a=nn.LeakyReLU(0.2)
        self.LReLU_b=nn.LeakyReLU(0.2)

    def forward(self, x, t, labels):                                                                     
        c = self.label_embedding(labels)
        t = self.time_embed(t)
        n = len(x)

        # N1 -> N2
        out=x
        out=out + self.te_N1(t).reshape(n, -1) + self.le_N1(c).reshape(n, -1)
        out=self.lin_N1N2(out) 
        out=self.LReLU_a(out)
        #
        # N2 -> N2
        out=out + self.te_N2(t).reshape(n, -1) + self.le_N2(c).reshape(n, -1)
        out=self.lin_N2N2(out) 
        out=self.LReLU_b(out)
        # N2 -> N1
        out=out + self.te_N2a(t).reshape(n, -1) + self.le_N2a(c).reshape(n, -1)
        out=self.lin_N2N1(out)
        return out

    def _make_te(self, dim_in, dim_out): 
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )

    
###

def training_loop(ddpm, loader, n_epochs, optim, device, display=False, store_path="ddpm_model.pt"):
    mse = nn.MSELoss()
    best_loss = float("inf")
    n_steps = ddpm.n_steps

    summary_writer = SummaryWriter(args.output_directory) # Usage: tensorboard --logdir=./
    color1="#5171A5"
    color2="#2E4052"
    first_loss_determined=False
    n_params=0
    for param in ddpm.parameters():
        n_params+=1
    print(f"{n_params=}")
    for epoch in tqdm(range(n_epochs), desc=f"Training progress", colour=color1):
        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(loader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour=color2)):
            # Loading data
            x0 = batch[0].to(device)
            lab = batch[1].to(device)
            n = len(x0)

            # Picking some noise for each of the structures in the batch, a timestep and the respective alpha_bars
            eta = torch.randn_like(x0).to(device) 
            t = torch.randint(0, n_steps, (n,)).to(device)

            # Computing the noisy structures based on x0 and the time-step (forward process)
            noisy_strcs = ddpm(x0, t, lab, eta)
            
            # Getting model estimation of noise based on the structures and the time-step
            eta_theta = ddpm.backward(noisy_strcs, t.reshape(n, -1), lab)
            
            # Optimizing the MSE between the noise plugged and the predicted noise
            loss = mse(eta_theta, eta)
            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item() * len(x0) / len(loader.dataset)
            summary_writer.add_scalar('Loss',torch.FloatTensor([epoch_loss]),global_step=epoch)

        log_string = f"\nLoss at epoch {epoch + 1}: {epoch_loss:.3f}"

        if first_loss_determined==False:
            first_loss=epoch_loss
            first_loss_determined=True

        training_wrong=False
        if epoch_loss > 10*first_loss:
            log_string += " --> Training gone wrong. Model will be re-loaded"
            training_wrong=True
            ddpm.load_state_dict(torch.load(store_path, map_location=device)) 
            
        # Storing the model
        if not training_wrong:
            if best_loss > epoch_loss:
                best_loss = epoch_loss
                torch.save(ddpm.state_dict(), store_path)
                log_string += " --> Best model ever (stored)"

        print(log_string)


############################# MAIN

#if __name__ == "__main__":   
args = parser.parse_args()
print(f"{args=}")

print("AC: starting main")
# Setting reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

###

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
dataset = generate_training_data(prm_top_files=prmfs, traj_files=trajfs, frames_i=[0,0], frames_f=frames_f, backbone=backbone, output_dir=args.output_directory)  #HERE

# Load data                                                              
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(f"{device=}")

# Defining model
n_steps, min_beta, max_beta = 1000, 10 ** -4, 0.02  # Originally used by the authors
if args.input_shape=='batch_3Nat':
    ddpm = inpcrd_DDPM_cond(inpcrd_Seq_Conditioned(max_size=box_s,n_steps=args.n_steps,time_emb_dim=args.time_emb_dim,Natoms=N_at,dims=3), n_steps=args.n_steps, min_beta=args.min_beta, max_beta=args.max_beta, device=device) 

if not args.no_train:
    if args.load_for_train:
        ddpm.load_state_dict(torch.load(store_path, map_location=device)) 
    training_loop(ddpm, loader, args.n_epochs, optim=Adam(ddpm.parameters(), args.learning_rate), device=device, store_path=store_path)

print("Loading the trained model")
best_model=ddpm

best_model.load_state_dict(torch.load(store_path, map_location=device)) 
best_model.eval()
print("Model loaded")
print("Generating new conformations")
for i_l,label in enumerate(list(range(len(prmfs)))):
    generate_new_inpcrds(ddpm, prmfs[i_l], n_samples=args.num_samples, device=device, ats=N_at, dims=3, what_label=torch.tensor(args.num_samples*[i_l]).to(device)) 
    

