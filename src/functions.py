
import numpy as np

import parmed

import MDAnalysis as mda
from MDAnalysis.analysis import align
from numpy.linalg import norm

import torch
import torch.nn as nn

def generate_training_data(prm_top_files, traj_files, frames_i, frames_f, backbone, output_dir, scaling_factor=1.):  
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
            positions=sel.positions
            pos_resh=positions.reshape(positions.shape[0]*3)
            input_dats.append((torch.tensor(pos_resh/scaling_factor),label)) 
            
            if (ts.frame == 0):
                write_inpcrd(sel.positions/scaling_factor,outname=output_dir+'initial_lab'+str(label)+'.inpcrd')
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
    Deviation of bonds with respect to their equilibrium value (defined by the force field).
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
    Deviation of angles with respect to their equilibrium value (defined by the force field). 
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


def generate_new_inpcrds(ddpm, prm_top_file, output_directory, n_samples=1, device=None, ats=71, dims=3, what_label=0, scaling_factor=1.):
    """Given a DDPM model, a number of samples to be generated and a device, returns some newly generated samples"""

    frames_per_gif=10 
    frame_idxs = np.linspace(0, ddpm.n_steps, frames_per_gif).astype(np.uint) 
    #frames = [] 

    #atoms=ats
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
                    outname=output_directory+'output_frame-'+str(idx)+'_sample-'+str(i)+'_lab'+str(what_label[0].item())+'.inpcrd' 
                    write_inpcrd((x0*scaling_factor).cpu().detach().numpy().reshape(ats,dims),outname=outname)
                    b_dev=bonds_deviation(prm_top_file,outname)
                    #a_dev=angles_deviation(prm_top_file,outname)
                    if idx==(frame_idxs[-1]-1):
                        ac_counts+=1
                        if b_dev<0.5:
                            acc+=1.
                    #print("DONE:",outname)
                if idx==(frame_idxs[-1]-1):
                    acc/=ac_counts
            if idx==(frame_idxs[-1]-1): print("Label: {} accuracy: {}".format(what_label[0].item(),acc))
