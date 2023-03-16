# FrameFromNoise

<!-- TO DO: 
- Perche a volte la loss sale di colpo a ordini di grandezza anche se ormai si era raggiunto il plateau?
- x0[0] nel write_inpcrd forse e sbagliato, se gli altri sono gli altri dei 5 n_samples! - no, era per batch_1_Nat_3
- perche lab 1 (MMMM) ha sempre meno accuracy di label 0 (HHHH)???
- outname=args.output_directory+'output_frame-'+str(idx)+'_sample-'+str(i)+'_lab'+str(what_label[0].item())+'.inpcrd' #QUIQUI what_label[0] e' per fare veloce
- remove dset, batch_1_Nat_3
- __main__
- desired_labels: apply!
-->

FrameFromNoise is a conditional denoising diffusion probabilistic model (DDPM) which, given the AMBER parameter and topology file of a biological system (.prmtop, .top) and an associated molecular dynamics (MD) trajectory (.dcd, .nc, ...), reconstructs a frame belonging to the same distribution of the sampled trajectory. Moreover, it is given a label (0 or 1) which allows to distinguish between frames with an observed condition from those without.

The trajectory is aligned and centered around the origin in the pre-processing stage, as this is necessary for the algorithm. The first frame is outputted as "initial.inpcrd"

Instead of a U-Net, as typical for diffusion models, the code makes use of a more traditional architecture composed of linear layers, where both time and label embeddings are added in every layer.

The code is adapted from this tutorial: https://medium.com/mlearning-ai/enerating-images-with-ddpms-a-pytorch-implementation-cef5a2ba8cb1

The output .inpcrd files can be visualized with VMD (or pymol): load the parameter and topology files as "AMBER7 Parm", then the .incprd as "AMBER7 Restart"

# Required Libraries

* numpy >= 1.22.3

* torch >= 1.12.1+cu116 (pip install torch==1.12.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html) 

* matplotlib >= 3.4.3

* MDAnalysis >= 2.3.0

* ParmEd >= 3.4.3

* tqdm >= 4.63.0

* tensorboardX >= 2.5.1

# Case Study

As a case-study, the software LEaP and CPPTRAJ (AmberTools21) were used to generate 30,000 conformations of a tetrapeptide, and arranged in a trajectory file. Approximately half of the frames regard a peptide of sequence HHHH, whereas the others regard a peptide of sequence MMMM. Given that histidine and methionine share the same number of atoms, their atomic position can give rise to a 3*N tensor of identical size (N being the number of atoms) that can be taken as input. However, for simplicity reasons, frames beloning to the two labels were assigned to two separate sets of .prmtop and .dcd files.

On 250 samples generated per label after 200 epochs of training, accurarcies of 0.992 and 0.916 were obtained for label 0 (HHHH) and label 1 (MMMM), where the accuracy measures what fraction of the total generated structures respect a certain quality check condition (in this case, maximum bond length deviation of 0.5 Å).

<p align="center">
<img width="500" src=https://github.com/alescrnjar/FrameFromNoise/blob/main/example_output/Example_generated_structures.png>
</p>
<p align="center">
<em> Two examples of generated structures (left: label 0, right: label 1). </em>
</p>
