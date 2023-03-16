#!/bin/bash

### ### BASH script for generation of input parameter+topology and trajectory ### ###

nconf_per_label=15000 # Number of desired conformations per label
length=4 # number of amino acids
log_freq=500 # print out info every this many conformations

outdir=../example_input # Output directory

rm $outdir/gen_mdcrd*cpptraj
rm $outdir/all_conformations*pdb

c_lab_0=0
c_lab_1=0
# Loop over the two labels
for label in 0 1
do
# Loop over index of conformation
for idx in `seq 0 $((nconf_per_label-1))`
do
rm leap.log
echo $c_lab_0 $c_lab_1 | awk '(($1+$2)%'$log_freq'==0){print "Label 0:",$1,"Label 1:",$2,"Total:",$1+$2,"/",'$((nconf_per_label*2))'}'

# Generate bonds, angles, and dihedrals randomly in valid range
val0=$( echo "scale=10; $RANDOM/32767*180" | bc )
val1=$( echo "scale=10; $RANDOM/32767*180" | bc )

# Define differences between the two labels
if [ $label = 0 ] ; then
    aminos="HIS HIS HIS HIS"
    idx=$c_lab_0
    c_lab_0=$((c_lab_0+1))
fi
if [ $label = 1 ] ; then
    aminos="MET MET MET MET"
    idx=$c_lab_1
    c_lab_1=$((c_lab_1+1))
fi

# Make LEaP input script
echo \
'source leaprc.gaff2
source leaprc.protein.ff14SB 
source leaprc.water.tip3p 
loadamberparams frcmod.ionsjc_tip3p

UN = sequence { '$aminos' }

impose UN { {1 '$length'} } { { "C" "N" "CA" "C" '$val1' } }

check UN

saveamberparm UN '$outdir'/peptide_lab'$label'.prmtop '$outdir'/peptide_lab'$label'_'$idx'.inpcrd

quit
' > leap.in

# Execute LEaP
tleap -s -f leap.in &> /dev/null
grep -e 'Errors = ' leap.log | grep -v 0 && echo "ERROR in LEaP." | exit

# Append line to cpptraj script
echo 'trajin '$outdir'/peptide_lab'$label'_'$idx'.inpcrd' >> $outdir/gen_mdcrd_lab$label.cpptraj
done 
done

for label in 0 1
do
# Collate all structures into a single trajectory file
echo "center @CA,C,N origin" >> $outdir/gen_mdcrd.cpptraj
echo "trajout "$outdir/"all_conformations_lab"$label".mdcrd mdcrd" >> $outdir/gen_mdcrd_lab$label.cpptraj
cpptraj -p $outdir/peptide_lab"$label".prmtop < $outdir/gen_mdcrd_lab"$label".cpptraj > /dev/null
echo "Successfully put structures in $outdir/all_conformations_lab$label.mdcrd"
done
rm $outdir/peptide*lab0*inpcrd
rm $outdir/peptide*lab1*inpcrd

