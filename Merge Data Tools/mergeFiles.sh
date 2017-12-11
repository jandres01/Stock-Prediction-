#!/bin/sh

##Place PBS directives here
#PBS -N jandres_job
#PBS -o /home/jandres/Output
#PBS -e /home/jandres/Output
#PBS -l nodes=1:ppn=36
#PBS -l walltime=72:00:00
#PBS -M jandres@trinity.edu
#PBS -m ae

cd Merge\ Data\ Tools/
R --vanilla < revised-monthly-momentum-data-generator.R #momentum.R #data-generator.R


