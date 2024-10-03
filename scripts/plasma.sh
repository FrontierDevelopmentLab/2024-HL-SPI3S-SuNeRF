#!/bin/bash -l

#PBS -N SuNeRF
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=16:ngpus=2:mem=64gb
#PBS -l walltime=12:00:00

module load conda/latest
module load cuda/11.7.1
conda activate lightning

cd /glade/u/home/rjarolim/projects/SuNeRF

#python -m sunerf.data.euv.prep_aia --data_path "/glade/work/rjarolim/data/sunerf/2012_08/aia/*.fits" --out_path "/glade/work/rjarolim/data/sunerf/2012_08_prep/aia" --resolution 512
#python -m sunerf.data.euv.prep_euvi --data_path "/glade/work/rjarolim/data/sunerf/2012_08/euvi_prep/*.fts" --out_path "/glade/work/rjarolim/data/sunerf/2012_08_prep/euvi"
#python -m sunerf.data.euv.prep_psi --data_path "/glade/work/rjarolim/data/sunerf/psi_data/psi_data/**/*.fits" --out_path "/glade/work/rjarolim/data/sunerf/psi_data_prep" --resolution 1024


#python -i -m sunerf.run_plasma --config "config/aia_2012_08-193.yaml"
#python -i -m sunerf.run_plasma --config "config/all_2012_08-193.yaml"
python -i -m sunerf.run_plasma --config "config/psi.yaml"

# Debugging
#python -i -m sunerf.run_plasma --config "config/psi_plasma_193.yaml"


########### EVALUATION ###########

#python -i -m  sunerf.evaluation.video --chk_path "/glade/work/rjarolim/sunerf/psi_v01/save_state.snf" --video_path "/glade/work/rjarolim/sunerf/psi_v01/video"
