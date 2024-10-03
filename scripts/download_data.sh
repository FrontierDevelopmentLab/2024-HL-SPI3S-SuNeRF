#!/bin/bash -l

#PBS -N sst_13392
#PBS -A P22100000
#PBS -q preempt
#PBS -l select=1:ncpus=8:ngpus=2:mem=24gb
#PBS -l walltime=24:00:00

module load conda/latest
module load cuda/11.7.1
conda activate lightning

cd /glade/u/home/rjarolim/projects/SuNeRF

python -m sunerf.data.download.download_aia --download_dir '/glade/work/rjarolim/data/sunerf/2012_08/aia' --email 'robert.jarolim@uni-graz.at' --t_start '2012-08-01T00:00:00' --t_end '2012-08-05T00:00:00'
python -m sunerf.data.download.download_aia --download_dir '/glade/work/rjarolim/data/sunerf/2012_08/aia' --email 'robert.jarolim@uni-graz.at' --t_start '2012-08-05T00:00:00' --t_end '2012-08-10T00:00:00'
python -m sunerf.data.download.download_aia --download_dir '/glade/work/rjarolim/data/sunerf/2012_08/aia' --email 'robert.jarolim@uni-graz.at' --t_start '2012-08-10T00:00:00' --t_end '2012-08-15T00:00:00'
python -m sunerf.data.download.download_aia --download_dir '/glade/work/rjarolim/data/sunerf/2012_08/aia' --email 'robert.jarolim@uni-graz.at' --t_start '2012-08-15T00:00:00' --t_end '2012-08-20T00:00:00'
python -m sunerf.data.download.download_aia --download_dir '/glade/work/rjarolim/data/sunerf/2012_08/aia' --email 'robert.jarolim@uni-graz.at' --t_start '2012-08-20T00:00:00' --t_end '2012-08-25T00:00:00'


python -m sunerf.data.download.download_euvi --download_dir '/glade/work/rjarolim/data/sunerf/2012_08/euvi' --t_start '2012-08-01T00:00:00' --t_end '2012-08-05T00:00:00'
python -m sunerf.data.download.download_euvi --download_dir '/glade/work/rjarolim/data/sunerf/2012_08/euvi' --t_start '2012-08-05T00:00:00' --t_end '2012-08-10T00:00:00'
python -m sunerf.data.download.download_euvi --download_dir '/glade/work/rjarolim/data/sunerf/2012_08/euvi' --t_start '2012-08-10T00:00:00' --t_end '2012-08-15T00:00:00'
python -m sunerf.data.download.download_euvi --download_dir '/glade/work/rjarolim/data/sunerf/2012_08/euvi' --t_start '2012-08-15T00:00:00' --t_end '2012-08-20T00:00:00'
python -m sunerf.data.download.download_euvi --download_dir '/glade/work/rjarolim/data/sunerf/2012_08/euvi' --t_start '2012-08-20T00:00:00' --t_end '2012-08-25T00:00:00'
