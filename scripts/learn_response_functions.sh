

module load conda/latest
module load cuda/11.7.1
conda activate lightning

cd /glade/u/home/rjarolim/projects/SuNeRF

python -m sunerf.train.learn_temperature_response_function --response_file '/glade/work/rjarolim/data/sunerf/temperature_response/aia_resonses.npz' --out_file '/glade/work/rjarolim/sunerf/response/aia.pt'
python -m sunerf.train.learn_temperature_response_function --response_file '/glade/work/rjarolim/data/sunerf/temperature_response/stereo_ahead_resonses.npz' --out_file '/glade/work/rjarolim/sunerf/response/stereo_ahead.pt'
python -m sunerf.train.learn_temperature_response_function --response_file '/glade/work/rjarolim/data/sunerf/temperature_response/stereo_behind_resonses.npz' --out_file '/glade/work/rjarolim/sunerf/response/stereo_behind.pt'

python -m sunerf.train.learn_temperature_response_function --response_file '/glade/work/rjarolim/data/sunerf/temperature_response/aia_resonses.npz' --out_file '/glade/work/rjarolim/sunerf/response/psi_mean.pt' --channels 2 3 4
python -m sunerf.train.learn_temperature_response_function --response_file '/glade/work/rjarolim/data/sunerf/temperature_response/aia_resonses.npz' --out_file '/glade/work/rjarolim/sunerf/response/psi_193.pt' --channels 3
