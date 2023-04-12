#
set -e



#
python main_ddi.py --sbatch gpu --device cuda
python main_ddi.py --sbatch gpu --device cuda --positional



#
sbatch sbatch/ddi_structure.sh
sbatch sbatch/ddi_position.sh
