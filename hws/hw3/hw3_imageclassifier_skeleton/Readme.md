# CNN

Question 1
python main.py --cnn --batch-size 100 --device cuda

Question 2


python main.py --kernel 3 --stride 3 --cnn --batch-size 100 --device cuda

python main.py --kernel 14 --stride 1 --cnn --batch-size 100 --device cuda

Question 3 (a)
python main.py --device cuda --cnn

Question 3 (b)
python main.py --cnn --device cuda --sbatch account:gpu
python main.py --shuffle-label --cnn --lr 1e-2 --device cuda --sbatch account:gpu

Question 4
python main.py --cnn --amprec --device cuda --batch-size 100 --sbatch account:gpu
python main.py --cnn --amprec --device cuda --batch-size 500 --sbatch account:gpu

# CGCNN

python structures.py --size 28

python main.py --cgcnn --batch-size 100 --device cuda

python main.py --cgcnn --rot-flip --kernel 5 --stride 1 --batch-size 100 --device cuda 


python main.py --cnn --rot-flip --device cuda
python main.py --cgcnn --rot-flip --device cuda
