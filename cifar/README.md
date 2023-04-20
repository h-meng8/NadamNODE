# Experiments with NODE-based models on CIFAR10

## Training
The code must be run from main directory of the repository, not from this directory. For example, if your working directory is ```cifar```, first go up one level
```
cd ..
```
Then run the training process using this command:
```
python cifar/main.py --tol 0.0001  --batch-size 64 --model <model-name>
```
For example, if you want to run the experiment with NadamNODE, then the command is as follows:
```
python cifar/main.py --tol 0.0001 --batch-size 64 --model nadamnode3
```
Modify this command if you want to run with other models.
