# Experiments with NODE-based models on MNIST

## Training
The code must be run from main directory of the repository, not from this directory. For example, if your working directory is ```mnist```, first go up one level
```
cd ..
```
Then run the training process using this command:
```
python mnist/mnist-full-run.py --names <model-name> --log-file <model-log-file-name>
```
For example, if you want to run the experiment with NadamNODE with training log file name nadamnode, then the command is as follows:
```
python mnist/mnist-full-run.py --gpu 0 --names nadamnode3 --log-file nadamnode
```
Modify this command if you want to run with other models.

