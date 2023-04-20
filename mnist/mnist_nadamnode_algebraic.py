from os import path
import sys

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from anode_data_loader import mnist
from base import *
from mnist.mnist_train import train
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--tol', type=float, default=1e-4)
parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--visualize', type=eval, default=True)
parser.add_argument('--niters', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--start', type=float, default=1.0)
parser.add_argument('--end', type=float, default=2.0)
parser.add_argument('--timeout', type=int, default=10000, help='Maximum time/iter to do early dropping')
parser.add_argument('--nfe-timeout', type=int, default=300, help='Maximum nfe (forward or backward) to do early dropping')
parser.add_argument('--names', nargs='+', default=['node', 'anode', 'sonode', 'hbnode', 'ghbnode', 'nnode', 'nesterovnode', 'gnesterovnode'], help="List of models to run")
parser.add_argument('--log-file', default="outdat1", help="name of the logging csv file")
args = parser.parse_args()


randomSeed = 1
torch.manual_seed(randomSeed)
torch.cuda.manual_seed(randomSeed)
torch.cuda.manual_seed_all(randomSeed)  # if use multi-GPU
np.random.seed(randomSeed)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
# shape: [time, batch, derivatives, channel, x, y]
dim = 5
hidden = 50
trdat, tsdat = mnist(batch_size=64)


class initial_velocity(nn.Module):

    def __init__(self, in_channels, out_channels, nhidden):
        super(initial_velocity, self).__init__()
        assert (3 * out_channels >= in_channels)
        self.actv = nn.LeakyReLU(0.3)
        self.fc1 = nn.Conv2d(in_channels, nhidden, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(nhidden, nhidden, kernel_size=3, padding=1)
        self.fc3 = nn.Conv2d(nhidden, 2 * out_channels - in_channels, kernel_size=1, padding=0)
        self.out_channels = out_channels
        self.in_channels = in_channels

    def forward(self, x0):
        x0 = x0.float()
        # out = repeat(torch.zeros_like(x0[:,:1]), 'b 1 ... -> b d c ...', d=2, c=self.out_channels)
        # out[:, 0, :self.in_channels] = x0

        # """
        out = self.fc1(x0)
        out = self.actv(out)
        out = self.fc2(out)
        out = self.actv(out)
        out = self.fc3(out)
        out = torch.cat([x0, out], dim=1)
        out = rearrange(out, 'b (d c) ... -> b d c ...', d=2)
        # """
        return out

class adamnode_initial_velocity(nn.Module):

    def __init__(self, in_channels, out_channels, nhid):
        super(adamnode_initial_velocity, self).__init__()
        assert (3 * out_channels >= in_channels)
        self.actv = nn.LeakyReLU(0.3)
        self.fc1 = nn.Conv2d(in_channels, nhid, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(nhid, nhid, kernel_size=3, padding=1)
        self.fc3 = nn.Conv2d(nhid, 7, kernel_size=1, padding=0)
        self.out_channels = out_channels
        self.in_channels = in_channels

    def forward(self, x0):
        x0 = x0.float()
        out = self.fc1(x0)
        out = self.actv(out)
        out = self.fc2(out)
        out = self.actv(out)
        out = self.fc3(out)
        out_v = out * out
        # import pdb; pdb.set_trace()
        out = torch.cat([x0, out, out_v], dim=1)
        out = rearrange(out, 'b (d c) ... -> b d c ...', d=3)
        # import pdb; pdb.set_trace()
        return out

class DF(nn.Module):

    def __init__(self, in_channels, nhidden):
        super(DF, self).__init__()
        #self.activation = nn.LeakyReLU(0.01)
        self.activation = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(in_channels + 1, nhidden, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(nhidden + 1, nhidden, kernel_size=3, padding=1)
        self.fc3 = nn.Conv2d(nhidden + 1, in_channels, kernel_size=1, padding=0)

    def forward(self, t, x0):
        out = rearrange(x0, 'b 1 c x y -> b c x y')
        t_img = torch.ones_like(out[:, :1, :, :]).to(device=args.gpu) * t
        out = torch.cat([out, t_img], dim=1)
        out = self.fc1(out)
        out = self.activation(out)
        out = torch.cat([out, t_img], dim=1)
        out = self.fc2(out)
        out = self.activation(out)
        out = torch.cat([out, t_img], dim=1)
        out = self.fc3(out)
        out = rearrange(out, 'b c x y -> b 1 c x y')
        return out


#class predictionlayer(nn.Module):
#    def __init__(self, in_channels):
#        super(predictionlayer, self).__init__()
#        self.dense = nn.Linear(in_channels * 28 * 28, 10)
#
#    def forward(self, x):
#        x = rearrange(x[:, 0], 'b c x y -> b (c x y)')
#        x = self.dense(x)
#        return x
class predictionlayer(nn.Module):
    def __init__(self, in_channels, truncate=False, dropout=0.0):
        super(predictionlayer, self).__init__()
        self.dense = nn.Linear(in_channels * 28 * 28, 10)
        self.truncate = truncate
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.truncate:
            x = rearrange(x[:, 0], 'b ... -> b (...)')
        else:
            x = rearrange(x, 'b ... -> b (...)')
        x = self.dropout(x)
        x = self.dense(x)
        return x

#file = open('./data/0.txt', 'a')

for name in args.names:
    print(args.start)
    print(args.end)
    hblayer = NODElayer(NadamNODE2(DF(dim, hidden), None, use_h=True, sign=-1), adam=True, nesterov_algebraic=True, time_requires_grad=False, tol=args.tol, evaluation_times=torch.Tensor([args.start, args.end]))
    model = nn.Sequential(adamnode_initial_velocity(1, dim, hidden), hblayer, predictionlayer(dim, truncate=True)).to(device=args.gpu)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.000)
    evaluation_times_folder = "1_2"
    train(model, optimizer, trdat, tsdat, args, name, 0, evalfreq=1,
                        csvname=f'imgdat/{evaluation_times_folder}/{args.log_file}_{args.tol}_seed_{randomSeed}.csv')
    


