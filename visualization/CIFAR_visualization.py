import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

font = {'size'   : 30}
plt.rc('font', **font)

tolerance = 0.0001
names = ["nesterovnode", "gnesterovnode", "nadamnode"]
alt_names = ["NesterovNODE", "GNesterovNODE", "NadamNODE"]
df_names = {}
for name in names:
	filepath = f"../output/cifar/{name}/{name}_{tolerance}_.csv"
	print("filepath:", filepath)
	# if name == "ghbnode":
	# 	filepath = f"../imgdat/1_2/backup/{name}_{tolerance}.csv"
	df = pd.read_csv(filepath, header=None, names=["iter", "loss", "acc", "totalnfe", "forwardnfe", "time/iter", "time_elapsed"])
	df["train/test"] = np.where(pd.isnull(df["totalnfe"]), "test", "train")
	df["backwardnfe"] = np.where(df["train/test"] == "test", 0, df["totalnfe"] - df["forwardnfe"])
	df["forwardnfe"] = np.where(df["train/test"] == "test", df["totalnfe"], df["forwardnfe"])
	df_names[name] = df

print(df_names[names[-1]].head(20))

colors = [
	"mediumvioletred",
	"red",
	"deepskyblue"
]
line_styles = [
	':',
	'--',
	'-.'
]
line_widths = [
	5,
	5,
	5
]

fig = plt.figure(figsize=(25, 15))
gs = fig.add_gridspec(2, 2, hspace=0.8, wspace=0.5)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

ax4 = fig.add_subplot(gs[1, 0])
ax5 = fig.add_subplot(gs[1, 1])

axes = (ax1, ax2)
alt_attr_names = ["Train Forward NFEs", "Train Backward NFEs"]
for j, attribute in enumerate(["forwardnfe", "backwardnfe"]):
	for i, name in enumerate(names):
		df_name = df_names[name]
		df_name_train = df_name.loc[df_name["train/test"] == "train"]
		if attribute == "acc":
			df_name_train = df_name.loc[df_name["train/test"] == "test"]
		attr_arr = df_name_train[attribute]
		iteration_arr = df_name_train["iter"]
		assert attr_arr.shape[0] <= 40 # max number of iterations
		axes[j].plot(iteration_arr, attr_arr, line_styles[i], linewidth=line_widths[i], color=colors[i], label=alt_names[i])
	if attribute == "acc":
		axes[j].set_xlim((0, 11))
		axes[j].set_ylim((0.5, 0.7))

	axes[j].set(xlabel="Epoch", ylabel=f"{alt_attr_names[j]}")
	axes[j].grid()


axes = (ax5, ax4)
alt_attr_names = ["Test Accuracy", "Train Loss"]
for j, attribute in enumerate(["acc", "loss"]):
	for i, name in enumerate(names):
		df_name = df_names[name]
		df_name_train = df_name.loc[df_name["train/test"] == "train"]
		if attribute == "acc":
			df_name_train = df_name.loc[df_name["train/test"] == "test"]
		attr_arr = df_name_train[attribute]
		iteration_arr = df_name_train["iter"]
		assert attr_arr.shape[0] <= 40 # max number of iterations
		axes[j].plot(iteration_arr, attr_arr, line_styles[i], linewidth=line_widths[i], color=colors[i], label=alt_names[i])
	if attribute == "acc":
		axes[j].set_xlim((0, 11))
		axes[j].set_ylim((0.5, 0.65))
	if attribute == "loss":
		axes[j].set_xlim((0, 11))
		axes[j].set_ylim((0.6, 1.8))

	axes[j].set(xlabel="Epoch", ylabel=f"{alt_attr_names[j]}")
	axes[j].grid()


axbox = axes[-1].get_position()
plt.legend(bbox_to_anchor=(0.5, axbox.y0+0.28), loc="lower center",
		   bbox_transform=fig.transFigure, ncol=4, handletextpad=0.5, columnspacing=0.6, borderpad=0.3)
#plt.savefig(f"visualization/cifar.pdf", transparent = True, bbox_inches = 'tight', pad_inches = 0)
plt.show()
