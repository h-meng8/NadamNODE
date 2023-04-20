import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

font = {'size'   : 30}

plt.rc('font', **font)

tolerance = 0.0001
names = [
    "gnesterovnode",
    "nadamnode"
]
alt_names = [
    "GNesterovNODE Euler 0.2",
    "NadamNODE Euler 0.2"
]
df_names = {}
for name in names:
    filepath = f"../imgdat/1_2/{name}_mnist_euler_{tolerance}_.csv"
    temp_df = pd.read_csv(filepath, header=None, names=["model", "test#", "train/test", "iter", "loss", "acc", "forwardnfe", "backwardnfe", "time/iter", "time_elapsed"])
    df_names[name] = temp_df
df_names[names[-1]].head()

colors = [
    "red",
    "deepskyblue",
]
line_styles = [
    '--',
    '-.'
]
line_widths = [
    5,
    5
]

fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(1, 2, hspace=0.70, wspace=0.4)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
axes = (ax1, ax2)
height_width_ratio = "auto"

alt_attr_names = ["Accuracy", "Loss"]
offset = 2
for j, attribute in enumerate(["acc", "loss"]):
    axes[j].set_aspect(height_width_ratio)
    for i, name in enumerate(names):
        df_name = df_names[name]
        df_name_train = df_name.loc[df_name["train/test"] == "test"]
        if attribute == "loss":
            df_name_train = df_name.loc[df_name["train/test"] == "train"]
        attr_arr = df_name_train[attribute]
        if attribute == "acc":
            print(f"Accuracy of {name}: {np.max(attr_arr)}")
        iteration_arr = df_name_train["iter"]
        assert attr_arr.shape[0] <= 40 # max number of iterations
        axes[j].plot(iteration_arr, attr_arr, line_styles[i], color=colors[i], linewidth=line_widths[i], label=alt_names[i])
    if attribute == "acc":
        axes[j].set(xlabel="Epoch", ylabel=f"Test {alt_attr_names[j]}")
        #axes[j].set_ylim(0.85, 0.985)
    else:
        axes[j].set(xlabel="Epoch", ylabel=f"Train {alt_attr_names[j]}")

    # if attribute == "forwardnfe":
    #     axes[j].set_ylim(30, 90)
    # plt.legend()
    axes[j].grid()
axbox = axes[0].get_position()
l5 = plt.legend(bbox_to_anchor=(0.5, axbox.y0-0.02), loc="lower center",
                bbox_transform=fig.transFigure, ncol=3)
plt.savefig(f"mnist.pdf", transparent = True, bbox_inches = 'tight', pad_inches = 0)
plt.show()

for i, name in enumerate(names):
    df_name = df_names[name]
    df_name_train = df_name.loc[df_name["train/test"] == "test"]
    attr_arr = df_name_train["acc"]
    print(f"Accuracy of {name}: {np.max(attr_arr)}")

