from itertools import product
from pprint import pprint

OPTIMIZED = False
RESUME = False
save_folder = "test"
max_epoch = 2
basic_cmd = (
    lambda config, trainer_name, save_dir, seed:
    f"python {'-O' if OPTIMIZED else ''} main.py Config={config} Trainer.name={trainer_name} Trainer.save_dir={save_folder}/{save_dir} Trainer.max_epoch={max_epoch} Seed={seed}"
    if not RESUME
    else f"python {'-OO' if OPTIMIZED else ''} main.py Config={config} Trainer.name={trainer_name} Trainer.save_dir={save_dir} Trainer.checkpoint_path=runs/{save_dir} Trainer.max_epoch={max_epoch}"
)

trainer_names = [
    "iicgeo",
    "iicmixup",
    "iicvat",
    "iicgeovat",
    "iicgeovatmixup",
    "imsat",
    "imsatvat",
    "imsatmixup",
    "imsatvatmixup",
    "imsatvatgeo",
    "imsatvatgeomixup",
]
save_dirs = trainer_names
datasets = ["mnist", "cifar", "svhn"]
datasets = ["mnist"]
randoms = [1]
randoms = list(range(1, randoms[0] + 1))
cmds = []

for dataset, (trainername, save_dir), rand in product(
        datasets, zip(trainer_names, save_dirs), randoms
):
    cmds.append(
        basic_cmd(
            "config/config_" + dataset.upper() + ".yaml",
            trainername,
            "/".join([dataset + "_" + str(rand), save_dir]),
            rand
        )
    )
cmds = ['"' + item + '" \\' for item in cmds]
# cmds = '\n'.join(cmds)
pprint(cmds, width=120)
with open("cmds.txt", "w") as f:
    for item in cmds:
        f.write("%s\n" % item)
