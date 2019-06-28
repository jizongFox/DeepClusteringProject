from pprint import pprint

OPTIMIZED = True
RESUME = False
max_epoch = 2
basic_cmd = lambda config, trainer_name, save_dir: \
    f"python {'-O' if OPTIMIZED else ''} main.py Config={config} Trainer.name={trainer_name} Trainer.save_dir={save_dir} Trainer.max_epoch={max_epoch}" if not RESUME else \
        f"python {'-OO' if OPTIMIZED else ''} main.py Config={config} Trainer.name={trainer_name} Trainer.save_dir={save_dir} Trainer.checkpoint_path=runs/{save_dir} Trainer.max_epoch={max_epoch}"

trainer_names = [
    'iicgeo',
    'iicmixup',
    'iicvat',
    'iicgeovat',
    'imsatvat',
    'imsatmixup',
    'imsatvatgeo',
    'imsatvatgeomixup',
]
save_dirs = [
    'iicgeo',
    'iicmixup',
    'iicvat',
    'iicgeovat',
    'imsatvat',
    'imsatmixup',
    'imsatvatgeo',
    'imsatvatgeomixup',
]
datasets = [
    'mnist',
    'cifar',
    'svhn'
]
randoms = [
    1
]
cmds = []
from itertools import product

for dataset, (trainername, save_dir), rand in product(datasets, zip(trainer_names, save_dirs), randoms):
    cmds.append(basic_cmd("config/config_" + dataset.upper() + '.yaml', trainername,
                          '/'.join([dataset + "_" + str(rand), save_dir])))
cmds = ['"'+item+'" \ ' for item in cmds]
# cmds = '\n'.join(cmds)
pprint(cmds, width=120)
with open('cmds.txt', 'w') as f:
    for item in cmds:
        f.write("%s\n" % item)
