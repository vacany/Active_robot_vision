import argparse
from torchpack.utils.config import configs

from core import builder

# pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
parser = argparse.ArgumentParser()
parser.add_argument('config', metavar='FILE', help='config file')
parser.add_argument('--run-dir', metavar='DIR', help='run directory')
args, opts = parser.parse_known_args()

configs.load(args.config, recursive=True)
configs.update(opts)

# dataset = builder.make_dataset()

model = builder.make_model()


criterion = builder.make_criterion()
optimizer = builder.make_optimizer(model)
scheduler = builder.make_scheduler(optimizer)
