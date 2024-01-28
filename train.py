from functools import partial

from datasets import dataloaders
from trainers import trainer, MLI_train
from utils.util import *

args = trainer.train_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
assert args.gpu_num > 0

fewshot_path = dataset_path(args)
pm = trainer.Path_Manager(fewshot_path=fewshot_path, args=args)
train_loader = dataloaders.\
    meta_train_dataloader(data_path=pm.train,
                          way=args.train_way,
                          shots=[args.train_shot, args.train_query_shot],
                          transform_type=args.train_transform_type)

args.save_folder = get_save_path(args)

if args.model == 'MLI':
    train_func = partial(MLI_train.default_train, train_loader=train_loader)

tm = trainer.Train_Manager(args, path_manager=pm, train_func=train_func)

model = load_model(args)
if args.resume:
    model = load_resume_point(args, model)
tm.train(model)
tm.evaluate(model)