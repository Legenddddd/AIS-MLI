import argparse
from utils.util import *
from trainers.eval import meta_test



def test_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resnet", action="store_true")
    parser.add_argument("--model", choices=[ 'MLI'], default='MLI')
    parser.add_argument("--dataset", choices=['cub_cropped', 'cub_raw',
                                              'stanford_dog',
                                              'meta_iNat', 'tiered_meta_iNat'], default='cub_cropped')
    parser.add_argument("--train_way", help="training way", type=int, default=10)

    parser.add_argument("--train_shot", help="number of support images per class for meta-training and meta-testing during validation", type=int,default=5)
    parser.add_argument("--train_query_shot", help="number of query images per class during meta-training", type=int, default=15)
    parser.add_argument("--test_transform_type", help="size transformation type during inference", type=int)
    parser.add_argument("--pre", help="whether use pre-resized 84x84 images for val and test", action="store_true")

    parser.add_argument("--disturb_num", help="channel number", type=int, default=3)
    parser.add_argument("--short_cut_weight", help="short cut weight", type=float, default=0)

    parser.add_argument("--model_path", type=str)
    args = parser.parse_args()

    return args


args = test_parser()

test_path = dataset_path(args)
if args.pre:
    test_path = os.path.join(test_path, 'test_pre')
else:
    test_path = os.path.join(test_path, 'test')

gpu = 3
torch.cuda.set_device(gpu)

model = MLI(args=args)
model.cuda()
model.load_state_dict(torch.load(args.model_path, map_location=get_device_map(gpu)),strict=True)
model.eval()


with torch.no_grad():
    for way in [5]:
        for shot in [1,5]:
            mean,interval = meta_test(data_path=test_path,
                                model=model,
                                way=way,
                                shot=shot,
                                pre=args.pre,
                                transform_type=args.test_transform_type,
                                gpu_num = 1,
                                trial=2000)
            print('%d-way-%d-shot acc: %.3f\t%.3f'%(way,shot,mean,interval))