python test.py --model MLI --dataset cub_cropped      --disturb_num 5 --short_cut_weight 0.05  --model_path /data2/zhaolijun/PycharmProjects/AIS-MLI-main/checkpoints/CUB_fewshot_cropped/model_Conv-4.pth  --pre
python test.py --model MLI --dataset stanford_dog     --disturb_num 5 --short_cut_weight 0.05  --model_path /data2/zhaolijun/PycharmProjects/AIS-MLI-main/checkpoints/StanfordDogs_fewshot/model_Conv-4.pth  --pre
python test.py --model MLI --dataset meta_iNat        --disturb_num 5 --short_cut_weight 0.05  --model_path /data2/zhaolijun/PycharmProjects/AIS-MLI-main/checkpoints/meta_iNat/model_Conv-4.pth  --test_transform_type 2
python test.py --model MLI --dataset tiered_meta_iNat --disturb_num 5 --short_cut_weight 0.05  --model_path /data2/zhaolijun/PycharmProjects/AIS-MLI-main/checkpoints/tiered_meta_iNat/model_Conv-4.pth  --test_transform_type 2
python test.py --model MLI --dataset cub_cropped      --disturb_num 1 --short_cut_weight 0.3   --model_path /data2/zhaolijun/PycharmProjects/AIS-MLI-main/checkpoints/CUB_fewshot_cropped/model_ResNet-12.pth  --pre  --resnet
python test.py --model MLI --dataset cub_raw          --disturb_num 1 --short_cut_weight 0.3   --model_path /data2/zhaolijun/PycharmProjects/AIS-MLI-main/checkpoints/CUB_fewshot_raw/model_ResNet-12.pth  --pre  --resnet
python test.py --model MLI --dataset stanford_dog     --disturb_num 1 --short_cut_weight 0.3   --model_path /data2/zhaolijun/PycharmProjects/AIS-MLI-main/checkpoints/StanfordDogs_fewshot/model_ResNet-12.pth  --pre  --resnet