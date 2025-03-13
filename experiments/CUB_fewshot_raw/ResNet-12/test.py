import sys
import os
import torch
import yaml
sys.path.append('../../../')
from models.MyModel import FMSFSDI
from utils import util
from trainers.eval import meta_test


with open('../../../config.yml', 'r') as f:
    temp = yaml.safe_load(f)
data_path = os.path.abspath(temp['data_path'])

test_path = os.path.join(data_path,'CUB_fewshot_raw/test_pre')
model_path = '../../../trained_model_weights/cub_raw_ResNet12.pth'

gpu = 0
torch.cuda.set_device(gpu)

model = FMSFSDI(resnet=True,ffc=True,enable_lfu=True)
model.cuda()
model.load_state_dict(torch.load(model_path,map_location=util.get_device_map(gpu)),strict=True)
model.eval()

with torch.no_grad():
    way = 5
    for shot in [1,5]:
        mean,interval = meta_test(data_path=test_path,
                                model=model,
                                way=way,
                                shot=shot,
                                pre=True,
                                transform_type=None,
                                trial=10000)
        print('%d-way-%d-shot acc: %.3f\t%.3f'%(way,shot,mean,interval))
