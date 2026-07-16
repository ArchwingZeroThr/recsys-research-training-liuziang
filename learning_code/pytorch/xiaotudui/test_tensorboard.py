#用于展示训练中的事件（比如loss函数图像）
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
#在命令行输入tensorboard --logdir=logs,可以通过--port=6007修改端口名
writer = SummaryWriter("logs")

img_path= '../dataset/train/ants/0013035.jpg'
img_PIL=Image.open(img_path)
img_array=np.array(img_PIL)
writer.add_image("test",img_array,1,dataformats='HWC')
'''其需要tensor或numpy.array格式,因此不常用PIL,而是使用Opencv,此处还有一个问题从PIL到numpy，需要在add_imageO中
指定shape中每一个数字/维表示的含义'''


# for i in range(100):
#     writer.add_scalar("y=x",i,i)#其中scalar_value是y轴,global_step=None是x轴


writer.close()
