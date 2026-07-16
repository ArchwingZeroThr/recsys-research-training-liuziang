from torch.utils.data import Dataset
#此时使用jupyter中使用help(Dataset)查看作用,也可以ctrl+左键
from PIL import Image
import os

#定义类，必须需要重写这三个函数
class MyData(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir=root_dir
        self.label_dir=label_dir
        self.path=os.path.join(self.root_dir,self.label_dir)
        self.img_path=os.listdir(self.path)

    def __getitem__(self, idx):
        img_name=self.img_path[idx]
        img_item_path=os.path.join(self.root_dir,self.label_dir,img_name)
        #注意此处不可使用self.img_path，因为这是一个列表，而join需要字符串
        img=Image.open(img_item_path)
        label=self.label_dir
        return img,label

    def __len__(self):
        return len(self.img_path)

#实例化——蚂蚁
root_dir= "../dataset/train"
ants_label_dir="ants"
bees_label_dir="bees"
#一般来说data和label分文件夹放置
ants_dataset = MyData(root_dir,ants_label_dir)
bees_dataset = MyData(root_dir,bees_label_dir)
#展示图片与label
img1,label1=ants_dataset[0]
img2,label2=bees_dataset[0]

img1.show()
img2.show()
print(f"第一张图是{label1},第二张图是{label2}")