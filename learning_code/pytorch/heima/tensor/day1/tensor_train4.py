"""
案例：
    创建指定类型的张量。
涉及到的两数：
    type(torch支持的数据类型)
    half()/double()/float()/short()/int()/long()
掌握:
    type
"""

import torch

t1=torch.tensor([1,2,3,4,5,6],dtype=torch.float)
print(f"t1: {t1}, type(t1): {type(t1)}, dtype(t1): {t1.dtype}")

t2=t1.type(torch.int16)
print(f"t2: {t2}, type(t2): {type(t2)}, dtype(t2): {t2.dtype}")

