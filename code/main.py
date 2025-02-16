import os

try:
    cmd = 'python dfgp_sam_new_mixup_main_pmnist.py'
    os.system(cmd)
except:
    print(cmd + "Error!!")

try:
    cmd = 'python dfgp_sam_new_mixup_main_cifar100.py'
    os.system(cmd)
except:
    print(cmd + "Error!!")
