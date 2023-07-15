
# VoteNet for Single Target.

## synthetic_dataset

生成 1w 个虚拟场景，每个场景点云大小应为 512 ~ 10000!!!!!

__data 读取三个文件：__

xxxxxx_bbox.npy
1,6 (x,y,z,euler)

xxxxxx_pc.npz ['pc']
N,3 (x,y,z)

xxxxxx_votes.npz
N,4 (bool,dx,dy,dz)

## log

__loss:__

使用 tensorboard 查看 events.out.tfevents 文件。






