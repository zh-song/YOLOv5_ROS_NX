## NVIDIA SDK Manager 1.8.4 deb
1. host install this Manager.deb
2. Terminal input "sdkmanager", at the same time connneting Host and Xavier with UAB tpye-C and press Power button
3. setp 1 -> 4 :

    - speicially, 重新插入电源线，先按 force recovery 按钮不放开，然后再按 Power 按钮，直到灯亮起来进入 recover mode
    - step 3 先等 Xavier 中 Linux 系统安装好（可通过查看 HDMI 显示器），再输入用户及密码
    
4. sudo jtop (刷机自带TensorRt)
5. IP 192.168.55.1 usename zhsong password 123

## Miniforge.sh 
scp 命令传输文件 miniforge.sh 至 Xavier 上，其作用与 conda 相同（可不安装，此处不安装）

## Pytorch
1. Pytorch 与 Python 版本相对应
2. Pytorch 从 [Pytorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048) 中下载
3. 并根据 torch 版本安装 torchvision
4. 如果采用 miniforge 需将 cv2 链接到虚拟环境

## Deepstream

```
sudo apt install libssl1.1 \\
deepstream-app --version-all
```
```
deepstream-app version 6.1.1
DeepStreamSDK 6.1.1
CUDA Driver Version: 11.4
CUDA Runtime Version: 11.4
TensorRT Version: 8.4
cuDNN Version: 8.4
libNVWarp360 Version: 2.0.1d3
```
