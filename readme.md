创建好虚拟环境后，可以安装包了
一般来说，你的pip版本是不够的，需要先升级以下
```python
python -m pip install --upgrade pip
```
然后安装以下numpy（如果不先安装numpy直接安装torch的时候，会比较慢）
```python
pip install numpy
```
然后安装pytorch
```python
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
在文中，有一个地方，如果你要查看数据集图片，你需要安装以下opencv
```python
pip install opencv-python
```