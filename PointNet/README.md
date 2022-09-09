## PointNet

![teaser](C:\Users\Mike\Desktop\rep-pointnet\PointNet\misc\teaser.png)

### ModelNet40

```bash
cd PointNet
```

下载 modelnet40_normal_resampled 数据集

```bash
!unzip -oq ./modelnet40_normal_resampled.zip -d .
```

然后运行

```bash
python train_cls.py
```

### ShapeNet

```bash
cd PointNet
```

下载 modelnet40_normal_resampled 数据集

```bash
!unzip -oq ./shapenet.zip -d .
```

然后运行

```bash
python train_partseg.py
```

