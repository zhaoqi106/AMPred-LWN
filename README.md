# AMPred-LWN



---
You can get full model at
通过网盘分享的文件：AMPred-LWN.zip
链接: https://pan.baidu.com/s/1MA7SbC4cCVkumNGu6qmHFg?pwd=airi 提取码: airi 
--来自百度网盘超级会员v9的分享
which include Mamba architecture locally.



### Mamba-2 installation (important)

AMPred-LWN uses Mamba-related CUDA extensions for speed (e.g., `mamba-ssm`, `causal-conv1d`).
 If you have a CUDA-enabled environment, **install/build CUDA kernels** (recommended):
Mamba-ssm we have already packed a full folder in baidunetdisk.

```
# Build CUDA extensions (recommended for speed)
MAMBA_FORCE_BUILD=1 CAUSAL_CONV1D_FORCE_BUILD=1 \
pip install  --no-binary causal-conv1d \
  --force-reinstallcausal-conv1d
```

---
Happy modelling! 🎉
