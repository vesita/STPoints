# STPoints

基于SUSTechPoint进行的简单修改，更符合个人的开发习惯。

## 快速开始

```bash
  uv sync
  uv pip install -r requirement.txt
  wget https://github.com/naurril/SUSTechPOINTS/releases/download/0.1/deep_annotation_inference.h5  -P algos/models
```

```fish
  source ./.venv/bin/activate.fish
```

```bash
  source ./.venv/bin/activate.sh
```

最后启动main.py,使用浏览器打开<http://0.0.0.0:8081>

## 说明

目前python版本刚需3.8,否则会因为兼容性问题无法启动main.py
