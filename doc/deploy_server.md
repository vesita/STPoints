# uWSGI 部署

## 安装

```
pip install uwsgi
```

## 配置

uwsgi 配置文件 `uwsgi.ini`（默认端口 8092）：

```ini
[uwsgi]
http = 0.0.0.0:8092
chdir = /root/SUSTechPOINTS
module = main:application
master = true
buffer-size = 65536
processes = 4
threads = 2
```

## 使用 CherryPy 直接运行（推荐，默认端口 8081）

```
python main.py
```

然后访问 http://127.0.0.1:8081

## uwsgi 运行

```
uwsgi --ini uwsgi.ini
```

然后访问 http://127.0.0.1:8092
