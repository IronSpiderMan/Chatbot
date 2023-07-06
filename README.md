# 环境搭建
## 1. 安装milvus
### （1）下载yaml文件
```bash
wget https://github.com/milvus-io/milvus/releases/download/v2.2.11/milvus-standalone-docker-compose.yml -O docker-compose.yml
```
### （2）开启milvus
```bash
sudo docker-compose up -d
```

## 2. 安装模块
```bash
pip install -r requirements.txt
```
