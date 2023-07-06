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

# 运行应用

## 1. 下载

```bash
git clone https://github.com/IronSpiderMan/Chatbot.git 
```

## 2. 准备question_answer.csv文件

格式如下：

```csv
id,question,answer
0,this is question,this is answer
...
```

## 3、导入数据

```bash
python insert.py
```

## 4. 运行项目

```bash
uvicorn main:app --reload
```

## 5. 查看api

```
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/redoc
```
