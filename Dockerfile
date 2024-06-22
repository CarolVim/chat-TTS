# 使用官方的 Python 3.10 镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 复制当前目录内容到工作目录
COPY . .

# 安装 virtualenv
RUN pip install --no-cache-dir virtualenv

# 复制并激活现有的虚拟环境
COPY py310 /app/py310
ENV VIRTUAL_ENV=/app/py310
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 清除 pip 缓存以避免哈希冲突
RUN pip cache purge

# 确保虚拟环境中的 pip 是最新的
RUN pip install --no-cache-dir --upgrade pip

# 安装依赖项
RUN pip install --no-cache-dir -r requirements.txt

# 运行 api5.py 文件
CMD ["python", "api5.py"]
