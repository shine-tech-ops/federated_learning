#!/bin/bash

# Regional Node 启动脚本

echo "🚀 启动 Regional Node..."

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "📦 创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
echo "🔧 激活虚拟环境..."
source .regional_venv/bin/activate

# 安装依赖
echo "📥 安装依赖..."
pip install -r requirements.txt

# 创建 .env 文件（如果不存在）
if [ ! -f ".env" ]; then
    echo "📄 创建 .env 文件..."
    cp env.example .env
    echo "✅ .env 文件已创建，请根据需要修改配置"
fi

# 启动服务
echo "🎯 启动 Regional Node..."
python run.py
