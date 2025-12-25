##
## This script is sourced by /data/data/com.termux/files/usr/bin/login before executing shell.
##

# shellcheck disable=SC2028
echo "正在进行初始化安装"

# 更新系统 禁用所有交互提示
apt-get update -y --allow-unauthenticated --allow-downgrades --allow-remove-essential --allow-change-held-packages

# 检查并安装 proot-distro
echo "🔴 正在检查 proot-distro..."
if ! command -v proot-distro > /dev/null; then
    echo "❌ 错误：proot-distro 未安装，请先安装proot-distro"
    echo "🔴 正在安装 proot-distro..."
    pkg install proot-distro -y
    echo "✅ 安装 proot-distro 完成"
else
    echo "✅ 检查 proot-distro 完成"
fi

# 通过检查Ubuntu目录是否存在来判断是否已安装
echo "🔴 检查Ubuntu子系统是否已安装..."
if [ -d "/data/data/com.termux/files/usr/var/lib/proot-distro/installed-rootfs/ubuntu" ]; then
    echo "✅ Ubuntu子系统已安装，跳过安装"
else
    echo "🔴 Ubuntu子系统未安装，开始安装..."
    if ! proot-distro install ubuntu; then
        echo "❌ Ubuntu安装失败，请检查网络连接和存储空间"
        exit 1
    fi
    echo "✅ Ubuntu安装完成"
fi

proot-distro login ubuntu -- bash -c "
echo '🔴 开始更新软件包列表...'
apt-get update -y --allow-unauthenticated --allow-downgrades --allow-remove-essential --allow-change-held-packages

echo '🔴 检查并安装Python...'
if ! command -v python3 &> /dev/null; then
    echo '🔴 Python未安装，开始安装...'
    apt install python3 -y
else
    echo '✅ Python已安装，跳过安装'
fi

echo '🔴 检查并安装pip...'
if ! command -v pip3 &> /dev/null; then
    echo '🔴 pip未安装，开始安装...'
    apt install python3-pip -y
else
    echo '✅ pip已安装，跳过安装'
fi

echo '🔴 安装虚拟环境：python3-venv'
if [ ! -d 'pytorch_env' ]; then
    echo '🔴 正在安装 python3-venv...'
    apt install -y python3-venv
    python3 -m venv pytorch_env
    #echo -e \"\n# 自动激活虚拟环境\nif [ -f \"pytorch_env/bin/activate\" ]; then\n    source pytorch_env/bin/activate\nfi\" >> ~/.bashrc
fi
source pytorch_env/bin/activate

echo '🔴 安装PyTorch及相关机器学习库...'
if pip3 list | grep -q 'torch'; then
    echo '✅ PyTorch已安装，跳过安装'
else
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    echo '✅ PyTorch及相关机器学习库安装完成'
fi

echo '🔴 安装常用数据科学库...'
if pip3 list | grep -q 'numpy'; then
    echo '✅ 常用数据科学库已安装，跳过安装'
else
    pip3 install numpy pandas matplotlib scikit-learn jupyter
    echo '✅ 常用数据科学库安装完成'
fi


echo '🔴 安装其他实用工具...'
if pip3 list | grep -q 'requests'; then
    echo '✅ 其他实用工具已安装，跳过安装'
else
    pip3 install requests beautifulsoup4 flask django
    echo '✅ 其他实用工具安装完成'
fi

echo '🔴 验证PyTorch安装...'
python3 -c \"
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
print('✅ PyTorch安装成功!')
\"

echo '🔴 安装示例代码：训练1'
mkdir -p train_example_1
cp --update=none /data/data/com.termux/files/home/train_example_1/* ~/train_example_1/

echo '✅ 所有安装完成！'
"
