#!/bin/bash

# 设置颜色变量
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 显示主菜单
show_menu() {
    echo -e "\n${YELLOW}请选择要执行的操作：${NC}"
    echo -e "${GREEN}1${NC} \t 检查并安装环境"

    echo -e "${GREEN}2${NC} \t 运行训练"
    echo -e "${GREEN}3${NC} \t 运行模型"

    echo -e "${GREEN}login${NC} \t 登录到子系统"
    echo -e "${GREEN}exit${NC} \t 退出"
    echo -en "${YELLOW}\n请输入：${NC}"
}

# 运行安装脚本功能
run_install_script() {
    echo -e "\n${BLUE}=== 运行安装脚本 ===${NC}"

    # 检查 install.sh 文件是否存在
    if [[ ! -f "install.sh" ]]; then
        echo -e "${RED}错误：未找到 install.sh 文件！${NC}"
        echo -e "${YELLOW}请确保 install.sh 文件存在于当前目录：$(pwd)${NC}"
        read -r -p "按回车键继续..."
        return
    fi

    # 检查 install.sh 是否有执行权限
    if [[ ! -x "install.sh" ]]; then
        echo -e "${YELLOW}install.sh 没有执行权限，正在添加执行权限...${NC}"
        chmod +x install.sh
        # shellcheck disable=SC2181
        if [[ $? -ne 0 ]]; then
            echo -e "${RED}错误：无法为 install.sh 添加执行权限！${NC}"
            read -r -p "按回车键继续..."
            return
        fi
        echo -e "${GREEN}✓ 已添加执行权限${NC}"
    fi

    echo -e "${YELLOW}即将运行 install.sh 脚本...${NC}"
    echo -e "${YELLOW}脚本路径：$(pwd)/install.sh${NC}"

    # 确认运行
    read -r -p "确定要运行安装脚本吗？(y/N): " confirm_run
    if [[ $confirm_run != "y" && $confirm_run != "Y" ]]; then
        echo -e "${YELLOW}已取消运行安装脚本${NC}"
        read -r -p "按回车键继续..."
        return
    fi

    echo -e "\n${GREEN}开始执行 install.sh...${NC}"
    echo -e "${BLUE}================================${NC}"

    # 运行 install.sh 脚本
    ./install.sh

    # 获取脚本执行结果
    install_result=$?

    echo -e "${BLUE}================================${NC}"

    if [[ $install_result -eq 0 ]]; then
        echo -e "${GREEN}✓ install.sh 执行完成！${NC}"
    else
        echo -e "${RED}✗ install.sh 执行失败，退出代码：$install_result${NC}"
        echo -e "${YELLOW}请检查 install.sh 脚本内容${NC}"
    fi

    read -r -p "按回车键继续..."
}

# 检查并设置设备名称、环境
create_venv() {
    # 获取目录名称
    read -r -p "请输入设备名称: " device_name

    # 定义要在 Ubuntu 环境中执行的命令
    UBUNTU_COMMANDS="
    echo '=== 在 Ubuntu 环境中 ==='
    echo '当前目录：$PWD'
    echo '用户：$(whoami)'

    echo -e \"\n${BLUE}=== 创建设备目录 ===${NC}\"
    # 检查设备目录是否已存在
    if [[ -z \"$device_name\" ]]; then
        echo -e \"${RED}错误：设备目录名称不能为空！${NC}\"
        read -r -p \"按回车键继续...\"
    fi
    # 创建设备目录
    if mkdir -p \"./$device_name\"; then
        echo -e \"${RED}✗ 设备目录创建成功！${NC}\"
    else
        echo -e \"${RED}✗ 设备目录创建失败！${NC}\"
        echo -e \"${YELLOW}请检查路径权限或磁盘空间${NC}\"
    fi
    "

    # 使用 proot-distro 登录到 Ubuntu 并执行命令
    proot-distro login ubuntu -- bash -c "$UBUNTU_COMMANDS"

    read -r -p "按回车键继续..."
}

# 拉取模型
pull_model() {
    read -r -p "按回车键继续..."
}

# 运行训练
run_train() {
    # 检查当前目录是否为 /root
    if [ "$PWD" != "/root" ]; then
        echo "当前不在 /root 目录，切换到 Ubuntu 环境并执行命令..."

        # 定义要在 Ubuntu 环境中执行的命令
        UBUNTU_COMMANDS="
        echo '=== 在 Ubuntu 环境中 ==='
        echo '当前目录：$PWD'
        echo '用户：$(whoami)'

        source pytorch_env/bin/activate
        cd train_example_1
        python3 start_device.py
        "

        # 使用 proot-distro 登录到 Ubuntu 并执行命令
        proot-distro login ubuntu -- bash -c "$UBUNTU_COMMANDS"
    else
        echo "当前已在 /root 目录"
        echo "当前目录：$PWD"
    fi

    read -r -p "按回车键继续..."
}

# 运行模型
run_model() {
    # TODO 自生成小模型
    # TODO HuggingFace大模型
    echo -e "${GREEN}3${NC} \t 加载自生成小模型"
    echo -en "${YELLOW}\n请选择：${NC}"

    read -r -p "按回车键继续..."
}

# 主循环
main() {
    while true; do
        clear
        show_menu

        read -r choice

        case $choice in
            1)
                run_install_script
                ;;
            2)
                run_train
                ;;
            3)
                run_model
                ;;

            login|LOGIN)
                proot-distro login ubuntu
                exit 0
                ;;
            exit|EXIT)
                echo -e "${GREEN}感谢使用西湖创建工具！再见！${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}无效的选择，请重新输入${NC}"
                read -r -p "按回车键继续..."
                ;;
        esac
    done
}

# 脚本入口
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main
fi
