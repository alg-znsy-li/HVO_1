#!/bin/bash

# 源文件夹和目标文件夹
SOURCE_DIR="/lpai/output/models/grpo/runs/"
DEST_DIR="/lpai/output/tensorboard/"

# 循环每30秒复制一次
while true
do
    # 检查源文件夹是否存在
    if [ -d "$SOURCE_DIR" ]; then
        # 复制文件夹内容到目标文件夹
        cp -r "$SOURCE_DIR"/* "$DEST_DIR"
        echo "Files copied from $SOURCE_DIR to $DEST_DIR at $(date)"
    else
        echo "Source directory does not exist!"
    fi
    
    # 等待30秒
    sleep 30
done

