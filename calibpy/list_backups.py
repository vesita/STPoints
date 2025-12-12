#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
列出配置文件备份的简单脚本
"""

import os


def list_backups(backup_dir="./config/backups"):
    """
    列出所有配置文件备份
    
    Args:
        backup_dir: 备份目录路径
        
    Returns:
        list: 备份文件列表
    """
    if not os.path.exists(backup_dir):
        print(f"备份目录 {backup_dir} 不存在")
        return []
    
    backups = []
    for file in os.listdir(backup_dir):
        if file.startswith("config_backup_") and file.endswith(".json"):
            backups.append(os.path.join(backup_dir, file))
    
    # 按时间排序
    backups.sort(reverse=True)
    return backups


def main():
    backups = list_backups()
    if backups:
        print("可用的配置文件备份:")
        for i, backup in enumerate(backups):
            print(f"{i+1}. {os.path.basename(backup)}")
    else:
        print("没有找到配置文件备份")


if __name__ == "__main__":
    main()