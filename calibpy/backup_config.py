#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
备份和恢复配置文件的工具脚本
"""

import json
import os
import shutil
from datetime import datetime


def backup_config(config_file_path="./config/config.json", backup_dir="./config/backups"):
    """
    备份配置文件
    
    Args:
        config_file_path: 配置文件路径
        backup_dir: 备份目录路径
        
    Returns:
        bool: 备份是否成功
    """
    # 检查配置文件是否存在
    if not os.path.exists(config_file_path):
        print(f"配置文件 {config_file_path} 不存在")
        return False
    
    # 创建备份目录
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    # 生成备份文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"config_backup_{timestamp}.json"
    backup_path = os.path.join(backup_dir, backup_filename)
    
    try:
        # 复制配置文件到备份目录
        shutil.copy2(config_file_path, backup_path)
        print(f"配置文件已备份到: {backup_path}")
        return True
    except Exception as e:
        print(f"备份配置文件时出错: {e}")
        return False


def restore_config(backup_file_path, config_file_path="./config/config.json"):
    """
    从备份恢复配置文件
    
    Args:
        backup_file_path: 备份文件路径
        config_file_path: 配置文件路径
        
    Returns:
        bool: 恢复是否成功
    """
    # 检查备份文件是否存在
    if not os.path.exists(backup_file_path):
        print(f"备份文件 {backup_file_path} 不存在")
        return False
    
    # 检查配置文件目录是否存在
    config_dir = os.path.dirname(config_file_path)
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    
    try:
        # 复制备份文件到配置文件位置
        shutil.copy2(backup_file_path, config_file_path)
        print(f"配置文件已从 {backup_file_path} 恢复")
        return True
    except Exception as e:
        print(f"恢复配置文件时出错: {e}")
        return False


def list_backups(backup_dir="./config/backups"):
    """
    列出所有备份文件
    
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
    """
    主函数 - 提供简单的命令行界面
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="配置文件备份和恢复工具")
    parser.add_argument("action", choices=["backup", "restore", "list"], 
                        help="操作类型: backup(备份), restore(恢复), list(列出备份)")
    parser.add_argument("--backup-file", help="要恢复的备份文件路径")
    parser.add_argument("--config-file", default="./config/config.json", 
                        help="配置文件路径 (默认: ./config/config.json)")
    parser.add_argument("--backup-dir", default="./config/backups", 
                        help="备份目录路径 (默认: ./config/backups)")
    
    args = parser.parse_args()
    
    if args.action == "backup":
        backup_config(args.config_file, args.backup_dir)
    elif args.action == "restore":
        if not args.backup_file:
            print("恢复操作需要指定 --backup-file 参数")
            return
        restore_config(args.backup_file, args.config_file)
    elif args.action == "list":
        backups = list_backups(args.backup_dir)
        if backups:
            print("备份文件列表:")
            for backup in backups:
                print(f"  {backup}")
        else:
            print("没有找到备份文件")


if __name__ == "__main__":
    main()