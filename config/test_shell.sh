#!/bin/bash

# 重定向所有输出到shell_run.log
exec &> shell_run.log

# 下面的命令输出将会写入到shell_run.log
echo "这是一个测试信息。"
ls -l /一个不存在的目录/  # 这将产生一个错误信息，并且也会被写入到shell_run.log

# 其他命令...
