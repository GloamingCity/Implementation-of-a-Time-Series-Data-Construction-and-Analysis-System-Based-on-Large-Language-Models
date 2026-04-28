@echo off

REM 删除当前目录中（不包括子目录）的所有PNG文件
echo 删除所有PNG文件...
del /q *.png

REM 删除当前目录中（不包括子目录）的所有JSONL文件
echo 删除所有JSONL文件...
del /q *.jsonl

REM 删除同级目录中的文件夹根目录下的PNG文件
echo 删除同级目录中的文件夹根目录下的PNG文件...
for /d %%i in (*) do (
    del /q "%%i\*.png"
)

REM 删除同级目录中的文件夹根目录下的JSONL文件
echo 删除同级目录中的文件夹根目录下的JSONL文件...
for /d %%i in (*) do (
    del /q "%%i\*.jsonl"
)

echo 清理完成！