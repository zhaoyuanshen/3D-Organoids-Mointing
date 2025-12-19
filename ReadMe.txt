cellposesam.py——分割细胞代码，可输出4个子图
cellposesam_4image.py——输出4合1分割对比图
GUI.py——图形化界面主程序
main.py——启动窗口，拉起GUI
model_train.py——训练分化预测模型
read_history.py——重新读取分化训练数据
rename_and_random_select——按顺序重命名切割后的图像，并按组随机挑选训练集和验证集
tif_process——将大图切割成224*224小图，以原图文件名加后缀命名
week_predict——分化预测代码

main和GUI关联，其他代码可独立使用

环境：
python 3.11
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install cellpose[gui]
pip install PySide6

打包时排除 PyQt6