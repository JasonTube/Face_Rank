文件及文件夹列表如下：

Face_Rank.ipynb：项目的jupyter notebook文件，内附有集中的项目依赖项
Face_Rank_Report.pdf：实验报告，项目的jupyter notebook导出的pdf文件
face_rank.py：从jupyter notebook中复制出来的代码，只运行此文件可进行动态颜值预测
my_net.py：为了调用的方便从jupyter notebook中复制出来的代码，顺序运行jupyter可以不用此文件

test.xlsx：测试集的文件名和标签
train.xlsx：训练集的文件名和标签
loss.npy：储存loss曲线的值

checkpoints：保存网络参数文件的文件夹，内含best model和last model的参数，为了节省传输文件的大小，已经删去了后者
face_test_set：储存测试集预处理后图像的文件夹，已清空，可由jupyter重新生成
face_train_set：储存训练集预处理后图像的文件夹，已清空，可由jupyter重新生成
original_set：来自网络上的数据集 SCUT-FBP5500_v2 
original_test_set：按标签储存测试集图像的文件夹，已清空，可由jupyter重新生成
original_train_set：按标签储存训练集图像的文件夹，已清空，可由jupyter重新生成
predict_set：储存待打分图像的文件夹
reset：用来清空重置face_test_set、face_train_set、original_test_set、original_train_set文件夹的

runs：tensorboard库生成的文件，用来可视化网络结构，需要按jupyter中注释的要求操作
jupyter_image：jupyter中插入的图片，与代码运行无关

