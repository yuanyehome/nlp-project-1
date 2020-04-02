原始数据见：`data/data.zip`

整理过的二进制数据见：`data/all_data.pkl`

运行KNN：`python src/KNN.py`

默认随机种子为0，可以使用`python src/KNN.py -seed time`使用当前时间作为随机种子。

`-debug`参数用于输出调试信息，请先建立`dbg/`文件夹。

`-save`参数用于保存结果，请先建立`res/`文件夹。