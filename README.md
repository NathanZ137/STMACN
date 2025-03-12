# Traffic Flow Forcasting

## Project Structure

```yaml
- data 数据集
    - dataset_name
        - SE_{dataset_name}.txt [num_vertex, dims=64]
        - TE_{dataset_name}.npz [seq_len,2]
        - {dataset_name}.npz [seq_len, num_vertex]
    - y_hat
        x       [num_samples, steps, num_vertex]
        y       [num_samples, steps, num_vertex]
        y_hat   [num_samples, steps, num_vertex]
- config 配置文件
- ckpt 模型保存文件
- model 模型
- scripts
    - node2vec 生成SE文件，图节点向量表示
    - notebook
        调试代码，数据预处理，可视化
        - cost2adj_files.ipynb 将数据集中csv文件处理后生成adj files
        - meta-la_process.ipynb TE生成示例，weekofday[0,7),timeofday[0,T),T表示一天的时间步
- utils 数据集、功能函数定义
- trainer 模型训练、推理过程封装
- main.py 运行主代码
- checkpath.sh 补全项目目录结构
- train.sh 模型训练执行脚本，挂服务器后台，输出重定向至日志文件
- eval.sh 模型推理执行脚本
```