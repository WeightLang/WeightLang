# 原生数据集加载

## 原生数据集加载器（wgt 0.5，当前源码）

原生脚本由 Rust 执行，无 Python datasets 依赖；解释和编译模式共用读取实现。
模型构造不会自动读取数据；脚本显式打开流并调用 batch。

| 函数 | 完整契约 |
| --- | --- |
| `jsonl(path)` | 本地 UTF-8 JSONL，path 相对脚本目录（绝对路径也可用）；每行一个 JSON 值，空行报错；打开时读取文件句柄 |
| `hf_dataset(dataset,config,split)` | 三个非空字符串全部必填，无默认配置/split；例如 `hf_dataset("scikit-learn/iris","default","train")`；创建流，在首次 batch 时请求网络 |
| `batch(reader,count)` | count 为 1..65536 的整数；保留尾批，到末尾反复返回 []；返回原始 JSON 值列表，HF 每行是列名到值的对象；不自动分词、转张量、shuffle 或补齐 |
| `close(reader)` | 释放文件/网络客户端；重复关闭无操作，关闭后 batch 报错 |

HF 使用 [Dataset Viewer rows API](https://huggingface.co/docs/dataset-viewer/rows)，
每次请求最多 100 行，较大 batch 自动分页，读取偏移由流持有。
只支持该 API 可访问且有 Parquet 导出的配置/split；不是 Python load_dataset 的全功能替代。
通过环境变量 HF_TOKEN 提供可选 Bearer 凭据，遵循服务端访问权限；
没有配置发现、revision 固定、离线缓存、下载脚本执行、本地 Parquet 解码或媒体文件自动下载。
图像/音频等字段保留服务端 JSON 描述，不转换成像素/波形。
不在磁盘生成数据副本，也不创建模型制品。

单行最大 1 MiB、单批有效载荷最大 16 MiB；HF 单次 HTTP 响应也限制 16 MiB，
这些是载荷边界，不是进程内存上限。请求超时 30 秒；HTTP 失败（含 401/403/429）、
无效 JSON、缺失字段、截断单元格、错序/提前空页均报错，无自动重试或静默丢行。
HF 的一次 batch 失败不提交读取偏移；总行数变化会报错，但不能检测所有远端内容更新，
需要严格复现时应使用固定的本地 JSONL 快照。
JSONL 错误可能已经消费输入；应关闭并重新打开。
每次运行累计最多创建 1024 个流（包括已关闭的流）。

完整可运行示例：[huggingface/model.wgt](../examples/native_training/huggingface/model.wgt)。
它只加载五条公开 Iris 样本，不训练；需要网络：

```sh
cargo run -p weightlang-wgt --bin wgt -- script examples/native_training/huggingface/model.wgt
# 同一脚本也支持 script FILE --compiled
```

训练时可将数值列在 WGT 中组成 [batch,features] 输入和 [batch,outputs] 目标，
显式交给 fit；字符串/媒体不隐式转成数值张量。
