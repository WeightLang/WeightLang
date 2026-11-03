# 原生测试约定

以 docs/native-testing.md 和 script/testing.rs、script/benchmark.rs 为准。
普通无参 fn test_*() 放 tests/*.wgt；不需要 main，不新增 test 关键字。
文件/函数排序，每次独立运行，main 不执行；filter 匹配路径::函数名。
assert 失败保留源行号，其他测试继续，汇总失败非零退出。AOT 测试入口明确拒绝。
模型加载和训练分离，测试先加载显式保存的模型；JSONL 分批读取并归约 MSE/准确率。
完整例子 examples/native_evaluation/model.wgt 和 tests/evaluation.wgt。
benchmark 第六参数可选：cpu 默认、gpu/适配器 ID、train:cpu。
CPU/GPU 前向输入为一维向量；训练 workload=[inputs,targets,rate,"mse"]，warmup/repeats 均更新模型。
GPU 为同步上传到回读的主机计时，非内核时间；无设备报错，不回退。
GPU 训练计时未接入，不使用其他会话未提交的 training API。
