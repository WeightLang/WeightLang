# 原生测试约定

以 docs/native-testing.md、`crates/wgt/src/` 中的 CLI 测试/benchmark 实现及 `crates/wgt-language/src/` 中的语法实现为准。
普通无参 fn test_*() 放 tests/*.wgt；不需要 main，不新增 test 关键字。
文件/函数排序，每次独立运行，main 不执行；filter 匹配路径::函数名。
assert 失败保留源行号，其他测试继续，汇总失败非零退出。AOT 测试入口明确拒绝。
模型定义解析与服务实例化分离；测试显式连接服务、初始化或加载模型后再推理/训练。JSONL 分批读取并归约 MSE/准确率。Rust facade 与服务的具体边界见 `docs/native-backend.md`。
完整例子 examples/native_evaluation/model.wgt 和 tests/evaluation.wgt。
benchmark 第六参数可选：cpu 默认、gpu/适配器 ID、train:cpu。
CPU/GPU 前向输入为一维向量；训练 workload=[inputs,targets,rate,"mse"]，warmup/repeats 均更新模型。
GPU 为同步上传到回读的主机计时，非内核时间；无设备报错，不回退。
GPU 训练计时未接入，不使用其他会话未提交的 training API。
