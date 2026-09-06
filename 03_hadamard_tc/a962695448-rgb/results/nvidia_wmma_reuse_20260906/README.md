# WMMA 实验原始归档与离线复算

从本目录执行以下命令（Python 3.9 或更新版本，仅使用标准库）：

```bash
python analyze.py --raw-root raw --source-root sources --output ../wmma-offline-recomputed
```

输出目录必须尚不存在。命令只读取本归档，在相邻新目录生成 RESULTS.md、summary.json、全部配置轮次与负例 CSV；不访问 GPU、不编译、不联网，也不修改原始文件。

`raw/` 保留取回的 377 个原始文件及传输清单；`sources/` 提供冻结的七个源/协议文件。离线程序重算文件 size/SHA、120 个配置和 21,600 条事件数据。原传输 ZIP 没有重复放入本归档，复算会明确记录 ZIP 未在本地重算，数据分析结论不受影响。

三轮是在每个配置的同一个 C++ 进程内执行，不能视为三次独立进程重复。统计解释与负例见 RESULTS.md。二进制不包含在归档内。

`.gitattributes` 使用 `** -text`，避免 Git 自动转换换行符而破坏原字节与校验值；请保留该文件。`archive_manifest.json` 列出其余全部归档文件的摘要，清单不自包含。
