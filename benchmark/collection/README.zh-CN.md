[English](README.md) | **中文**

# 采集工作区

本目录是 CUDA→SYCL benchmark 的**采集阶段工作区**。我们在这里登记候选、
保存 CUDA 源码快照，之后再把入选的案例适配成最终的案例单元格式
（`original/main.cu` + `CMakeLists` + 确定性输入 + `tests/verify.py` +
`metadata.json`）。适配好的可运行案例**不**放在这里：统一放到
`benchmark/cases/<类别 slug>/<案例>/`，并使用共享的
`benchmark/tools/verify_lib.py`。

**一个类别一个文件夹，一个类别一个负责人。** 每人只在自己的类别文件夹里
干活，登记表互不冲突。在 `candidates.csv` 里登记**不等于**该案例入选——
是否入选由 `final_decision` 列决定。

## 目录结构

```
collection/
├── README.md                 <- 英文贡献指南
├── README.zh-CN.md           <- 本文件：中文贡献指南
├── _TEMPLATE/                <- 新类别从这里复制
│   ├── README.md
│   └── candidates.csv
└── <category-slug>/          <- 每个类别一个（见下方认领表）
    ├── README.md             <- 负责人 + 覆盖矩阵 + 缺口
    ├── candidates.csv         <- 该类别的登记表（列见下文）
    └── sources/
        ├── <id>/             <- 每个非排除候选一份快照
        │   ├── <精简后的上游 CUDA 源码>
        │   └── SOURCE.txt    <- 出处（格式见下文）
        ├── _deps/            <- 案例编译所需的共享头文件（可选）
        └── _licenses/        <- 上游 license 全文，被 SOURCE.txt 引用
```

## 如何加入你的类别

1. `cp -r _TEMPLATE <你的类别-slug>`（slug 用下方认领表里的）。
2. 编辑 `<slug>/README.md`：写上你的名字作为负责人，定义你的覆盖矩阵
   （你这个类别应覆盖的维度）。采集是**覆盖驱动，不是数量驱动**——当新
   候选不再点亮新的矩阵格子时就停，而不是凑够某个固定数量。
3. 把每个候选登记为 `<slug>/candidates.csv` 里的一行，用下文的列。ID 用
   `<slug前缀>-NN`（如 `graph-01`、`md-01`）。
4. 对每个 `final_decision` 不是 `exclude` 的候选，把它的 CUDA 源码快照放进
   `<slug>/sources/<id>/`，并加一个 `SOURCE.txt`（格式见下文）。
5. 提 PR。合并前由一名队友检查格式是否合规。

## candidates.csv 的列

`id, source_repo, source_url, license, kernel_application_name, domain,
cuda_features_used, estimated_difficulty, build_status, run_status,
correctness_oracle, input_size, reason_selected, migration_notes,
final_decision`

- `estimated_difficulty`：A（直接）/ B / C（最难）。
- `build_status` / `run_status`：在装了工具链的机器上验证前一律
  `not_attempted`（见下方说明）；验证后填 `ok` / `fail:<原因>` / `skipped`。
- `correctness_oracle`：迁移后的版本怎么验证对错（自带 reference、CPU 重算、
  checksum、容差/统计检查）。无法设计 oracle 的案例应排除。
- `final_decision`：`candidate` / `exclude` / 或路由备注，如某案例属于别的
  类别时写 `transfer-to-stencil`。

## SOURCE.txt 格式

每份快照一个，精确记录代码出处，便于任何人重新拉取或扩展：

```
id:        graph-01
upstream:  <仓库 URL> @ <短 commit sha>
path:      <在上游仓库中的子路径>
license:   <名称> (full text: ../_licenses/<source>-LICENSE.txt)
retrieved: YYYY-MM-DD
notes:     <输入数据位置、删掉了什么、编译注意事项>
```

## 快照规则

- 在 `SOURCE.txt` 里钉死上游 **commit**；对记录的 `path` 做 sparse-checkout
  即可复现快照。
- 保持精简：剔除 `.git`、`doc/`、IDE 配置、大的输出转储。保留上游自带的小
  输入（如示例图），否则在 `notes` 里写下载 URL。
- 上游 license 全文放在 `sources/_licenses/` 里一份，由各 `SOURCE.txt` 引用。
  源码内自带的 per-case `LICENSE` 文件要保留。
- 多个案例共用的头文件（如某运行时库）放进 `sources/_deps/`，并在相关案例的
  `SOURCE.txt` 里引用。

## 验证说明

build/run 验证需要 nvcc + SYCL 工具链。本机没有工具链的成员把
`build_status`/`run_status` 留为 `not_attempted`，在全组指定的 GPU 机器上验证。
桌面审查（来源、license、特性、难度、oracle 方案）不需要 GPU，先做这一步。

## 类别与负责人

| Slug | 类别 | 负责人 |
|---|---|---|
| `simple-kernels` | simple-but-not-trivial kernels | yuepan |
| `memory-movement` | memory movement & layout | yuepan |
| `stencil-convolution` | stencil / convolution / image processing | Zijian |
| `reductions-scans` | reductions and scans | Zijian |
| `graph-irregular` | graph / irregular access | liqui |
| `molecular-dynamics` | molecular dynamics / simulation | liqui |
| `linear-algebra` | linear algebra | TBD |
| `multi-kernel-pipelines` | multi-kernel pipelines | TBD |
| `cuda-library-usage` | CUDA library usage | TBD |
| `streams-atomics-templates` | streams, events, shared memory, atomics, templates, macros | TBD |

跨类别案例（如某个本质是 stencil 的 simulation 案例）只归一个负责人；在
`final_decision` 里注明移交，并在周会上提出。
