# Collection workspace

This is the **collection-phase working area** for the CUDA-to-SYCL
benchmark. It holds the candidate registries and CUDA source snapshots we
gather before adapting selected cases into the final pilot-style format
(`pilot_benchmark/` layout: `original/main.cu` + `CMakeLists` +
deterministic inputs + `tests/verify.py` + `metadata.json`).

**One folder per category. One owner per category.** Each member works
inside their own category folder, so registries never collide.
Registration in a `candidates.csv` is *not* a decision to include a case —
the `final_decision` column is.

## Layout

```
collection/
├── README.md                 <- this file: the shared contribution guide
├── _TEMPLATE/                <- copy this to start a new category
│   ├── README.md
│   └── candidates.csv
└── <category-slug>/          <- one per category (see table below)
    ├── README.md             <- owner + coverage matrix + gaps for this category
    ├── candidates.csv         <- this category's registry (columns defined below)
    └── sources/
        ├── <id>/             <- one snapshot per non-excluded candidate
        │   ├── <upstream CUDA source, trimmed>
        │   └── SOURCE.txt    <- provenance (format below)
        ├── _deps/            <- shared headers a case compiles against (optional)
        └── _licenses/        <- upstream license texts, referenced by SOURCE.txt
```

## How to add your category

1. `cp -r _TEMPLATE <your-category-slug>` (use the slug from the table below).
2. Edit `<slug>/README.md`: put your name as owner and define your
   coverage matrix (the dimensions your category should span). Collection
   is **coverage-driven, not count-driven** — you stop when new candidates
   stop lighting up new matrix cells, not at a fixed number.
3. Register every candidate as a row in `<slug>/candidates.csv` using the
   columns below. IDs are `<slug-prefix>-NN` (e.g. `graph-01`, `md-01`).
4. For each candidate whose `final_decision` is not `exclude`, snapshot
   its CUDA source under `<slug>/sources/<id>/` and add a `SOURCE.txt`
   (format below).
5. Open a PR. A teammate reviews format compliance before merge.

## candidates.csv columns

`id, source_repo, source_url, license, kernel_application_name, domain,
cuda_features_used, estimated_difficulty, build_status, run_status,
correctness_oracle, input_size, reason_selected, migration_notes,
final_decision`

- `estimated_difficulty`: A (straightforward) / B / C (hardest).
- `build_status` / `run_status`: `not_attempted` until validated on a
  machine with the toolchain (see note below); then `ok` / `fail:<why>` /
  `skipped`.
- `correctness_oracle`: how a migrated version is checked (built-in
  reference, CPU recompute, checksum, tolerance/statistical check). A case
  with no designable oracle should be excluded.
- `final_decision`: `candidate` / `exclude` / or a routing note such as
  `transfer-to-stencil` when a case belongs to another category.

## SOURCE.txt format

One per snapshot, recording exactly where the code came from so anyone can
re-fetch or extend it:

```
id:        graph-01
upstream:  <repo URL> @ <short commit sha>
path:      <subpath within the upstream repo>
license:   <name> (full text: ../_licenses/<source>-LICENSE.txt)
retrieved: YYYY-MM-DD
notes:     <inputs location, what was stripped, build caveats>
```

## Snapshot rules

- Pin the upstream **commit** in `SOURCE.txt`; sparse-checkout of the
  recorded `path` is enough to reproduce a snapshot.
- Keep it minimal: strip `.git`, `doc/`, IDE configs, and large output
  dumps. Keep small upstream-shipped inputs (e.g. a sample graph);
  otherwise record the download URL in `notes`.
- Put the upstream license text once in `sources/_licenses/` and reference
  it from each `SOURCE.txt`. Preserve any per-case `LICENSE` file that
  ships inside the source.
- Headers shared by several cases (e.g. a runtime library) go in
  `sources/_deps/` and are referenced from the cases' `SOURCE.txt`.

## Validation note

Build/run validation needs an nvcc + SYCL toolchain. Members without a
local toolchain leave `build_status`/`run_status` as `not_attempted` and
validate on the team's designated GPU machine. Desk review (sources,
licenses, features, difficulty, oracle plan) needs no GPU and comes first.

## Categories and owners

| Slug | Category | Owner |
|---|---|---|
| `simple-kernels` | simple-but-not-trivial kernels | TBD |
| `memory-movement` | memory movement & layout | TBD |
| `stencil-convolution` | stencil / convolution / image processing | Zijian |
| `reductions-scans` | reductions and scans | Zijian |
| `graph-irregular` | graph / irregular access | liqui |
| `molecular-dynamics` | molecular dynamics / simulation | liqui |
| `linear-algebra` | linear algebra | TBD |
| `multi-kernel-pipelines` | multi-kernel pipelines | TBD |
| `cuda-library-usage` | CUDA library usage | TBD |
| `streams-atomics-templates` | streams, events, shared memory, atomics, templates, macros | TBD |

Cross-category cases (e.g. a simulation case that is really a stencil) are
assigned to exactly one owner; note the hand-off in `final_decision` and
raise it at the weekly sync.

---

# 采集工作区(中文版）

本目录是 CUDA→SYCL benchmark 的**采集阶段工作区**。我们在这里登记候选、
保存 CUDA 源码快照，之后再把入选的案例适配成最终的 pilot 格式
（`pilot_benchmark/` 布局：`original/main.cu` + `CMakeLists` + 确定性输入
+ `tests/verify.py` + `metadata.json`）。

**一个类别一个文件夹，一个类别一个负责人。** 每人只在自己的类别文件夹里
干活，登记表互不冲突。在 `candidates.csv` 里登记**不等于**该案例入选——
是否入选由 `final_decision` 列决定。

## 目录结构

```
collection/
├── README.md                 <- 本文件：全组共享的贡献指南
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

slug 与上方英文表一致；负责人以英文表为准（此处随之更新）。当前：
`stencil-convolution`、`reductions-scans` = Zijian；`graph-irregular`、
`molecular-dynamics` = liqui；其余待认领（TBD）。

跨类别案例（如某个本质是 stencil 的 simulation 案例）只归一个负责人；在
`final_decision` 里注明移交，并在周会上提出。
