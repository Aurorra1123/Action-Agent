# 跨环境可扩展智能体系统：抽象设计方案（修订版）

本文档在初稿基础上，结合本地参考代码进一步对齐与改进：
- Benchmark 基于 `Example_code/benchmark.py` 的 `BaseBenchmark` 风格并补充 Pass@k；
- Engine 对齐 `Example_code/AutoEnv/autoenv/engine/async_llm.py` 的 `AsyncLLM` 能力与用法；
- 能力抽象对齐 `Example_code/AutoEnv/autoenv/agent/base/*` 的 Pydantic 基类（Action/Workflow/Agent）；
- ActionSpace 明确从 AutoEnv 环境规范与 `action_space.txt`/YAML `transition.actions` 自动导入与注册。

## 1. 设计目标与原则

- 跨环境通用：同时覆盖“已提供动作空间”的环境（如 ALFWorld/ScienceWorld/WebArena）与“未提供动作空间”的环境（如 GAIA/BrowseComp/SWE）。
- 动作即一切：把可执行工具、Agentic Workflow、Agent 本身都抽象为可参数化、可组合、可注册的「动作」。
- 面向创造与检索：支持从环境/网络/人类提供的先验中自动创造、注册、索引与检索动作，并按任务上下文进行动态装配。
- 可评测与可比较：统一评测框架与指标（含 Pass@k），适配 GAIA / SWE-bench / ALFWorld 三类风格任务。
- 安全可控：在受限/沙箱环境中执行不可信动作，提供权限、资源与副作用控制。

## 2. 总体架构

架构分为六个核心层：

1) Benchmark 层（适配器与评测）
- 采用 `Example_code/benchmark.py` 的 `BaseBenchmark` 异步评测骨架（含数据加载、并发、日志与 CSV 输出），新增 Pass@k 统计与多尝试聚合接口。
- 兼容 GAIA / SWE-bench / ALFWorld 的输入输出与判分差异。
- 参考代码已经下载在Example_code/benchmark.py

2) Engine 层（模型/执行引擎）
- 对齐 `AsyncLLM`：支持 `__call__`、`call_with_format(formatter)`、`create_llm_instance`、多模型配置（`LLMsConfig`）、Token 使用与成本跟踪（`TokenUsageTracker`）。
- 提供系统提示注入（system_msg）、`max_completion_tokens`、`temperature/top_p` 控制与统一日志。
- 参考代码已经下载在本地路径Example_code/AutoEnv/autoenv/engine

3) 能力抽象层（Action / Workflow / Agent）
- 对齐 Pydantic 基类：`autoenv/agent/base/base_action.py`、`base_workflow.py`、`base_agent.py`。
- “Agent 即动作”：`BaseAgent` 继承 `BaseAction`，支持 `agent-as-function` 的 `to_param()` 元信息导出，便于作为工具被编排与调用。
- 参考代码已经下载在本地路径Example_code/AutoEnv/autoenv/base

4) Action Space 层（创造/注册/检索/使用）
- 动作来源（具体化）：
  - From 环境：解析 AutoEnv 基准中的 `action_space.txt` 与环境 YAML 中的 `transition.actions`，自动生成带 `parameters` Schema 的 `BaseAction` 子类。
  - From Web/检索：将常见操作流程标准化为 `BaseWorkflow`；配合 `formatter` 规范输入输出。
  - From 人类/库：导入手写工具/Agent，并以 `to_param()` 暴露给路由器与 Planner。
  - From 合成：用 LLM 依据 schema 生成候选动作/工作流，并调用 `call_with_format` 自检。
- 动作索引：语义嵌入 + 标签/环境元数据的混合检索；支持版本、来源与依赖过滤。
- 动作注册：按环境/任务/Agent 维度维护子集；记录依赖、安全需求与验证得分。
- 动作使用：执行前（依赖/权限/沙箱）校验，执行中（日志/成本）观察，执行后（成功率/偏好）回灌。

5) Agent Creator（主循环）
- 以 `AsyncLLM` + Prompt 片段库为核心，按任务检索/合成 `BaseAction/BaseWorkflow/BaseAgent`，并进行少量自测。
- 成功方案以 `agent-as-function` 形式注册到 Action Space，形成可被其他 Agent 调用的工具。

6) 运行与观测层
- 统一日志、轨迹与事件流；指标采集；可视化与复现；缓存与重试策略。

## 3. 核心抽象与接口（与参考代码对齐）

### 3.1 Benchmark 抽象（扩展 BaseBenchmark + Pass@k）

对齐 `Example_code/benchmark.py` 的接口：
- `async def evaluate_problem(self, problem, agent) -> Tuple[...]`
- `def calculate_score(self, expected, prediction) -> Tuple[float, Any]`
- `async def run_evaluation(...)` / `run_baseline(...)` 异步并发评测与 CSV 落盘（包含 `score`、`cost` 等列）。

改进点（统一返回协议与指标口径）：
- 统一单样本多尝试协议：对外暴露 `run_sample(sample, k)`，固定返回
  - `tries: list[dict]`（长度≤k），每个元素包含：
    - `ok: bool`（本次尝试是否成功，按各基准的成功判定器定义）
    - `final: Any`（本次尝试的最终输出，用于可视化/复核）
    - `cost: dict`（本次尝试的成本：`tokens`, `latency_s`, `tool_calls`, `usd` 等）
    - `meta: dict`（可选：轨迹摘要、错误原因、seed 等）
  - `context: dict`（可选：该样本层面的辅助信息）
- Benchmark 层集中计算：
  - Pass@k：`ANY_SUCCESS(tries, k)`，即在至多 k 次独立尝试中是否至少一次成功。
  - 单位成功成本：`unit_success_cost(tries)` 定义为“达到首次成功所需的累计成本”；若全部失败则返回 `None` 并另行统计 `total_cost_until_fail`。
- 标准化评价记录：每条样本输出包含 `(id, pass_at_k, unit_success_cost, total_cost, tries, meta)`；`score`（若适用）与提取值 `extracted` 保持可选字段。
- 成功判定口径：
  - “一次尝试是否成功”由基准特定的判定器提供（GAIA/ALFWorld/SWE-bench 各异）；
  - Pass@k 的聚合口径固定为 ANY_SUCCESS；若需更严格统计，暴露“每次尝试的 success 判定器”而不是改动 Pass@k 本身。

新增方法建议：

```python
def compute_pass_at_k(self, successes: list[bool], k: int) -> float:
    # ANY_SUCCESS 口径，布尔→浮点（1.0 或 0.0），便于聚合平均
    return 1.0 if any(successes[:k]) else 0.0

def compute_unit_success_cost(self, tries: list[dict]) -> Optional[float]:
    # 累计至首次成功的成本；无成功则返回 None
    acc = 0.0
    for t in tries:
        acc += float(t.get("cost", {}).get("usd", 0.0))
        if t.get("ok"):
            return acc
    return None

def summarize_metrics(self, results: list[dict], k_list=(1,5)) -> dict:
    # 统一产出：avg_cost、pass@k、单位成功成本等
    out = {}
    total_cost = sum(r.get("total_cost", 0.0) for r in results)
    out["avg_cost"] = total_cost / max(1, len(results))
    for k in k_list:
        out[f"pass@{k}"] = sum(r.get(f"pass@{k}", 0.0) for r in results) / max(1, len(results))
    unit_costs = [r.get("unit_success_cost") for r in results if r.get("unit_success_cost") is not None]
    out["avg_unit_success_cost"] = (sum(unit_costs) / max(1, len(unit_costs))) if unit_costs else None
    return out
```

环境适配：
- GAIA：抽取/匹配期望答案，可能需要 `extract_answer_code` 辅助；继续复用 `log_mismatch`。
- SWE-bench：以测试通过为判定；支持 `k` 次不同补丁尝试。
- ALFWorld：以任务完成为判定；支持基于策略的多条轨迹尝试。

成本与可比性：统一的 `tries` 返回与集中化 Pass@k/单位成功成本计算，确保 GAIA / SWE / ALFWorld 间的可比性与复现实验成本的对齐。

### 3.2 Engine 抽象（对齐 AsyncLLM）

关键能力（与 `autoenv/engine/async_llm.py` 一致）：
- `LLMsConfig.default()`：从 YAML 或环境变量加载多模型配置。
- `AsyncLLM(llm_config, system_msg=None, max_completion_tokens=None)`：
  - `await llm(prompt)`：标准对话补全；
  - `await llm.call_with_format(prompt, formatter)`：结构化输出，基于 `BaseFormatter` 校验；
  - Token 使用与成本统计：`TokenUsageTracker`；
  - 统一日志记录（`logs.py`）。

扩展建议：
- 可在 `AsyncLLM` 外封装轻量的工具执行器（例如脚本/函数执行、文件系统等），并统一成本/权限记录（见 6. 安全与沙箱）。

成本记账集中化：
- 由 Engine 提供 `merge_usage_into_cost()`，将 `TokenUsageTracker` 输出、调用延迟与工具次数折算/合并为统一成本结构：

```python
def merge_usage_into_cost(usage: dict, *, latency_s: float = 0.0, tool_calls: int | dict = 0, usd_rate: dict | None = None) -> dict:
    """
    返回统一成本：{"tokens": {"prompt": int, "completion": int, "total": int},
                 "latency_s": float, "tool_calls": int, "usd": float}
    其中 usd 可由 `usage` × `usd_rate` 估算；缺省时返回 0。
    """
```

所有动作与工作流调用完成后，将该成本统一回填至 Benchmark 层的 `tries[i].cost`，避免遗漏。

### 3.3 BaseAction / BaseWorkflow / BaseAgent（Pydantic 对齐）

统一理念：基类用 Pydantic 声明参数与校验，`Agent 即动作` 可被当作工具调用与编排。

关键点（与参考实现一致）：
- `BaseAction(BaseModel)`：`name/description/parameters`，抽象 `async __call__(**kwargs)` 与 `to_param()`（函数调用参数描述）。
- `BaseWorkflow(BaseAction)`：含 `llm_config/dataset/llm`，通过 `model_validator` 在实例化后构造 LLM。
- `BaseAgent(BaseAction)`：含系统与步进 Prompt、`llm`、`max_steps` 等；定义 `async step()` 与 `async run()`，并在 `__call__` 转发到 `run()`；`to_param()` 以 `agent-as-function` 暴露。

扩展字段（建议）：
- `version`、`provenance`、`security`（权限/沙箱需求）、`effects_schema`（副作用/外部影响声明）、`exec_policy`（超时/重试/预算），作为可选字段加入各基类，默认兼容参考实现。

### 3.4 Action Space（更具体的落地）

职责：动作的生命全周期管理（创造/发现 → 规格化 → 注册/索引 → 检索/装配 → 运行/反馈 → 版本演化）。

核心能力：
- 创建：
  - From 环境：解析环境提供的 API/动作集合。
  - From Web/检索：抓取常用操作流程模板并规格化为动作。
  - From 人类/库：导入手工编写的动作代码或工作流。
  - From 合成：用 LLM 在给定 schema 下合成候选动作并自测。
- 注册：为不同环境/Query 维护专属动作子集，管理依赖、冲突与版本。
- 检索：基于嵌入向量、标签与规则的混合检索（语义+元数据+上下文约束）。
- 使用：执行前校验（依赖、权限、沙箱）、执行中观测（日志、资源）、执行后反馈（奖励、偏好、成功率）。

数据模型（简化，增补治理字段）：

```python
class ActionSpec(TypedDict):
    id: str
    name: str
    version: str
    description: str
    inputs_schema: dict
    outputs_schema: dict
    requires: list[str]        # 依赖（其他动作/资源）
    environment_tags: list[str]
    security: dict             # 权限/沙箱最小授权声明
    effects_schema: dict       # 副作用/外部影响的结构化声明
    provenance: dict           # 来源/合成提示/评测引用等
    validation: dict           # 验证指标汇总
```

“Trick”（Prompt 注入 Agent Creator）：
- 维护一套高质量 Prompt 片段库（模式、反模式、约束、评估用例、示例轨迹），在合成 Agent/Workflow 时自动拼装注入，提高可用性与稳定性。
- 将“成功轨迹与失败教训”结构化回灌，作为下一次合成时的检索条件与 Few-shot 素材。

验证字段收敛（ActionSpec.validation）：
- 先实现：
  - `per_env_pass: {env: float}`（环境级通过率估计）
  - `last_verified_at: str`（ISO 时间）
- 其余（`attempts/successes/avg_cost/adaptation_required`）从评测日志异步/按需计算并懒更新，不进入热路径。

检索策略按环境分流：
- GAIA：语义检索 + Web/来源可信度加权；
- SWE-bench：以 schema/验证率（`per_env_pass`）优先；
- ALFWorld：以标签/已验证宏动作优先；
- 权重写在配置中，便于线下调参与环境切换。

检索排序公式（简化）：

```
score = a*semantic + b*per_env_pass + c*(1/avg_cost_norm)
```

其中 `a,b,c` 随环境通过配置切换；`avg_cost_norm` 建议以分位数缩放稳定数值范围。

失败自适应的最小策略：
- 失败归因三类：前置缺失 / 工具无效 / 高成本成功；
- 对应动作：插入前置（依赖/准备）、替代工具（同类动作）、对高成本动作降权；
- 投入小但能显著改善检索与后续执行稳定性。

## 4. 主循环：Agent Creator

主循环以“构-检-选-用-评-存”为主线：

1) 解析任务与环境上下文；构造检索 Query（文本+环境标签）。
2) 从 Action Space 检索候选动作/工作流；若不足则触发合成（`call_with_format` 校验输出）。
3) 注入 Prompt 片段库生成/改进 `Agent/Workflow`；进行轻量自测（环境仿真或离线校验）。
4) 选择 Top-N 方案实际执行；必要时分解为子任务并迭代（反思/纠错）。
5) 评估（含 Pass@k）；记录轨迹、成本与失败原因（统一日志+成本）。
6) 成功方案注册与版本化，沉淀可复用工具供后续任务调用。

触发合成（明确边界）：
- 固定两条触发条件，避免策略空转：
  1) 候选数 `< M`；
  2) Top-N 候选的 `per_env_pass` 平均值 `< θ`；
- 其余触发词（如 high_cost/low_diversity）暂不引入，后续如需再通过配置开启。

## 5. 评测与适配

统一评测流程：
- 基于扩展后的 `BaseBenchmark` 迭代样本，驱动主循环执行；
- 记录每个样本的多次尝试结果，计算 Pass@k 与成本；
- 输出可复现的运行包（包含随机种子、版本、轨迹与预测）。

Pass@k 定义（修正）：
- 一次尝试是否成功 = 基准特定的成功判定；
- Pass@k = k 次独立尝试中是否至少有一次成功（ANY_SUCCESS），用于可比性；
- 更严格统计需求通过暴露“每次尝试的 success 判定器”满足，而不改变 Pass@k 的组合口径。

环境适配要点：
- GAIA：需要网页/外部工具的桥接与安全控制；强调多步推理与工具链。
- SWE-bench：需要仓库克隆/补丁生成/测试执行；强调沙箱运行与资源隔离。
- ALFWorld：需要将环境动作封装为 `BaseAction`，支持策略学习式的序列决策。

## 6. 安全与沙箱

- 运行域隔离：默认在受限容器/沙箱执行外部动作，限制网络、文件与 CPU/内存。
- 权限模型：动作声明所需权限，通过策略引擎在执行前评估并最小授权。
- 审计与回滚：对有副作用的动作提供审计日志与回滚策略（若可行）。

## 7. 目录与模块建议（贴合参考代码）

```
Agent/
  benchmarks/
    base.py            # 继承 Example_code 的 BaseBenchmark，扩展 Pass@k
    gaia.py            # GAIA 适配器
    swe_bench.py       # SWE-bench 适配器
    alfworld.py        # ALFWorld 适配器

  engine/
    async_llm.py       # 直接复用/轻改 Example_code 的 AsyncLLM
    formatter.py       # 结构化格式校验（与 AsyncLLM.call_with_format 对齐）
    exec.py            # ToolExecutor 与沙箱适配
    costs.py           # 统一成本折算：merge_usage_into_cost()

  core/
    actions.py         # 细化动作规范（在 pydantic 基类之上添加可选元数据）
    action_space.py    # ActionSpace 管理：创造/注册/检索/使用
    creator.py         # Agent Creator 主循环
    prompts/
      patterns.md      # Prompt 片段库（模式/反模式/示例）

  runtime/
    sandbox.py         # 沙箱策略与执行隔离
    telemetry.py       # 轨迹、日志、指标、事件
    cache.py           # 结果缓存与重试

  experiments/
    configs/           # 评测与运行配置
    runners/           # 统一入口脚本
```

## 8. 渐进式落地计划（里程碑）

- M1 基础骨架：
  - 以 `Example_code/benchmark.py` 为基类实现 `benchmarks/base.py`，加入 Pass@k 与统一指标汇总；
  - 统一 `run_sample(sample, k)` 返回协议：`tries=[{ok, final, cost, meta}]` 并在 Benchmark 层计算 `pass@k` 与 `unit_success_cost`；
  - 复用 `AsyncLLM`，补充最小 `exec.py`（命令/脚本/函数执行，含沙箱标志）；
  - 提供 `engine.costs.merge_usage_into_cost()` 并将成本回填到 `tries[i].cost`；
  - 引入 Pydantic 基类（直接依赖 Example_code 的 `agent/base`）。

- M2 Action Space V1：
  - 从 AutoEnv 基准解析 `action_space.txt` 与 YAML `transition.actions` 自动生成动作，并注册；
  - 基于嵌入的简单检索（环境/标签过滤）；
  - 引入治理字段（`provenance/validation/security/effects_schema`）；
  - 配置化的环境分流检索与评分权重（`a,b,c`）；
  - 执行前校验与最小沙箱（权限白名单、资源配额）。

- M3 Agent Creator V1：
  - 在 GAIA / SWE-bench / ALFWorld 上贯通评测；
  - 检索优先，合成为辅（失败再合成）；
  - 统一轨迹/成本采集与失败用例回灌；
  - 合成触发条件固定为“候选数<M 或 Top-N 均值<θ”。

- M4 强化与优化：
  - Prompt 片段库完善与自动回灌（基于 Formatter 的可验证模板）；
  - 偏好与稳定性建模（选择高通过率动作/工作流）；
  - 更细粒度的权限/副作用控制与审计。

## 9. 开放问题与权衡

- 合成动作的正确性验证成本：如何在低成本下获得高置信？（自测/仿真/对抗样本）
- 跨环境的动作规格统一粒度：过粗影响复用，过细增加编排复杂度。
- 检索与路由策略：何时优先检索既有动作，何时触发合成与自改进？
- 评测可比性：同一 Pass@k 下不同尝试成本如何对齐？
- 安全边界：对不可信代码与外部系统操作的最小必要权限与审计强度。

---

与需求对齐的结论与改进摘要：
- 对齐了参考代码的三大关键面：`BaseBenchmark` 评测范式、`AsyncLLM` 引擎接口、Pydantic 基类（Action/Workflow/Agent）。
- 将 Action Space 与 AutoEnv 的 `action_space.txt` 与 YAML `transition.actions` 打通，明确环境→动作的自动化导入路径。
- 在 Benchmark 层补充了 Pass@k 与多尝试聚合，满足“跨环境评测可比较”的需求。
- 在 Engine 层强调 `call_with_format` 与 `formatter` 的结构化约束，提升动作/工作流生成与自检的可靠性。
 - 统一 `run_sample(..., k)` → `tries=[{ok, final, cost,...}]` 返回协议，在 Benchmark 层集中计算 Pass@k 与单位成功成本。
 - ActionSpec 增补治理字段：`provenance/validation/security/effects_schema`，并收敛 `validation` 的最小必需子集。
 - 检索策略按环境分流并引入简化排序公式，权重可配置化；加入失败自适应的最小策略。
 - 成本记账集中化：Engine 暴露 `merge_usage_into_cost()`，统一回填 tokens/latency/tool_calls 与 USD 估算。
 - Creator 合成触发边界明确化：仅以候选数与 Top-N 通过率阈值为准。


---

以上方案在不追求单一基准最优性能的前提下，优先保障跨环境的通用性、扩展性与可运维性，并为后续性能优化与能力沉淀提供稳定的抽象与演进路径。
