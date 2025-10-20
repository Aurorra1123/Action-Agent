# 跨环境智能体抽象骨架（详细版）

本仓库实现了一个面向 GAIA / SWE‑bench / ALFWorld 等多风格环境的统一抽象骨架，满足“动作即一切”的设计理念：把工具、工作流、甚至 Agent 本身，统一抽象为可参数化、可注册、可检索、可复用的动作，并能在不同环境下复用与编排。

本方案基于你提供的参考代码与要求落地：
- Benchmark：对齐 AFlow 风格（参考文件已下载 `Example_code/benchmark.py`），并补充 Pass@k 聚合。
- Engine：对齐 AutoEnv 的 `AsyncLLM` 能力（参考路径 `Example_code/AutoEnv/autoenv/engine`）。
- BaseAction/Workflow/Agent：对齐 AutoEnv 的 Pydantic 抽象（参考路径 `Example_code/AutoEnv/autoenv/agent/base`）。
- ActionSpace：实现创建/检索/注册/使用；支持从环境规范导入（`action_space.txt` / YAML `transition.actions`）。
- Main Loop：以 Agent Creator 为主循环，检索优先、合成为辅，支持 prompt 注入的增强策略。

—

一、需求映射（对应 plan.md 6 点）

1) Benchmark（+Pass@k）
- `benchmarks/base.py` 提供异步评测骨架、结果落盘（CSV/JSON）、`compute_pass_at_k()`/`summarize_metrics()`；
- `benchmarks/gaia.py`, `benchmarks/swe_bench.py`, `benchmarks/alfworld.py` 提供三类适配。

2) Engine（AsyncLLM 对齐）
- `engine/async_llm.py` 优先导入 Example_code 的 `AsyncLLM/LLMConfig/LLMsConfig`；
- `engine/formatter.py` 支持 `BaseFormatter` 与 `JSONListOfActionSpecsFormatter`；
- `engine/exec.py` 提供带超时/沙箱标志的异步执行器。

3) BaseAction / BaseWorkflow / BaseAgent 参数化
- `core/bases.py`：优先导入 Example_code 的 Pydantic 基类；保留 `to_param()` 与 “Agent 即动作”语义；
- 示例动作/Agent：`core/examples/actions.py`, `core/examples/agents.py`。

4) ActionSpace（创建/检索/注册/使用）
- `core/action_space.py`：ActionSpec 元数据；`register/search/use`；
- 导入器：`import_from_action_space_txt()`/`import_from_env_yaml()`；
- 预留 `security/depends/version` 等字段。

5) 注入 Prompt（Trick）
- `core/prompts/patterns.md`：提示片段库；
- `core/creator.py` 的 `synthesize_action_specs()` 接入 `JSONListOfActionSpecsFormatter`，用于结构化合成与自检。

6) Main Loop = AgentCreator
- `core/creator.py`：检索候选 → 不足则合成 → 执行 → 评估 →（可扩展）注册回灌。

—

二、目录结构

- Benchmarks：`benchmarks/base.py`, `benchmarks/gaia.py`, `benchmarks/swe_bench.py`, `benchmarks/alfworld.py`
- Engine：`engine/async_llm.py`, `engine/formatter.py`, `engine/exec.py`
- Core：`core/bases.py`, `core/action_space.py`, `core/creator.py`, `core/examples/*`, `core/prompts/patterns.md`
- Runtime：`runtime/sandbox.py`, `runtime/telemetry.py`, `runtime/cache.py`
- Runners & Data：`experiments/runners/run_{gaia,swe,alfworld}.py`, `data/*.jsonl`

—

三、核心接口（摘要）

1) Benchmarks（`benchmarks/base.py`）
- `async def evaluate_problem(problem, agent) -> Tuple[...]`：单题评测，返回与 `get_result_columns()` 对齐的行；
- `def get_result_columns() -> List[str]`：列名（至少包含 `score` 与 `cost`）；
- `async def run_evaluation(...)` / `run_baseline(...)`：按索引或全量评测；
- `compute_pass_at_k(successes, k)` / `summarize_metrics(rows, columns, k_list)`：Pass@k 聚合。

2) Engine
- `engine/async_llm.py`：`AsyncLLM(config)`，`await llm(prompt)`，`await llm.call_with_format(prompt, formatter)`；
- `engine/formatter.py`：`BaseFormatter`（`prepare_prompt/validate_response/format_error_message`），`JSONListOfActionSpecsFormatter`；
- `engine/exec.py`：`ToolExecutor(timeout_s)`，`await executor.run(action, params, sandbox=True)`。

3) BaseAction / BaseWorkflow / BaseAgent（`core/bases.py`）
- `BaseAction(BaseModel)`：`name/description/parameters`，`async __call__(**kwargs)`，`to_param()`；
- `BaseWorkflow(BaseAction)`：`llm_config/dataset/llm`（与 Example_code 对齐）；
- `BaseAgent(BaseAction)`：`system_prompt/next_step_prompt/max_steps`、`async step()/run()`；“Agent 即动作”。

4) ActionSpace（`core/action_space.py`）
- `register/get/spec/list_actions/search/use`；
- 导入：`import_from_action_space_txt()` / `import_from_env_yaml()`；
- `ActionSpec` 字段：`name/version/inputs_schema/outputs_schema/environment_tags/security/provenance`。

5) Agent Creator（`core/creator.py`）
- `AgentCreator(ActionSpace, CreatorConfig)`；
- `retrieve_candidates()` → `synthesize_action_specs()` → `choose_and_run()` → `main()`。

—

四、主要流程（端到端）

1) 构建/导入动作：从环境规范导入或使用我们提供的动作/Agent，注册到 `ActionSpace`。
2) 任务解析与候选检索：`AgentCreator.retrieve_candidates(query, tags)`。
3) 动作合成（可选）：`synthesize_action_specs()` 用 LLM + Formatter 生成结构化动作规格（后续可绑定实现并注册）。
4) 执行与评测：`choose_and_run()` 执行候选；`benchmarks/*` 评测并落盘（支持 Pass@k 聚合）。

—

五、最小示例（GAIA）

- 数据：`data/gaia.jsonl`（已提供 2 条样例）。
- 运行：`python experiments/runners/run_gaia.py`。
- 流程：注册 `demo:echo_answer` 动作 → `GAIAEchoAgent` 调用动作 → `GAIABenchmark` 评测 → 写出日志。

—

六、适配说明（GAIA / SWE‑bench / ALFWorld）

- GAIA：文本推理与工具链；当前以精确匹配示例为评测，可扩展为提取/归一化后判分，保留 mismatch 日志。
- SWE‑bench：以测试通过为准；当前为占位 Runner；后续可接入仓库克隆、补丁生成与测试执行器，支持 k 次不同补丁尝试。
- ALFWorld：以环境成功为准；当前为占位 Runner；后续可把环境动作封装为 `BaseAction`，由策略生成执行序列。

—

七、安全与成本

- 沙箱：`runtime/sandbox.py` 为占位（无隔离）；`engine/exec.py` 已支持超时；建议后续加入权限白名单、资源配额与副作用审计。
- 成本：`benchmarks/base.py` 结果中保留 `cost`；可与 Example_code 的 TokenUsageTracker 对齐并聚合。

—

八、Roadmap（建议）

- M1：当前骨架（已完成）。
- M2：Action 导入器增强（自动绑定实现/依赖）、嵌入检索与上下文路由、最小安全策略。
- M3：Creator 合成熟练度提升（few‑shot/反思/自测回灌）、Pass@k 端到端串联。
- M4：对接真实 SWE‑bench/ALFWorld 环境，完善成本与指标面板。

—

备注
- 若存在本地 Example_code，本仓库会自动优先导入其实现（Engine 与 Bases）；缺失时使用最小桩实现。
- 无 pandas 时，评测结果自动回退为 JSON 落盘。
- 导入器支持 `action_space.txt` 与 AutoEnv 风格 YAML `transition.actions`（若安装了 PyYAML）。
