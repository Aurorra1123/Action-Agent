理解我这个IDEA的想法

一部分智能体环境会提供给智能体动作空间，如ALFWorld, ScienceWolrd, WebArena。一部分智能体环境不会给智能体提供动作空间，如GAIA，BrowseComp，SWE。对于这两者，智能体都可以通过自行提出，组合，改进的方式组合动作空间。

我们希望构建一套智能体系统，理论上能够解决所有环境的任务（不在意Performance），只需要通过动作创造，注册，检索，使用。值得注意的是，动作并不是局限于可执行的，不带LLM调用的python code，也包含Agentic Workflow, 以及Agent本身

我们希望解的Benchmark包括GAIA，SWE-Bench，ALFWorld这三个风格迥异的Benchmark。


这篇的Code在写的时候，整体上要Follow一个原则，我们希望在动作空间的角度尽可能的实现跨环境的自动，SOTA的性能不是第一指标，核心还是要跨环境。

整体来说，抽象需要包含这几个类
1. Benchmark，可以参考AFlow的Benchmark，只是需要添加Pass@k的计算方式。https://github.com/FoundationAgents/AFlow/blob/main/benchmarks/benchmark.py 参考代码已经下载在Example_code/benchmark.py
2. Engine, 参考OpenManus-DR就可以，还算比较好用。https://github.com/didiforgithub/AutoEnv/blob/main/autoenv/engine/async_llm.py 参考代码已经下载在本地路径Example_code/AutoEnv/autoenv/engine

3. BaseAction, BaseWorkflow, BaseAgent，这里的核心是让所有可能成为工具的东西都能够被参数化。https://github.com/didiforgithub/AutoEnv/tree/main/autoenv/agent/base Example_code/AutoEnv/autoenv/base
4. ActionSpace 的基础功能需要实现创建（from 环境/web搜索参考构建/From我们提供），检索，注册（为不同环境/Query的Agent注册不同的动作），使用（考虑是否需要依赖Sandbox）
5. ActionSpace 的一个Trick是，需要有地方注入能够prompting Agent Creator 生成比较好用的Agent的内容
6. Main Loop 似乎应该是AgentCreater