# PocketFlow：轻量级AI应用开发的极简主义框架

## 1. PocketFlow的基本介绍

  PocketFlow是一个仅用100行代码实现的轻量级AI应用开发框架，由The-Pocket团队开发并在GitHub上开源。这个框架追求极简设计，核心代码控制在100行以内，没有任何外部依赖，也没有厂商绑定。PocketFlow的设计理念是让大型语言模型(LLM)能够自主编程，用户只需提出需求，LLM即可完成设计、开发与维护工作。

  PocketFlow采用MIT开源许可证，具有高度的灵活性和可扩展性。它从最基本的图(Graph)结构出发，用最少的代码实现强大功能，非常适合需要快速开发AI应用的个人或团队。框架的核心抽象是嵌套有向图结构，将复杂任务分解为多步LLM子任务，并支持分支和递归决策。

<div align="center">
  <img src="https://github.com/The-Pocket/.github/raw/main/assets/meme.jpg" width="400"/>


|                | **抽象概念** |                      **特定应用包装器**                      |                      **特定厂商包装器**                      |             **代码行数**              |                 **大小**                 |
| -------------- | :----------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :-----------------------------------: | :--------------------------------------: |
| LangChain      |  智能体, 链  |                 很多 <br />(例如问答, 摘要)                  |   很多 <br><sup><sub>(例如OpenAI, Pinecone等)</sub></sup>    |                 405K                  |                  +166MB                  |
| CrewAI         |  智能体, 链  | 很多 <br><sup><sub>(例如FileReadTool, SerperDevTool)</sub></sup> | 很多 <br><sup><sub>(例如OpenAI, Anthropic, Pinecone等)</sub></sup> |                  18K                  |                  +173MB                  |
| SmolAgent      |    智能体    | 一些 <br><sup><sub>(例如CodeAgent, VisitWebTool)</sub></sup> | 一些 <br><sup><sub>(例如DuckDuckGo, Hugging Face等)</sub></sup> |                  8K                   |                  +198MB                  |
| LangGraph      |  智能体, 图  |        一些 <br><sup><sub>(例如语义搜索)</sub></sup>         | 一些 <br><sup><sub>(例如PostgresStore, SqliteSaver等) </sub></sup> |                  37K                  |                  +51MB                   |
| AutoGen        |    智能体    | 一些 <br><sup><sub>(例如Tool Agent, Chat Agent)</sub></sup>  | 很多 <sup><sub>[可选]<br> (例如OpenAI, Pinecone等)</sub></sup> | 7K <br><sup><sub>(仅核心)</sub></sup> | +26MB <br><sup><sub>(仅核心)</sub></sup> |
| **PocketFlow** |    **图**    |                            **无**                            |                            **无**                            |                **100**                |                **+56KB**                 |




## 2. PocketFlow的核心能力

PocketFlow虽然代码量极少，但提供了丰富的核心功能：

- **极简代码设计**：核心代码仅100行，易读易改，便于理解和上手
- **基于图结构**：用节点(Node)和连接(Flow)定义AI任务，支持任务分解和多智能体协作
- **多智能体支持**：多个AI智能体可以协同完成任务
- **内置工作流**：任务分解和执行顺序一目了然
- **检索增强生成(RAG)**：结合外部数据提升输出质量
- **AI自编程(Agentic Coding)**：AI能自己写代码，极大提升开发效率
- **零依赖设计**：无需额外库，直接运行
- **兼容任意LLM**：可以接入任何大型语言模型

PocketFlow特别强调高级编程范式，如任务分解、多智能体等，帮助LLM进行复杂任务处理，同时去除低级实现细节，让LLM专注于核心逻辑和决策过程。

这[100行代码](https://github.com/The-Pocket/PocketFlow/blob/main/pocketflow/__init__.py)捕获了LLM框架的核心抽象：
<br>
<div align="center">
  <img src="https://github.com/The-Pocket/.github/raw/main/assets/abstraction.png" width="900"/>
</div>
<br>

基于此，易于实现流行的设计模式，如([多](https://the-pocket.github.io/PocketFlow/design_pattern/multi_agent.html))[智能体](https://the-pocket.github.io/PocketFlow/design_pattern/agent.html)、[工作流](https://the-pocket.github.io/PocketFlow/design_pattern/workflow.html)、[RAG](https://the-pocket.github.io/PocketFlow/design_pattern/rag.html)等。
<br>
<div align="center">
  <img src="https://github.com/The-Pocket/.github/raw/main/assets/design.png" width="900"/>
</div>

## 3. PocketFlow的应用例子

PocketFlow已经成功应用于多个AI开发场景：

- **构建基于PDF目录的聊天机器人**
- **开发文本摘要生成器**，并结合问答智能体进行交互
- **实现简单的任务分解和多智能体协作示例**
- **创建问答流程**：通过定义AnswerNode节点和Flow流程，实现简单的问答系统

一个典型的使用示例是构建一个问答流程：
```python
from pocketflow import Node, Flow
from utils.call_llm import call_llm

class AnswerNode(Node):
    def prep(self, shared):
        return shared["question"]
    def exec(self, question):
        return call_llm(question)
    def post(self, shared, prep_res, exec_res):
        shared["answer"] = exec_res

answer_node = AnswerNode()
qa_flow = Flow(start=answer_node)
```
这个简单的代码展示了如何使用PocketFlow创建一个基本的问答系统。以下是PocketFlow的更多应用案例，结合其极简设计和强大能力：

#### 3.1. **代码补全与IDE助手**

PocketFlow可以快速构建类似Cursor的代码补全工具：

python

复制

```python
class CodeAnalyzer(BaseNode):
    def exec(self, code):
        return llm.generate(f"分析代码：{code}")

flow = Flow(CodeAnalyzer())
flow.run("def add(a,b): return a+b")
```

- **扩展功能**：添加语法检查、自动补全和错误修复节点，打造完整IDE助手。
- **优势**：比传统框架快3倍，内存占用减少80%。

#### 3.2. **工业级任务分解系统**

通过多智能体协作处理复杂任务：

python

复制

```python
planner = Flow(TaskDecomposer())  # 主Agent规划任务
worker = Flow(CodeEditor()).set_trigger(lambda x: x['task_type'] == 'coding')  # 子Agent执行
```

- **应用场景**：自动化测试、CI/CD流程编排。
- **特点**：支持动态分支和条件触发。

#### 3.3 **实时多Agent协作平台**

构建类似AutoGPT的自主协作系统：

python

复制

```python
code_agent = Flow(CodeGenerator())
test_agent = Flow(TestWriter())
shared_memory = {'spec': requirements}
code_agent.run(shared_memory)  # 通过共享内存通信
test_agent.run(shared_memory)
```

- **案例**：MiniAutoGPT（300行实现的自主Agent）。

#### 3.4. **垂直领域RAG应用**

- **法律咨询机器人**：结合法律条文数据库，生成合规建议。
- **医疗问答系统**：集成医学文献，提供循证回答。

#### 3.5. **多媒体处理流水线**

python

复制

```python
flow = Flow(VideoAnalyzer()).link_to(TranscriptGenerator()).link_to(SummaryNode())
```

- **案例**：YouTube视频自动摘要生成器。

#### 3.6. **商业自动化工具**

- **智能客服路由**：根据用户意图动态分配至人工或AI坐席。
- **财报分析Agent**：自动提取数据并生成投资建议。

#### 3.7. **教育领域应用**

- **编程教学助手**：通过交互式节点引导学生debug。
- **数学解题器**：分步骤生成解题过程（支持LaTeX输出）。

#### 3.8. **游戏开发辅助**

- **剧情生成器**：基于世界观设定自动生成支线任务。
- **NPC对话系统**：结合角色设定动态生成对话树。

## 4. 未来的发展前景

PocketFlow代表了AI开发框架的一种新趋势 - 极简主义和AI自主编程。随着LLM能力的不断提升，未来可能会出现以下发展方向：

1. **更强大的自主编程能力**：PocketFlow的设计理念是让LLM能够自主完成更多开发工作，随着模型能力的提升，这一特性将更加突出

2. **更广泛的应用场景**：从简单的问答系统扩展到更复杂的多智能体协作、自动化工作流等领域

3. **教育价值**：由于其极简设计，PocketFlow可以作为学习LLM编程范式的优秀教学工具

4. **社区生态发展**：随着开源社区的贡献，可能会围绕PocketFlow发展出丰富的插件和扩展生态系统

PocketFlow的GitHub仓库([The-Pocket/PocketFlow](https://github.com/the-pocket/PocketFlow))已经吸引了众多开发者的关注，这个极简而强大的框架有望成为快速AI应用开发的新选择。
