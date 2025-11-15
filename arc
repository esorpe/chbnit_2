
Measuring Minds: An Analysis of Computational Paradigms for Solving the Abstraction and Reasoning Corpus (ARC-AGI)


Introduction: Redefining the Problem of Intelligence

For decades, the field of artificial intelligence (AI) measured progress by benchmarking systems on specific, narrow skills—such as mastering chess or captioning images.1 This approach, however, measures the output of intelligence, not intelligence itself.2 It was discovered that with unlimited training data and developer priors, researchers could effectively "buy" arbitrary levels of skill for a system, masking its true, underlying generalization power.3
In his 2019 paper, "On the Measure of Intelligence," François Chollet proposed a fundamental paradigm shift. He formally defined intelligence not as the possession of skill, but as the efficiency of skill-acquisition.3 This reframing led to the creation of the Abstraction and Reasoning Corpus (ARC-AGI), a benchmark designed not to test what an AI knows, but how well it can learn.1

Section 1: The ARC-AGI Benchmark

The ARC-AGI benchmark is a collection of grid-based visual reasoning puzzles.4 Each task presents a solver with 2-3 "demonstration" input/output pairs. The solver must infer the abstract transformation rule from these examples and apply it to a new "test" input to generate the correct output.6
ARC-AGI is a robust benchmark for general intelligence for several key reasons:
It Measures Few-Shot Generalization: With only a handful of examples, "memorization" is impossible. The system must generalize from minimal data.7
It Neutralizes Pre-Training Priors: The tasks are designed to be "Easy for Humans, Hard for AI".3 They are built on a minimal set of "Core Knowledge Priors"—innate human concepts like objectness, basic number sense, and geometry.3 By excluding complex cultural knowledge (like written language), the benchmark "forces the test-taker (human or AI) to demonstrate genuine problem-solving ability".3 This levels the playing field, ensuring a system is not just leveraging a massive "bought" prior (like the entire internet's text).3
It Targets Compositionality: The benchmark's core difficulty lies in "compositional generalization"—the ability to combine known rules or concepts in novel ways.7 For instance, a task might require applying "Rule A to red objects" and "Rule B to blue objects" simultaneously.9 AI systems have historically struggled with this, even when they can solve single-rule problems.9

Section 2: A Chronological Evolution of Model Architectures

The quest to solve ARC-AGI has produced a clear evolution in computational approaches, moving from rigid, engineered systems to highly adaptive, neuro-symbolic hybrids. This progression can be broadly divided into two main paradigms: Induction (Program Synthesis), which tries to find the abstract program that solves the task, and Transduction (Direct Prediction), which tries to predict the output grid directly.5

2.1 Early Models (circa 2020-2023)

For the first five years after its 2019 release, ARC-AGI remained unbeaten, with the state-of-the-art (SOTA) score stagnating at around 33%. This era was defined by two underperforming approaches:
Symbolic Domain-Specific Languages (DSLs): This was the dominant and most successful early paradigm, exemplified by the winner of the 2022 ARCathon.3
Model Architecture: This approach is not a neural network. Instead, the developer manually codes a massive library of possible grid-transformation functions (e.g., move_object_up, mirror_x_axis, count_objects) into a Domain-Specific Language (DSL).12 A separate search algorithm (like a genetic algorithm) then attempts to find the correct sequence of these pre-built functions that, when executed, transforms the demonstration inputs into the outputs.12
Limitations: This method is "engineered, not learned".12 The "intelligence" is not in the AI but in the human developer who anticipated and "manually coded" the transformation functions.12 These systems are inherently brittle and fail completely on tasks that require a novel transformation not included in their hard-coded DSL.
Classical Deep Learning (ConvNets): Early attempts to apply standard deep learning models, such as Convolutional Neural Networks (ConvNets), were a "failure".12
Model Architecture: A standard ConvNet architecture, which excels at image classification by identifying local patterns.
Limitations: ConvNets are designed to find local patterns (e.g., a pixel's immediate neighbors).12 They are "not so capable of solving tasks with complex transformations that require some sort of global information," which is essential for ARC's abstract, holistic rules.12

2.2 Intermediate Models (The 2024 Breakthrough)

The 2024 ARC Prize saw the SOTA score jump from 33% to over 53%, driven by the winning team, "the ARChitects".
Model: "the ARChitects" (2024)
Model Architecture: This solution used a "transductive" transformer model (a model that directly generates the output grid). The core innovation was not the base model itself, but a new engineering method for applying it: Test-Time Training (TTT).
Methodology Deep Dive (TTT): The "theARChitects" system is adaptive. For each new task it encounters at test time, it performs the following loop:
Augmentation: It takes the 2-3 demonstration pairs and uses a "novel augmentation-based validation system" to generate hundreds (e.g., ~560) of new, similar training samples.
Adaptation (TTT): It rapidly fine-tunes its transformer model on these ~560 new, task-specific samples for a small number of steps. This "adapts" the model on-the-fly to the specific rule of the new task.
Inference: It then uses this newly adapted model to make a prediction, employing advanced techniques like "depth-first search for token selection" to find the most likely correct output grid.
Significance: This was the first solution to "break" the 5-year stagnation. It proved that static, pre-trained models were insufficient, and that dynamic, "test-time" adaptation was a critical component for solving ARC-AGI.

2.3 Recent Models (Frontier Engineering & New Architectures)

Following the 2024 prize, SOTA models have pushed into two distinct directions: massive-scale, compute-heavy refinement systems and hyper-efficient, specialized recursive networks.
Iterative Refinement (Neuro-Symbolic Hybrids): This is the current SOTA on the ARC leaderboards, used by systems from J. Berman (2025) and E. Pang (2025).13
Model Architecture: This is a neuro-symbolic pipeline that uses a frontier Large Language Model (LLM), like GPT-4o or Claude 3.5, as a reasoning and code-generation engine.14
Methodology Deep Dive ("Evolutionary Test-time Compute"):
Generate (Neural): An LLM is prompted with the ARC task and asked to "generate a bunch of Python transform functions" that might solve it. One method generated ~8,000 candidate Python programs.15
Test (Symbolic): These thousands of generated Python programs are executed and tested against the task's demonstration pairs.14
Refine (Loop): This is the critical step. Most programs fail. The system then enters a refinement loop. In one implementation, the "fittest" programs (those that almost work) are used to "create new prompts for generating even better solutions". In another, the LLM is shown its own incorrect output (the "diff" between its output and the expected output) and is prompted to revise its own code to fix the errors.15 This generate-test-refine loop iterates, "evolving" the code until a correct solution is found.
Significance: This is the highest-performing method to date, achieving ~80% on ARC-AGI-1.13 It is, however, extremely costly in terms of computation.
Tiny Recursive Models (TRM): A new line of research has emerged that directly challenges the "scale is all you need" paradigm.
Model: Tiny Recursive Model (TRM)
Model Architecture: This is a non-transformer model. It is a "tiny" 2-layer network with only 7 million parameters (compared to billions or trillions in LLMs).16 It is a recurrent-style architecture designed for recursive reasoning.
Methodology Deep Dive (Recursive Reasoning): The TRM "recursively improves its predicted answer" over multiple steps. It starts with the input (x) and an initial blank answer (y), along with an internal "latent reasoning state" (z). In each step, it first updates its internal reasoning state (z) based on the input, the current answer, and its previous state. Then, it updates its answer (y) based on this new reasoning state. This loop allows the model to "progressively improve its answer" by "thinking" step-by-step, in an extremely parameter-efficient way.
Significance: With less than 0.01% of the parameters of a large LLM, the TRM achieves 45% on ARC-AGI-1—a score "higher than most LLMs" (like Deepseek R1 or Gemini 2.5 Pro) in a standard inference setting. This provides strong evidence that small, specialized, recursive architectures may be a more efficient and promising path for abstract reasoning than pure scale.

Section 3: State-of-the-Art and the Enduring Human Gap


3.1 The 2024 Leap and the ARC-AGI-2 "Chasm"

The 2024 ARC Prize was a major breakthrough, jumping the SOTA score on the original ARC-AGI-1 dataset from ~33% to 55.5%, driven by the TTT and program synthesis methods.5
However, this success was immediately tempered by the release of ARC-AGI-2 in 2025. This new version was specifically engineered to be "less brute-forcible" 10 and to test the "deeper levels of compositional generalization" that AI systems were failing.7
The result was a performance "chasm".9 The very same SOTA systems failed catastrophically on this new, harder dataset:
Pure LLMs (e.g., GPT-4) score 0% on ARC-AGI-2.19
Advanced AI Reasoning Systems (using TTT, refinement, etc.) achieve only single-digit percentages.19
Humans, in contrast, solve 100% of the ARC-AGI-2 tasks.13
This data provides a stark empirical rebuttal to naive scaling hypotheses. The "gap" between AI performance on ARC-1 and ARC-2 proves that the benchmark has successfully isolated a cognitive hurdle (compositionality) that current architectures cannot overcome simply with more compute or data.9

3.2 Current Leaderboard Analysis: The Price of Intelligence

The official ARC Prize leaderboard provides the most concrete data for this analysis. Crucially, it has been designed to "explicitly quantify the cost of intelligence" by tracking not just the score, but the computational "Cost/Task".19 This "visualizes the critical relationship between cost-per-task and performance" 13 and represents the final, practical implementation of Chollet's 2019 "efficiency" definition.13
The goal is no longer just to solve the tasks, but to solve them efficiently.
Table 1: ARC-AGI Leaderboard State (as of Q3/Q4 2025)

13
System
Organization
System Type
ARC-AGI-1 Score
ARC-AGI-2 Score
Cost/Task
Human Panel
Human
N/A
98.0%
100.0%
$17.00
J. Berman (2025)
Bespoke
Refinement
79.6%
29.4%
$30.40
E. Pang (2025)
Bespoke
Refinement
77.1%
26.0%
$3.97
GPT-5 Pro
OpenAI
CoT
70.2%
18.3%
$7.14
Grok 4 (Thinking)
xAI
xAI
66.7%
16.0%
$2.17
Claude Sonnet 4.5
Anthropic
CoT
63.7%
13.6%
$0.759

This data reveals a new "Pareto frontier" for AI research.
The Human Panel sets the gold standard: 100% accuracy for a cost of $17.13
J. Berman's (2025) "Refinement" system holds the highest AI score (79.6% / 29.4%), but is also the most expensive at $30.40 per task.13
E. Pang's (2025) system achieves a slightly lower score (77.1% / 26.0%) but at a fraction of the cost ($3.97).13
This leaderboard forces a crucial question: which system is "more intelligent"? By Chollet's definition, E. Pang's more efficient system may be closer to true intelligence than Berman's more costly, brute-force approach. The leaderboard is no longer a simple list; it is a scatter plot where the goal is to reach the top-left corner (high score, low cost). This design "forces the field to compete on intelligence (efficiency), not just brute-force compute".19

3.3 Insights from Open Competitions (Kaggle)

The open-to-the-public Kaggle competitions (ARC Prize 2024 and 2025) provide a complementary view by operating under strict computational constraints.13 The 2025 competition, for instance, enforced a tight compute budget (e.g., "$50 compute budget for 120 evaluation tasks").13
The 2024 Kaggle competition was won by the team "the ARChitects" with a 53.5% score. The 2025 Kaggle competition (based on the much harder ARC-AGI-2) showed far lower scores, with the top team (NVARC) achieving 27.64%.21 This discrepancy highlights the extreme cost of the SOTA "Refinement" methods seen on the main leaderboard, confirming that high-scoring approaches are still heavily reliant on "brute-force" computation.19

Section 4: Concluding Analysis: Implications for AGI


4.1 Scaling vs. Reasoning: The ARC-AGI Rebuttal

The results from ARC-AGI provide the strongest empirical evidence to date against the "scale is all you need" hypothesis. That hypothesis is inextricably linked to a "definition of AGI... oriented around AI's skill" 22—the very "output fallacy" that ARC was designed to correct. By defining intelligence as skill acquisition and efficiency, and by designing a benchmark that measures these properties, ARC-AGI reveals precisely where scaling fails.
The "ARC-AGI-2 Chasm" (Section 3.1) is the definitive proof. The fact that the most powerful, scaled LLMs (like GPT-5 Pro) drop from 70% on ARC-1 to 18% on ARC-2 13, while humans score 100% 13, demonstrates that "general fluid intelligence" is not an emergent property of model size. Rather, it appears to be the product of a specific set of cognitive architectures—likely the neuro-symbolic hybrids 23 or specialized recursive networks 16—that are capable of the discrete, compositional, and abstract reasoning that humans find trivial.

4.2 The Future of Measurement: ARC-AGI-3 and Interactive Agency

The ARC-AGI project is not complete. Its clear, long-term roadmap demonstrates a tiered, cognitive-science-based approach to measuring intelligence.25 The field is now looking toward the next planned cognitive hurdle: ARC-AGI-3.26
This next iteration will be the "first interactive reasoning benchmark".25 It will move beyond the static, "input-output" reasoning of ARC-1 and ARC-2 to test interactive agency.25 In ARC-AGI-3, solvers will be AI "agents" that must interact with novel, abstract game environments, testing a new suite of capabilities, including "exploration, goal directedness, and memory".26
This roadmap reveals a clear, logical progression for defining and measuring AGI:
ARC-AGI-1: Test basic fluid intelligence and the ability to generalize from core priors.
ARC-AGI-2: Test robust, static compositional reasoning.
ARC-AGI-3: Test interactive, goal-directed agency over time.

4.3 Revisiting the "Quest for Human Nature"

This analysis began with the "quest to human nature and information processing." The ARC-AGI benchmark is, itself, a novel "intellectual information process"—a tool created by human intelligence to measure and guide the creation of artificial intelligence. It operationalizes abstract concepts from psychometrics (fluid intelligence) 8 into a concrete, falsifiable, and competitive framework.
By establishing a benchmark that is not solvable by "buying" skill with unlimited data and priors 3, the ARC Prize Foundation is "guiding researchers, industry, and regulators" 26 away from the benchmark treadmill and the "output fallacy." It is forcing the field to confront the hard problems of efficiency, generalization, and compositionality. In doing so, it provides the "appropriate feedback signal" 3 necessary to move beyond mere skill replication and toward a true, human-like general intelligence.
Works cited
[1911.01547] On the Measure of Intelligence - arXiv, accessed November 15, 2025, https://arxiv.org/abs/1911.01547
accessed November 15, 2025, https://arxiv.org/pdf/2501.07458#:~:text=approach%20as%20intelligent.-,Chollet%20(2019%2C%20pp.,that%20are%20not%20intelligence%20itself.
What is ARC-AGI? - ARC Prize, accessed November 15, 2025, https://arcprize.org/arc-agi
fchollet/ARC-AGI: The Abstraction and Reasoning Corpus - GitHub, accessed November 15, 2025, https://github.com/fchollet/ARC-AGI
ARC Prize 2024: Technical Report, accessed November 15, 2025, https://arcprize.org/media/arc-prize-2024-technical-report.pdf
arXiv:2311.09247v3 [cs.AI] 11 Dec 2023, accessed November 15, 2025, https://arxiv.org/pdf/2311.09247
ARC-AGI-2: A New Challenge for Frontier AI Reasoning Systems - arXiv, accessed November 15, 2025, https://arxiv.org/html/2505.11831v1
Paper Summary: On the Measure of Intelligence (Chollet 2019) - Liam Wellacott, accessed November 15, 2025, https://liamwellacott.github.io/artificial%20intelligence/on-the-measure-of-intelligence/
ARC-AGI-2 - ARC Prize, accessed November 15, 2025, https://arcprize.org/arc-agi/2/
ARC-AGI-2 A New Challenge for Frontier AI Reasoning Systems, accessed November 15, 2025, https://arcprize.org/blog/arc-agi-2-technical-report
François Chollet on OpenAI o-models and ARC - YouTube, accessed November 15, 2025, https://www.youtube.com/watch?v=w9WE1aOPjHc
The Measure of Intelligence: Abstract and Reasoning - Temple CIS, accessed November 15, 2025, https://cis.temple.edu/~pwang/5603-AI/Project/2021S/Malhotra/The%20Measure%20of%20Intelligence_%20Abstract%20and%20Reasoning.pdf
Leaderboard - ARC Prize, accessed November 15, 2025, https://arcprize.org/leaderboard
[2412.04604] ARC Prize 2024: Technical Report - arXiv, accessed November 15, 2025, https://arxiv.org/abs/2412.04604
Getting 50% (SoTA) on ARC-AGI with GPT-4o — LessWrong, accessed November 15, 2025, https://www.lesswrong.com/posts/Rdwui3wHxCeKb7feK/getting-50-sota-on-arc-agi-with-gpt-4o
Less is More: Recursive Reasoning with Tiny Networks - arXiv, accessed November 15, 2025, https://arxiv.org/abs/2510.04871
accessed January 1, 1970, https://arxiv.org/html/2510.04871v1
ARC Prize 2024: Technical Report - arXiv, accessed November 15, 2025, https://arxiv.org/html/2412.04604v1
Announcing ARC-AGI-2 and ARC Prize 2025, accessed November 15, 2025, https://arcprize.org/blog/announcing-arc-agi-2-and-arc-prize-2025
2025 Competition Details - ARC Prize, accessed November 15, 2025, https://arcprize.org/competitions/2025/
Leaderboard - ARC Prize 2025 | Kaggle, accessed November 15, 2025, https://www.kaggle.com/competitions/arc-prize-2025/leaderboard
How to Beat ARC-AGI by Combining Deep Learning and Program Synthesis, accessed November 15, 2025, https://arcprize.org/blog/beat-arc-agi-deep-learning-and-program-synthesis
Guide - ARC Prize, accessed November 15, 2025, https://arcprize.org/guide
NSA: Neuro-symbolic ARC Challenge, accessed November 15, 2025, https://arxiv.org/abs/2501.04424
François Chollet: The ARC Prize & How We Get to AGI : YC Startup Library | Y Combinator, accessed November 15, 2025, https://www.ycombinator.com/library/Md-fran-ois-chollet-the-arc-prize-how-we-get-to-agi
ARC Prize, accessed November 15, 2025, https://arcprize.org/
