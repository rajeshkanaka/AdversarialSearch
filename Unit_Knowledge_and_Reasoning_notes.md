# Lecture Notes: Knowledge and Reasoning (Unit IV) - AI for SPPU SE AI-DS & CSE (AI)

**Course:** Artificial Intelligence  
**Target Students:** Second Year (SE) AI & Data Science / Computer Science & Engineering (AI)  
**University:** Savitribai Phule Pune University (SPPU)  
**Academic Year:** 2024-25  
**Total Teaching Time:** 9 Hours (Theory + Short Activities)

---

## Why Should We Study This Unit?
Knowledge and reasoning tell us how an intelligent agent can **store facts about the world**, **think about them logically**, and **make correct decisions**. This unit connects directly to topics like expert systems, chatbots, and question-answering machines that we see around us (e.g., Alexa, Siri, Google Assistant). Understanding this unit helps us build systems that are not just reactive but **think before acting**.

---

## Chapter Map (Quick Overview)
1. [Logical Agents](#1-logical-agents)
2. [Knowledge-Based Agents](#2-knowledge-based-agents)
3. [The Wumpus World Problem](#3-the-wumpus-world-problem)
4. [Propositional Logic: A Very Simple Logic](#4-propositional-logic-a-very-simple-logic)
5. [Propositional Theorem Proving](#5-propositional-theorem-proving)
6. [Effective Propositional Model Checking](#6-effective-propositional-model-checking)
7. [Agents Based on Propositional Logic](#7-agents-based-on-propositional-logic)
8. [First-Order Logic (FOL)](#8-first-order-logic-fol)
9. [Representation Revisited](#9-representation-revisited)
10. [Syntax and Semantics of First-Order Logic](#10-syntax-and-semantics-of-first-order-logic)
11. [Using First-Order Logic](#11-using-first-order-logic)
12. [Knowledge Engineering in First-Order Logic](#12-knowledge-engineering-in-first-order-logic)
13. [Inference in First-Order Logic](#13-inference-in-first-order-logic)
14. [Propositional vs. First-Order Inference](#14-propositional-vs-first-order-inference)
15. [Unification and First-Order Inference](#15-unification-and-first-order-inference)
16. [Forward Chaining](#16-forward-chaining)
17. [Backward Chaining](#17-backward-chaining)
18. [Resolution](#18-resolution)
19. [Knowledge Representation](#19-knowledge-representation)
20. [Case Study: BBC & Amazon Alexa – AI-Driven Interactive Media](#20-case-study-bbc--amazon-alexa--ai-driven-interactive-media)

---

## Study Plan (Suggested)
| Lecture | Topics | Teaching Style | Mini Activity |
|--------|--------|----------------|---------------|
| 1 | Logical agents, KB agents | Storytelling, simple diagrams | Identify AI systems around you |
| 2 | Wumpus World | Board drawing, group game | Students play the Wumpus game |
| 3 | Propositional logic basics | Truth tables, examples | Create own propositions |
| 4 | Theorem proving & model checking | Worked examples | Solve sample proofs |
| 5 | Agents with propositional logic | Case demonstration | Design a rule-based agent |
| 6 | FOL introduction, syntax, semantics | Comparative charts | Translate sentences |
| 7 | Using FOL, knowledge engineering | Real-world examples | Model a family tree |
| 8 | Inference, unification, chaining | Step-by-step derivations | Practice problems |
| 9 | Resolution, KR, case study | Discussion + video | Analyze Alexa case |

---

## 1. Logical Agents
**Definition:** Agents that make decisions based on **logical reasoning** with the help of a knowledge base.  
**Key Points:**
- Use logic to reason about actions and outcomes.
- Work in environments where **thinking matters** (e.g., planning, diagnosis).  
**Simple Example:** A medical diagnosis system that asks symptoms and uses logic to suggest diseases.  
**Important Terms:**
- **Percepts:** What the agent observes.
- **Knowledge base (KB):** Stored information.
- **Inference:** Deriving new facts from old facts.

**YouTube Support:** [Knowledge-Based Agents Explained (Neso Academy)](https://www.youtube.com/watch?v=Uz0bHX2Gjzw)

---

## 2. Knowledge-Based Agents
**Definition:** Agents that use an internal knowledge base (collection of facts + rules) to make decisions.  
**Steps of the KB-Agent Loop:**
1. **Tell**: Add new percepts to the KB.  
2. **Ask**: Query the KB to decide action.  
3. **Perform Action**: Execute the chosen action.  
4. Repeat.

**Advantages:**
- More flexible and intelligent than simple reflex agents.  
- Can explain their actions (why a decision was taken).

**Exam Tip:** Draw the KB agent architecture diagram. Label sensors, KB, inference engine, and actuators.

**Reference Video:** [Knowledge-Based Agents (Gate Smashers)](https://www.youtube.com/watch?v=N9F5ypwDV8A)

---

## 3. The Wumpus World Problem
**Why famous?** Classic example to show how logical agents work in an uncertain world.  
**Environment:**
- 4x4 grid, Wumpus (monster), pits, gold.  
- Agent perceives **stench** near Wumpus, **breeze** near pits, **glitter** near gold.  
- Goal: Grab gold and exit safely.

**Important Observations:**
- No direct view of Wumpus/pits; agent must deduce using logic.
- Knowledge is incomplete; reasoning fills the gaps.

**Typical Exam Question:** "Explain how the agent uses percepts to infer a safe square." Use the following template:
1. Agent perceives breeze in (1,2).  
2. Therefore, there may be a pit in (2,2) or (1,3).  
3. If later (2,1) also has breeze, agent narrows possibilities.

**Learning Aid:** [Wumpus World Simulation (YouTube - aiGuru)](https://www.youtube.com/watch?v=L_7XcxFW0r4)

---

## 4. Propositional Logic: A Very Simple Logic
**Also called:** Boolean logic or sentential logic.  
**Components:**
- **Propositions:** Statements that are true or false. Example: "It is raining".
- **Logical connectives:** AND (∧), OR (∨), NOT (¬), IMPLIES (→), BICONDITIONAL (↔).
- **Syntax:** Rules to form valid sentences (e.g., P ∧ Q is valid).
- **Semantics:** Rules to assign truth values (e.g., truth tables).

**Truth Table Example:**
| P | Q | P → Q |
|---|---|-------|
| T | T | T |
| T | F | F |
| F | T | T |
| F | F | T |

**Common Mistakes:**
- Confusing P → Q with Q → P.
- Thinking implication is reversible (it is not).

**Reference Video:** [Propositional Logic in AI (Gate Smashers)](https://www.youtube.com/watch?v=7V4Isiu1RiI)

---

## 5. Propositional Theorem Proving
**Goal:** Show that a statement is logically true using inference rules.  
**Approaches:**
- **Proof by enumeration:** Check all models (truth assignments).  
- **Proof using inference rules:** Use Modus Ponens, And-Elimination, etc.

**Important Inference Rules:**
- **Modus Ponens:** If P → Q and P are true, then Q is true.
- **Modus Tollens:** If P → Q and ¬Q are true, then ¬P is true.
- **Resolution (for propositional logic):** Combine two clauses to remove a variable.

**Example Problem:** Prove that from (P → Q) and (Q → R) we can infer (P → R).  
**Solution Outline:**
1. Assume P.  
2. By Modus Ponens on (P → Q), get Q.  
3. Again using Modus Ponens on (Q → R), get R.  
4. Therefore, P implies R.

**Practice:** Try converting small statements into **Conjunctive Normal Form (CNF)** for resolution-based proofs.

**Reference Video:** [Inference Rules & Theorem Proving (EduPoint)](https://www.youtube.com/watch?v=r2ZRV-6zJbI)

---

## 6. Effective Propositional Model Checking
**What is it?** Automatically verifying if a knowledge base satisfies certain sentences.  
**Methods:**
- **Truth table checking:** Slow for many symbols.  
- **Improved methods:** Backtracking search, using heuristics.

**Important Terms:**
- **Model:** An assignment of truth values to symbols.  
- **Entailment:** KB ⊨ α means every model of KB is also a model of α.

**Exam Reminder:** Explain why propositional model checking has exponential complexity in the worst case.

**Extra Learning:** [Model Checking Introduction (UPenn CIS)](https://www.youtube.com/watch?v=6QE6RZ5q-7M)

---

## 7. Agents Based on Propositional Logic
**How do they work?**
1. Maintain a KB of propositional sentences.  
2. Update KB with new percepts (tell).  
3. Use inference to find safe actions (ask).  
4. Perform action.

**Example:** Wumpus world agent using propositional rules to decide which cell is safe to move.

**Pros:** Easy to implement for small worlds.  
**Cons:** Hard to scale when number of objects increases.

**Video Aid:** [Rule-Based Systems (Simple Snippets)](https://www.youtube.com/watch?v=CDw8A-dmhTg)

---

## 8. First-Order Logic (FOL)
**Why needed?** Propositional logic cannot talk about objects, their properties, and relationships. FOL fixes this by adding **quantifiers** and **predicates**.

**Key Components:**
- **Objects:** People, places, things (e.g., `Ramesh`, `Pune`).
- **Predicates:** Properties/relations (e.g., `Student(Ramesh)`, `Loves(Ramesh, AI)`).
- **Functions:** Return objects (e.g., `FatherOf(Ramesh)`).
- **Quantifiers:**
  - **Universal (∀):** "for all"
  - **Existential (∃):** "there exists"

**Example Sentences:**
- ∀x (Student(x) → Studies(x, AI))  → "Every student studies AI."
- ∃x (Professor(x) ∧ Teaches(x, AI)) → "Some professor teaches AI."

**Reference Video:** [First Order Logic Basics (Gate Smashers)](https://www.youtube.com/watch?v=VEtXvdHGCDA)

---

## 9. Representation Revisited
**Why representation matters?** The way we represent knowledge affects how easily we can infer new knowledge.

**Good Representation Should:**
- Be **correct** (captures truth).  
- Be **complete enough** for the problem.  
- Be **computationally feasible** (fast enough to reason).

**Representation Styles:**
- **Declarative:** What is true (facts, rules).  
- **Procedural:** How to do things (algorithms).

**Tip for Exams:** Give a short example of representing family relations in both propositional logic and FOL.

---

## 10. Syntax and Semantics of First-Order Logic
**Syntax (Grammar Rules):**
- **Terms:** variables (x), constants (Pune), functions (FatherOf(x)).
- **Atomic sentences:** Predicate(terms) like `Loves(Ram, AI)`.
- **Complex sentences:** Formed using connectives and quantifiers.

**Semantics (Meaning Rules):**
- **Domain:** Set of objects we talk about (e.g., all students in class).
- **Interpretation:** Assign meaning to constants, predicates, functions.
- **Satisfaction:** When a sentence is true under a given interpretation.

**Example:**
- Domain: {Ramesh, Sita}  
- Interpretation: Student(x) is true for Ramesh & Sita; Loves(x, AI) true only for Sita.  
- Sentence: ∃x Loves(x, AI) → True because Sita loves AI.

**Reference Video:** [Syntax & Semantics of FOL (GATE Lectures)](https://www.youtube.com/watch?v=B0elOAX-H7I)

---

## 11. Using First-Order Logic
**Typical Steps:**
1. **Identify domain** (what objects we talk about).  
2. **List predicates/functions** needed.  
3. **Translate sentences** from English to FOL.  
4. **Add rules** that link predicates.  
5. **Use inference** to answer questions.

**Example: Mini Family Domain**
- Facts: `Parent(Ravi, Meera)`, `Female(Meera)`  
- Rule: ∀x ∀y (Parent(x, y) ∧ Female(y) → Daughter(y, x))
- Query: `Daughter(Meera, Ravi)`? → True by applying the rule.

**Practice Tip:** Make your own knowledge base for "Students, Courses, and Teachers" and try answering questions like "Who teaches AI?" using FOL rules.

**Extra Resource:** [FOL Examples (Jenny's Lectures)](https://www.youtube.com/watch?v=8fEkITPNF3M)

---

## 12. Knowledge Engineering in First-Order Logic
**Meaning:** Building a knowledge base step-by-step for a particular domain.

**Process:**
1. **Understand the domain** (talk to experts, read documents).  
2. **Choose vocabulary** (predicates, functions, constants).  
3. **Encode general knowledge** (rules).  
4. **Encode specific facts** (data).  
5. **Check for consistency** (no conflicts).  
6. **Perform inference** to answer queries.  
7. **Update** when new information comes.

**Real Example:** Designing a knowledge base for a university timetable system.
- Predicates: `Course(c)`, `Teaches(teacher, course)`, `Slot(course, time)`  
- Rules: Two courses cannot be in same slot for same class.

**Reference Video:** [Knowledge Engineering (Ravindrababu Ravula)](https://www.youtube.com/watch?v=ikfRkNvztGU)

---

## 13. Inference in First-Order Logic
**Goal:** Derive new sentences that are logically implied by existing knowledge.

**Challenges vs Propositional Logic:**
- Need to handle variables and quantifiers.
- Must **instantiate** variables with constants (or other terms).

**Common Methods:**
- **Forward chaining** (data-driven).  
- **Backward chaining** (goal-driven).  
- **Resolution** (refutation-based).

**Key Concept:** **Generalized Modus Ponens** for FOL.  
Example: From `∀x (Student(x) → Studies(x, AI))` and `Student(Riya)`, infer `Studies(Riya, AI)`.

**Video Aid:** [Inference in FOL (Gate Smashers)](https://www.youtube.com/watch?v=RGN7yaPzm9c)

---

## 14. Propositional vs First-Order Inference
| Feature | Propositional | First-Order |
|---------|---------------|-------------|
| Symbols | Whole statements | Predicates with variables |
| Domain | No internal structure | Objects & relations |
| Inference | Simpler, faster | More powerful but heavier |
| Example | Rain → WetGround | ∀x (Rain(x) → Wet(x)) |

**Exam Question Pattern:** "Compare propositional logic and first-order logic in terms of expressiveness and complexity." Use the above table.

---

## 15. Unification and First-Order Inference
**Unification:** Process of **making two FOL expressions look the same** by finding a suitable substitution.

**Example:**
- Want to unify `Loves(x, y)` and `Loves(Ram, AI)`  
- Substitution: {x/Ram, y/AI}

**Most General Unifier (MGU):** The simplest substitution that works.

**Why needed?** For applying inference rules like resolution and chaining in FOL.

**Practice:** Try unifying `Parent(x, MotherOf(y))` with `Parent(Anita, MotherOf(Riya))`. MGU is {x/Anita, y/Riya}.

**Helpful Video:** [Unification in AI (NPTEL)](https://www.youtube.com/watch?v=lVLtZL1NVrY)

---

## 16. Forward Chaining
**Idea:** Start from known facts and apply rules to reach the goal.

**Steps:**
1. Put all known facts in the agenda.  
2. Apply rules whose premises match facts (after unification).  
3. Add conclusions as new facts.  
4. Repeat until goal is found or no new facts.

**Use Cases:** Production systems, expert systems.

**Advantages:**
- Good when there are many facts and few goals.  
- Automatically explores everything implied by known data.

**Example:**
- Facts: `Parent(Ram, Sita)`, `Parent(Sita, Kiran)`  
- Rule: `Parent(x, y) ∧ Parent(y, z) → Grandparent(x, z)`  
- Forward chaining derives `Grandparent(Ram, Kiran)`.

**Reference Video:** [Forward Chaining (Jenny's Lectures)](https://www.youtube.com/watch?v=10lLbD1KZGo)

---

## 17. Backward Chaining
**Idea:** Start from goal and work backwards to known facts.

**Steps:**
1. Start with the goal (query).  
2. Find rules that conclude the goal.  
3. Make the premises of that rule new sub-goals.  
4. Continue until you reach known facts.

**Use Cases:** Rule-based expert systems like medical diagnosis.

**Advantages:**
- Focused search when there are many possible facts but few goals.  
- Useful for question-answering systems.

**Example:**
- Goal: `Grandparent(Ram, Kiran)`  
- Rule: `Parent(x, y) ∧ Parent(y, z) → Grandparent(x, z)`  
- Sub-goals: `Parent(Ram, Sita)` and `Parent(Sita, Kiran)`  
- If facts match, goal is proven.

**Reference Video:** [Backward Chaining (Gate Smashers)](https://www.youtube.com/watch?v=kg_SpqnXlcU)

---

## 18. Resolution
**Purpose:** A complete inference rule for both propositional and first-order logic.

**Propositional Resolution:** Combine two clauses to eliminate a literal.
- Example: From (P ∨ Q) and (¬Q ∨ R) we can derive (P ∨ R).

**FOL Resolution Steps:**
1. Convert sentences to **clause form (CNF)**.  
2. Standardize variables (rename to avoid clash).  
3. Skolemize existential quantifiers (replace with functions/constants).  
4. Drop universal quantifiers.  
5. Use resolution with unification until contradiction (empty clause) is found.

**Why important?** Works for automated theorem proving.

**Tip:** Always show the intermediate clauses in the exam to get full marks.

**Video:** [Resolution in AI (Neso Academy)](https://www.youtube.com/watch?v=hiE5IQg3dbg)

---

## 19. Knowledge Representation
**Meaning:** How we store information so that a computer can reason about it.

**Key Requirements:**
- **Representation power** (expressiveness).  
- **Reasoning support** (easy inference).  
- **Practicality** (efficient enough).

**Types of Knowledge:**
1. **Declarative Knowledge:** Facts, propositions. Example: "Pune is in Maharashtra."  
2. **Procedural Knowledge:** How-to rules. Example: "If fever & cough → suggest rest."  
3. **Meta-Knowledge:** Knowledge about knowledge. Example: "Temperature > 100 means fever."  
4. **Heuristic Knowledge:** Thumb rules. Example: "If user asks for news, play top headlines."

**Common KR Techniques in AI:**
- Logic (propositional, FOL)  
- Semantic networks  
- Frames and scripts  
- Ontologies (OWL)  
- Production rules

**Exam Preparation:** Draw a small semantic network showing relationships (e.g., Animal → Mammal → Dog).

**Reference Video:** [Knowledge Representation (Simplilearn)](https://www.youtube.com/watch?v=j1E_mVi2q8g)

---

## 20. Case Study: BBC & Amazon Alexa – AI-Driven Interactive Media
**Background:** BBC created an interactive voice drama for Amazon Alexa called "The Inspection Chamber". Users participate through voice and become part of the story.

**How Logic-Based Chatbots Are Used:**
1. **Knowledge Base:** Contains story scripts, possible user inputs, character responses.  
2. **Inference Engine:** Chooses next story branch based on user replies.  
3. **Logic Rules:** Ensure story consistency (e.g., "If user chooses to escape → play escape scene").  
4. **Natural Language Understanding (NLU):** Converts user speech to logical forms.

**Why It Matters to Us:**
- Shows how knowledge representation + reasoning create interactive experiences.  
- Highlights the need for careful **knowledge engineering**.  
- Demonstrates use of **forward and backward chaining** to pick next dialogue.

**Key Takeaways for Exams:**
- Mention how KB stores story states.  
- Explain that inference helps personalize user experience.  
- Connect to topics like **logic rules**, **knowledge engineering**, and **reasoning**.

**Further Reading/Watching:**
- [BBC on Alexa Interactive Story (Official Video)](https://www.youtube.com/watch?v=Lo-53VCkCqM)  
- [Voice Assistants & Logic-Based Dialogue Systems](https://www.amazon.science/latest-news/the-making-of-the-inspection-chamber)

---

## Quick Revision: 10-Point Checklist
1. Understand the architecture of a knowledge-based agent.  
2. Be able to explain Wumpus world percepts and inferences.  
3. Revise truth tables and inference rules in propositional logic.  
4. Practice CNF conversion for theorem proving.  
5. Know how model checking works and its limitations.  
6. Learn the vocabulary of FOL (predicates, quantifiers, terms).  
7. Translate at least 5 English statements into FOL and vice versa.  
8. Practice unification with simple predicate pairs.  
9. Compare forward chaining, backward chaining, and resolution.  
10. Review the case study to connect theory with real-world systems.

---

## Expected University Exam Questions (Practice)
1. **Short Notes (4-5 marks):** Logical agents, Wumpus world, Forward chaining, Unification.  
2. **Medium Questions (6-8 marks):** Compare propositional and first-order logic; Explain knowledge engineering steps.  
3. **Long Questions (10-12 marks):** Design a knowledge-based agent for a given scenario; Use resolution to prove a statement; Discuss the Alexa case study with logic reasoning.

**Tip:** Always include diagrams/flowcharts wherever possible. Write point-wise answers in simple sentences.

---

## Additional Learning Resources (Handpicked)
| Topic | Type | Link |
|-------|------|------|
| Knowledge-Based Agents | Video | [https://www.youtube.com/watch?v=N9F5ypwDV8A](https://www.youtube.com/watch?v=N9F5ypwDV8A) |
| Propositional Logic Basics | Video | [https://www.youtube.com/watch?v=7V4Isiu1RiI](https://www.youtube.com/watch?v=7V4Isiu1RiI) |
| First-Order Logic | Video | [https://www.youtube.com/watch?v=VEtXvdHGCDA](https://www.youtube.com/watch?v=VEtXvdHGCDA) |
| Forward/Backward Chaining | Video | [https://www.youtube.com/watch?v=10lLbD1KZGo](https://www.youtube.com/watch?v=10lLbD1KZGo) |
| Resolution | Video | [https://www.youtube.com/watch?v=hiE5IQg3dbg](https://www.youtube.com/watch?v=hiE5IQg3dbg) |
| Case Study (Alexa) | Article | [https://www.amazon.science/latest-news/the-making-of-the-inspection-chamber](https://www.amazon.science/latest-news/the-making-of-the-inspection-chamber) |

---

## Final Words for Students
- Focus on **conceptual clarity** with simple examples.  
- Practice **diagram-based answers** (agent architecture, reasoning flow).  
- Use the linked videos/articles for deeper understanding and revision.  
- During exam preparation, create your own mini knowledge bases to build confidence.

**All the best! Keep learning with logic and reason.**
