# Chapter 2 Literature Review

## 2.1 Introduction

This chapter provides a critical review of the theoretical foundations and existing research landscape regarding smart contract vulnerability detection. It begins by establishing the fundamental concepts of blockchain and smart contracts, emphasizing the inherent security risks posed by immutability and decentralization. Subsequently, it critically analyzes the evolution of detection methodologies, transitioning from traditional static analysis and rule-based tools to advanced Artificial Intelligence (AI) approaches. The review categorizes contemporary AI studies into semantic (NLP-based), structural (graph-based), and hybrid frameworks, scrutinizing the strengths and limitations of each. Finally, the chapter synthesizes these findings to identify significant research gaps, specifically regarding multimodal fusion, cross-contract interactions, and interpretability that this proposed study aims to address.

## 2.2 Background: Blockchain and Smart Contract

### 2.2.1 Overview of Blockchain Technology

Blockchain technology functions as a decentralized, distributed ledger where no single entity controls the network, thereby eliminating central points of failure and censorship (Nakamoto, 2008). The architecture is defined by three core pillars: decentralization, transparency, and immutability.

- **Decentralization:** Ensures that the network operates via distributed nodes rather than a central authority.
- **Transparency:** All transactions are visible to network participants, fostering accountability (Buterin, 2014).
- **Immutability:** Once data is recorded, it cannot be altered retroactively without the consensus of the network, ensuring robust data integrity.

Security is maintained through cryptographic hashing and consensus mechanisms, such as Proof of Work or Proof of Stake, which secure the network against tampering and fraud.

### 2.2.2 Smart Contracts and The Immutability Paradox

A smart contract is a self-executing program stored on a blockchain that automatically executes predefined actions when specific conditions are met. First conceptualized by Nick Szabo and popularized by Ethereum, smart contracts eliminate intermediaries, significantly reducing operational costs and settlement time (Buterin, 2014).

However, the "immutability" feature of blockchain introduces a critical security paradox. While it guarantees the integrity of executed transactions, it creates a catastrophic risk for the code itself: smart contracts cannot be patched or modified once deployed. Any vulnerability or logic error present at deployment becomes a permanent, public attack vector (Ozdag, 2025). This rigidity means that unlike traditional software, where bugs can be fixed with updates, smart contract vulnerabilities often lead to irreversible financial losses.

Common vulnerability types include:

- Reentrancy
- Integer Overflow/Underflow
- Access Control
- Timestamp Dependency
- Insecure Randomness

### 2.2.3 Critical Need for Vulnerability Detection

The proactive identification of security weaknesses in smart contracts is not merely a technical requirement but a fundamental necessity for the viability of the blockchain ecosystem. This necessity is driven by four compounding factors:

- **Permanent Risks (Immutability):** Once a smart contract is deployed to the mainnet, its code cannot be altered. Embedded bugs or vulnerabilities become permanent fixtures of the contract unless the contract is migrated or destroyed, which is often impractical.
- **Financial Stakes:** Smart contract vulnerabilities often translate directly into severe financial losses. Notable examples include The DAO attack and the Ronin Network breach.
- **Evolving Threats:** Attackers continuously develop methods to exploit complex logical flaws that go beyond simple coding errors. These attack vectors often outpace traditional security tools.
- **Trust and Adoption:** Security is foundational for user trust. Robust vulnerability detection is therefore necessary for wider adoption of blockchain-based systems.

### 2.2.4 Overview of Ethereum

Ethereum is a decentralized, open-source blockchain platform launched in 2015, designed to enable the execution of Turing-complete smart contracts (Buterin, 2014). Unlike Bitcoin, Ethereum’s core innovation lies in its ability to host self-executing programs that automate complex agreements without intermediaries, making it the de facto standard for decentralized applications (DApps).

The platform operates on the Ethereum Virtual Machine (EVM), a runtime environment that executes smart contract code written in Solidity, the most widely adopted language for Ethereum development (Dannen, 2017). Ethereum’s dominance, tool maturity, and availability of large-scale datasets make it the optimal target platform for this research.

## 2.3 Analysis of Vulnerability Detection Techniques

### 2.3.1 Traditional Vulnerability Detection Tools

#### 2.3.1.1 Rule-Based and Static Analysis Tools

Traditional tools used in smart contract vulnerability detection include:

- **Mythril:** Uses symbolic execution to detect common vulnerabilities such as reentrancy and integer overflow.
- **Slither:** Uses abstract syntax trees (ASTs) to identify insecure Solidity patterns.
- **Oyente:** Explores execution paths symbolically but struggles with complex contracts.
- **Manticore:** Supports deep symbolic execution for contracts and binaries but is comparatively slow.
- **Securify:** Uses abstract interpretation / formal verification, but may over-approximate and generate false positives.
- **MythX:** A cloud-based hybrid tool combining static analysis, fuzzing, and symbolic execution, but proprietary and difficult to reproduce.
- **SmartCheck:** A lightweight AST-based analyzer that is easy to integrate but weak at semantic and logical flaw detection.

#### 2.3.1.2 Limitations of Traditional Tools

Traditional rule-based and static analysis tools exhibit four critical limitations when detecting vulnerabilities in Ethereum smart contracts.

##### 1. Rule-Based Rigidity

Tools like Mythril and Slither rely on fixed signature-based detection logic, meaning they only identify vulnerabilities that match predefined syntactic patterns. This rigidity is problematic because attackers continuously adapt and exploit novel vulnerability variants that fall outside existing rule sets. The literature highlights that logic-based vulnerabilities and cross-contract exploit patterns are frequently missed.

##### 2. High False Positives

Static analysis tools generate excessive false positives, especially for pre-0.8.x contracts that rely on SafeMath or ReentrancyGuard rather than native compiler protections. The reviewed literature reports notable false-positive rates for:

- integer overflow alerts in SafeMath-protected code
- reentrancy alerts in contracts protected by ReentrancyGuard

This increases manual audit burden and reduces trust in automated tools.

##### 3. Inability to Capture Context

Traditional tools lack semantic understanding of Ethereum smart contract logic, failing to distinguish between malicious patterns and legitimate design choices. For example, they may flag all `delegatecall` usage as dangerous, even when protected by `onlyOwner`, or misclassify benign proxy architectures as vulnerable.

##### 4. Poor Scalability

Mythril is particularly slow due to symbolic execution and can become infeasible on large datasets or highly complex contracts. Slither is faster, but still suffers from timeouts and scalability bottlenecks on nested or complex pre-0.8.x contracts. This creates challenges for large-scale dataset curation and real-time scanning.

### 2.3.2 AI-Driven Vulnerability Detection Approaches

To address traditional tools’ limitations, researchers have adopted AI and machine learning models, including Transformer-based semantic models and GNN-based structural models.

The chapter reviews studies using:

- **Semantic analysis with Transformer models**
- **Structural analysis with graph-based methods**
- **Hybrid models combining semantic and structural learning**

It also presents comparative tables of:

- cross-contract interaction support
- approach type
- algorithm / technique
- interpretability support
- dataset used
- modality used
- reported accuracy / F1-score

#### Table 2.1 Method Analysis of Related Work

The reviewed works compare characteristics such as:

- whether cross-contract interaction is supported
- whether the approach is semantic, structural, or hybrid
- whether interpretability is included
- whether the method is static or hybrid in analysis behavior

Representative studies discussed include:

- Wang et al. (2025)
- Liu et al. (2023a)
- Duan et al. (2023)
- Chen et al. (2023)
- Osei et al. (2025)
- Zhang et al. (2022a, 2022b)
- Liu et al. (2023b)
- Xu et al. (2025)
- Gong et al. (2023)
- Tan et al. (2023)

#### Table 2.2 Analysis of Smart Contract Datasets, Data Modality and Key Results

The literature review notes that:

- **SmartBugs Wild** is widely used as a multimodal benchmark dataset.
- **Ethereum Smart Contracts (ESC)** is frequently used in semantic or structural studies.
- studies report a wide range of performance values depending on task design and data modality.

### 2.3.2.1 Semantic Analysis with Transformer Models

Transformer-based models such as BERT and CodeBERT represent a paradigm shift in automated vulnerability detection. Unlike traditional static analysis, these models use self-attention to process source code as sequential text, allowing them to capture semantic relationships and long-range dependencies between tokens.

These models are strong at:

- inferring developer intent
- detecting subtle logic errors
- capturing contextual meaning beyond keyword matching

However, their limitations include:

- weak modeling of structural execution behavior
- limited ability to capture control-flow-dependent vulnerabilities such as reentrancy
- lack of built-in interpretability in many implementations

### 2.3.2.2 Structural Analysis (Graph-Based Methods)

Graph Neural Networks (GNNs) are well suited for vulnerability detection because they model structural relationships of code using:

- Control Flow Graphs (CFGs)
- Abstract Syntax Trees (ASTs)
- transaction graphs

These methods are effective for:

- reentrancy
- integer overflow
- cross-contract dependency patterns
- flow-dependent vulnerabilities

Their main limitation is lack of semantic understanding. Without context regarding developer intent or meaning, structurally similar benign and malicious code may be treated the same, leading to false positives.

### 2.3.2.3 Hybrid Models (Transformer + GNN/CNN)

Hybrid models aim to integrate the semantic understanding of Transformers with the structural reasoning of GNNs or CNNs. These models seek to create a unified feature space that captures both:

- what the code intends to do
- how it executes

The review finds that hybrid approaches generally improve detection accuracy across diverse vulnerability types. However, they still face persistent challenges:

- ineffective multimodal fusion
- limited interpretability
- high computational overhead
- weak handling of cross-contract interactions in some architectures

### 2.3.3 Analysis Methodology and Criteria

To evaluate the performance, limitations, and practical utility of vulnerability detection techniques for Ethereum smart contracts, the chapter defines a multi-faceted analysis methodology integrating:

- empirical tool testing
- literature synthesis
- real-world exploit validation

#### 2.3.3.1 Core Evaluation Criteria

The core evaluation criteria include:

1. **Vulnerability Coverage**
2. **Detection Performance**

Target SWC-aligned vulnerabilities include:

- SWC-101 Integer Overflow/Underflow
- SWC-104 Unchecked External Calls
- SWC-107 Reentrancy
- SWC-109 Uninitialized Storage Pointer
- SWC-112 Dangerous Delegatecall
- SWC-114 Transaction-Ordering Dependence
- SWC-116 Timestamp Dependency

Key performance metrics include:

- Precision
- Recall
- F1-score
- False Positive Rate

## 2.4 Research Gaps

This section identifies the main gaps in the reviewed studies.

### 2.4.1 Limited Vulnerability Scope

Most existing tools and models focus heavily on well-known vulnerabilities such as reentrancy and integer overflow. There is a lack of comprehensive coverage for:

- complex business logic flaws
- emerging attack vectors
- less common but high-impact issues

The chapter’s analysis notes that several studies cover standard vulnerabilities but do not extensively address novel logic errors or more complex attack scenarios in a unified framework.

### 2.4.2 Multimodal Fusion Complexity

Integrating diverse data modalities such as source code, ASTs, CFGs, and opcodes into a unified representation remains technically challenging. Existing hybrid attempts often concatenate features without properly aligning semantic and structural information, leading to computational overhead and suboptimal performance.

### 2.4.3 Lack of Cross-Contract Interaction Analysis

A significant limitation in the current literature is that many methods analyze contracts in isolation. Modern DeFi protocols depend on interactions between multiple contracts, and many existing approaches fail to model vulnerabilities that arise specifically from inter-contract dependencies and calls.

### 2.4.4 Lack of Interpretability and the “Black Box” Problem

Deep learning models, especially hybrid architectures, often function as black boxes. Although some studies attempt to include diagnostic reports or post-hoc explanations, most state-of-the-art models still lack intrinsic interpretability mechanisms. This lack of transparency limits developer trust and practical adoption in professional auditing workflows.

## 2.5 Summary

The literature confirms that while AI-driven approaches have surpassed traditional tools in adaptability, no single existing framework perfectly balances semantic understanding, structural reasoning, cross-contract context, and interpretability. This thesis addresses these gaps with an implemented **Hybrid CodeBERT–GCN** system: **Slither-only** function-level labels on **ESC** and **SmartBugs Wild**, **cross-modal attention** for semantic–structural fusion, an optional **cross-contract** branch for multi-file projects, and **SHAP**-based explanations validated through case studies and expert review (Chapters 3–5).
