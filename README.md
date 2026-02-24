# Encrypted Traffic Classification Methods

A survey of specific methods currently used in the academic field for classifying encrypted network traffic, including methods for assessing the robustness of these systems against adversarial attacks.

---

## Background

As encryption protocols such as TLS 1.3, QUIC, and VPNs become ubiquitous, traditional traffic classification techniques — port-based identification and deep packet inspection (DPI) — have become ineffective. The academic community has responded with a rich body of work spanning statistical analysis, classical machine learning, deep learning, and hybrid approaches. Below is a structured overview of the major method families and specific techniques in active use.

---

## 1. Non-ML Heuristic and Metadata-Based Methods

These methods do not rely on machine learning models and instead exploit protocol metadata or deterministic rules.

### 1.1 Port-Based Classification (Legacy)
- Maps well-known port numbers (e.g., 443 for HTTPS, 853 for DNS-over-TLS) to application protocols.
- **Limitation:** Modern applications use dynamic ports, port multiplexing, and tunneling, making this approach unreliable. It is considered the "first generation" of traffic classification and is largely obsolete as a standalone method.

### 1.2 Server Name Indication (SNI) Inspection
- During the TLS handshake (ClientHello), the client transmits the target hostname in plaintext via the SNI field.
- Classifiers match the SNI value against known domain-to-application mappings.
- **Limitation:** TLS 1.3 with Encrypted Client Hello (ECH) encrypts the SNI field, rendering this method ineffective. Even without ECH, CDNs and shared hosting cause many applications to share SNI values.

### 1.3 Fingerprinting Methods (JA3, JA4, JARM)
- **JA3 / JA3S:** Creates an MD5 hash fingerprint from TLS ClientHello parameters (cipher suites, extensions, elliptic curves) and ServerHello responses. Widely used for identifying client applications and malware.
- **JA4+:** An evolution of JA3 with improved fingerprint granularity covering TLS, HTTP, TCP, and other layers.
- **JARM:** Active fingerprinting of TLS servers by sending crafted ClientHello messages and hashing the ServerHello responses.
- **Limitation:** Fingerprint databases must be continuously maintained. TLS library updates and randomization features in modern browsers reduce fingerprint stability.

### 1.4 DNS-Based Association
- Correlates DNS queries preceding a TLS connection with the resulting encrypted flow to infer the application.
- **Limitation:** DNS-over-HTTPS (DoH) and DNS-over-TLS (DoT) encrypt DNS traffic, and cached DNS responses may not produce observable queries.

---

## 2. Classical Machine Learning Methods

These approaches extract hand-crafted statistical features from encrypted flows and feed them into traditional ML classifiers.

### 2.1 Feature Categories

| Feature Type | Examples |
|---|---|
| **Flow-level statistics** | Total bytes, duration, mean/variance of inter-arrival times, number of packets per flow |
| **Packet-level statistics** | Packet size distribution, payload length histogram, direction sequences |
| **Burst-level features** | Burst size, burst duration, number of bursts per flow |
| **TLS handshake metadata** | Cipher suites offered, extensions, certificate chain length, handshake message sizes |
| **Time-series features** | Autocorrelation, spectral density of packet timing |

### 2.2 Specific Classifiers in Use

- **Random Forest (RF):** The most widely adopted classical ML method for encrypted traffic classification. Ensembles of decision trees offer robustness, interpretability, and strong performance with statistical flow features. Frequently used as a baseline in academic studies.
- **Gradient Boosted Trees (XGBoost, LightGBM, CatBoost):** Boosted tree ensembles that often outperform RF, especially on imbalanced datasets.
- **k-Nearest Neighbors (k-NN):** Simple instance-based classifier; effective when feature spaces are low-dimensional but does not scale well.
- **Support Vector Machines (SVM):** Effective in higher-dimensional feature spaces with kernel tricks (RBF, polynomial). Used in earlier encrypted traffic studies.
- **Naive Bayes:** Fast probabilistic classifier; serves as a lightweight baseline in many comparative studies.
- **Hidden Markov Models (HMM):** Used to model temporal sequences of packet sizes or directions as state transitions, capturing traffic dynamics.

### 2.3 Feature Selection and Reduction
- **Principal Component Analysis (PCA)** and **Linear Discriminant Analysis (LDA)** are applied to reduce high-dimensional feature sets.
- **Information Gain**, **Chi-squared tests**, and **Mutual Information** are used for feature ranking and selection.

---

## 3. Deep Learning Methods

Deep learning methods can automatically learn representations from raw traffic data, reducing the need for manual feature engineering.

### 3.1 Convolutional Neural Networks (CNN)

- **1D-CNN on raw bytes:** Treats the first N bytes of a flow's payload as a 1D signal and applies convolutional filters to learn local byte patterns. This is one of the most common approaches.
- **2D-CNN on traffic images:** Reshapes raw packet bytes into 2D grayscale images (e.g., 28×28 pixel grids) and applies image classification architectures. Notable works include methods that convert flows into "traffic images."
- **NetConv (2025):** A pre-trained convolutional model using stacked traffic convolution layers with window-wise byte scoring and sequence-wise byte gating. Demonstrated 6.88% improvement over transformer models and 7.41× higher throughput with linear complexity.
- CNNs are frequently reported as the strongest single-model performer, with studies showing up to **99.34% accuracy** on HTTPS traffic classification benchmarks.

### 3.2 Recurrent Neural Networks (RNN)

- **LSTM (Long Short-Term Memory):** Models the sequential nature of packet arrivals, capturing long-range temporal dependencies in traffic flows. Applied to both packet-level features and raw byte sequences.
- **GRU (Gated Recurrent Unit):** A lighter variant of LSTM; often achieves comparable accuracy with lower computational cost.
- **Bidirectional RNNs:** Process packet sequences in both forward and backward directions for richer temporal context.

### 3.3 Transformer-Based Models

- **TransECA-Net (2025):** A transformer architecture specifically designed for encrypted traffic classification, using self-attention over byte sequences.
- **ET-BERT:** A pre-trained transformer model that learns contextual representations of encrypted traffic bytes, then fine-tunes for downstream classification tasks.
- **Limitation:** Quadratic complexity in self-attention limits scalability to long byte sequences; positional encodings may not generalize beyond training sequence lengths.

### 3.4 Autoencoders and Generative Models

- **Variational Autoencoders (VAE):** Used for unsupervised or semi-supervised feature learning from unlabeled encrypted traffic.
- **Generative Adversarial Networks (GANs):** Applied for data augmentation — generating synthetic traffic samples to address class imbalance, and for anomaly detection in encrypted flows.

### 3.5 Graph Neural Networks (GNN)

- **Traffic Interaction Graphs:** Model hosts and flows as nodes and edges in a communication graph. GNNs learn structural patterns that individual flow analysis cannot capture.
- **S2-ETR (2024):** Integrates semantic feature extraction with communication topology via a Hyper-Bipartite Graph framework, outperforming 15 baselines by 2.4%–17.1%.
- **AppNet and FlowPrint:** Build device-level or app-level traffic graphs and use graph-based clustering or classification for application identification.

---

## 4. Hybrid and Ensemble Methods

### 4.1 Stacked Deep Ensembles
- Combine multiple DL architectures (e.g., CNN + LSTM + GRU) in a stacked ensemble. A 2025 study in *Scientific Reports* demonstrated that stacked deep ensembles improve robustness over single-model classifiers for HTTPS traffic.

### 4.2 Multi-Stage / Cascaded Classifiers
- Use a lightweight first-stage classifier (e.g., SNI or fingerprint matching) and fall back to an ML/DL model only for unresolved flows, balancing accuracy with computational efficiency.
- The **DPC (Determine whether Plaintext is enough to be Classified)** selector optimizes whether to classify using plaintext metadata alone or to invoke a heavier encrypted-content model.

### 4.3 Multi-Modal Fusion
- Combine heterogeneous feature types (flow statistics + raw bytes + graph structure) using attention-based fusion layers.
- Joint training on packet-level and flow-level representations.

---

## 5. Pre-Training and Self-Supervised Learning

A significant trend in 2024–2025 is leveraging large-scale **unlabeled** encrypted traffic for pre-training.

- **Masked byte modeling:** Analogous to masked language modeling in NLP — random bytes in a packet are masked, and the model learns to predict them, acquiring general traffic representations.
- **Contrastive learning:** Trains encoders to produce similar representations for augmented views of the same flow and dissimilar representations for different flows.
- **Transfer learning:** Pre-trained models (ET-BERT, NetConv) are fine-tuned on small labeled datasets, drastically reducing the labeled data requirement.

---

## 6. Specific Application Domains

| Task | Typical Methods |
|---|---|
| **Application identification** (e.g., YouTube vs. Netflix) | CNN on raw bytes, Random Forest on flow statistics, GNN on traffic graphs |
| **Malware / intrusion detection** | LSTM on packet sequences, autoencoders for anomaly detection, JA3 fingerprinting |
| **Website fingerprinting** (over Tor/VPN) | Deep fingerprinting with CNN, k-NN with cumulative features, triplet networks |
| **VPN vs. non-VPN detection** | Gradient boosted trees on statistical features, 1D-CNN on packet sizes |
| **QoS-aware classification** (video, VoIP, browsing) | Multi-task learning with shared DL backbones, HMMs |
| **IoT device identification** | Random Forest on flow metadata, GNN on device communication graphs |

---

## 7. Publicly Available Benchmark Datasets

| Dataset | Description |
|---|---|
| **ISCX VPN-nonVPN (2016)** | Labeled VPN and non-VPN traffic for application classification |
| **ISCX Tor-nonTor (2016)** | Tor and non-Tor encrypted traffic |
| **USTC-TFC2016** | 20 classes of malware and benign encrypted traffic |
| **CIRA-CIC-DoHBrw-2020** | DNS-over-HTTPS traffic dataset |
| **CESNET-TLS22 / CESNET-QUIC22** | Large-scale TLS and QUIC datasets from a national research network |
| **Cross-Platform (2025)** | Annotated TLS communications for popular desktop and mobile applications |

---

## 8. Open Challenges

1. **Concept drift:** Traffic patterns evolve as applications update, requiring models that adapt over time (online / continual learning).
2. **Encrypted Client Hello (ECH) and QUIC:** New protocol features eliminate metadata previously used for classification.
3. **Class imbalance:** Long-tail distributions where a few applications dominate traffic volume.
4. **Privacy vs. classification trade-offs:** Ethical and legal concerns around traffic analysis.
5. **Generalizability:** Models trained on one network often perform poorly on another (domain shift).
6. **Real-time constraints:** Production deployment requires low-latency inference at line rate.

---

## 9. Adversarial Robustness Assessment Methods

A growing body of academic work focuses specifically on evaluating how resilient encrypted traffic classifiers are to adversarial manipulation. Unlike image or text domains, network traffic imposes hard protocol and semantic constraints on what an adversary can change, making this a distinct subfield. The methods below cover both the attack techniques used to probe classifiers and the defense/certification frameworks used to harden them.

### 9.1 Threat Models and Adversary Capabilities

Academic robustness assessments operate under clearly defined threat models that specify what the adversary can observe and modify:

| Threat Model | Adversary Knowledge | Typical Use |
|---|---|---|
| **White-box** | Full access to model architecture, weights, and gradients | Upper-bound robustness evaluation; gradient-based attacks |
| **Black-box (query-based)** | Can query the model and observe outputs only | Realistic deployment scenarios; transferability studies |
| **Black-box (transfer-based)** | No query access; uses a surrogate model | Worst-case practical attack; tests cross-model generalization |
| **Gray-box** | Partial knowledge (e.g., feature set but not weights) | Intermediate realism between white-box and black-box |

A critical distinction from other adversarial ML domains is the **network-semantic constraint**: perturbations must produce valid traffic (correct checksums, valid protocol state machines, preserving application-layer functionality). This rules out arbitrary Lp-bounded perturbations common in image adversarial ML.

### 9.2 Adversarial Attack Methods Used for Robustness Probing

These are specific attack techniques researchers apply to encrypted traffic classifiers to measure their vulnerability.

#### 9.2.1 Gradient-Based Attacks (White-Box)

- **FGSM (Fast Gradient Sign Method):** Single-step perturbation along the gradient direction. Adapted for traffic by constraining perturbations to valid packet modifications (e.g., appending padding bytes rather than changing encrypted payload content).
- **PGD (Projected Gradient Descent):** Iterative version of FGSM with projection back onto the feasible perturbation set after each step. The standard benchmark for white-box robustness in both image and traffic domains.
- **C&W (Carlini & Wagner) Attack:** Optimization-based attack that minimizes perturbation magnitude while achieving misclassification. Used to establish lower bounds on classifier robustness.
- **AdvTraffic (2022):** A traffic-specific adversarial framework that applies gradient-based perturbations to encrypted flow features while respecting protocol constraints. Demonstrated effective evasion of CNN and LSTM-based classifiers.

#### 9.2.2 Universal Adversarial Perturbations (UAP)

A single perturbation pattern is crafted that, when applied to any input, degrades classifier performance across all classes. Three domain-specific variants have been proposed:

- **AdvPad:** Injects a universal adversarial perturbation into packet content/padding fields to attack packet-level classifiers.
- **AdvPay:** Injects UAPs into dummy packet payloads to target flow-content classifiers that analyze sequences of packet payloads.
- **AdvBurst:** Injects crafted dummy packets with adversarial statistical properties into flow bursts to fool time-series-based classifiers.

These three variants collectively test robustness across different feature abstraction levels (packet, payload, flow).

#### 9.2.3 Practical Constraint-Aware Attacks

- **PANTS (Practical Adversarial Network Traffic Samples, 2025):** A white-box framework that combines adversarial ML with **Satisfiability Modulo Theories (SMT) solvers** to generate adversarial traffic samples that satisfy all network-semantic constraints (valid headers, correct protocol state, functional application behavior). PANTS achieves a **70% higher median success rate** than prior baselines (Amoeba, BAP) in finding adversarial inputs. It addresses non-differentiable components in traffic processing pipelines that prevent naive gradient-based attacks.
- **Amoeba (CoNEXT 2023):** A reinforcement-learning-based approach that learns to craft adversarial packet sequences against ML-based censorship classifiers through a black-box trial-and-error process, achieving 94% average attack success rate with transferability to unseen models.
- **BAP (Blind Adversarial Perturbations):** The Nasr et al. (USENIX Security 2021) method — pre-computes universal adversarial perturbation patterns offline (white-box), then deploys them "blind" on live traffic in real-time without per-flow re-optimization. Used as a baseline in the PANTS comparison.

#### 9.2.4 Traffic Morphing and Shaping Attacks

These methods modify observable traffic characteristics (without touching encrypted content) to make one traffic class resemble another:

- **Traffic Morphing (Wright et al.):** Uses convex optimization to find the minimal set of packet padding and splitting operations that transform the statistical distribution of one traffic class to match a target class. Reduces classifier accuracy with significantly less overhead than naive constant-rate padding.
- **TANTRA (Timing-Based Adversarial Network Traffic Reshaping Attack, 2022):** Trains an LSTM on benign traffic to learn legitimate inter-packet timing distributions, then reshapes malicious traffic timing to match. Achieves ~99.99% evasion rate against NIDS classifiers while preserving packet content.
- **Packet padding/splitting:** Appending random bytes to TLS records (within protocol limits) or splitting payloads across multiple packets to alter packet-size distributions.
- **Dummy packet injection:** Inserting decoy packets that are ignored by the application but alter flow-level statistical features visible to classifiers.
- **Timing perturbation:** Adding random delays to packets to disrupt inter-arrival time features.

#### 9.2.5 GAN-Based Adversarial Traffic Generation

- **Generative Adversarial Networks** are trained to produce synthetic traffic flows that are indistinguishable from a target class according to the classifier but carry the content of a different (or malicious) class. The generator learns to produce realistic perturbations while the discriminator enforces traffic realism.

### 9.3 Defense and Robustness Enhancement Methods

#### 9.3.1 Adversarial Training

The most widely used defense: the classifier is retrained on a mixture of clean and adversarial examples.

- **PANTS adversarial training loop:** Iteratively generates adversarial traffic using the PANTS framework, augments the training set, and retrains the classifier. Demonstrated **52.7% improvement** in robustness without sacrificing clean accuracy, with improved resilience even against attack methods not seen during training.
- **PGD adversarial training:** The min-max optimization approach of Madry et al., adapted for traffic features. The classifier is trained to minimize worst-case loss under PGD perturbations.
- **Ensemble adversarial training:** Incorporates adversarial examples generated against multiple surrogate models to improve robustness against transfer attacks.

#### 9.3.2 Certified Robustness

These methods provide mathematical guarantees (not just empirical observations) that a classifier's prediction will not change within a defined perturbation radius.

- **CertTA (Certified Robustness for Traffic Analysis, USENIX Security 2025):** The first certified robustness framework designed specifically for traffic analysis models. It introduces a **multi-modal randomized smoothing** mechanism that generates provable robustness regions against three simultaneous perturbation types:
  - Additive perturbations (packet padding, timing delays)
  - Discrete alterations (dummy packet insertion/deletion)
  - Combined multi-modal attacks
  CertTA is universally applicable across six different traffic analysis model architectures and provides stronger guarantees than generic smoothing approaches.

- **PROSAC (Provably Safe Certification, 2024):** A general-purpose certification framework based on hypothesis testing that validates whether a model satisfies (α, ζ)-safety criteria — ensuring adversarial population risk remains below an acceptable threshold with statistical confidence. Applicable to traffic classifiers as a post-hoc verification tool.

- **Randomized Smoothing (generic):** Constructs a smoothed classifier by averaging predictions over random noise added to the input. Provides Lp-norm robustness certificates. Adapted for traffic by defining appropriate noise distributions over packet sizes and timing features.

#### 9.3.3 Input Preprocessing Defenses

- **Feature squeezing:** Reduces the precision of input features (e.g., quantizing packet sizes to bins) to eliminate adversarial perturbation space.
- **Traffic normalization:** Standardizes packet sizes and timing to fixed patterns, removing adversarial modifications at the cost of reduced classification granularity.
- **Anomaly detection filters:** A secondary model detects inputs that deviate from the training distribution before they reach the classifier, flagging potential adversarial samples.

#### 9.3.4 Architecture-Level Defenses

- **Defensive distillation:** Trains a student network using soft probability outputs from a teacher network, smoothing the loss surface to make gradient-based attacks less effective.
- **Input gradient regularization:** Penalizes large input gradients during training, making the model less sensitive to small perturbations.
- **Randomized inference:** Introduces stochasticity at inference time (e.g., random input transformations, dropout) so that adversarial perturbations tuned to a deterministic model fail against the randomized version.

### 9.4 Website Fingerprinting Defenses as Robustness Benchmarks

Website fingerprinting (WF) over Tor/VPN is one of the most active testbeds for adversarial robustness. Defenses developed in this subfield are widely used as robustness baselines:

- **WTF-PAD (Adaptive Padding):** Injects dummy packets according to an adaptive padding strategy, reducing state-of-the-art WF attack accuracy from 91% to 20% with zero latency overhead and <80% bandwidth overhead.
- **Walkie-Talkie:** Converts browser communication to half-duplex mode, producing moldable burst sequences. Defeats all known WF attacks with 31% bandwidth overhead and 34% time overhead.
- **FRONT:** Obfuscates the feature-rich initial portion of a traffic trace with randomized dummy packets. Achieves superior performance to WTF-PAD with only 33% data overhead.
- **GLUE:** Adds dummy packets between consecutive traces to make separate page loads appear as one continuous stream, preventing trace boundary detection.
- **Mockingbird (adversarial examples for WF defense):** Generates adversarial traffic traces that resist adversarial training by moving randomly through the space of viable traces rather than following predictable gradient directions. Reduces deep-learning WF attack accuracy from 98% to 42–58%.
- **TrafficSliver:** Splits traffic across multiple Tor circuits, reducing the information available to any single network observer.

### 9.5 Evaluation Metrics for Adversarial Robustness

| Metric | What It Measures |
|---|---|
| **Attack success rate (ASR)** | Fraction of adversarial inputs that cause misclassification |
| **Accuracy under attack** | Clean accuracy minus accuracy degradation when adversarial inputs are applied |
| **Perturbation budget / overhead** | Amount of modification required (bandwidth overhead %, added latency, number of dummy packets) |
| **Certified radius** | Provable Lp-norm (or domain-specific) radius within which predictions are guaranteed stable |
| **Transferability rate** | Fraction of adversarial examples crafted against one model that also fool a different model |
| **Robustness-accuracy trade-off** | Clean accuracy loss incurred to achieve a given level of adversarial robustness |
| **Semantic validity** | Whether adversarial traffic remains protocol-compliant and application-functional |
| **Defense overhead** | Computational cost, latency, and bandwidth cost of applying defenses |

### 9.6 Robustness Evaluation Methodology

A standard academic robustness evaluation typically follows this protocol:

1. **Baseline measurement:** Train and evaluate the target classifier on clean data, reporting standard accuracy, precision, recall, and F1.
2. **White-box attack battery:** Apply FGSM, PGD (multiple ε values), and C&W attacks to establish the worst-case robustness under full adversary knowledge.
3. **Constraint-aware attack:** Use a domain-specific tool (PANTS, traffic morphing) to generate attacks that satisfy network-semantic constraints, measuring practical evasion rates.
4. **Black-box transferability:** Generate adversarial examples against surrogate models and evaluate transfer success against the target classifier.
5. **Defense integration:** Apply adversarial training or certified smoothing and re-evaluate under the same attack battery.
6. **Overhead analysis:** Report bandwidth overhead, latency impact, and computational cost of both attacks and defenses.
7. **Ablation studies:** Vary perturbation budgets, feature subsets, and model architectures to understand robustness drivers.

---

## 10. Multi-Class Classification Under Adversarial Conditions

The majority of adversarial robustness work surveyed in Section 9 includes substantial multi-class evaluation (website fingerprinting in closed-world is inherently multi-class, and many attack papers evaluate on multi-class application/service datasets). However, the user may wonder which papers go beyond binary anomaly/malware detection to tackle adversarial robustness for **multi-class** encrypted traffic classification tasks — e.g., distinguishing among many application types, service categories, or individual websites simultaneously under adversarial conditions. This section highlights those papers explicitly, together with additional multi-class-focused work not covered earlier.

### 10.1 Which Existing Section-9 Papers Are Multi-Class?

Most of the papers already listed in Section 9 evaluate multi-class scenarios. The table below clarifies the classification granularity for each.

| Paper | Multi-Class Task | Number of Classes | Dataset(s) |
|---|---|---|---|
| **[B1] Sadeghzadeh et al. (IEEE TIFS 2021)** | Application-type classification | 12 classes (VPN + non-VPN app categories: chat, VoIP, streaming, file transfer, email, P2P, browsing) | ISCX-VPN-nonVPN 2016, USTC-TFC2016 (20 malware + benign classes) |
| **[B2] Nasr et al. (USENIX Security 2021)** | Website fingerprinting (closed-world multi-class) | 95+ monitored websites | Custom Tor WF dataset |
| **[B4] PANTS (USENIX Security 2025)** | App identification + VPN detection + QoE inference | 18 app classes; 11 resolution classes; binary VPN | UTMobileNetTraffic2021, VCAML, ISCXVPN2016 |
| **[B5] TANTRA (IEEE TDSC 2022)** | Multi-class attack-type identification | 8 attack types + benign | CIC-IDS-2017, CSE-CIC-IDS-2018 |
| **[B8] Chehade et al. (arXiv 2025)** | Multi-class application classification | Multiple classes (VPN app categories) | ISCX-VPN-nonVPN 2016 |
| **[C1] CertTA (USENIX Security 2025)** | Website fingerprinting + app classification (multi-class) | 95–100 monitored websites; application categories | Tor WF datasets, encrypted app traffic |
| **[D1] HyperVision (NDSS 2023)** | Multi-class malicious traffic type identification | 48+ attack types | 92 datasets (48 encrypted malicious traffic types) |
| **[D2] ReTrial (IEEE TIFS 2025)** | Multi-class encrypted malware family classification | Multiple malware families + benign | USTC-TFC2016, CIC-DoHBrw-2020 |
| **[E1] Deep Fingerprinting (CCS 2018)** | Multi-class website identification | 100 monitored websites (closed-world) | Custom Tor dataset |
| **[E4] Tik-Tok (PETS 2020)** | Multi-class website identification | 100 monitored sites + 9,000 unmonitored | Custom Tor dataset |
| **[E8] TrafficSliver (CCS 2020)** | Multi-class website identification | 100+ websites | Custom Tor dataset |
| **[E11] Cherubin et al. (USENIX Security 2022)** | Multi-class website identification | 5–25+ monitored websites (scaling study) | Real-world Tor traffic (>60M flows) |

### 10.2 Additional Multi-Class Adversarial Robustness Papers

The following papers focus specifically on multi-class encrypted traffic classification under adversarial conditions and were not fully covered in Section 9. They are prioritized by venue impact.

---

**[M1]** Kai Wang, Zhiliang Wang, Dongqi Han, Wenqi Chen, Jiahai Yang, Xingang Shi, and Xia Yin.
"BARS: Local Robustness Certification for Deep Learning based Traffic Analysis Systems."
In *Proceedings of the 30th Network and Distributed System Security Symposium (NDSS 2023)*, San Diego, CA, USA, February 27–March 3, 2023.
- **Venue:** NDSS (top-4 security conference).
- **Multi-class task:** Evaluated on three practical DL-based traffic analysis systems performing multi-class classification (website fingerprinting with 100 classes, encrypted application identification).
- **Contribution:** Introduced **Boundary-Adaptive Randomized Smoothing (BARS)** with a novel Distribution Transformer that optimizes smoothing noise converging on classification boundaries. Provides tighter local robustness certificates than generic smoothing. Demonstrated on five practical application cases with gradient-based noise optimization.
- **Code:** github.com/KaiWangGitHub/BARS

**[M2]** Xiyuan Zhao, Xinhao Deng, Qi Li, Yunpeng Liu, Zhuotao Liu, Kun Sun, and Ke Xu.
"Towards Fine-Grained Webpage Fingerprinting at Scale."
In *Proceedings of the 31st ACM Conference on Computer and Communications Security (CCS 2024)*, Salt Lake City, UT, USA, October 14–18, 2024. DOI: 10.1145/3658644.3690211.
- **Venue:** ACM CCS (top-4 security conference).
- **Multi-class task:** Fine-grained **webpage-level** identification (not just website-level) — 1,000 monitored webpages + 9,000+ unmonitored webpages from encrypted Tor traffic. Far exceeds the class count of typical WF studies.
- **Contribution:** Proposed **Oscar**, a multi-label metric learning framework combining proxy-based and sample-based losses. Achieved **88.6% improvement in Recall@5** over state-of-the-art. Addresses the challenge of distinguishing subpages within the same website where traffic patterns are highly similar.

**[M3]** Jiaxing Zhang, Xinhao Deng, Qi Li, and Ke Xu.
"Transformer-based Model for Multi-tab Website Fingerprinting Attack."
In *Proceedings of the 2023 ACM SIGSAC Conference on Computer and Communications Security (CCS 2023)*, Copenhagen, Denmark, November 26–30, 2023. DOI: 10.1145/3576915.3623107.
- **Venue:** ACM CCS (top-4 security conference).
- **Multi-class task:** Identifies **multiple websites** simultaneously from mixed multi-tab encrypted Tor traffic traces. Formulated as set prediction over N classes — a significantly harder multi-class problem than single-tab WF.
- **Contribution:** Proposed **TMWF**, a Transformer-based model inspired by object detection that uses single-tab queries to extract features from multi-tab traces. Eliminates the unrealistic assumption that tab count is known. Includes a new open-world multi-tab dataset.
- **Code:** github.com/jzx-bupt/TMWF

**[M4]** Quanliang Jing, Xinxin Fan, Yanyan Liu, and Jingping Bi.
"Adversarial Pre-Padding: Generating Evasive Network Traffic Against Transformer-Based Classifiers."
*arXiv preprint*, arXiv:2510.25810, October 2025 (revised December 2025).
- **Multi-class task:** Evaluated on three multi-class encrypted traffic datasets — **CSTNET-TLS 1.3** (120 application classes), **ISCX-VPN** (application and service categories, 12+ classes), and **ISCX-Tor** (8 classes) — targeting pre-trained transformer models.
- **Contribution:** First adversarial attack specifically designed for **pre-trained transformer-based encrypted traffic classifiers** (ET-BERT, YaTC, NetMamba). Uses a **pre-padding strategy** (modifying bytes before the encrypted payload) combined with **reinforcement learning** (MDP formulation) to generate adversarial byte sequences. Reduced multi-class classification accuracy from **99% to as low as 25.68%** across all three datasets and three model architectures.

**[M5]** Ahmed M. Elshewey and Ahmed M. Osman.
"Enhancing Encrypted HTTPS Traffic Classification Based on Stacked Deep Ensembles Models."
*Scientific Reports*, vol. 15, Article 35230, 2025. DOI: 10.1038/s41598-025-21261-6.
- **Venue:** Scientific Reports (Nature Portfolio, IF ~4.6).
- **Multi-class task:** 6-class HTTPS traffic type classification (Download, Live Video, Music, Player, Upload, Website) on 145,671 flows with 88 features.
- **Contribution:** Stacked deep ensemble (DNN + CNN + RNN + LSTM + GRU with multinomial logistic regression meta-learner) achieving **99.49% accuracy, 0.9932 macro-F1, 0.9998 macro-AUC**. Outperformed individual models (best single: CNN at 99.34%). Demonstrated ensemble robustness advantage over single-model classifiers for multi-class encrypted traffic.

**[M6]** Zheng Li, Yanbei Liu, Changqing Zhang, Wanjin Shan, Haifeng Zhang, and Xiaoming Zhu.
"Trustworthy Deep Learning for Encrypted Traffic Classification."
*Soft Computing*, vol. 29, no. 2, pp. 645–662, February 2025. DOI: 10.1007/s00500-025-10462-w.
- **Venue:** Soft Computing (Springer).
- **Affiliations:** Tiangong University (Tianjin), Tianjin University, and the 54th Research Institute of CETC (Shijiazhuang).
- **Multi-class task:** Application and service classification on **ISCX VPN-nonVPN** (12 classes: VPN/non-VPN × 6 app types) and **USTC-TFC2016** (20 classes: 10 malware families + 10 benign apps).
- **Contribution:** Introduced **ConfidNet** — a confidence-calibration network trained alongside the traffic classifier (ConvNet + ClassifyNet) to provide reliable confidence scores. Specifically designed to identify misclassified multi-class samples. Addresses the gap where high-accuracy classifiers give overconfident wrong predictions, which is critical when adversarial perturbations cause targeted misclassification.

**[M7]** Thulfiqar Mahmood Tawfeeq and Mohsen Nickray.
"Adversarial Training for Improved VPN Traffic Classification Using EfficientNet-B0 and Projected Gradient Descent."
*International Journal of Intelligent Engineering and Systems*, vol. 18, no. 1, 2025. DOI: 10.22266/ijies2025.0229.87.
- **Multi-class task:** VPN application-type classification on ISCX 2016 dataset (multiple VPN app categories: browsing, email, chat, streaming, file transfer, VoIP).
- **Contribution:** Combined **EfficientNet-B0 + biLSTM** architecture with **PGD adversarial training**. Achieved **99.81% accuracy on clean traffic** and **99.35% on adversarial traffic**, demonstrating that adversarial training preserves multi-class accuracy while adding robustness against perturbations (packet delays, congestion, adversarial noise).

**[M8]** MTDecipher Authors (various affiliations).
"MTDecipher: Robust Encrypted Malicious Traffic Detection via Multi-Task Graph Neural Networks."
*Cybersecurity* (Springer Nature), vol. 9, Article 112, January 2026. DOI: 10.1186/s42400-025-00522-x.
- **Venue:** Cybersecurity (Springer Nature).
- **Multi-class task:** Simultaneous **edge classification** (flow-level malicious/benign) and **node classification** (multi-class malware type/family identification) in a multi-task formulation.
- **Contribution:** Bidirectional attentive sequence encoder (Bi-GRU + attention) with edge-block dual sampling for robust encrypted traffic detection. Multi-task GNN jointly optimizes edge and node losses to reduce structural bias. Outperforms eight existing methods on two real-world datasets with traffic obfuscation.

**[M9]** Cuong Dao, Van Tong, Nam Thang Hoang, Hai Anh Tran, and Truong X. Tran.
"Enhancing Encrypted Traffic Classification with Deep Adaptation Networks."
In *Proceedings of the 48th IEEE Conference on Local Computer Networks (LCN 2023)*, Daytona Beach, FL, USA, October 2–5, 2023. DOI: 10.1109/LCN58197.2023.10223333.
- **Venue:** IEEE LCN.
- **Multi-class task:** Multi-class encrypted application traffic classification using domain adaptation for cross-network generalization.
- **Contribution:** Deep Adaptation Network with Mean Embedding Test for multi-class encrypted traffic classification when training and deployment networks differ (domain shift). Addresses a robustness challenge where classifiers fail on traffic from new networks — relevant to adversarial robustness as domain shift and adversarial perturbation are closely related failure modes.

### 10.3 Summary of Multi-Class Coverage

The combined paper set (Sections 9 + 10) covers adversarial robustness across a wide range of multi-class encrypted traffic tasks:

| Multi-Class Task | Class Count | Representative Papers |
|---|---|---|
| **Application identification** (e.g., YouTube, Skype, Netflix) | 6–120 classes | [B1], [B4], [M4], [M5], [M6], [M7], [M9] |
| **Service-type classification** (chat, VoIP, streaming, browsing) | 6–12 classes | [B1], [B8], [M6], [M7] |
| **Website fingerprinting** (closed-world) | 25–1,000+ sites | [B2], [C1], [E1], [E4], [E8], [E11], [M1], [M2] |
| **Multi-tab website identification** | N websites from mixed traces | [M3] |
| **Malware family classification** | 10–48 types | [D1], [D2], [M8] |
| **Attack-type detection** (DDoS, brute-force, botnet, etc.) | 8+ types | [B5] |
| **Operating system / browser identification** | Multiple OS × browser | IEEE DataPort dataset |
| **Certified robustness for multi-class traffic** | 100+ classes | [C1] CertTA, [M1] BARS |

---

## 11. Literature Review: Multi-Class Encrypted Traffic Classification Under Adversarial Conditions

This section provides a focused, structured analysis of papers that address **multi-class** classification of encrypted network traffic under adversarial conditions. For each paper, we examine: (1) the multi-class scenario, (2) the threat model and target classifier architecture, (3) the traffic features exploited, (4) the concrete steps for adversarial data generation, and (5) the optimization method employed.

### 11.1 Sadeghzadeh et al. — AdvPad / AdvPay / AdvBurst (IEEE TIFS, 2021)

**Citation:** Amir Mahdi Sadeghzadeh, Saeed Shiravi, and Rasool Jalili. "Adversarial Network Traffic: Towards Evaluating the Robustness of Deep-Learning-Based Network Traffic Classification." *IEEE TIFS*, vol. 16, pp. 3940–3955, 2021. DOI: 10.1109/TIFS.2021.3053093.

**(1) Multi-class scenario.** The paper evaluates on three multi-class datasets: **ISCX-VPN-App** (application-level: browsing, email, chat, streaming, file transfer, VoIP — 12 classes combining VPN and non-VPN), **ISCX-VPN-Service** (service-level categories), and **USTC-TFC2016** (20 classes: 10 malware families + 10 benign applications). The classifiers are partitioned into three categories by input type: packet classifiers, flow content classifiers, and flow time-series classifiers — each evaluated on the multi-class tasks.

**(2) Threat model & target architectures.**
- **Threat model:** **White-box.** The attacker has full access to model weights and gradients.
- **Target classifiers:**
  - *Packet classifiers:* 1D-CNN operating on raw byte sequences of individual packets (inspired by Deep Packet).
  - *Flow content classifiers:* 1D-CNN / 2D-CNN processing concatenated payload bytes of the first N packets in a flow.
  - *Flow time-series classifiers:* 2D-CNN on FlowPic-style images (packet-size × time 2D histograms), and LSTM models on sequences of packet-size/inter-arrival-time feature vectors.

**(3) Traffic features exploited for adversarial generation.**
- **AdvPad:** Raw byte content of packet payloads (the padding region appended after the encrypted payload in each packet).
- **AdvPay:** Payload bytes of injected dummy packets (entire new packets whose payloads carry the perturbation, added to the flow).
- **AdvBurst:** Statistical features of flow bursts — packet sizes, inter-arrival times, and burst boundaries. Adversarial dummy packets are crafted to alter these statistical properties.

**(4) Adversarial data generation steps.**
1. Select target classifier category (packet / flow content / flow time-series) and corresponding input representation.
2. Compute a **Universal Adversarial Perturbation (UAP)** — a single perturbation vector effective across all inputs — using an iterative algorithm adapted from DeepFool:
   - For each training sample, compute the minimal perturbation that crosses the classifier's nearest decision boundary.
   - Accumulate perturbations across samples and project back onto an Lp-norm ball.
   - Iterate until a target fooling rate is reached on a validation set.
3. Apply the UAP to traffic according to the attack variant:
   - *AdvPad:* Append the UAP bytes into the padding field of each packet.
   - *AdvPay:* Inject new dummy packets whose payloads contain the UAP.
   - *AdvBurst:* Insert dummy packets at chosen burst positions so the resulting burst-level statistics match the perturbation target.
4. Verify that modified traffic remains protocol-compliant (valid TLS records, correct TCP checksums).

**(5) Optimization method.**
- **UAP via iterative DeepFool projected onto an Lp-ball.** For each sample x_i: solve `argmin ||δ||_p s.t. f(x_i + δ) ≠ f(x_i)` using the signed gradient of the nearest decision hyperplane. The accumulated perturbation δ is projected onto `{δ : ||δ||_p ≤ ε}` after each update. The procedure is iterated over the dataset until the fooling rate (fraction of misclassified samples) exceeds a threshold. This is an **untargeted** attack — the goal is misclassification to any wrong class.

---

### 11.2 Nasr et al. — Blind Adversarial Perturbations (USENIX Security, 2021)

**Citation:** Milad Nasr, Alireza Bahramali, and Amir Houmansadr. "Defeating DNN-Based Traffic Analysis Systems in Real-Time With Blind Adversarial Perturbations." *USENIX Security 2021*, pp. 2705–2722.

**(1) Multi-class scenario.** Evaluated on **website fingerprinting (WF)** — a closed-world multi-class task where the classifier identifies which of **95 monitored websites** a Tor user is visiting. Also evaluated on **flow correlation** (binary, but the WF component is fully multi-class).

**(2) Threat model & target architectures.**
- **Threat model:** **White-box** for perturbation design (access to model gradients during the offline training phase), but the resulting perturbations are applied **blind** — they are independent of the specific target flow and can be deployed in real-time without re-optimization.
- **Target classifiers:**
  - **Deep Fingerprinting (DF):** A CNN architecture with convolutional, batch normalization, max-pooling, and fully connected layers — the state-of-the-art WF attack at time of publication.
  - **DeepCorr:** A CNN for flow correlation.
  - **Var-CNN:** A variant CNN architecture for WF.

**(3) Traffic features exploited.**
- **Packet direction sequences:** Ordered sequence of +1 (outgoing) and −1 (incoming) per packet.
- **Packet sizes:** Byte length of each packet.
- **Inter-packet timing (IAT):** Time intervals between consecutive packets.
- **Burst-level features:** Aggregated statistics of consecutive same-direction packet groups.
- Key insight: these features are *interdependent* — e.g., inserting a dummy packet changes not only the packet count but also all subsequent IATs and burst boundaries.

**(4) Adversarial data generation steps.**
1. **Offline perturbation design:** Using a training set, compute an **adversarial perturbation pattern** (a "patch" of dummy packets and timing delays) that is *universal* — effective across all input flows.
2. **Remapping technique:** Because traffic features are mutually dependent, a perturbation in one feature (e.g., inserting a dummy packet) cascades to others (IAT changes, burst changes). The remapping function maps the desired feature-space perturbation back to realizable traffic-space operations: `traffic_ops = Remap(feature_perturbation)`, ensuring all dependent features remain consistent.
3. **Real-time application:** The pre-computed perturbation pattern is applied to **live traffic** by injecting dummy packets and adding timing delays at predetermined positions — without buffering or inspecting the target flow content.
4. The perturbation pattern consists of: (a) a sequence of dummy packet insertions at specified positions, (b) specified delays added to real packets.

**(5) Optimization method.**
- **Gradient-based optimization of an adversarial loss function** over the training set. The loss encourages the classifier's output distribution to be uniform (maximum confusion) when the perturbation is applied. Formally: `min_δ E_x[L_CE(f(x + Remap(δ)), y_uniform)]` where L_CE is cross-entropy loss, Remap enforces feature dependency constraints, and y_uniform is the uniform distribution over classes. Standard **Adam optimizer** with backpropagation through the classifier. After convergence, δ is fixed and deployed as a blind perturbation.

---

### 11.3 PANTS — Jin & Apostolaki (USENIX Security, 2025)

**Citation:** Minhao Jin and Maria Apostolaki. "PANTS: Practical Adversarial Network Traffic Samples against ML-powered Networking Classifiers." *USENIX Security 2025*. ISBN: 978-1-939133-52-6.

**(1) Multi-class scenario.** Evaluated on three distinct traffic classification tasks:
- **VPN detection** (binary: VPN vs. non-VPN) using **ISCXVPN2016** — 8,577 bi-directional flows.
- **APP (Application identification)** — **18 mobile application classes** (e.g., Dropbox, Google Drive, Facebook) using **UTMobileNetTraffic2021** — 7,134 bi-directional flows.
- **QoE (Quality of Experience inference)** — **11 video resolution classes** (e.g., 720p, 360p) from video conferencing apps (Google Meet, Microsoft Teams, Cisco Webex) using **VCAML** — 37,274 samples.
The APP and QoE tasks are multi-class. PANTS does **not** use the nPrint benchmark suite.

**(2) Threat model & target architectures.**
- **Threat model:** **White-box.** The adversary has access to the classifier's architecture, weights, and training data. PANTS evaluates two attacker positions: **end-host** (can delay packets, inject packets, append dummy payload) and **in-path** (can only delay packets of one direction).
- **Target classifiers:** For each application, four classifiers are independently trained:
  - **Multilayer Perceptron (MLP)**
  - **Random Forest (RF)**
  - **Transformer (TF)**
  - **Convolutional Neural Network (CNN)**
  - The framework is architecture-agnostic — it works with any differentiable classifier plus a non-differentiable traffic-engineering pipeline.

**(3) Traffic features exploited.**
- **Flow-level statistical features** extracted by a feature engineering module specific to each application (detailed in the paper's Appendix B). Key features include **packet sizes**, **inter-arrival times**, **flow duration**, and **byte counts** — not raw packet header fields or nPrint representations.
- The perturbation space is defined over realizable traffic operations: **packet delay** (up to 20% of flow duration), **dummy packet injection** (up to 20 packets), and **dummy payload appending** (to up to 20% of packets).

**(4) Adversarial data generation steps.**
1. **Formalize the adversarial search problem:** Given a target classifier f, an input flow x, and the traffic-engineering pipeline T (non-differentiable), find a perturbation δ such that: `f(T(x + δ)) ≠ f(T(x))` and `δ satisfies semantic constraints C`.
2. **Decompose the pipeline:** Separate the differentiable component (the ML model f) from the non-differentiable component (traffic engineering T, e.g., packet reordering, fragmentation).
3. **Gradient-based perturbation search:** For the differentiable part, compute gradients of the classification loss w.r.t. input features to identify promising perturbation directions using **PGD** (Projected Gradient Descent).
4. **SMT constraint solving:** Encode the semantic constraints (valid IP headers, consistent TCP state, preserved application functionality) as **Satisfiability Modulo Theories (SMT)** formulas using the **Z3 solver**. The gradient-suggested perturbation is projected onto the feasible set defined by these constraints.
5. **Iterative refinement:** Alternate between gradient steps and SMT projection until an adversarial example is found or a budget is exhausted.
6. **Adversarial training loop (defense):** The generated adversarial examples are added to the training set, the classifier is retrained, and the process repeats for multiple rounds to robustify the model.

**(5) Optimization method.**
- **Hybrid: PGD + SMT projection.** Gradient-based search (PGD with cross-entropy loss) provides the optimization direction; the Z3 SMT solver enforces hard network-semantic constraints that cannot be captured by simple box constraints or Lp projections. This is the key innovation — it bridges the gap between continuous gradient optimization and discrete protocol-validity constraints that make naive FGSM/PGD inapplicable to network traffic. The adversarial training defense uses a standard **min-max robust optimization** formulation.

---

### 11.4 Adversarial Pre-Padding — Jing et al. (arXiv, 2025)

**Citation:** Quanliang Jing, Xinxin Fan, Yanyan Liu, and Jingping Bi. "Adversarial Pre-Padding: Generating Evasive Network Traffic Against Transformer-Based Classifiers." arXiv:2510.25810, 2025.

**(1) Multi-class scenario.** Evaluated on three multi-class encrypted traffic datasets:
- **CSTNET-TLS 1.3:** ~120 application classes from the Chinese Science and Technology Network.
- **ISCX-VPN:** 12+ classes (VPN application and service categories).
- **ISCX-Tor:** 8 classes (Tor traffic types).
This is the first adversarial attack study specifically targeting pre-trained transformer-based encrypted traffic classifiers on multi-class tasks.

**(2) Threat model & target architectures.**
- **Threat model:** Both **white-box** (full gradient access) and **black-box** (query-only access, using a reward signal).
- **Target classifiers:**
  - **ET-BERT:** Pre-trained BERT-style transformer operating on raw byte sequences (datagrams). Uses masked byte prediction pre-training on unlabeled traffic, then fine-tunes for multi-class classification.
  - **YaTC:** Yet another Traffic Classifier — a transformer variant with byte-level tokenization.
  - **NetMamba:** A state-space model (Mamba architecture) for traffic classification.
  - Also evaluated on traditional models (1D-CNN, LSTM) for comparison.

**(3) Traffic features exploited.**
- **Raw byte sequences:** The first N bytes of each packet's payload (the input format for transformer-based classifiers). The pre-padding strategy specifically targets the **bytes before the encrypted payload** — protocol-header padding fields and optional TLS record-layer padding — which are mutable without breaking the encrypted content.
- **Key constraint:** Encrypted payload bytes cannot be modified (they would fail integrity checks). Only bytes in header/padding positions are perturbable.

**(4) Adversarial data generation steps.**
1. **Identify mutable positions:** Parse each packet to locate bytes that can be modified without breaking TLS/TCP integrity — specifically, pre-payload padding fields, optional header extensions, and TLS record padding.
2. **Formulate as MDP:** Model the byte-by-byte modification as a **Markov Decision Process**:
   - *State:* Current traffic representation (modified byte sequence so far).
   - *Action:* Choose a byte value (0–255) for the next mutable position.
   - *Reward:* In the white-box setting, reward = decrease in the classifier's confidence for the true class. In the black-box setting, reward = 1 if the classifier's predicted label changes (misclassification), 0 otherwise.
   - *Episode:* One complete pass through all mutable positions in a packet or flow.
3. **Train RL agent:** Train a deep RL agent (policy network) to learn the optimal byte-value selection policy across the mutable positions.
4. **Generate adversarial traffic:** Deploy the trained policy to produce adversarial byte sequences for new, unseen traffic flows. The modified bytes are inserted into the pre-padding positions of real packets.
5. **Validate:** Ensure TLS handshake and TCP connection remain functional after modification.

**(5) Optimization method.**
- **Deep Reinforcement Learning.** The RL agent uses a policy-gradient method to maximize the expected cumulative reward (misclassification rate). For the white-box setting, the reward integrates the classifier's loss gradient to guide exploration. For the black-box setting, the agent learns purely from query feedback. This approach elegantly handles the **non-differentiable** nature of the byte-selection problem (discrete action space of 256 values per position) that gradient-based methods like FGSM/PGD cannot directly address.

---

### 11.5 TANTRA — Sharon et al. (IEEE TDSC, 2022)

**Citation:** Yam Sharon, David Berend, Yang Liu, Asaf Shabtai, and Yuval Elovici. "TANTRA: Timing-Based Adversarial Network Traffic Reshaping Attack." *IEEE TDSC*, vol. 19, no. 6, pp. 3723–3738, 2022. DOI: 10.1109/TDSC.2022.3199100.

**(1) Multi-class scenario.** Evaluated on **multi-class intrusion detection**: 8 attack types (Botnet, DDoS, DoS GoldenEye, DoS Hulk, DoS Slowhttptest, DoS Slowloris, FTP-Patator, SSH-Patator) plus benign traffic. The classifier must correctly assign each flow to one of these categories. Datasets: **CIC-IDS-2017** and **CSE-CIC-IDS-2018**. *Note:* These are general NIDS datasets containing a mix of encrypted and unencrypted traffic, not exclusively encrypted traffic. TANTRA's timing manipulation technique is encryption-agnostic — it applies equally regardless of whether payload content is encrypted.

**(2) Threat model & target architectures.**
- **Threat model:** **Black-box.** The attacker has no access to the NIDS model internals. The LSTM is trained independently on benign traffic from the target network — no knowledge of the classifier is needed.
- **Target classifiers (NIDS being evaded):**
  - **Kitsune:** An ensemble of autoencoders.
  - **LUCID:** A CNN-based NIDS.
  - **Custom DNN:** A fully connected deep neural network.
  All three are multi-class classifiers distinguishing benign from multiple attack types.

**(3) Traffic features exploited.**
- **Inter-packet timing (IPT):** The sole feature manipulated. TANTRA does not modify packet content, sizes, directions, or counts — only the time intervals between consecutive packets.
- The NIDS classifiers use flow-level statistical features derived from timing (mean IAT, std IAT, flow duration, etc.), packet sizes, byte counts, and flag counts. By reshaping timing alone, the attack indirectly alters multiple derived statistical features.

**(4) Adversarial data generation steps.**
1. **Collect benign traffic** from the target network.
2. **Train an LSTM model** on benign inter-packet time differences: given the sequence of IPTs in a benign flow, the LSTM learns to predict the next IPT. This captures the temporal patterns characteristic of legitimate traffic.
3. **At attack time:** For each malicious packet to be sent, query the LSTM with the sequence of IPTs so far. The LSTM predicts the IPT that would make this look like benign traffic.
4. **Delay the packet** by the predicted amount: `actual_send_time = previous_packet_time + LSTM_predicted_IPT`. If the predicted IPT is shorter than the natural one, the packet is sent immediately (no artificial delay).
5. **Result:** The malicious flow's timing profile statistically matches benign traffic, evading the NIDS without modifying any packet content.

**(5) Optimization method.**
- **Supervised LSTM training** on benign traffic sequences. The loss function is **Mean Squared Error (MSE)** between predicted and actual inter-packet times in the benign training set: `L = (1/N) Σ(IPT_predicted - IPT_actual)²`. Standard **Adam** optimizer with backpropagation through time (BPTT). This is not a traditional adversarial optimization (no adversarial loss against the NIDS) — instead, it is a **generative approach** that learns the benign timing distribution and replays it onto malicious traffic.

---

### 11.6 Chehade et al. — Input Structure and Adversarial Robustness (arXiv, 2025)

**Citation:** Adel Chehade, Edoardo Ragusa, Paolo Gastaldo, and Rodolfo Zunino. "Adversarial Robustness of Traffic Classification under Resource Constraints: Input Structure Matters." arXiv:2512.02276, 2025.

**(1) Multi-class scenario.** Evaluated on:
- **ISCX-VPN-nonVPN:** Multi-class classification of VPN and non-VPN application types (browsing, email, chat, streaming, file transfer, VoIP — up to 12 classes).
- **USTC-TFC2016:** 20-class classification (10 malware families + 10 benign applications).
The study specifically compares how different input representations affect adversarial vulnerability in multi-class settings.

**(2) Threat model & target architectures.**
- **Threat model:** **White-box.** FGSM and PGD attacks with varying ε.
- **Target classifiers:** Two **1D-CNN architectures** designed via **Hardware-aware Neural Architecture Search (HW-NAS)** — compact models (≤65K parameters, ≤2M FLOPs) suitable for edge deployment:
  - *Flat-byte model:* Takes a flattened 1D vector of the first N raw bytes of a flow as input.
  - *Time-series model:* Takes a 2D matrix where each row is a packet represented by [packet_size, IAT, direction, ...] features over time.
  Both achieve >99% clean accuracy on USTC-TFC2016.

**(3) Traffic features exploited.**
- **Flat-byte input:** Raw byte values of the first N bytes of the flow. Perturbations are added directly to byte values within an Lp-ball.
- **Time-series input:** Per-packet statistical features — packet size, inter-arrival time, direction, payload length — arranged as a time series. Perturbations modify these statistical values.
- The key finding is that the **choice of input representation** dramatically changes the adversarial surface.

**(4) Adversarial data generation steps.**
1. Select the target classifier (flat-byte or time-series variant).
2. **FGSM attack:** Compute `x_adv = x + ε · sign(∇_x L(f(x), y))` — a single-step perturbation along the loss gradient.
3. **PGD attack:** Initialize `x_adv = x`, then iterate for T steps: `x_adv ← Π_{ε-ball}(x_adv + α · sign(∇_x L(f(x_adv), y)))`, where Π projects back onto the ε-ball and α is the step size.
4. Evaluate accuracy on adversarial examples across multiple ε values (0.01, 0.05, 0.1, etc.).
5. **Adversarial fine-tuning (defense):** Re-train the model on a mixture of clean and PGD-generated adversarial examples.

**(5) Optimization method.**
- Standard **FGSM** (single-step, L∞) and **PGD** (multi-step, L∞) with cross-entropy loss. No traffic-semantic constraints are enforced — this is a pure feature-space evaluation. The comparison between input structures reveals that **flat-byte models are inherently more robust** (>85% accuracy at ε=0.1) because perturbations distribute across a high-dimensional byte space, while **time-series models are fragile** (<35% at ε=0.1) because perturbations concentrate on a few highly informative statistical features.

---

### 11.7 BARS — Wang et al. (NDSS, 2023)

**Citation:** Kai Wang, Zhiliang Wang, Dongqi Han, Wenqi Chen, Jiahai Yang, Xingang Shi, and Xia Yin. "BARS: Local Robustness Certification for Deep Learning based Traffic Analysis Systems." *NDSS 2023*.

**(1) Multi-class scenario.** Evaluated on three practical DL-based traffic analysis systems performing multi-class classification:
- **Website fingerprinting** with 100 monitored websites (100-class).
- **Encrypted application identification.**
- **Anomaly detection with fine-grained attack categorization.**

**(2) Threat model & target architectures.**
- **Threat model:** This is a **defense/certification** paper — it provides robustness guarantees regardless of the attacker's capabilities. The certificates are valid against any attacker (including worst-case white-box).
- **Target classifiers (being certified):**
  - **ACID:** Autoencoder-based classifier for anomaly detection.
  - **CADE:** Contrastive autoencoder-based detector.
  - **Kitsune:** Ensemble of autoencoders for NIDS.
  The framework is architecture-agnostic.

**(3) Traffic features.**
- The framework operates on **any feature representation** used by the target classifier — packet-level features, flow statistics, byte sequences, etc. BARS handles the **heterogeneous feature space** challenge (mixed continuous, discrete, and categorical features) through its Distribution Transformer.

**(4) Certification steps (not adversarial generation, but robustness measurement).**
1. For a given input x and classifier f, define the perturbation set `B(x, r)` of all inputs within radius r.
2. **Add smoothing noise:** Sample N noisy copies of x: `x_i = x + η_i` where η_i is drawn from an optimized noise distribution.
3. **Classify all noisy copies** and compute the empirical class distribution.
4. **Certify:** If the most-frequent class has a sufficiently large margin over the runner-up, the prediction is certifiably robust within radius r.
5. **Optimize the noise distribution** using the Distribution Transformer: instead of fixed Gaussian noise, BARS learns noise shape and scale that converge on the classification boundary, maximizing the certified radius.

**(5) Optimization method.**
- **Boundary-Adaptive Randomized Smoothing.** The noise distribution parameters (shape, scale) are optimized via **gradient-based search** (gradient descent on the certification objective) using special distribution functions that enable differentiable optimization through the sampling process. This yields tighter robustness certificates than generic Gaussian smoothing — the noise is concentrated where it matters most (near decision boundaries) rather than uniformly distributed.

---

### 11.8 CertTA — Yan et al. (USENIX Security, 2025)

**Citation:** Jinzhu Yan, Zhuotao Liu, Yuyang Xie, Shiyu Liang, Lin Liu, and Ke Xu. "CertTA: Certified Robustness Made Practical for Learning-Based Traffic Analysis." *USENIX Security 2025*.

**(1) Multi-class scenario.** Evaluated on:
- **Website fingerprinting:** 95–100 monitored websites (multi-class).
- **Encrypted application traffic classification.**
Evaluated across **six traffic analysis model architectures** and **two datasets**.

**(2) Threat model & target architectures.**
- **Threat model:** **Certification** — provides provable guarantees valid against **any** attacker, including worst-case white-box.
- **Target classifiers (being certified):** Six diverse architectures including CNNs, LSTMs, and transformer-based models for traffic analysis. The framework is universally applicable.

**(3) Traffic features and perturbation modalities.**
CertTA is unique in handling **three simultaneous perturbation types** that map to realistic traffic manipulation:
- **Additive continuous perturbations:** Packet length padding (adding bytes to packets) and timing delays (adding latency to packets). These modify continuous-valued features.
- **Discrete perturbations:** Dummy packet insertion or deletion — changes the number of elements in the feature sequence.
- **Multi-modal combination:** Simultaneous application of both additive and discrete perturbations.

**(4) Certification steps.**
1. Define the multi-modal perturbation set: `B(x) = {x' : x' can be obtained from x by padding ≤ Δ_p bytes, adding ≤ Δ_t delay, and inserting/deleting ≤ Δ_d dummy packets}`.
2. **Multi-modal smoothing:** Inject noise into the input through a composite noise model: (a) Gaussian/Laplacian noise added to continuous features (packet sizes, timing), (b) random insertion/deletion noise applied to the packet sequence (for discrete perturbation certification).
3. **Classify many noisy copies** and compute empirical class probabilities.
4. **Derive the certified radius:** Using the smoothing theory, compute the maximum (Δ_p, Δ_t, Δ_d) within which the classifier's prediction is guaranteed to remain unchanged.
5. Report the **certified accuracy** at various perturbation radii.

**(5) Optimization method.**
- **Multi-modal randomized smoothing** with tailored noise distributions for each perturbation type. The key innovation is the mathematical derivation of certification bounds that simultaneously account for continuous (Lp-norm) and discrete (edit-distance) perturbation spaces — prior work could only handle one type at a time. The noise parameters are set analytically based on the desired certification radius.

---

### 11.9 Oscar — Zhao et al. (ACM CCS, 2024)

**Citation:** Xiyuan Zhao, Xinhao Deng, Qi Li, Yunpeng Liu, Zhuotao Liu, Kun Sun, and Ke Xu. "Towards Fine-Grained Webpage Fingerprinting at Scale." *ACM CCS 2024*. DOI: 10.1145/3658644.3690211.

**(1) Multi-class scenario.** The most extreme multi-class WF study: **1,000 monitored webpages + 9,000+ unmonitored webpages** from encrypted Tor traffic. Unlike typical WF that identifies websites (domains), Oscar identifies individual webpages (URLs) — requiring discrimination among subpages with highly similar traffic patterns. This is formulated as **multi-label metric learning** rather than standard softmax classification.

**(2) Threat model & target architectures.**
- **Threat model:** Passive adversary observing encrypted Tor traffic at the entry guard (standard WF threat model). The adversary has a **trained classifier** on collected traffic samples — effectively a **white-box** setting for the classifier's use but no active perturbation of traffic.
- **Architecture:** Oscar uses a **deep embedding network** (CNN backbone) that maps traffic traces into a metric embedding space. Classification is by nearest-neighbor lookup against class-representative embeddings. The training objective combines:
  - *Proxy-based metric loss:* Maintains class-representative proxies in embedding space.
  - *Sample-based metric loss:* Enforces same-class samples to be closer than different-class samples (triplet-style).

**(3) Traffic features exploited.**
- **Packet direction sequences:** +1/−1 encoding per packet.
- **Packet sizes.**
- **Timing information:** Inter-arrival times and burst durations.
- Features are extracted from the first N packets of each Tor circuit.

**(4) This is primarily an attack paper, not an adversarial-perturbation paper.** However, it is directly relevant because it establishes the multi-class attack benchmark that adversarial defenses must defeat. Oscar's 88.6% Recall@5 improvement quantifies how much harder it has become for defenses (padding, morphing, adversarial traces) to protect against fine-grained multi-class WF.

**(5) Optimization method.**
- **Combined metric learning loss:** `L = L_proxy + λ · L_sample`, where L_proxy is a proxy-NCA loss over class centroids and L_sample is a multi-similarity loss over training pairs. Optimized with **SGD + momentum**. The embedding dimension and loss weighting λ are tuned via cross-validation.

---

### 11.10 TMWF — Zhang et al. (ACM CCS, 2023)

**Citation:** Jiaxing Zhang, Xinhao Deng, Qi Li, and Ke Xu. "Transformer-based Model for Multi-tab Website Fingerprinting Attack." *ACM CCS 2023*. DOI: 10.1145/3576915.3623107.

**(1) Multi-class scenario.** **Multi-tab website fingerprinting** — the classifier must identify **multiple websites simultaneously** from a single mixed traffic trace produced by a user browsing multiple tabs concurrently over Tor. This is formulated as **set prediction** (predicting an unordered set of up to K website labels from one trace), making it a significantly harder multi-class problem than single-tab WF.

**(2) Threat model & target architectures.**
- **Threat model:** Passive adversary at the Tor entry guard (standard WF setting). The number of open tabs is **unknown** to the attacker (a realistic assumption that prior work did not make).
- **Architecture:** **Encoder-decoder Transformer** inspired by DETR (object detection):
  - *Encoder:* Processes the mixed traffic trace (packet direction + timing features) through self-attention layers to build a contextual representation.
  - *Decoder:* Takes K learnable **query embeddings** (one per potential website) and attends to the encoder output. Each query independently predicts a website label (or "no website" if fewer than K tabs are open).
  - *Prediction head:* Each decoder output is passed through a classification MLP.
  - *Training loss:* Hungarian matching between predicted set and ground-truth set, then cross-entropy on matched pairs.

**(3) Traffic features exploited.**
- **Packet direction** (+1/−1 per packet).
- **Packet timing** (absolute timestamps or inter-arrival times).
- **Packet sizes** (though less informative for Tor due to cell-size rounding).
Features are tokenized into a sequence fed to the Transformer encoder.

**(4) This is an attack paper.** The adversarial robustness angle: TMWF establishes that multi-tab browsing **does not inherently defend** against WF (a common assumption), raising the bar for adversarial defense design.

**(5) Optimization method.**
- **Bipartite matching loss** (Hungarian algorithm) + **cross-entropy classification loss**, optimized with **AdamW** optimizer and learning rate scheduling. The set-prediction formulation avoids the need for combinatorial enumeration of tab assignments.

---

### 11.11 DeepRed — Hajizadeh et al. (USENIX WOOT, 2025)

**Citation:** Mehrdad Hajizadeh et al. "DeepRed: A Deep Learning-Powered Command and Control Framework for Multi-Stage Red Teaming Against ML-based Network Intrusion Detection Systems." *USENIX WOOT 2025*, pp. 103–127.

**(1) Multi-class scenario.** Evaluated against multi-class NIDS classifiers that distinguish benign traffic from multiple attack categories in **CIC-IDS-style datasets** (DDoS, brute-force, infiltration, botnet, web attacks, etc.). The C2 traffic must evade detection while maintaining its malicious functionality.

**(2) Threat model & target architectures.**
- **Threat model:** **Black-box** for the GAN-based evasion (no direct gradient access to the NIDS); the discriminator serves as a surrogate.
- **Target classifiers (NIDS being evaded):**
  - **FlowTransformer:** Transformer-based NIDS.
  - **SSCL-IDS:** Self-supervised contrastive learning IDS.
  - Additional ML-NIDS benchmarks.
- **GAN architecture:**
  - *Generator:* Takes real malicious traffic features as input and outputs perturbed features designed to be classified as benign.
  - *Discriminator:* Distinguishes between real benign traffic and generator-produced perturbed malicious traffic.

**(3) Traffic features exploited.**
- **Packet-level features:** IP header fields, TCP flags, payload sizes, TTL.
- **Flow-level features:** Duration, packet count, byte count, inter-arrival time statistics, flag distributions.
- Two novel perturbation strategies constrain which features are modified:
  - **SPSF (Single-Packet Single-Feature):** Perturbs only one feature in one packet — the most constrained.
  - **SFP (Single-Feature Perturbation):** Perturbs one feature across multiple packets in a flow.

**(4) Adversarial data generation steps.**
1. **Collect real traffic:** Capture benign traffic from the target network and malicious C2 traffic from the red-team tool.
2. **Train the GAN:** The generator learns to transform malicious flow features so they resemble benign flows. The discriminator provides gradient feedback.
3. **Enforce TCP/IP constraints:** After each generator update, verify that perturbed features map to valid TCP/IP packets (correct checksums, valid flag combinations, feasible size/timing values). Invalid samples are rejected or clipped.
4. **Select attack strategy:** Choose SPSF (minimal perturbation) or SFP (single-feature across packets) based on the desired stealth level.
5. **Deploy:** The generator outputs perturbed C2 traffic in real time as part of a multi-stage red-team operation.

**(5) Optimization method.**
- **GAN adversarial training** with standard minimax objective: `min_G max_D [E_benign[log D(x)] + E_malicious[log(1 - D(G(x)))]]`. The generator and discriminator are alternately optimized with **Adam**. The TCP/IP constraint enforcement acts as a post-hoc projection step (not integrated into the gradient). The SPSF/SFP strategies further constrain the perturbation budget.

---

### 11.12 Comparative Summary Table

| Paper | Venue | Multi-Class Task | Classes | Threat Model | Classifier Architecture | Features Exploited | Optimization Method |
|---|---|---|---|---|---|---|---|
| Sadeghzadeh et al. [B1] | IEEE TIFS 2021 | App/service/malware ID | 12–20 | White-box | 1D-CNN, 2D-CNN (FlowPic), LSTM | Raw bytes, payload, burst statistics | UAP via iterative DeepFool + Lp projection |
| Nasr et al. [B2] | USENIX Security 2021 | Website fingerprinting | 95 | White-box (offline) → blind deployment | DF (CNN), Var-CNN, DeepCorr | Packet direction, size, IAT, bursts | Gradient-based (Adam) on adversarial loss + feature remapping |
| PANTS [B4] | USENIX Security 2025 | App ID, VPN, QoE | 2–18 | White-box | MLP, RF, Transformer, CNN | Flow statistics (pkt size, IAT, duration) | PGD + Z3 SMT solver (hybrid) |
| Adv. Pre-Padding [M4] | arXiv 2025 | App classification | 8–120 | White-box & black-box | ET-BERT, YaTC, NetMamba, 1D-CNN | Raw byte sequences (pre-payload padding) | Deep RL (policy gradient, MDP) |
| TANTRA [B5] | IEEE TDSC 2022 | Attack-type detection | 8+1 | Black-box | Kitsune, LUCID, custom DNN | Inter-packet timing only | Supervised LSTM (MSE on benign IPT) |
| Chehade et al. [B8] | arXiv 2025 | App/malware ID | 12–20 | White-box | 1D-CNN (HW-NAS, flat & time-series) | Raw bytes or packet-stat time series | FGSM / PGD (L∞) + adversarial fine-tuning |
| BARS [M1] | NDSS 2023 | WF + app ID | 100+ | Certification (any attacker) | ACID, CADE, Kitsune | Any (heterogeneous) | Boundary-adaptive randomized smoothing |
| CertTA [C1] | USENIX Security 2025 | WF + app classification | 95–100 | Certification (any attacker) | 6 architectures (CNN, LSTM, Transformer) | Packet size, timing, packet count | Multi-modal randomized smoothing |
| Oscar [M2] | ACM CCS 2024 | Fine-grained webpage FP | 1,000+ | Trained classifier (passive) | CNN + metric learning embedding | Direction, size, timing | Proxy-NCA + multi-similarity loss (SGD) |
| TMWF [M3] | ACM CCS 2023 | Multi-tab WF | N×100 sites | Trained classifier (passive) | Encoder-decoder Transformer | Direction, timing, size | Hungarian matching + cross-entropy (AdamW) |
| DeepRed [B6] | USENIX WOOT 2025 | Multi-class NIDS evasion | 5–15 | Black-box (GAN surrogate) | FlowTransformer, SSCL-IDS | Packet headers, flow statistics | GAN minimax (Adam) + TCP/IP projection |

---

## Key References

### Encrypted Traffic Classification (Sections 1–8)

- Lotfollahi et al., "Deep Packet: A Novel Approach For Encrypted Traffic Classification Using Deep Learning," *Soft Computing*, 2020.
- Rezaei & Liu, "Deep Learning for Encrypted Traffic Classification: An Overview," *IEEE Communications Magazine*, 2019.
- Shapira & Shavitt, "FlowPic: Encrypted Internet Traffic Classification is as Easy as Image Recognition," *IEEE INFOCOM Workshops*, 2019.
- Lin et al., "ET-BERT: A Contextualized Datagram Representation with Pre-training Transformers for Encrypted Traffic Classification," *WWW*, 2022.
- Wang et al., "Machine Learning-Powered Encrypted Network Traffic Analysis: A Comprehensive Survey," *IEEE Communications Surveys & Tutorials*, 2022.
- He et al., "TransECA-Net: A Transformer-Based Model for Encrypted Traffic Classification," *Applied Sciences*, 2025.
- Yu et al., "Convolutions are Competitive with Transformers for Encrypted Traffic Classification with Pre-training," *arXiv*, 2025.
- Akbari et al., "A Look Behind the Curtain: Traffic Classification in an Increasingly Encrypted Web," *ACM SIGMETRICS*, 2021.
- Ben-David et al., "Towards Identification of Network Applications in Encrypted Traffic," *Annals of Telecommunications*, 2025.

### Robustness Assessment / Adversarial Attacks on Encrypted Network Traffic Security (Section 9)

The papers below are specifically focused on adversarial attacks, robustness evaluation, and defenses concerning the security of **encrypted network traffic**. Each entry provides complete bibliographic information: full author list, title, venue, year, volume/pages (where applicable), DOI/identifier, and a summary of contributions. Papers are organized by topic and sorted within each category by venue impact and recency.

**Venue key:**
- **Top-4 Security:** IEEE S&P, ACM CCS, USENIX Security, NDSS
- **Top Security Journals:** IEEE TIFS (IF ~6.8), IEEE TDSC (IF ~7.3)
- **Top Survey Journals:** IEEE COMST (IF ~35)
- **Top Privacy:** PETS/PoPETs
- **Top ML:** NeurIPS, ICLR
- **Top Networking:** ACM SIGCOMM, IEEE/ACM ToN

---

#### A. Foundational Adversarial ML Methods (Widely Applied to Encrypted Traffic Research)

These canonical papers introduced the attack primitives and training paradigms that all subsequent encrypted-traffic adversarial work builds upon.

**[A1]** Ian J. Goodfellow, Jonathon Shlens, and Christian Szegedy.
"Explaining and Harnessing Adversarial Examples."
In *Proceedings of the 3rd International Conference on Learning Representations (ICLR 2015)*, San Diego, CA, USA, May 7–9, 2015. arXiv:1412.6572.
- **Venue:** ICLR (top-tier ML conference).
- **Relevance:** Introduced the Fast Gradient Sign Method (FGSM) and the concept of adversarial training. Foundation for all gradient-based robustness evaluations applied to encrypted traffic classifiers (referenced by [B1–B4]).

**[A2]** Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu.
"Towards Deep Learning Models Resistant to Adversarial Attacks."
In *Proceedings of the 6th International Conference on Learning Representations (ICLR 2018)*, Vancouver, BC, Canada, April 30–May 3, 2018. arXiv:1706.06083.
- **Venue:** ICLR (top-tier ML conference).
- **Relevance:** Introduced Projected Gradient Descent (PGD) adversarial training via min-max robust optimization. The standard white-box robustness benchmark in traffic analysis studies (used by [B1], [B4], [C1], [E1]).

**[A3]** Nicholas Carlini and David Wagner.
"Towards Evaluating the Robustness of Neural Networks."
In *Proceedings of the 2017 IEEE Symposium on Security and Privacy (IEEE S&P 2017)*, San Jose, CA, USA, May 22–26, 2017, pp. 39–57. DOI: 10.1109/SP.2017.49.
- **Venue:** IEEE S&P (top-4 security conference).
- **Relevance:** Introduced the C&W optimization-based attack for establishing tight lower bounds on classifier robustness. Applied in encrypted traffic work to measure worst-case resilience of DL classifiers.

---

#### B. Adversarial Attacks Targeting Encrypted Traffic Classifiers

**[B1]** Amir Mahdi Sadeghzadeh, Saeed Shiravi, and Rasool Jalili.
"Adversarial Network Traffic: Towards Evaluating the Robustness of Deep-Learning-Based Network Traffic Classification."
*IEEE Transactions on Information Forensics and Security (TIFS)*, vol. 16, pp. 3940–3955, 2021. DOI: 10.1109/TIFS.2021.3053093. IEEE Xplore: 9328496.
- **Venue:** IEEE TIFS (top-tier security journal, IF ~6.8).
- **Contribution:** Proposed three universal adversarial perturbation (UAP) attacks partitioned by classifier input type: **AdvPad** (injects UAP into packet content targeting packet classifiers), **AdvPay** (injects UAP into dummy packet payloads targeting flow-content classifiers), and **AdvBurst** (injects crafted dummy packets with adversarial statistical features targeting flow time-series classifiers). Demonstrated that minimal UAP injection significantly degrades all three categories of DL-based encrypted traffic classifiers.
- **Code:** github.com/amsadeghzadeh/AdversarialNetworkTraffic

**[B2]** Milad Nasr, Alireza Bahramali, and Amir Houmansadr.
"Defeating DNN-Based Traffic Analysis Systems in Real-Time With Blind Adversarial Perturbations."
In *Proceedings of the 30th USENIX Security Symposium (USENIX Security 2021)*, August 11–13, 2021, pp. 2705–2722. ISBN: 978-1-939133-24-3.
- **Venue:** USENIX Security (top-4 security conference).
- **Contribution:** First demonstration of real-time adversarial attacks on live encrypted traffic analysis systems (website fingerprinting and flow correlation). Introduced **blind adversarial perturbations** that are pre-computed and independent of the target connection, enabling application without buffering. Designed a **remapping technique** to preserve correctness of dependent traffic features (e.g., total bytes, flow duration) after perturbation.

**[B3]** Chenglong Hou, Shuping Dang, Zhen Wang, and Gaoning Pan.
"AdvTraffic: Obfuscating Encrypted Traffic with Adversarial Examples."
In *Proceedings of the 42nd IEEE International Conference on Distributed Computing Systems (ICDCS 2022)*, Bologna, Italy, July 10–13, 2022, pp. 1269–1270. DOI: 10.1109/ICDCS54860.2022.00134. IEEE Xplore: 9812875.
- **Venue:** IEEE ICDCS (top distributed systems conference).
- **Contribution:** Gradient-based adversarial perturbation framework specifically designed for encrypted traffic that respects TLS/HTTPS protocol constraints while evading CNN and LSTM-based classifiers.

**[B4]** Minhao Jin and Maria Apostolaki.
"PANTS: Practical Adversarial Network Traffic Samples against ML-powered Networking Classifiers."
In *Proceedings of the 34th USENIX Security Symposium (USENIX Security 2025)*, Seattle, WA, USA, August 13–15, 2025. ISBN: 978-1-939133-52-6. arXiv:2409.04691.
- **Venue:** USENIX Security (top-4 security conference).
- **Contribution:** Combined adversarial ML with **Satisfiability Modulo Theories (SMT) solvers** to generate adversarial traffic samples that satisfy all network-semantic constraints (valid headers, correct protocol state, functional application-layer behavior). Achieved **70% higher median success rate** than prior baselines (Amoeba, BAP). Adversarial training loop improved classifier robustness by **52.7%** without sacrificing clean accuracy. Robustified models showed improved resilience even against unseen attack methods.
- **Code:** github.com/jinminhao/PANTS

**[B5]** Yam Sharon, David Berend, Yang Liu, Asaf Shabtai, and Yuval Elovici.
"TANTRA: Timing-Based Adversarial Network Traffic Reshaping Attack."
*IEEE Transactions on Dependable and Secure Computing (TDSC)*, vol. 19, no. 6, pp. 3723–3738, November–December 2022. DOI: 10.1109/TDSC.2022.3199100. IEEE Xplore: 9865980.
- **Venue:** IEEE TDSC (top-tier security journal, IF ~7.3).
- **Contribution:** LSTM-based attack that learns inter-packet timing distributions from benign traffic and reshapes malicious encrypted traffic timing to match. Achieved **~99.99% evasion rate** against three state-of-the-art NIDS across eight attack types, without modifying packet content. Also proposed a novel mitigation technique.

**[B6]** Mehrdad Hajizadeh, Pegah Golchin, Ehsan Nowroozi, Maria Rigaki, Veronica Valeros, Sebastian García, Mauro Conti, and Thomas Bauschert.
"DeepRed: A Deep Learning-Powered Command and Control Framework for Multi-Stage Red Teaming Against ML-based Network Intrusion Detection Systems."
In *Proceedings of the 19th USENIX Workshop on Offensive Technologies (WOOT 2025)*, Seattle, WA, USA, August 11–12, 2025, pp. 103–127. ISBN: 978-1-939133-50-2.
- **Venue:** USENIX WOOT (co-located with USENIX Security).
- **Contribution:** GAN-based C2 framework generating adversarial encrypted traffic that evades ML-based NIDS. Introduced **Single-Packet Single-Feature (SPSF)** and **Single-Feature Perturbation (SFP)** attack strategies that maintain TCP/IP compliance and preserve attack functionality at the packet level. Detection rate reductions of up to 20%.

**[B7]** Bolor-Erdene Zolbayar, Ryan Sheatsley, Patrick McDaniel, Michael J. Weisman, Sencun Zhu, Shitong Zhu, and Srikanth Krishnamurthy.
"Generating Practical Adversarial Network Traffic Flows Using NIDSGAN."
*arXiv preprint*, arXiv:2203.06694, March 2022.
- **Contribution:** GAN-based adversarial traffic generation respecting network domain constraints. Achieved **99% success** in white-box, **85%** in black-box, and **70%** in restricted-black-box settings against DNN-based NIDS.

**[B8]** Adel Chehade, Edoardo Ragusa, Paolo Gastaldo, and Rodolfo Zunino.
"Adversarial Robustness of Traffic Classification under Resource Constraints: Input Structure Matters."
*arXiv preprint*, arXiv:2512.02276, December 2025.
- **Contribution:** Showed that input representation (flat byte sequence vs. 2D packet-wise time series) dramatically affects adversarial vulnerability in hardware-efficient compact models. Flat-input models retained **>85% accuracy** under perturbation (ε=0.1) while time-series variants dropped **below 35%**. Adversarial fine-tuning recovered robustness without sacrificing efficiency.

---

#### C. Certified and Provable Robustness for Encrypted Traffic Analysis

**[C1]** Jinzhu Yan, Zhuotao Liu, Yuyang Xie, Shiyu Liang, Lin Liu, and Ke Xu.
"CertTA: Certified Robustness Made Practical for Learning-Based Traffic Analysis."
In *Proceedings of the 34th USENIX Security Symposium (USENIX Security 2025)*, Seattle, WA, USA, August 13–15, 2025.
- **Venue:** USENIX Security (top-4 security conference).
- **Contribution:** **First certified robustness framework specifically for traffic analysis.** Introduced a **multi-modal randomized smoothing** mechanism generating provable robustness regions against three simultaneous perturbation types: additive perturbations (packet length padding, timing delays), discrete alterations (dummy packet insertion/deletion), and combined multi-modal attacks. Evaluated on six traffic analysis model architectures across two datasets, providing **stronger guarantees** than generic smoothing approaches. Universally applicable across diverse model architectures and flow representations.

**[C2]** Zhuoqun Huang, Neil G. Marchant, Keane Lucas, Lujo Bauer, Olga Ohrimenko, and Benjamin I. P. Rubinstein.
"RS-Del: Edit Distance Robustness Certificates for Sequence Classifiers via Randomized Deletion."
In *Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*, New Orleans, LA, USA, December 10–16, 2023. arXiv:2302.01757.
- **Venue:** NeurIPS (top-tier ML conference).
- **Contribution:** Randomized smoothing adapted for variable-length discrete sequences via randomized deletion. Applied to **MalConv malware detector**: achieved **91% certified accuracy** at an edit distance radius of 128 bytes. Novel proof technique organized around longest common subsequences (not Neyman-Pearson). Directly applicable to byte-sequence-based encrypted traffic classifiers.
- **Code:** github.com/Dovermore/randomized-deletion

---

#### D. Robust Detection of Encrypted Malicious Traffic

**[D1]** Chuanpu Fu, Qi Li, and Ke Xu.
"Detecting Unknown Encrypted Malicious Traffic in Real Time via Flow Interaction Graph Analysis."
In *Proceedings of the 30th Network and Distributed System Security Symposium (NDSS 2023)*, San Diego, CA, USA, February 27–March 3, 2023. Also published in *IEEE/ACM Transactions on Networking*, 2024. arXiv:2301.13686.
- **Venue:** NDSS (top-4 security conference) + IEEE/ACM ToN (top networking journal).
- **Contribution:** Proposed **HyperVision**, an unsupervised system that detects unknown encrypted malicious traffic by analyzing flow interaction graphs. Captures graph structural features (connectivity, sparsity) without requiring labeled datasets. Achieved **≥0.92 AUC**, **≥0.86 F1**, **80.6 Gb/s throughput**, and **0.83s average latency** on 92 datasets including 48 encrypted malicious traffic attacks.

**[D2]** ReTrial Authors (Beijing Institute of Technology, Beijing University of Posts and Telecommunications, Purdue University), including Bharat K. Bhargava.
"ReTrial: Robust Encrypted Malicious Traffic Detection via Discriminative Relation Incorporation and Misleading Relation Correction."
*IEEE Transactions on Information Forensics and Security (TIFS)*, vol. 20, pp. 677–692, 2025. DOI: 10.1109/TIFS.2024.3515821. IEEE Xplore: 10792981.
- **Venue:** IEEE TIFS (top-tier security journal, IF ~6.8).
- **Contribution:** Constructs a relational multigraph of encrypted flows and uses a **Graph Attention Network (GAT)** to selectively incorporate contextual information while correcting misleading relations via multi-order neighborhood similarity graphs. Demonstrated robustness under adversarial conditions: **maximum 5.88% F1 reduction** under random packet dropping and greedy perturbation edge injection, compared to significantly greater degradation in competing methods.

**[D3]** TrafCL Authors (Beijing Institute of Technology, Purdue University).
"TrafCL: Robust Encrypted Malicious Traffic Detection via Contrastive Learning."
In *Proceedings of the 33rd ACM International Conference on Information and Knowledge Management (CIKM 2024)*, Boise, ID, USA, October 21–25, 2024, pp. 2910–2919. DOI: 10.1145/3627673.3679839.
- **Venue:** ACM CIKM (top-tier data management conference).
- **Contribution:** Contrastive learning framework with Session Augmentation, Triple-aspect Session Feature Extraction, and Co-attention Session Encoder for detecting encrypted malware C2 traffic. Pre-trained on unlabeled data and fine-tuned on labeled data. Achieved **11.35% and 6.71% F1 improvements** over baselines on two encrypted traffic datasets.

---

#### E. Adversarial Attacks and Defenses for Encrypted Website Fingerprinting

**[E1]** Payap Sirinam, Mohsen Imani, Marc Juárez, and Matthew Wright.
"Deep Fingerprinting: Undermining Website Fingerprinting Defenses with Deep Learning."
In *Proceedings of the 25th ACM Conference on Computer and Communications Security (CCS 2018)*, Toronto, ON, Canada, October 15–19, 2018, pp. 1928–1943. DOI: 10.1145/3243734.3243768.
- **Venue:** ACM CCS (top-4 security conference).
- **Contribution:** CNN-based website fingerprinting attack on Tor achieving **>98% accuracy** on undefended traffic and **>90% against WTF-PAD**. Established the attack benchmark that all subsequent adversarial defenses are measured against.

**[E2]** Vera Rimmer, Davy Preuveneers, Marc Juárez, Tom Van Goethem, and Wouter Joosen.
"Automated Website Fingerprinting through Deep Learning."
In *Proceedings of the 25th Network and Distributed System Security Symposium (NDSS 2018)*, San Diego, CA, USA, February 18–21, 2018. DOI: 10.14722/ndss.2018.23105.
- **Venue:** NDSS (top-4 security conference).
- **Contribution:** Demonstrated automated feature learning via deep learning for WF attacks on encrypted Tor traffic, removing the need for hand-crafted features. Evaluated robustness across multiple network conditions and concept drift.

**[E3]** Mohammad Saidur Rahman, Mohsen Imani, Nate Mathews, and Matthew Wright.
"Mockingbird: Defending Against Deep-Learning-Based Website Fingerprinting Attacks With Adversarial Traces."
*IEEE Transactions on Information Forensics and Security (TIFS)*, vol. 16, pp. 1594–1609, 2021. DOI: 10.1109/TIFS.2020.3039691. IEEE Xplore: 9265277.
- **Venue:** IEEE TIFS (top-tier security journal, IF ~6.8).
- **Contribution:** Defense using adversarial trace generation that moves randomly through the space of viable traces rather than following predictable gradient directions, making it **resistant to adversarial retraining** by the attacker. Reduces Deep Fingerprinting accuracy from **98% to 42–58%** with 58% bandwidth overhead.
- **Code:** github.com/msrocean/mockingbird

**[E4]** Mohammad Saidur Rahman, Payap Sirinam, Nate Mathews, Kantha Girish Gangadhara, and Matthew Wright.
"Tik-Tok: The Utility of Packet Timing in Website Fingerprinting Attacks."
*Proceedings on Privacy Enhancing Technologies (PoPETs)*, vol. 2020, issue 3, pp. 5–24, 2020. DOI: 10.2478/popets-2020-0043.
- **Venue:** PoPETs/PETS (top privacy conference).
- **Contribution:** Demonstrated that packet timing features (burst-level timing, directional timing) substantially improve WF attacks on encrypted Tor traffic. Achieved **98.4% on undefended**, **93.5% on WTF-PAD-defended**, and **64.7% on onion-site** traffic. The Tik-Tok attack became a standard benchmark for evaluating subsequent defenses.
- **Code:** github.com/msrocean/Tik_Tok

**[E5]** Marc Juárez, Mohsen Imani, Stephen Schiavoni, and Claudia Díaz.
"WTF-PAD: Toward an Efficient Website Fingerprinting Defense."
In *Proceedings of the 21st European Symposium on Research in Computer Security (ESORICS 2016)*, Heraklion, Crete, Greece, September 26–30, 2016, Lecture Notes in Computer Science, vol. 9878, pp. 27–46. DOI: 10.1007/978-3-319-45744-4_2.
- **Venue:** ESORICS (top European security conference).
- **Contribution:** Adaptive padding defense for Tor. Reduced state-of-the-art WF attack accuracy from **91% to 20%** in closed-world and **1% precision** in open-world, with **zero latency overhead** and **<80% bandwidth overhead**.

**[E6]** Tao Wang and Ian Goldberg.
"Walkie-Talkie: An Efficient Defense Against Passive Website Fingerprinting Attacks."
In *Proceedings of the 26th USENIX Security Symposium (USENIX Security 2017)*, Vancouver, BC, Canada, August 16–18, 2017, pp. 1375–1390. ISBN: 978-1-931971-40-9.
- **Venue:** USENIX Security (top-4 security conference).
- **Contribution:** Converts browser communication to half-duplex mode to produce moldable burst sequences. Defeats all known WF attacks with **31% bandwidth overhead** and **34% time overhead** — substantially less than other effective defenses.

**[E7]** Jiajun Gong and Tao Wang.
"Zero-delay Lightweight Defenses against Website Fingerprinting."
In *Proceedings of the 29th USENIX Security Symposium (USENIX Security 2020)*, August 12–14, 2020, pp. 717–734. ISBN: 978-1-939133-17-5.
- **Venue:** USENIX Security (top-4 security conference).
- **Contribution:** Introduced **FRONT** (randomized dummy packets at trace front) and **GLUE** (inter-trace dummy packets preventing trace boundary detection). Zero latency overhead. FRONT outperforms WTF-PAD at **33% data overhead**; GLUE achieves heavyweight-defense performance at **22–44% overhead**.

**[E8]** Wladimir De la Cadena, Asya Mitseva, Jens Hiller, Jan Pennekamp, Sebastian Reuter, Julian Filter, Thomas Engel, Klaus Wehrle, and Andriy Panchenko.
"TrafficSliver: Fighting Website Fingerprinting Attacks with Traffic Splitting."
In *Proceedings of the 27th ACM Conference on Computer and Communications Security (CCS 2020)*, Virtual, November 9–13, 2020, pp. 1971–1985. DOI: 10.1145/3372297.3423351.
- **Venue:** ACM CCS (top-4 security conference).
- **Contribution:** Splits encrypted Tor traffic across multiple circuits to reduce the information available to any single entry-node observer. Network-layer defense reduces WF classifier accuracy from **>98% to below 16%** without artificial delays or dummy traffic. Application-layer defense reduces detection by **~50 percentage points**.

**[E9]** James K. Holland, Jason Carpenter, Se Eun Oh, and Nicholas Hopper.
"DeTorrent: An Adversarial Padding-only Traffic Analysis Defense."
*Proceedings on Privacy Enhancing Technologies (PoPETs)*, vol. 2024, issue 1, pp. 98–115. DOI: 10.56553/popets-2024-0007.
- **Venue:** PoPETs/PETS (top privacy conference).
- **Contribution:** Uses competing neural networks to generate padding-only defenses for encrypted Tor traffic. Reduces WF attacker accuracy by **61.5%** (10.5% better than next-best padding-only defense). Also effective against flow correlation attacks (TPR reduced to ~0.12 at 10⁻⁵ FPR).
- **Code:** github.com/jkhollandjr/PETS_DeTorrent

**[E10]** James K. Holland and Nicholas Hopper.
"RegulaTor: A Straightforward Website Fingerprinting Defense."
*Proceedings on Privacy Enhancing Technologies (PoPETs)*, vol. 2022, pp. 344–362. DOI: 10.2478/popets-2022-0049.
- **Venue:** PoPETs/PETS (top privacy conference).
- **Contribution:** Exploits common patterns in web browsing to create a lightweight defense. Reduces Tik-Tok attack accuracy from **66% to 25.4%** (closed-world) and F-score to **0.135** (open-world) with only **6.6% latency overhead** and **39.3% less bandwidth overhead** than comparable defenses.

**[E11]** Giovanni Cherubin, Rob Jansen, and Carmela Troncoso.
"Online Website Fingerprinting: Evaluating Website Fingerprinting Attacks on Tor in the Real World."
In *Proceedings of the 31st USENIX Security Symposium (USENIX Security 2022)*, Boston, MA, USA, August 10–12, 2022. ISBN: 978-1-939133-31-1.
- **Venue:** USENIX Security (top-4 security conference). **Distinguished Paper Award; Internet Defense Prize.**
- **Contribution:** First rigorous real-world evaluation of WF attacks on Tor. Found that while attacks achieve **>95% accuracy on 5 websites**, accuracy drops **below 80% when monitoring 25 websites**, revealing a significant gap between laboratory and realistic attack effectiveness. Critical for understanding adversarial robustness in deployment.

---

#### F. Encrypted Traffic Fingerprinting and Obfuscation Evasion

**[F1]** Diwen Xue, Michalis Kallitsis, Amir Houmansadr, and Roya Ensafi.
"Fingerprinting Obfuscated Proxy Traffic with Encapsulated TLS Handshakes."
In *Proceedings of the 33rd USENIX Security Symposium (USENIX Security 2024)*, Philadelphia, PA, USA, August 14–16, 2024. ISBN: 978-1-939133-44-1.
- **Venue:** USENIX Security (top-4 security conference).
- **Contribution:** Identified a protocol-agnostic fingerprint for detecting obfuscated proxy (encrypted) traffic through **nested TLS handshakes** inherent in tunneling protocols. Deployed in a mid-size ISP serving >1 million users with minimal false positives. Showed that random padding and multiple encapsulation layers are insufficient countermeasures.

**[F2]** Diwen Xue, Robert Stanley, Piyush Kumar, and Roya Ensafi.
"The Discriminative Power of Cross-layer RTTs in Fingerprinting Proxy Traffic."
In *Proceedings of the 32nd Network and Distributed System Security Symposium (NDSS 2025)*, San Diego, CA, USA, February 23–28, 2025.
- **Venue:** NDSS (top-4 security conference).
- **Contribution:** Exploited **RTT discrepancies** between transport and application layers caused by proxy routing to fingerprint encrypted circumvention traffic. Protocol-agnostic — targets multiple proxy protocols simultaneously. Evaluated on both testbed and real ISP traffic.

**[F3]** Ryan Wails, George Arnold Sullivan, Micah Sherr, and Rob Jansen.
"On Precisely Detecting Censorship Circumvention in Real-World Networks."
In *Proceedings of the 31st Network and Distributed System Security Symposium (NDSS 2024)*, San Diego, CA, USA, February 26–March 1, 2024.
- **Venue:** NDSS (top-4 security conference).
- **Contribution:** Combined deep-learning flow-based classifier with host-based multi-flow detection for encrypted circumvention traffic. Achieved **perfect recall and zero false positives** on >60 million real-world flows to >600K destinations. Showed existing methods have **>94% false positive rates** at realistic base rates. Provided design recommendations for increasing circumvention resistance.

**[F4]** Charles V. Wright, Scott E. Coull, and Fabian Monrose.
"Traffic Morphing: An Efficient Defense Against Statistical Traffic Analysis."
In *Proceedings of the 16th Network and Distributed System Security Symposium (NDSS 2009)*, San Diego, CA, USA, February 8–11, 2009.
- **Venue:** NDSS (top-4 security conference).
- **Contribution:** Foundational work on traffic-shaping evasion. Uses **convex optimization** to compute the minimal set of packet padding and splitting operations that transform the statistical distribution of one encrypted traffic class to match a target class, with significantly less overhead than naive constant-rate padding.

---

#### G. Robustness of Encrypted Traffic Evaluation Methodology

**[G1]** Zhao, Dettori, Boffa, Vassio, and Mellia.
"The Sweet Danger of Sugar: Debunking Representation Learning for Encrypted Traffic Classification."
In *Proceedings of ACM SIGCOMM 2025*.
- **Venue:** ACM SIGCOMM (top networking conference).
- **Contribution:** Critically examined representation learning models (BERT-inspired) for encrypted traffic classification. Revealed that reported high performance (up to 98%) is driven by **data preparation artifacts and spurious correlations** (shortcuts) that do not generalize to realistic deployments. Proposed corrected evaluation methodology.

**[G2]** Nimesha Wickramasinghe, Arash Shaghaghi, Gene Tsudik, and Sanjay Jha.
"SoK: Decoding the Enigma of Encrypted Network Traffic Classifiers."
In *Proceedings of the 2025 IEEE Symposium on Security and Privacy (IEEE S&P 2025)*. DOI: 10.1109/SP61157.2025.00165.
- **Venue:** IEEE S&P (top-4 security conference).
- **Contribution:** Systematization of Knowledge analyzing ML-based encrypted network traffic classifiers. Conducted **348 feature occlusion experiments** revealing that many classifiers exploit unintended features and dataset artifacts. Introduced **CipherSpectrum**, a TLS 1.3 encrypted traffic dataset (120K sessions, 40 domains, 3 cipher suites).

**[G3]** Mohamed elShehaby and Ashraf Matrawy.
"Introducing Perturb-ability Score (PS) to Enhance Robustness Against Problem-Space Evasion Adversarial Attacks on Flow-based ML-NIDS."
*arXiv preprint*, arXiv:2409.07448, September 2024 (revised January 2025).
- **Contribution:** Introduced the **Perturb-ability Score (PS)**, a metric quantifying how susceptible individual NIDS features are to adversarial manipulation given network domain constraints. PS-guided feature selection maintains detection performance while significantly reducing vulnerability to problem-space evasion attacks.

---

#### H. Comprehensive Surveys Covering Adversarial Robustness in Encrypted Traffic

**[H1]** Yulong Wang, Tong Sun, Shenghong Li, Xin Yuan, Wei Ni, Ekram Hossain, and H. Vincent Poor.
"Adversarial Attacks and Defenses in Machine Learning-Empowered Communication Systems and Networks: A Contemporary Survey."
*IEEE Communications Surveys & Tutorials (COMST)*, vol. 25, no. 4, pp. 2245–2298, Fourth Quarter 2023. DOI: 10.1109/COMST.2023.3319492.
- **Venue:** IEEE COMST (IF ~35, highest-impact journal in networking/communications).
- **Contribution:** Comprehensive taxonomy of adversarial attack methods, defense techniques, and robustness enhancement strategies for ML in communication networks including encrypted traffic classification. Covers gradient masking, transferability, clean accuracy trade-offs, and robustness certification.

**[H2]** Islam Debicha, Benjamin Cochez, Tayeb Kenaza, Thibault Debatty, Jean-Michel Dricot, and Wim Mees.
"Adversarial Machine Learning for Network Intrusion Detection Systems: A Comprehensive Survey."
*IEEE Communications Surveys & Tutorials (COMST)*, vol. 25, no. 1, pp. 538–566, First Quarter 2023. DOI: 10.1109/COMST.2022.3233793. IEEE Xplore: 10005100.
- **Venue:** IEEE COMST (IF ~35).
- **Contribution:** Systematic review of adversarial evasion, poisoning, and exploratory attacks against NIDS. Emphasizes domain-specific constraints (semantic and syntactic) unique to network traffic that invalidate naive adversarial perturbations from the image domain.

**[H3]** Mayra Macas, Chunming Wu, and Walter Fuertes.
"Adversarial Examples: A Survey of Attacks and Defenses in Deep Learning-Enabled Cybersecurity Systems."
*Expert Systems with Applications*, vol. 238, Part E, Article 122223, March 2024. DOI: 10.1016/j.eswa.2023.122223.
- **Venue:** Expert Systems with Applications (IF ~8.5).
- **Contribution:** Comprehensive taxonomy covering adversarial ML across cybersecurity applications including network traffic classification, malware detection, and intrusion detection. Curated list of cybersecurity datasets and systematic review of attack generation and defense methods.

**[H4]** Giovanni Apruzzese, Pavel Laskov, and Johannes Schneider.
"SoK: Pragmatic Assessment of Machine Learning for Network Intrusion Detection."
In *Proceedings of the 8th IEEE European Symposium on Security and Privacy (EuroS&P 2023)*, Delft, Netherlands, July 3–7, 2023. DOI: 10.1109/EuroSP57164.2023.00042. IEEE Xplore: 10190520. arXiv:2305.00550.
- **Venue:** IEEE EuroS&P (top security conference).
- **Contribution:** Systematization of Knowledge evaluating ML-NIDS under hundreds of configurations and diverse adversarial scenarios across four hardware platforms. Introduced the concept of **"pragmatic assessment"** bridging academic research and real-world deployment. Validated through a user study with security practitioners.

**[H5]** Shufan Peng et al.
"A Comprehensive Survey of Website Fingerprinting Attacks and Defenses in Tor: Advances and Open Challenges."
*arXiv preprint*, arXiv:2510.11804, October 2025.
- **Contribution:** Most recent comprehensive survey of WF attacks and defenses for encrypted Tor traffic. Covers datasets, attack models (k-NN, CNN, transformer-based), and defense mechanisms (adaptive padding, traffic morphing, adversarial perturbation, circuit splitting). Discusses multi-tab browsing and real-world deployment challenges.

---

#### I. Multi-Class Encrypted Traffic Classification Under Adversarial Conditions (Section 10)

**[M1]** Kai Wang, Zhiliang Wang, Dongqi Han, Wenqi Chen, Jiahai Yang, Xingang Shi, and Xia Yin. "BARS: Local Robustness Certification for Deep Learning based Traffic Analysis Systems." NDSS 2023. [Full citation in Section 10.2]

**[M2]** Xiyuan Zhao, Xinhao Deng, Qi Li, Yunpeng Liu, Zhuotao Liu, Kun Sun, and Ke Xu. "Towards Fine-Grained Webpage Fingerprinting at Scale." ACM CCS 2024. DOI: 10.1145/3658644.3690211. [Full citation in Section 10.2]

**[M3]** Jiaxing Zhang, Xinhao Deng, Qi Li, and Ke Xu. "Transformer-based Model for Multi-tab Website Fingerprinting Attack." ACM CCS 2023. DOI: 10.1145/3576915.3623107. [Full citation in Section 10.2]

**[M4]** Quanliang Jing, Xinxin Fan, Yanyan Liu, and Jingping Bi. "Adversarial Pre-Padding: Generating Evasive Network Traffic Against Transformer-Based Classifiers." arXiv:2510.25810, 2025. [Full citation in Section 10.2]

**[M5]** Ahmed M. Elshewey and Ahmed M. Osman. "Enhancing Encrypted HTTPS Traffic Classification Based on Stacked Deep Ensembles Models." *Scientific Reports*, vol. 15, Article 35230, 2025. DOI: 10.1038/s41598-025-21261-6. [Full citation in Section 10.2]

**[M6]** Zheng Li, Yanbei Liu, Changqing Zhang, Wanjin Shan, Haifeng Zhang, and Xiaoming Zhu. "Trustworthy Deep Learning for Encrypted Traffic Classification." *Soft Computing*, vol. 29, no. 2, pp. 645–662, 2025. DOI: 10.1007/s00500-025-10462-w. [Full citation in Section 10.2]

**[M7]** Thulfiqar Mahmood Tawfeeq and Mohsen Nickray. "Adversarial Training for Improved VPN Traffic Classification Using EfficientNet-B0 and Projected Gradient Descent." *IJIES*, vol. 18, no. 1, 2025. DOI: 10.22266/ijies2025.0229.87. [Full citation in Section 10.2]

**[M8]** "MTDecipher: Robust Encrypted Malicious Traffic Detection via Multi-Task Graph Neural Networks." *Cybersecurity* (Springer Nature), vol. 9, Article 112, 2026. DOI: 10.1186/s42400-025-00522-x. [Full citation in Section 10.2]

**[M9]** Cuong Dao, Van Tong, Nam Thang Hoang, Hai Anh Tran, and Truong X. Tran. "Enhancing Encrypted Traffic Classification with Deep Adaptation Networks." IEEE LCN 2023. DOI: 10.1109/LCN58197.2023.10223333. [Full citation in Section 10.2]
