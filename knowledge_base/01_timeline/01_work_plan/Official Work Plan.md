
> **Historical Document**: This reflects the original project scope and planning. For current online-stage direction, see [03_online_stage_design/README.md](../03_online_stage_design/README.md)

|  בי"ס להנדסת חשמל The School of Electrical Engineering             |  |
| ----- | :---- |
| ***Project Work Plan*** |  |
| **Project Name:** Memory Reactivation Decoder |  |
|  Students: |  |
| **Name**: Itai Paperno | **ID**: 211314471 |
| **Name**: Roi Guri | **ID**: 318835816 |
|  |  |
| Project carried out at: Nitzan Censor Laboratory, Sagol School of Neuroscience, TAU  ***For the project instructor:** I approve the submission of the following report* |  |
|  | Signature: \_\_\_\_\_\_\_\_ Name: \_\_\_\_\_\_\_ |

# **Abstract**

Memory reactivation, the sequential replay of experience-specific neural patterns, is a fundamental mechanism in cognitive neuroscience. This process is assumed to be essential for human episodic memory retrieval and the consolidation of memories during both sleep and wakefulness. 

Investigating these neural processes in humans presents a significant technical challenge. The reactivation events are rapid, transient, and often subtle, embedded within the complex data captured by non-invasive methods like Electroencephalography (EEG). The high temporal resolution of EEG is ideal for capturing these fast dynamics , but its utility is contingent on overcoming the low signal-to-noise ratio and physiological artifacts. To address this challenge this project will implement advanced computational methods, such as multivariate pattern analysis (MVPA) and machine learning, to reliably decode these neural signatures.

This project aims to develop a **Memory Reactivation Decoder**, a real-time neuro-feedback system capable of detecting and utilizing transient memory reactivation events in the human brain, typically measured via Electroencephalography (EEG).

The project is structured into two main phases:  
(1) Offline Decoder Development  
(2) Online Real-Time Decoder Implementation.

**The Offline phase** establishes a machine learning classifier. It involves an extensive literature review on neural reinstatement and computational decoding methods, in-depth familiarity with existing EEG datasets, benchmarking of preprocessing pipelines (e.g., artifact rejection, filtering, ICA, z-scoring), and training various machine learning architectures (e.g., linear models, SVM, regularized regression) to detect memory reactivation patterns. Performance is evaluated based on accuracy, sensitivity, specificity, and temporal generalization. Crucially, model interpretability tools (e.g., SHAP values, representational similarity) will be used to ensure the decoder indexes meaningful neural signals rather than noise or artifacts.

The offline stage will utilize pre-recorded EEG data. This data was gathered during a lab experiment where subjects performed a visual memory task where the association between  triads of visual stimuli were learned and subsequently tested for recall. Classifiers will be trained to detect the neuronal activation of stimuli from a specific visual category.  

**The Online phase** translates this validated model into a low-latency real-time system. This involves developing a dedicated real-time decoding pipeline compatible with continuous EEG streams, adapting the offline-trained models for online use (integrating buffering and sliding windows), and ensuring seamless Lab Integration with the existing EEG acquisition hardware.

The final step is Action-Driven Decoding, where the system is configured to output a real-time event marker (e.g., a digital trigger) based on the detected reactivation events. This output is designed to enable future closed-loop interventions (e.g., targeted stimulation or stimulus presentation) by the lab, and will be validated by rigorous live testing for latency and stability.

**Block Diagram**

![][image1]

1. # **Motivation**

Memory reactivation, the re-emergence of neural activity patterns from an initial experience, is a fundamental mechanism supporting human cognition. It is essential for two core functions: the retrieval of episodic memories during wakefulness and the long-term consolidation of those memories during sleep. This process forms the basis of the "hippocampal-neocortical dialogue," where the hippocampus is thought to trigger the reinstatement of memory-specific information in the neocortex, where they are permanently stored.

The lab possesses a strong research background in human memory, with active interests in finding ways to enhance memory for learning and, for clinical applications like PTSD, to target and weaken specific traumatic memories.

The primary aim of this project is to establish a tool for memory reactivation decoding. While the lab is interested in memory reactivation, it currently lacks a validated method to quantify this process from neural data. This gap creates two key technical opportunities that this project will solve. For offline analysis, researchers in the lab often develop project-specific tools. This project will centralize these methods into a single, structured, and validated software package for memory decoding. This will improve standardization and reusability for future research.

For real-time experiments, interventions are often timed based on assumptions of when memory reactivation occurs, rather than on empirical detection. This approach does not account for trial-to-trial variability. This project will provide a tool for data-driven detection, enabling next-generation closed-loop designs.

The final deliverable will be a validated Memory Reactivation Decoder package. This system will be capable of 'Action-Driven Decoding,' generating precise, real-time event markers (triggers) based on empirical data. These triggers can then be integrated into future lab experiments, providing a core component for closed-loop systems to test hypotheses about enhancing or weakening specific memories.

2. # **Statement of Work** 

#### **Theoretical Background and Sources**

The project requires a foundational understanding of the following key domains:

1. **EEG Signal Processing:** Fundamentals of time-series analysis, filtering (e.g., FIR/IIR), spectral analysis (e.g., Fast Fourier Transform, Wavelets), artifact rejection (e.g., ICA, EOG/EMG removal), and referencing techniques.

2. **Memory Reactivation/Neural Reinstatement:** Understanding the neural correlation of memory reactivation, including the role of oscillatory patterns (e.g., theta, gamma) and specific brain regions (e.g., medial temporal lobe, hippocampus) as measured non-invasively by EEG.

3. **Machine Learning (Model Training):** Theory behind supervised classification models (SVMs, Regularized Linear Models) for high-dimensional, noisy EEG data. Methods for feature engineering, model selection, cross-validation, and hyperparameter tuning.

4. **Machine Learning (Model Validation & Explainability):** This involves theories to validate the decoder, primarily using the Temporal Generalization Method (TGM) to distinguish stable representations from transient signals and interpretation methods (RSA, feature importance) to ensure it's decoding plausible neural features, not artifacts.

5. **Real-Time Theory:** This covers the engineering challenges of Domain Adaptation (applying an offline model to a live stream with different statistical properties), and Streaming Protocols (low-latency data and marker synchronization, e.g., LSL).

**Main Sources**: This project will draw its theoretical and technical foundation from several sources. Foundational academic literature in the field will be a primary source, including key review articles on memory reactivation, EEG artifact removal, and decoding methodologies, with other relevant papers to be used as necessary. The implementation will be heavily based on the official documentation for the core Python libraries: MNE-Python, Scikit-learn, and Lab Streaming Layer (LSL). This will be supplemented by knowledge from relevant academic coursework in signal processing and machine learning, as well as targeted online research.

#### **Project Requirements and Steps**

**Part 1: Offline Decoder (Building & Validating the Classifier)**

1. **Data Familiarization:** Assess the existing pre-recorded EEG dataset (binary animate/inanimate task) for noise levels, and data quality.

* **Tools:** MNE-Python (for data loading and inspection), Pandas (for metadata handling), Matplotlib (for data visualization and summary statistics).

2. **Preprocessing Pipeline Design:** Enhance existing preprocessing pipeline that includes segmenting the recording to epochs, filtering relevant oscillatory patterns and handling artifacts.

* **Tools:** MNE-Python (for FIR filtering, epoching, and implementing BSS methods like ICA), NumPy (for signal manipulation).

3. **Feature Engineering & Selection:** Extract and evaluate features relevant to the theoretical target signals.

* **Tools:** MNE-Python (for Power Spectral Density (PSD) and time-frequency analysis), Scikit-learn (for feature selection), NumPy (for feature matrix construction).

4. **Binary Classifier Training and Optimization:** Train and benchmark multiple decoding architectures (e.g., linear models, SVMs).

* **Tools:** Scikit-learn

5. **Performance Evaluation and Explainability:** Evaluate the classifier's performance to ensure the model is decoding meaningful neurophysiological signals.

* **Tools:** Scikit-learn (for metrics), MNE-Python, Matplotlib (to plot confusion matrices, TGM matrices, and RSA matrices), SHAP library (for feature importance analysis).

6. **(Optional) Go/No-Go Decision & Multiclass Path:**  If the current binary dataset is insufficient, a new experiment should be designed to collect an EEG dataset suitable for multiclass classification. This will involve training and validating a multiclass decoder.

* **Tools:** MNE-Python (for data acquisition and processing), Scikit-learn (for Multiclass Logistic Regression).

**Part 2: Online Real-Time Decoder (Translating to a Live System)**

1. **Real-Time Environment Setup:** Implement the validated offline pipeline in a new script capable of handling live data streams.

* **Tools:** Python, Lab Streaming Layer (LSL) library (for receiving data), NumPy (for real-time buffering).

2. **Pipeline Calibration and Latency Minimization:** Fine-tune the real-time data processing to ensure the end-to-end decoding latency (from EEG sample to output trigger) meets the required specification.

3. **Action-Driven Integration:** Develop the software interface to generate a timely event marker (trigger) based on the classifier's decision.

4. **Online Simulation (using offline data):** Validate the full real-time chain by streaming pre-recorded data to establish latency and accuracy-degradation baselines.

* **Tools:** Python scripts (to replay MNE-Python data over an LSL stream.

5. **(Optional) Live Testing:** Conduct tests with human subjects to validate system stability and performance under real-world conditions.

* **Tools:** Full stack: Lab EEG Hardware, LSL, Python real-time script.

* The lab's data acquisition setup uses the Bittium NeurOne Tesla EEG system, an fMRI-compatible amplifier, to acquire signals from Easycap EEG recording caps. The NeurOne software controls this hardware, monitors the incoming data, and provides the real-time data stream for analysis.

3. **Project deliverables**

**System Requirements:**

1. **Area Under the Curve (AUC) \- Offline Decoding performance**  
   Target: The classifier's AUC must exceed the chance level by a minimum of 20 percentage points for binomial tasks .

2. **Real-Time Decoding Latency**  
   Target: The end-to-end processing time, from the acquisition of an EEG sample window to the output of a classification decision in the online system, must be less than 100 milliseconds.

3. **Online vs. Offline Accuracy Degradation**  
   Target: The accuracy (AUC) of the online decoder during simulation with pre-recorded data should not degrade by more than 10% compared to the purely offline performance on the same data partition. 

**Deliverables List:**

1.  **Enhanced & Packaged Preprocessing Pipeline**

* Description: A documented Python software module that enhances the existing lab pipeline. It will integrate artifact handling methods and filtering, creating a ready-to-use package for future EEG experiments in the lab.

* Testing & Validation: The pipeline will be validated by comparing the signal quality before and after artifact removal through visual inspection and by benchmarking its impact on the performance of a baseline classifier.

2. **Offline Classification Model (Trained & Validated)**

* Description: The optimized machine learning model and associated Python scripts for training and testing on the offline dataset. This includes all code for generating performance metrics (accuracy, confusion matrices, etc.).

* Testing & Validation: Standard k-fold cross-validation and subject-out cross-validation will be used to assess generalizability.

3. **Model Explainability Analysis Report**

* Description: A report containing the results of model interpretation tools (e.g., feature importance, representational similarity, SHAP values) demonstrating the decoding results map onto meaningful neural signals rather than artifacts.

* Testing & Validation: Validation will be based on the consistency of the identified neural features (e.g., specific time-frequency patterns, sensor locations) with established findings in the memory reactivation literature

4. **(Optional): New Data Acquisition & Multiclass Models**

* Description: Should the existing binary dataset prove insufficient, this deliverable includes:

  * The experimental paradigm and script for collecting a new EEG dataset for multiclass classification (e.g., cued recall of specific items).

  * The trained and validated multiclass decoder (e.g., Multiclass Logistic Regression, Shallow CNN) designed to decode item-specific memory reactivation events.

* Testing & Validation: Performance will be evaluated using metrics suitable for multiclass problems, such as Mean AUC and Mean Recall across all classes. The Temporal Generalization Method (TGM) will be used to confirm sustained item-specific neural representations.

5. **Real-Time Decoder Software (Online System)**

* Description: The fully implemented, low-latency Python pipeline capable of receiving streamed EEG data, performing preprocessing and feature extraction, and outputting real-time classification triggers for action-driven decoding.

* Testing & Validation: The system will be tested via online simulation with the pre-recorded data and optionally in a live recording environment.

4. # **Project Schedule**

|  | Milestone | Description (2-3 lines) | Planned Date |
| :---: | ----- | ----- | :---: |
| 1\. | Initial Literature Review | Comprehensive survey of neural reinstatement and decoding methods. | 16.11.25 |
| 2\. | Workplan submission | Project Workplan according to given guidelines. | 16.11.25 |
| 3\. | Data Familiarity | Assess available datasets: size, number of subjects, trial structure, conditions. Evaluate the advantages (e.g., trial count, task design) and limitations (e.g., noise, imbalance, missing data). Identify data gaps that may require new experiments. | 30.11.25 |
| 4\. | Implementing Preprocessing Pipeline (D1) | Understand preprocessing pipelines (artifact rejection, filtering, ICA, z-scoring). Assess how each preprocessing choice influences downstream machine learning performance. Benchmark preprocessing alternatives. | 14.12.25 |
| 5\. | Offline Model Training | Train machine learning models on the existing data set. Explore multiple architectures (linear models, SVM, regularized regression). | 25.01.26 |
| **6\.** | Binary Model Evaluation & Explainability (D2, D3) | Evaluate classifier accuracy, sensitivity, specificity, and temporal generalization. Assess generalizability across subjects | 08.02.26 |
| **7\.** | Go/No-Go Decision for New Data Acquisition | Based on the performance and limitations of the binary classifier, make a final decision on whether to proceed with acquiring a new, multiclass dataset. | 08.02.26 |
| 8\. | **Progress Presentation Submission** | Deliver comprehensive presentation covering theory, methods, offline results, and the plan for the online phase. | 22.02.26 |
| **9\.** | Real-Time Pipeline Development | Implement the core software framework for low-latency processing, including real-time feature extraction from EEG streams (e.g., LSL) and buffering. | 23.03.26 |
| 10\. | Online Simulation & Validation (D5) | Validate the full real-time processing chain by streaming pre-recorded EEG data. Establish latency baselines and measure accuracy degradation relative to offline performance to meet targets. | 13.04.26 |
| 11\. | Lab Integration & Action-Driven Logic (D5) | Integrate the online decoder with the EEG acquisition hardware. Implement and test the logic for generating real-time classification triggers (e.g., LSL markers) intended for future closed-loop interventions. | 04.05.26 |
| 12\. | (Optional) Live Pilot Testing | Conduct controlled live testing with human subjects to validate system stability, end-to-end latency, and accuracy under real-world streaming conditions. Perform iterative refinement. | 18.05.26 |
| 12\. | **Poster Submission and finishing the work** | Complete the design and content of the final project poster. | 24.05.26 |
| **13\.** | Final Documentation & Report Writing | Compile all results, methods, and analysis into the final comprehensive project report. | 28.06.26 |
| **14\.** | **Final deliverables submission** | Submission of the final report, complete code, and all required project artifacts. | 01.07.26 |

**Optional Multiclass Path:**

If a decision is made that new data is necessary and the multiclass classification path is adopted, the schedule and planned dates will be updated with the requisite tasks. This work is anticipated to proceed concurrently with the binary classification efforts planned for the first semester.

1. **New Experiment Paradigm Design & Scripting:**  
   Design and implement the experimental script for a multiclass cued recall task, ensuring integration with lab hardware for stimulus presentation and EEG recording.

2. **New Data Acquisition (Experiment Execution):**   
   Oversee and conduct the experiment on subjects in the lab to generate the new item-specific EEG dataset for multiclass classification.

3. **Multiclass Classifier Development (D4):**  
   Develop and optimize the final multiclass decoder (e.g., One-vs-Rest Logistic Regression, Shallow CNN) on the newly acquired dataset. Evaluate using Mean AUC, Mean Recall, and the Temporal Generalization Method (TGM).

[image1]: <data:image/png;base64,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>