# Multimodal Emotion Recognition in Conversational Contexts Using the MELD Dataset

## Overview

This project presents a comprehensive multimodal approach to emotion recognition in conversational contexts using the MELD (Multimodal Multi-Party Dataset for Emotion Recognition in Conversation) dataset. The system integrates textual, audio, and visual modalities through a sophisticated pipeline that combines state-of-the-art feature extraction techniques with graph neural networks to capture both individual utterance characteristics and conversational context.

The architecture employs a multi-stage approach that first extracts modality-specific features using specialized models including RoBERTa for text processing, utterance-level features for audio analysis, and SlowFast R50 for video understanding. These features are then fused through a BiLSTM network and processed using Graph Neural Networks to model conversational dynamics before making emotion predictions.

## Dataset Information

### MELD Dataset Characteristics

The [`Multimodal EmotionLines Dataset (MELD)`](https://github.com/declare-lab/MELD.git) has been created by enhancing and extending the EmotionLines dataset to include audio and visual modalities alongside textual content. MELD contains the same dialogue instances available in EmotionLines while encompassing comprehensive multimodal information. The dataset includes more than 1400 dialogues and 13000 utterances extracted from the Friends TV series, featuring multiple speakers participating in natural conversations.

MELD contains approximately 13,000 utterances from 1,433 dialogues sourced from the television series Friends. The dataset demonstrates superiority over other conversational emotion recognition datasets such as SEMAINE and IEMOCAP due to its multiparty conversation structure and substantially larger number of utterances, nearly doubling the content available in comparable datasets.

### Dataset Organization and Emotion Categories

The MELD dataset is organized into three standard splits comprising a training set used for model training and parameter optimization, a development set utilized for hyperparameter tuning and model validation during development, and a test set reserved for final model evaluation and performance reporting.

Each utterance in the dataset has been labeled with one of seven distinct emotion categories. These categories include Anger representing expressions of frustration, irritation, or hostility; Disgust indicating expressions of revulsion or strong disapproval; Sadness encompassing expressions of sorrow, disappointment, or melancholy; Joy representing expressions of happiness, pleasure, or contentment; Neutral categorizing emotionally neutral expressions; Surprise indicating expressions of astonishment or unexpected reactions; and Fear representing expressions of anxiety, worry, or apprehension.

## Architecture Overview

The proposed system implements a comprehensive multimodal emotion recognition pipeline that processes textual, audio, and visual information from conversational videos. The architecture is designed to capture both individual utterance characteristics and the broader conversational context that influences emotional expression in multi-party dialogues.

The system follows a hierarchical approach where each modality is processed through specialized feature extractors, followed by temporal modeling through BiLSTM networks, and finally integrated through Graph Neural Networks that model the conversational structure and speaker interactions. This multi-level processing ensures comprehensive understanding of emotional content across different modalities and temporal contexts.

## Feature Extraction Pipeline

### Text Feature Processing

The textual component of each utterance is processed using RoBERTa (Robustly Optimized BERT Pretraining Approach), a state-of-the-art transformer-based language model that has demonstrated superior performance on various natural language understanding tasks through its optimized training procedure and architectural improvements over the original BERT model.

The system utilizes the roberta-base tokenizer and pre-trained RoBERTa base model configuration. Each utterance is tokenized and encoded to produce contextual embeddings that capture both semantic and syntactic information from the conversational text. The RoBERTa model provides rich contextual representations that understand the nuanced language patterns typical in conversational settings, including colloquialisms, emotional expressions, and context-dependent meanings that are crucial for accurate emotion recognition.

### Audio Feature Analysis

Audio features are extracted at the utterance level to capture prosodic and paralinguistic information that conveys emotional content through vocal characteristics. The audio processing pipeline focuses on extracting meaningful acoustic features that complement the textual information and provide additional emotional context not explicitly encoded in transcriptions.

The system extracts comprehensive audio features including prosodic characteristics such as pitch contours, rhythm patterns, and temporal dynamics; spectral features encompassing frequency domain characteristics including spectral centroid, rolloff, and mel-frequency cepstral coefficients; energy-based features capturing voice activity patterns and intensity variations; and paralinguistic features identifying voice quality indicators and emotional prosody markers. These features capture the acoustic manifestations of emotions that are often present in speech but not explicitly encoded in textual transcription, providing crucial complementary information for emotion recognition.

### Video Processing and Frame Extraction

The video processing pipeline implements a sophisticated frame extraction system specifically designed for the MELD dataset's conversational videos. This system addresses the challenges of multi-speaker scenarios, variable video lengths, and the need for consistent speaker focus across different conversational contexts.

The frame extraction process follows a detailed workflow that implements an intelligent speaker identification mechanism combining facial recognition with mouth movement analysis to ensure consistent focus on the primary speaker throughout each utterance. The system employs several key components including a speaker identification subsystem that identifies the main speaker across the entire video sequence through facial recognition and speech activity detection, utilizing face encoding comparison with similarity thresholds and dynamic focus tracking with confidence scoring to maintain consistent speaker identification.

The face processing pipeline employs DLib CNN face detection for high-accuracy face identification, followed by face recognition encoding that produces 128-dimensional face embeddings. A correlation tracker provides robust face tracking with quality assessment and automatic failure recovery mechanisms. The intelligent frame selection system extracts exactly 32 frames per video, implementing sophisticated logic to handle variable-length videos and generating masked frames when videos contain fewer than 32 useful frames to maintain consistent input dimensions while preserving temporal structure.

### Video Feature Extraction

Following frame extraction, visual features are extracted using SlowFast R50, a state-of-the-art video understanding model that implements a sophisticated dual-pathway architecture specifically designed for temporal video understanding. The SlowFast network processes the extracted 32-frame sequences through two complementary pathways that capture different aspects of temporal dynamics in emotional expressions.

The SlowFast R50 model implements a pretrained dual-pathway architecture that processes video sequences through two specialized channels. The Slow Pathway operates on 8 frames to capture detailed spatial information including facial expressions, micro-expressions, and static visual cues that convey emotional states, focusing on high-resolution spatial features crucial for understanding subtle emotional expressions. The Fast Pathway processes all 32 frames to capture rapid temporal changes and motion patterns, specializing in detecting dynamic facial movements, gesture patterns, and temporal evolution of expressions throughout the utterance duration.

The architecture incorporates sophisticated masking systems to handle variable-length sequences effectively, applying zero padding while maintaining temporal coherence through intelligent mask application when videos contain fewer than 32 frames. The system ensures consistent processing regardless of original video length while preserving the temporal structure necessary for accurate emotion recognition. The SlowFast model produces rich 2304-dimensional visual feature vectors that capture both spatial detail of emotional expressions and temporal dynamics of expression evolution throughout each conversational utterance.

## Model Architecture

### Multimodal Feature Fusion and Projection

The extracted features from all three modalities undergo sophisticated processing before integration. Each modality produces high-dimensional feature vectors that require careful alignment and normalization for effective fusion. The text features consist of 768-dimensional contextual embeddings from RoBERTa, audio features comprise 6373-dimensional comprehensive acoustic feature vectors including prosodic elements, and video features represent 2304-dimensional temporal visual features from SlowFast R50.

To ensure effective integration, each modality's features are projected to a common 256-dimensional space through specialized projection networks. The text projector implements a linear transformation from 768 to 256 dimensions with ReLU activation and LayerNorm for linguistic feature alignment. The audio projector employs a two-stage projection from 6373 to 512 to 256 dimensions with normalization for significant dimensionality reduction while preserving acoustic information. The video projector uses a two-stage projection from 2304 to 512 to 256 dimensions with activation and normalization to align visual temporal features.

The system implements an innovative adaptive gating mechanism that dynamically weights the contribution of each modality based on input characteristics. This mechanism combines all projected features into a unified 768-dimensional representation, employs a gate network that learns modality importance weights using a Linear transformation followed by Softmax normalization, and applies element-wise multiplication for adaptive modality weighting, allowing the model to focus on the most informative modalities for each specific utterance.

### Advanced Temporal Modeling with BiLSTM

The adaptively gated multimodal features undergo sophisticated temporal processing through a specialized Bidirectional Long Short-Term Memory network that captures complex temporal dependencies and contextual relationships within conversational sequences. This architecture is specifically designed to handle the unique challenges of conversational emotion recognition.

The BiLSTM network implements a robust architecture with LayerNorm preprocessing that prepares gated features for stable sequence processing, packed sequence processing that efficiently handles variable-length conversational sequences without unnecessary computational overhead, and bidirectional processing featuring 128 units per direction with 256 total output across 2 layers with 0.5 dropout between layers for forward and backward temporal context integration.

The bidirectional nature allows the model to understand how past conversational history and future context influence the current utterance's emotional content, which is crucial in conversational settings where emotional states often depend on broader dialogue context. The packed sequence mechanism efficiently processes conversations of different lengths without padding-related computational waste while maintaining proper gradient flow through the network. Multiple dropout layers with 0.5 dropout rate prevent overfitting while maintaining the model's ability to capture complex temporal patterns in conversational data.

### Graph Neural Network Integration

The output features from the BiLSTM are used to construct and process a Graph Neural Network that models the conversational structure and speaker interactions. Graph Neural Networks are particularly well-suited for conversational emotion recognition as they can explicitly model the relationships between different speakers, the influence of conversational history, and the complex dynamics that emerge in multi-party conversations.

The GNN architecture provides several advantages including conversational structure modeling that explicitly represents the graph structure of conversations including speaker relationships and temporal connections, speaker interaction modeling that captures how different speakers influence each other's emotional states throughout the conversation, context propagation that allows information to propagate through the conversational graph enabling the model to understand how emotional states evolve and influence each other, and flexible representation that can handle variable-length conversations and different numbers of speakers.

### Advanced Training Pipeline and Optimization

The training process implements a sophisticated pipeline that combines state-of-the-art optimization techniques with robust training management and comprehensive monitoring systems. This advanced training framework ensures optimal model performance while maintaining training stability and reproducibility.

The training pipeline begins with comprehensive environment setup including CUDA memory optimization, reproducible random seed management, and secure API authentication for external services. The system is configured with optimized parameters including 32 frames per video, 224×224 pixel resolution, and carefully tuned batch sizes that balance GPU memory usage with training efficiency.

The system employs an Adaptive Focal Loss function specifically designed to handle inherent class imbalance in emotional datasets, featuring gamma equals 2.0 focus parameter with alpha equals 0.8 weighting, class balancing through inverse square-root class weighting to address dataset imbalance, label smoothing for additional regularization to prevent overconfident predictions, and adaptive weighting with dynamic adjustment based on class difficulty.

The training process utilizes a multi-faceted optimization approach including AdamW Optimizer with weight decay, advanced learning rate scheduling featuring OneCycleLR Scheduler for SlowFast training with 30% warmup phase and cosine annealing and ReduceLROnPlateau for BiLSTM training with patience equals 5 for adaptive learning rate adjustment, mixed precision training with FP16 training and gradient scaling for improved memory efficiency and training speed, and comprehensive gradient management including gradient accumulation, gradient clipping, and NaN detection for training stability.

## Experimental Setup

### Multi-Seed Evaluation Framework

To ensure robust and reliable results, all experiments are conducted using multiple random seeds to account for variability in model initialization and training dynamics. The evaluation employs five distinct random seeds including 42, 2023, 7, 123, and 314 to provide comprehensive assessment across different initialization conditions.

This multi-seed approach provides statistical significance to the results and allows for proper assessment of model stability and consistency across different initialization conditions. The systematic evaluation framework ensures that reported performance metrics represent genuine model capabilities rather than artifacts of specific random initializations.

### Batch Size Analysis and Configuration

The system performance is evaluated across different effective batch sizes to understand the impact of batch size on model convergence and final performance. The evaluation systematically examines effective batch sizes of 8, 16, 32, and 64 to identify optimal training configurations.

This comprehensive batch size analysis helps identify optimal training configurations and provides insights into how batch size affects the learning dynamics of the multimodal architecture, particularly given the complexity of processing multiple modalities simultaneously. The analysis considers both computational efficiency and model performance to determine the most effective training parameters.

### Evaluation Metrics and Assessment

Model performance is assessed using comprehensive evaluation metrics that provide detailed insights into system capabilities across different dimensions. The evaluation framework includes confusion matrices presented in both normalized by true class format to show recall performance and absolute sample counts to demonstrate raw prediction patterns.

The assessment encompasses classification accuracy measuring overall classification performance across all emotion categories, per-class performance providing detailed analysis of performance for each emotion category, and statistical analysis conducting cross-seed performance analysis to assess model stability and reliability across different experimental conditions.

## Results and Analysis

### Performance Characteristics and Model Capabilities

The comprehensive evaluation approach provides detailed insights into system performance across multiple dimensions. The system demonstrates exceptional training stability and convergence properties with expected performance achieving weighted F1 scores exceeding 0.68 across the MELD dataset.

The system exhibits class-specific excellence particularly in recognizing Neutral emotions with exceptional recognition accuracy attributed to robust contextual understanding through BiLSTM temporal modeling and comprehensive multimodal feature fusion. Joy recognition demonstrates good performance leveraging high activation patterns detected in facial expressions via SlowFast R50 features and distinctive vocal prosody captured in utterance-level audio features. Surprise recognition shows good capability benefiting from rapid facial micro-expression changes and sudden acoustic pattern shifts effectively captured by the concatenated feature representation. The system maintains balanced recognition across all emotional states through advanced focal loss handling.

### Statistical Analysis and Model Stability

The rigorous multi-seed evaluation using seeds 42, 2023, 7, 123, and 314 provides comprehensive statistical insights into model performance and reliability. The evaluation reveals model stability through variance analysis across different random initializations demonstrating training consistency, reproducibility ensured through full RNG state preservation guaranteeing identical results for given configurations, statistical significance provided through multiple seed evaluation offering confidence intervals for performance metrics, and robustness assessment identifying potential overfitting or instability issues across different initialization conditions.

The systematic evaluation across different batch sizes reveals important training dynamics including memory efficiency achieved through gradient accumulation enabling effective larger batch sizes on constrained hardware, convergence characteristics showing how different batch sizes exhibit varying convergence patterns and final performance, optimization dynamics where larger batch sizes may provide more stable gradients but potentially slower convergence, and resource utilization optimization through optimal batch size selection balancing GPU memory usage with training efficiency.

### Confusion Matrix Analysis and Insights

The dual [`confusion matrix analysis`](./docs/documents/confusion_matrix.md) provides comprehensive understanding of model behavior through normalized confusion matrices that reveal per-class recognition capabilities independent of class frequency, highlighting which emotions are most challenging to distinguish and providing insights into model bias toward frequent classes. Absolute count confusion matrices show raw prediction patterns and highlight the most common misclassification pairs, providing practical insights into real-world deployment scenarios where class distribution matters.

The advanced diagnostic capabilities include gradient analysis through real-time gradient norm tracking preventing vanishing or exploding gradient issues, learning rate dynamics with adaptive scheduling providing optimal learning progression, sample prediction tracking through Weights & Biases integration allowing monitoring of specific sample predictions throughout training, and early stopping intelligence with sophisticated stopping criteria preventing both underfitting and overfitting.

## Architectural Diagrams and System Visualization

The project includes comprehensive architectural diagrams for each major component of the system. These diagrams encompass the complete multimodal emotion analysis architecture showing end-to-end system overview and integration of text, audio, and video processing pipelines; MELD video frame extraction detailing the sophisticated video preprocessing pipeline with intelligent speaker identification and robust face tracking; MELD emotion recognition training pipeline featuring comprehensive training workflow with SlowFast dual-pathway architecture and masked temporal processing; multimodal BiLSTM architecture illustrating advanced temporal modeling system with adaptive gating mechanisms and feature extraction capabilities; and MELD emotion recognition SlowFast masked pathways showing visual feature extraction architecture with dual-pathway processing and temporal masking.

The system is composed of multiple interconnected modules. The architectural diagrams and explanations for each major component are provided below.

1. [Complete Multimodal Emotion Analysis Pipeline](./docs/documents/complete_multimodal_pipeline.md)
2. [MELD Video Frame Extraction](./docs/documents/meld_video_frame_extraction.md)  
3. [MELD Emotion Recognition using SlowFast Masked Pathways](./docs/documents/meld_slowfast_masked_pathways.md)  
4. [Feature Extraction using BiLSTM on MELD Dataset](./docs/documents/meld_bilstm_feature_extraction.md)  
5. [Multimodal Emotion Analysis Using GNN](./docs/documents/emotion_analysis_gnn.md)  
   
## Future Work and Research Directions

### Potential Improvements and Enhancements

Several areas present opportunities for future development and improvement of the multimodal emotion recognition system. Advanced fusion techniques offer potential for investigation of attention-based fusion mechanisms to better integrate multimodal information and improve the understanding of cross-modal relationships. Transformer integration presents opportunities for exploration of transformer-based architectures for improved temporal modeling, potentially replacing or augmenting the current BiLSTM approach.

Speaker-aware modeling represents an important direction for enhanced speaker identification and speaker-specific emotion modeling, which could improve recognition accuracy in multi-party conversational settings. Cross-dataset evaluation provides opportunities for testing the approach on additional conversational emotion datasets to assess generalization capabilities and robustness across different domains. Real-time processing optimization offers potential for developing systems capable of real-time emotion recognition in live conversations, which would enable practical deployment in interactive applications.

### Research Directions and Long-term Goals

The field presents several promising research directions that could significantly advance conversational emotion recognition capabilities. Contextual emotion dynamics research offers opportunities for deeper investigation of how emotional states evolve throughout conversations, potentially leading to more sophisticated models of emotional contagion and influence in multi-party settings.

Cultural and linguistic variations research presents important directions for extension to multiple languages and cultural contexts, ensuring that emotion recognition systems can operate effectively across diverse populations and communication styles. Interactive emotion recognition research offers potential for integration with conversational AI systems for dynamic emotion-aware responses, creating more empathetic and contextually appropriate artificial intelligence systems. Privacy-preserving approaches research includes development of federated learning approaches for emotion recognition, enabling training on distributed data while maintaining privacy and security requirements.

## Citation and References

### MELD Dataset Citation

When using the MELD dataset, researchers should cite the original publication that introduced this comprehensive multimodal resource. The appropriate citation references the work by Poria, Soujanya and colleagues titled "MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversation," published in the Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 527-536, in 2019.

`MELD Dataset Citation`

When using the MELD dataset, please cite the original paper:

```bibtex
@inproceedings{poria2019meld,
    title={MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversation},
    author={Poria, Soujanya and Hazarika, Devamanyu and Majumder, Navonil and Naik, Gautam and Cambria, Erik and Mihalcea, Rada},
    booktitle={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
    pages={527--536},
    year={2019}
}
```

### Additional References and Methodological Foundations

For comprehensive understanding of the methodological foundations, researchers should consider citing related works that form the theoretical and technical basis of the approach. The RoBERTa text processing methodology is based on the work by Liu, Yinhan and colleagues titled "RoBERTa: A Robustly Optimized BERT Pretraining Approach," published as an arXiv preprint in 2019.

`RoBERTa for Text Processing`
```bibtex
@article{liu2019roberta,
    title={RoBERTa: A Robustly Optimized BERT Pretraining Approach},
    author={Liu, Yinhan and Ott, Myle and Goyal, Naman and Du, Jingfei and Mandar, Joshi and Chen, Danqi and Levy, Omer and Lewis, Mike and Zettlemoyer, Luke and Stoyanov, Veselin},
    journal={arXiv preprint arXiv:1907.11692},
    year={2019}
}
```

The SlowFast networks for video understanding represent a significant contribution to temporal video analysis, developed by Feichtenhofer, Christoph and colleagues in their work titled "SlowFast networks for video recognition," published in the Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6202-6211, in 2019.

`SlowFast Networks for Video Understanding`
```bibtex
@inproceedings{feichtenhofer2019slowfast,
    title={SlowFast networks for video recognition},
    author={Feichtenhofer, Christoph and Fan, Haoqi and Malik, Jitendra and He, Kaiming},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
    pages={6202--6211},
    year={2019}
}
```

## Acknowledgments and Community Contributions

This work builds upon the valuable contributions of the research community in multimodal emotion recognition and conversational AI. The project particularly acknowledges the creators of the MELD dataset for providing a comprehensive multimodal conversational emotion dataset that enables advanced research in this domain.

Recognition is extended to the developers of RoBERTa for advancing transformer-based language understanding, providing the foundation for sophisticated textual analysis in conversational contexts. The creators of SlowFast networks are acknowledged for their innovations in temporal video understanding, which have enabled sophisticated analysis of visual emotional expressions in conversational videos.

The broader research community working on emotion recognition and multimodal learning deserves acknowledgment for their ongoing contributions that have created the theoretical and methodological foundation upon which this work builds. The collaborative nature of scientific research in artificial intelligence and machine learning has made possible the development of increasingly sophisticated approaches to understanding human emotional expression in natural conversational settings.

## Repository Structure and Implementation Details

```
├── README.md                                                           # Main project overview and navigation
├── .github/                                                            # GitHub Actions and CI/CD
│   └── workflow/                                                       # Workflow configurations directory
├── artifacts/                                                          # Generated files (models, logs, results)
├── src/                                                                # Source code
│   └── EmotionAnalysis/                                                # Main project package
│       ├── __init__.py                                                 # Package initialization
│       ├── components/                                                 # Reusable modular components
│       │   ├── __init__.py                                             # Components package initialization
│       │   ├── data_ingestion.py                                       # Data collection components
│       │   ├── data_validation.py                                      # Data validation components
│       │   ├── data_transformation.py                                  # Data transformation components
│       │   ├── model_trainer.py                                        # Model training components
│       │   └── model_evaluation.py                                     # Model evaluation components
│       ├── data/                                                       # Data handling modules
│       │   ├── dataset.py                                              # Dataset loading and preprocessing
│       │   └── transforms.py                                           # Data transformation utilities
│       ├── models/                                                     # Model architectures
│       │   ├── base_model.py                                           # Base model class for inheritance
│       │   └── custom_model.py                                         # Task-specific model implementations
│       ├── training/                                                   # Training pipeline components
│       │   ├── trainer.py                                              # Main training orchestration
│       │   ├── loss.py                                                 # Loss function definitions
│       │   ├── optimizer.py                                            # Optimizer configurations
│       │   └── scheduler.py                                            # Learning rate scheduling
│       ├── evaluation/                                                 # Model evaluation framework
│       │   ├── evaluation.py                                           # Evaluation pipeline
│       │   └── metrics.py                                              # Performance metrics calculation
│       ├── utils/                                                      # Utility functions
│       │   ├── common_utils.py                                         # General-purpose utilities
│       │   ├── logger.py                                               # Custom logging setup
│       │   ├── config.py                                               # Configuration management
│       │   ├── checkpoint.py                                           # Model state persistence
│       │   └── visualizer.py                                           # Training visualization tools
│       ├── entity/                                                     # Data entities and schemas
│       │   └── __init__.py                                             # Entity package initialization
│       └── pipeline/                                                   # Modular ML pipeline stages
│           ├── __init__.py                                             # Pipeline package initialization
│           ├── stage_01_data_ingestion.py                              # Data collection stage
│           ├── stage_02_data_validation.py                             # Data quality validation
│           ├── stage_03_data_transformation.py                         # Feature engineering
│           ├── stage_04_model_trainer.py                               # Model training stage
│           └── stage_05_model_evaluation.py                            # Model assessment stage
├── scripts/                                                            # Setup and utility scripts
│   ├── install_system_deps.sh                                          # System dependencies installation
│   ├── download_models.sh                                              # Pre-trained model downloads
│   └── setup_environment.sh                                            # Complete environment setup
├── docs/                                                               # Documentation directory
│   ├── documents/                                                      # Architecture documentation files
│   │   ├── complete_multimodal_pipeline.md                             # Complete multimodal pipeline documentation
│   │   ├── meld_video_frame_extraction.md                              # Video frame extraction process
│   │   ├── meld_slowfast_masked_pathways.md                            # SlowFast masked pathways architecture
│   │   ├── meld_bilstm_feature_extraction.md                           # BiLSTM feature extraction methodology
│   │   └── emotion_analysis_gnn.md                                     # Graph Neural Network emotion analysis
│   └── artifacts/                                                      # Architecture diagrams and visualizations
│       ├── Multimodal_Emotion_Analysis_Architecture.svg                # Complete system architecture
│       ├── MELD_Video_Frame_Extraction.svg                             # Video frame extraction workflow
│       ├── MELD_Emotion_Recognition_SlowFast_Masked_Pathways_svg.svg   # SlowFast pathways visualization
│       ├── MELD_Feature_Extraction_Using_BiLSTM.svg                    # BiLSTM feature extraction
│       └── MELD_Feature_Extraction_Using_GNN.svg                       # GNN feature extraction
├── logs/                                                               # Application and training logs
├── notebooks/                                                          # Jupyter notebooks for experimentation
├── configs/                                                            # Configuration files
│   ├── default.yaml                                                    # Default project settings
│   ├── train.yaml                                                      # Training-specific configurations
│   └── eval.yaml                                                       # Evaluation parameters
├── requirements.txt                                                    # CPU-only Python dependencies
├── requirements-cuda.txt                                               # GPU-specific Python dependencies (CUDA)
├── .gitignore                                                          # Git ignore patterns
├── Dockerfile                                                          # Container deployment instructions
├── main.py                                                             # Primary application entry point
├── app.py                                                              # Web/API application entry point
└── setup.py                                                            # Package installation script
```

---

*`This documentation provides a comprehensive overview of the multimodal emotion recognition system developed for the MELD dataset. For detailed implementation information, please refer to the source code and configuration files in the repository.`*