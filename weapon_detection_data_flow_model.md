```mermaid
flowchart TD
    %% Main Data Flow
    Input[Input Image File] --> Load[Load Image with OpenCV]
    Load --> Preprocess[Preprocess Image]
    
    %% Preprocessing Branch
    Preprocess --> |For PyTorch| PrePyTorch[Convert to RGB, Resize, Normalize, ToTensor]
    Preprocess --> |For Keras| PreKeras[Resize, Normalize to [0,1]]
    Preprocess --> |For YOLO| PreYOLO[Direct Input]
    
    %% Model Inference
    PrePyTorch --> PyTorchModel[PyTorch CNN Model]
    PrePyTorch --> KaggleModel[Kaggle Model]
    PreKeras --> KerasModel[Keras Model]
    PreYOLO --> YOLOv5Model[YOLOv5s Model]
    PreYOLO --> YOLOv8Model[YOLOv8s Model]
    PreYOLO --> YOLOv11Model[YOLOv11 Model]
    
    %% Model Results
    PyTorchModel --> PyTorchResults[PyTorch Raw Results]
    KaggleModel --> KaggleResults[Kaggle Raw Results]
    KerasModel --> KerasResults[Keras Raw Results]
    YOLOv5Model --> YOLOv5Results[YOLOv5 Raw Results]
    YOLOv8Model --> YOLOv8Results[YOLOv8 Raw Results]
    YOLOv11Model --> YOLOv11Results[YOLOv11 Raw Results]
    
    %% Result Processing
    PyTorchResults --> ProcessPyTorch[Process PyTorch Results]
    KaggleResults --> ProcessKaggle[Process Kaggle Results]
    KerasResults --> ProcessKeras[Process Keras Results]
    YOLOv5Results --> ProcessYOLOv5[Process YOLOv5 Results]
    YOLOv8Results --> ProcessYOLOv8[Process YOLOv8 Results]
    YOLOv11Results --> ProcessYOLOv11[Process YOLOv11 Results]
    
    %% Standardized Results
    ProcessPyTorch --> StandardPyTorch[Standardized PyTorch Results]
    ProcessKaggle --> StandardKaggle[Standardized Kaggle Results]
    ProcessKeras --> StandardKeras[Standardized Keras Results]
    ProcessYOLOv5 --> StandardYOLOv5[Standardized YOLOv5 Results]
    ProcessYOLOv8 --> StandardYOLOv8[Standardized YOLOv8 Results]
    ProcessYOLOv11 --> StandardYOLOv11[Standardized YOLOv11 Results]
    
    %% Filtering
    StandardPyTorch --> FilterPyTorch[Filter Non-weapon Detections]
    StandardKaggle --> FilterKaggle[Filter Non-weapon Detections]
    StandardKeras --> FilterKeras[Filter Non-weapon Detections]
    StandardYOLOv5 --> FilterYOLOv5[Filter Non-weapon Detections]
    StandardYOLOv8 --> FilterYOLOv8[Filter Non-weapon Detections]
    StandardYOLOv11 --> FilterYOLOv11[Filter Non-weapon Detections]
    
    %% Model Agreement
    FilterPyTorch --> Agreement[Model Agreement Calculation]
    FilterKaggle --> Agreement
    FilterKeras --> Agreement
    FilterYOLOv5 --> Agreement
    FilterYOLOv8 --> Agreement
    FilterYOLOv11 --> Agreement
    
    %% Final Results
    Agreement --> Compile[Result Compilation]
    Compile --> Log[Log Results to JSON]
    Log --> Return[Return Final Results]
    
    %% Subgraphs for Detail
    subgraph "Model Initialization (Pre-run)"
        direction TB
        InitYOLOv5[Load YOLOv5s]
        InitYOLOv8[Load YOLOv8s]
        InitYOLOv11[Load YOLOv11]
        InitPyTorch[Load PyTorch CNN]
        InitKaggle[Load Kaggle Model]
        InitKeras[Load Keras Model]
    end
    
    subgraph "Result Processing Details"
        direction TB
        RP1[Convert Bounding Boxes]
        RP2[Normalize Confidence Scores]
        RP3[Map Class Indices to Names]
        RP4[Standardize Output Format]
    end
    
    subgraph "Model Agreement Details"
        direction TB
        MA1[Count Detecting Models]
        MA2[Calculate Agreement Percentage]
        MA3[Determine Majority Agreement]
        MA4[Calculate Confidence Metrics]
    end
    
    subgraph "Result Compilation Details"
        direction TB
        RC1[Filename]
        RC2[Model Statistics]
        RC3[Agreement Statistics]
        RC4[Weapon Detection Flag]
        RC5[Detailed Detections]
        RC6[Timestamp]
    end
    
    %% Data Storage
    Log --> |Store| DetectionLogs[detection_logs.json]
    DetectionLogs --> |Read| HistoricalData[Historical Detection Data]
``` 