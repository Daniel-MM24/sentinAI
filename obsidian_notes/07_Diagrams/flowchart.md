# SentinAI Architecture Flowchart

```mermaid
flowchart TB
    %% ============================================================
    %% SECTION 1: DATA SOURCES (Horizontal Flow)
    %% ============================================================
    subgraph S1["1. Data Sources"]
        direction LR
        D1[" M-Pesa<br/>Transactions"] --> D2["Core Banking<br/>Events"] --> D3["Regulatory<br/>Documents"]
    end

    %% ============================================================
    %% SECTION 2: INGESTION & PROCESSING (Horizontal Flow)
    %% ============================================================
    subgraph S2["2. Ingestion & Processing"]
        direction LR
        P1["PostgreSQL<br/>Queue + SKIP LOCKED"] --> P2["ETL Pipeline<br/>Bronze → Silver → Gold"]
    end

    %% ============================================================
    %% SECTION 3: STORAGE (Horizontal Flow)
    %% ============================================================
    subgraph S3["3. Storage Layer"]
        direction LR
        ST1["Bronze<br/>Raw JSON"] --> ST2["Silver<br/>Clean Parquet"] --> ST3["Gold<br/>Features"]
    end

    %% ============================================================
    %% SECTION 4: AI AGENTS (Horizontal Flow)
    %% ============================================================
    subgraph S4["4. AI Agents"]
        direction LR
        AG1["Researcher<br/>Finds Rules"] --> AG2["Auditor<br/>Verifies"] --> AG3["Analyst<br/>Reports"]
    end

    %% ============================================================
    %% SECTION 5: VALIDATION (Horizontal Flow)
    %% ============================================================
    subgraph S5["5. Model Validation"]
        direction LR
        V1["SHAP<br/>Attribution"] --> V2["Narrative<br/>Explanation"]
    end

    %% ============================================================
    %% SECTION 6: AUDIT & OVERSIGHT (Horizontal Flow)
    %% ============================================================
    subgraph S6["6. Audit & Human Oversight"]
        direction LR
        AU1["Immutable<br/>Log"] --> AU2["Human<br/>Review"]
    end

    %% ============================================================
    %% VERTICAL CONNECTIONS BETWEEN SECTIONS
    %% ============================================================
    S1 ==> S2 ==> S3 ==> S4 ==> S5 ==> S6

    %% ============================================================
    %% STYLING
    %% ============================================================
    classDef sources fill:#FFE4B5,stroke:#FF8C00,stroke-width:2px,color:#333
    classDef process fill:#DDA0DD,stroke:#8B008B,stroke-width:2px,color:#333
    classDef storage fill:#98FB98,stroke:#228B22,stroke-width:2px,color:#333
    classDef agents fill:#4ECDC4,stroke:#2C3E50,stroke-width:2px,color:#fff
    classDef validate fill:#F0E68C,stroke:#B8860B,stroke-width:2px,color:#333
    classDef audit fill:#FF6B6B,stroke:#C92A2A,stroke-width:2px,color:#fff
    
    class D1,D2,D3 sources
    class P1,P2 process
    class ST1,ST2,ST3 storage
    class AG1,AG2,AG3 agents
    class V1,V2 validate
    class AU1,AU2 audit