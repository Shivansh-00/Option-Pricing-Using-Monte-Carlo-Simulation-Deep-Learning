# Architecture & System Diagrams

### Intelligent Option Pricing & Risk Analytics Platform

> **Project:** Option Pricing Using Monte Carlo Simulation & Deep Learning  
> **Tech Stack:** Python · FastAPI · PyTorch · HTML/CSS/JS · SQLite · Kubernetes  

---

# Part 1 — Architecture Choice

## ✅ Chosen Architecture: Monolithic Architecture (Modular Monolith)

### Why Monolithic?

| # | Reason | Explanation |
|---|--------|-------------|
| 1 | **Single Deployment Unit** | The entire backend (Pricing Engine, ML, DL, Auth, Explainability, RAG) runs as one FastAPI application on one server process. |
| 2 | **Shared Data Layer** | All modules share a single SQLite database for authentication and in-memory model objects — no inter-service messaging or distributed storage needed. |
| 3 | **Tight Module Coupling** | The DL layer directly calls the Monte Carlo engine for residual learning; the Explainability module directly imports ML/DL models for SHAP analysis. These are in-process function calls, not network requests. |
| 4 | **Simplicity** | No need for service discovery, API gateways, or container orchestration complexity during development. A single `uvicorn app.main:app --reload` starts everything. |
| 5 | **Low Latency** | In-process function calls between Pricing → ML → DL → Explainability are sub-millisecond, whereas microservices would add network overhead for each inter-service call. |
| 6 | **Team Size** | A single developer/small team can manage one codebase more efficiently than 5+ separate microservices. |

### How It Works in This Project

```mermaid
flowchart TB
    subgraph SERVER["FastAPI Monolithic Server (uvicorn :8000)"]
        direction TB
        subgraph ROW1[" "]
            direction LR
            A["🔐 Auth Module"]
            B["💰 Pricing Engine"]
            C["🤖 ML Service"]
        end
        subgraph ROW2[" "]
            direction LR
            D["🧠 DL Service"]
            E["📊 Greeks Engine"]
            F["💡 Explainability"]
        end
        subgraph ROW3[" "]
            direction LR
            G["🌊 Vol Surface"]
            H["📋 Portfolio Risk"]
            I["🌐 Market Data"]
        end
        DB[("SQLite DB")] 
        Models[("Model Files .pt")]
    end
    FE["🖥️ Frontend (HTML/JS)\nServed by FastAPI"] -->|"HTTP / WebSocket"| SERVER
```

**Key characteristics:**

| # | Feature | Detail |
|---|---------|--------|
| 1 | **Single Process** | `uvicorn` starts FastAPI, loading all modules into one Python process |
| 2 | **Shared Memory** | ML/DL models loaded once, shared across all request handlers |
| 3 | **Modular Internals** | Clean separation: `pricing.py`, `ml.py`, `dl.py`, `auth.py`, `explain.py`, etc. |
| 4 | **Static Frontend** | HTML/CSS/JS served as static files from the same FastAPI server |
| 5 | **Embedded DB** | SQLite for authentication — no external DB server needed |

---

# Part 2 — System Diagrams

---

## Diagram 1: Use Case Diagram

> **Purpose:** Shows who uses the system (actors) and what actions they can perform.

```mermaid
flowchart LR
    User(("👤 Trader\nAnalyst"))
    Admin(("👨‍💻 Quant\nDeveloper"))
    Market(("🌐 Market\nData API"))

    subgraph OptionQuant["🏛️ OptionQuant Platform"]
        direction TB
        subgraph Auth["Authentication"]
            UC1["Register"]
            UC2["Login / Logout"]
        end
        subgraph Pricing["Option Pricing"]
            UC3["Black-Scholes\nPricing"]
            UC4["Monte Carlo\nSimulation"]
            UC5["View Greeks"]
            UC6["Jump Diffusion\nPricing"]
            UC7["View MC Paths\n& Convergence"]
        end
        subgraph AI["AI / ML / DL"]
            UC8["ML Predictions\n(IV, Regime)"]
            UC9["DL Forecasts\n(LSTM, Transformer)"]
            UC10["Train DL Models"]
            UC11["GPU Monte Carlo\nBenchmark"]
        end
        subgraph Analytics["Analytics & Insights"]
            UC12["Vol Surface\nGeneration"]
            UC13["Portfolio Risk\nAnalysis"]
            UC14["Arbitrage\nDetection"]
            UC15["AI Explanation\n(SHAP + RAG)"]
        end
        subgraph Realtime["Real-Time"]
            UC16["WebSocket\nLive Updates"]
            UC17["Comparison\nCharts"]
        end
    end

    User --> UC1
    User --> UC2
    User --> UC3
    User --> UC4
    User --> UC5
    User --> UC6
    User --> UC7
    User --> UC8
    User --> UC9
    User --> UC12
    User --> UC13
    User --> UC14
    User --> UC15
    User --> UC16
    User --> UC17

    Admin --> UC10
    Admin --> UC11
    Admin --> UC3
    Admin --> UC9

    Market -.->|"Stock Data\nVIX, Rates"| UC4
    Market -.->|"Options Chains"| UC8
```

### Actors

| Actor | Role | Use Cases |
|-------|------|-----------|
| **👤 Trader / Analyst** | Authenticated end-user | Price options, view Greeks, run ML/DL predictions, monitor portfolio, view charts |
| **👨‍💻 Quant Developer** | Admin / power user | Train DL models, run GPU benchmarks, administer platform |
| **🌐 Market Data API** | External data source | Supply stock prices, VIX index, treasury rates, options chains |

---

## Diagram 2: Class Diagram

> **Purpose:** Shows the main classes, their attributes, methods, and relationships.

```mermaid
classDiagram
    class PricingInputs {
        +float spot
        +float strike
        +float maturity
        +float rate
        +float volatility
        +string option_type
        +int steps
        +int paths
    }

    class MCResult {
        +float price
        +float std_error
        +float ci_lower
        +float ci_upper
        +int paths_used
        +int steps_used
        +string variance_reduction
        +float elapsed_ms
        +list convergence
        +list sample_paths
    }

    class MonteCarloEngine {
        +price_european(inputs) MCResult
        +price_with_antithetic(inputs) MCResult
        +price_with_control_variate(inputs) MCResult
        +generate_paths(inputs) ndarray
        +compute_convergence(payoffs) list
    }

    class BlackScholesEngine {
        +price(S, K, T, r, sigma, type) float
        +delta() float
        +gamma() float
        +vega() float
        +theta() float
        +rho() float
    }

    class GreeksEngine {
        +compute_all_greeks(inputs) GreeksResponse
        +finite_difference_greeks(inputs) dict
    }

    class MLService {
        +predict_iv(features) float
        +detect_regime(features) string
        +estimate_mispricing(features) float
        +get_feature_importance() dict
    }

    class DLService {
        +LSTMModel lstm_model
        +TransformerModel tf_model
        +forecast(inputs) DLForecastResponse
        +train(request) TrainResult
        +compute_residual(mc, dl) float
    }

    class LSTMModel {
        +LSTM lstm_layer
        +Linear fc_layer
        +forward(x) Tensor
    }

    class TransformerModel {
        +TransformerEncoder encoder
        +Linear fc_layer
        +forward(x) Tensor
    }

    class AuthService {
        +hash_password(pwd) string
        +verify_password(pwd, hash) bool
        +create_access_token(user) string
        +create_refresh_token(user) string
        +get_current_user(token) User
    }

    class ExplainabilityService {
        +shap_explain(model, data) dict
        +rag_query(question) string
        +generate_explanation(result) string
    }

    class VolSurfaceEngine {
        +generate_surface(params) ndarray
        +train_transformer(data) Model
        +interpolate(strike, expiry) float
    }

    class PortfolioRiskEngine {
        +compute_var(portfolio) float
        +stress_test(portfolio, scenarios) dict
        +position_greeks(positions) dict
    }

    class MarketDataService {
        +get_stock_data(ticker) dict
        +get_options_chain(ticker) dict
        +get_vix() float
        +get_treasury_rates() dict
    }

    PricingInputs --> MonteCarloEngine : feeds
    PricingInputs --> BlackScholesEngine : feeds
    MonteCarloEngine --> MCResult : produces
    BlackScholesEngine --> GreeksEngine : computes
    MonteCarloEngine --> GreeksEngine : computes
    DLService *-- LSTMModel : contains
    DLService *-- TransformerModel : contains
    DLService --> MonteCarloEngine : residual learning
    MLService --> ExplainabilityService : SHAP input
    DLService --> ExplainabilityService : SHAP input
    MarketDataService --> MLService : provides features
    MarketDataService --> DLService : provides features
    VolSurfaceEngine --> TransformerModel : uses
```

### Key Relationships

| From | To | Relationship |
|------|----|-------------|
| `DLService` | `MonteCarloEngine` | Residual learning — DL corrects MC base price |
| `DLService` | `LSTMModel`, `TransformerModel` | Composition — owns both neural network models |
| `MLService`, `DLService` | `ExplainabilityService` | SHAP feature attribution on trained models |
| `MarketDataService` | `MLService`, `DLService` | Provides real-time features for inference |

---

## Diagram 3: Data Flow Diagram (DFD)

### Level 0 — Context Diagram

> **Purpose:** Bird's-eye view of the entire system as a single process.

```mermaid
flowchart LR
    User["👤 User\n(Trader / Analyst)"]
    Market["🌐 Market Data\nProvider"]

    System["🏛️ OptionQuant\nPlatform"]

    User -->|"Login Credentials\nPricing Parameters\nModel Queries"| System
    System -->|"Option Prices\nGreeks & Risk\nForecasts\nAI Explanations"| User
    Market -->|"Stock OHLCV\nVIX Index\nTreasury Rates\nOptions Chains"| System
    System -->|"API Data\nRequests"| Market
```

### Level 1 — Major Processes

> **Purpose:** Shows the 8 major processes and how data flows between them.

```mermaid
flowchart TB
    User["👤 User"]
    Market["🌐 Market Data"]

    User -->|"credentials"| P1["1.0 Authentication\n(JWT)"]
    P1 -->|"JWT Token"| User
    P1 <-->|"read/write"| D1[("SQLite\nUser DB")]

    User -->|"S, K, T, r, sigma"| P2["2.0 Pricing\nEngine"]
    P2 -->|"Prices, Greeks\nMC Paths"| User

    Market -->|"Stock, VIX\nRates Data"| P3["3.0 Data\nProcessing"]
    P3 -->|"Cleaned\nFeatures"| P4["4.0 ML\nPrediction"]
    P3 -->|"Feature\nSequences"| P5["5.0 DL\nForecasting"]

    P4 -->|"IV, Regime\nMispricing"| User
    P5 -->|"LSTM/TF\nForecasts"| User

    P2 <-->|"MC Price\nResidual"| P5

    P4 -->|"Model\n+ Data"| P6["6.0 Explainability\n(SHAP + RAG)"]
    P5 -->|"Model\n+ Data"| P6
    P6 -->|"AI\nExplanation"| User

    P3 -->|"Vol Data"| P7["7.0 Vol Surface\nGeneration"]
    P7 -->|"3D Surface"| User

    P2 -->|"Prices"| P8["8.0 Portfolio\nRisk"]
    P8 -->|"VaR, Stress\nTest Results"| User

    D2[("Model Files\nlstm.pt\ntransformer.pt")] <-->|"Load/Save\nWeights"| P5
```

### Level 2 — Pricing Engine Detail

> **Purpose:** Expands Process 2.0 to show internal data flow within the pricing engine.

```mermaid
flowchart TB
    Input["📥 Pricing Parameters\nSpot, Strike, T, r, sigma, type"]
    Input --> Validate{"✅ Validate\nInputs"}

    Validate -->|"Invalid"| Error["❌ 422\nValidation Error"]
    Validate -->|"Valid"| BS["2.1 Black-Scholes\nAnalytical Pricing"]
    Validate -->|"Valid"| MC["2.2 Monte Carlo\nSimulation"]

    MC --> GBM["2.2.1 Standard\nGBM Paths"]
    MC --> AV["2.2.2 Antithetic\nVariates"]
    MC --> CV["2.2.3 Control\nVariate"]
    MC --> Heston["2.2.4 Heston\nStochastic Vol"]

    GBM --> Payoff["2.3 Payoff\nComputation\nmax(S-K, 0)"]
    AV --> Payoff
    CV --> Payoff
    Heston --> Payoff

    Payoff --> Stats["2.4 Statistics\nMean, CI, Std Error"]
    BS --> Greeks["2.5 Greeks\nDelta Gamma Vega\nTheta Rho"]
    Stats --> Greeks

    Stats --> Conv["2.6 Convergence\nAnalysis"]
    Stats --> Paths["2.7 Sample Path\nExtraction"]

    Greeks --> Response["📤 Pricing Response\nPrice + Greeks + Paths\n+ Convergence"]
    Conv --> Response
    Paths --> Response
```

---

## Diagram 4: Component Diagram

> **Purpose:** Shows the major software components/modules and how they are layered.

```mermaid
flowchart TB
    subgraph FE["🖥️ FRONTEND LAYER"]
        direction LR
        UI["index.html\nDashboard"]
        LoginUI["login.html\nAuth Page"]
        AppJS["app.js\nController"]
        Charts["charts/\nMC Paths\nGreeks\nComparison"]
        Styles["CSS\nstyles.css\npremium.css"]
    end

    subgraph ROUTES["🔌 API ROUTES (FastAPI)"]
        direction LR
        R1["auth_routes"]
        R2["pricing_routes\npricing_api"]
        R3["ml_routes"]
        R4["dl_routes"]
        R5["explain_routes"]
        R6["market_routes"]
        R7["quant_routes"]
        R8["ws_routes"]
    end

    subgraph CORE["⚙️ CORE BUSINESS LOGIC"]
        direction LR
        C1["pricing.py\nMC + BS"]
        C2["greeks.py"]
        C3["ml.py\nML Models"]
        C4["dl.py\nLSTM + TF"]
        C5["train_dl.py"]
        C6["vol_engine.py"]
        C7["jump_diffusion.py"]
        C8["stochastic_vol.py"]
        C9["gpu_monte_carlo.py"]
        C10["variance_reduction.py"]
    end

    subgraph INTEL["🧠 AI & ANALYTICS"]
        direction LR
        I1["shap_explain.py"]
        I2["rag/ module"]
        I3["regime.py"]
        I4["mispricing.py"]
        I5["portfolio_risk.py"]
        I6["rl_hedging.py"]
        I7["uncertainty.py"]
    end

    subgraph INFRA["🔧 INFRASTRUCTURE"]
        direction LR
        INF1["auth.py\nJWT + SQLite"]
        INF2["config.py"]
        INF3["schemas.py\nPydantic"]
        INF4["prometheus_metrics.py"]
        INF5["websocket_manager.py"]
        INF6["market_data.py"]
    end

    subgraph DATA["💾 DATA LAYER"]
        direction LR
        DB[("SQLite\nUsers")]
        Models[("models/\nLSTM.pt\nTransformer.pt")]
        KB[("rag/\nknowledge_base")]
    end

    FE ==>|"HTTP / WebSocket"| ROUTES
    ROUTES ==> CORE
    ROUTES ==> INTEL
    ROUTES ==> INFRA
    CORE ==> DATA
    INTEL ==> DATA
    INFRA ==> DATA
```

### Layer Summary

| Layer | Files | Responsibility |
|-------|-------|---------------|
| **Frontend** | `index.html`, `login.html`, `app.js`, `charts/` | User interface, visualization, input forms |
| **API Routes** | `*_routes.py`, `pricing_api.py` | HTTP endpoint definitions, request validation |
| **Core Logic** | `pricing.py`, `greeks.py`, `ml.py`, `dl.py`, etc. | Business algorithms (BS, MC, LSTM, Transformer) |
| **AI & Analytics** | `shap_explain.py`, `rag/`, `regime.py`, etc. | SHAP explainability, RAG, regime detection, portfolio risk |
| **Infrastructure** | `auth.py`, `config.py`, `schemas.py`, etc. | Auth, config, Pydantic schemas, Prometheus, WebSocket |
| **Data** | SQLite DB, `.pt` files, RAG knowledge base | Persistent storage for users, models, and documents |

---

## Diagram 5: Sequence Diagrams

### 5.1 — User Login Flow

> **Purpose:** Step-by-step interaction when a user logs in.

```mermaid
sequenceDiagram
    actor User as 👤 User
    participant FE as 🖥️ Frontend<br/>(login.html)
    participant API as 🔌 FastAPI<br/>Server
    participant Auth as 🔐 Auth<br/>Module
    participant DB as 💾 SQLite<br/>Database

    User->>FE: Enter username & password
    FE->>API: POST /api/v1/auth/login
    API->>Auth: authenticate(username, password)
    Auth->>DB: SELECT user WHERE username=?
    DB-->>Auth: User record (salt$hash)
    Auth->>Auth: PBKDF2 verify password

    alt ✅ Password Valid
        Auth->>Auth: Generate JWT access token (30min)
        Auth->>Auth: Generate JWT refresh token (7d)
        Auth-->>API: TokenPair {access, refresh}
        API-->>FE: 200 OK + JSON tokens
        FE->>FE: localStorage.setItem(token)
        FE-->>User: Redirect to Dashboard ✅
    else ❌ Password Invalid
        Auth-->>API: AuthenticationError
        API-->>FE: 401 Unauthorized
        FE-->>User: Display error message ❌
    end
```

### 5.2 — Monte Carlo Option Pricing

> **Purpose:** Step-by-step interaction for pricing an option via Monte Carlo.

```mermaid
sequenceDiagram
    actor User as 👤 Trader
    participant FE as 🖥️ Dashboard
    participant API as 🔌 FastAPI
    participant MC as 🎲 Monte Carlo<br/>Engine
    participant BS as 📐 Black-Scholes<br/>Engine
    participant GK as 📊 Greeks<br/>Engine

    User->>FE: Set S=100, K=105, T=1, r=5%, σ=20%
    FE->>API: POST /api/v1/pricing/mc/detailed
    Note over API: Verify JWT Token ✅

    rect rgb(240, 248, 255)
        Note over MC: Monte Carlo Simulation
        API->>MC: price_european(inputs)
        MC->>MC: Generate 20,000 GBM paths
        MC->>MC: Apply antithetic variates
        MC->>MC: Compute payoffs: max(S_T - K, 0)
        MC->>MC: Discount: e^(-rT) × mean(payoffs)
        MC->>MC: Calculate 95% CI & std error
        MC->>MC: Extract convergence curve
        MC->>MC: Sample 50 display paths
        MC-->>API: MCResult
    end

    rect rgb(255, 248, 240)
        Note over BS: Analytical Pricing
        API->>BS: price(S, K, T, r, σ)
        BS-->>API: BS price = 8.02
    end

    rect rgb(240, 255, 240)
        Note over GK: Risk Sensitivities
        API->>GK: compute_all_greeks(inputs)
        GK-->>API: Δ=0.54, Γ=0.02, ν=0.38, Θ=-0.01, ρ=0.45
    end

    API-->>FE: Complete response JSON
    FE->>FE: Render MC path visualization
    FE->>FE: Render convergence chart
    FE->>FE: Display BS vs MC comparison
    FE->>FE: Show Greeks dashboard
    FE-->>User: Interactive pricing view
```

### 5.3 — DL Forecast with Residual Learning

> **Purpose:** Shows how the Deep Learning service combines LSTM + Transformer + Monte Carlo for hybrid pricing.

```mermaid
sequenceDiagram
    actor User as 👤 User
    participant FE as 🖥️ Frontend
    participant API as 🔌 FastAPI
    participant DL as 🧠 DL Service
    participant LSTM as 📈 LSTM Model
    participant TF as 🔮 Transformer
    participant MC as 🎲 Monte Carlo
    participant SHAP as 💡 SHAP

    User->>FE: Request DL forecast
    FE->>API: POST /api/v1/dl/forecast

    rect rgb(245, 240, 255)
        Note over DL: Deep Learning Inference
        API->>DL: forecast(inputs)
        DL->>MC: mc_price(inputs)
        MC-->>DL: base_price = 8.15

        DL->>LSTM: forward(feature_sequence)
        LSTM-->>DL: lstm_pred = 8.42

        DL->>TF: forward(feature_sequence)
        TF-->>DL: sentiment = bullish

        DL->>DL: residual = lstm_pred - base_price
        DL->>DL: final = base_price + residual
        Note over DL: Hybrid Residual Learning
    end

    DL-->>API: DLForecastResponse
    API-->>FE: {price, residual, benchmarks}
    FE-->>User: Forecast with MC vs DL comparison
```

---

## Diagram 6: Deployment Diagrams

### 6.1 — Local Development

> **Purpose:** How the system runs on a developer's machine.

```mermaid
flowchart TB
    subgraph DEV["💻 LOCAL DEVELOPMENT (localhost)"]
        Browser["🌐 Chrome Browser\nhttp://localhost:8000"]

        subgraph Uvicorn["Python Process — uvicorn"]
            FastAPI["FastAPI App\nPort 8000"]
            Static["Static File Server\n(HTML/CSS/JS)"]
        end

        subgraph LocalStore["Local File System"]
            SQLite[("SQLite DB\noptiquant_users.db")]
            PTFiles[("Model Weights\nlstm_model.pt\ntransformer_model.pt")]
            RAGFiles[("RAG Knowledge Base\nFinancial Documents")]
        end

        Browser -->|"HTTP Requests\nWebSocket"| FastAPI
        Browser -->|"Static Assets"| Static
        FastAPI --> SQLite
        FastAPI --> PTFiles
        FastAPI --> RAGFiles
    end

    ExtAPI["🌐 External APIs\nYahoo Finance\nTreasury Rates"]
    FastAPI <-->|"Market Data\nHTTP Calls"| ExtAPI
```

### 6.2 — Production Kubernetes Deployment

> **Purpose:** How the system is deployed in a production Kubernetes cluster.

```mermaid
flowchart TB
    Users["👥 Users (HTTPS)"]

    subgraph CLOUD["☁️ PRODUCTION KUBERNETES CLUSTER"]
        subgraph INGRESS["🌐 Ingress Layer"]
            IG["Nginx Ingress\nHTTPS Termination\nTLS Certificate"]
        end

        subgraph SERVICES["🔌 Service Mesh"]
            BESvc["Backend Service\nClusterIP :8000"]
            FESvc["Frontend Service\nClusterIP :80"]
        end

        subgraph BACKEND["⚙️ Backend Pods (2-5 replicas)"]
            BE1["FastAPI Container\nPod 1"]
            BE2["FastAPI Container\nPod 2"]
            BE3["FastAPI Container\nPod N..."]
        end

        subgraph FRONTEND["🖥️ Frontend Pods (2 replicas)"]
            FE1["Nginx + Static\nPod 1"]
            FE2["Nginx + Static\nPod 2"]
        end

        subgraph CACHE["⚡ Cache"]
            Redis["Redis\nPort 6379\nSession Cache"]
        end

        subgraph AUTOSCALE["📈 Auto Scaling"]
            HPA["HPA\nCPU > 70%\nMin: 2, Max: 5"]
        end

        subgraph MON["📊 Monitoring Stack"]
            Prom["Prometheus\nMetrics Scraper"]
            Graf["Grafana\nDashboards"]
            Alert["AlertManager\nSlack + Email"]
        end
    end

    Users --> IG
    IG --> FESvc --> FRONTEND
    IG --> BESvc --> BACKEND
    BACKEND --> Redis
    HPA -.->|"scale"| BACKEND
    BACKEND --> Prom
    Prom --> Graf
    Prom --> Alert

    ExtMkt["🌐 Market APIs"]
    BACKEND <-->|"Data Feeds"| ExtMkt
```

### 6.3 — CI/CD Pipeline

> **Purpose:** Automated build, test, and deployment pipeline.

```mermaid
flowchart LR
    Dev["👨‍💻 Developer"]
    Dev -->|"git push"| GH["📦 GitHub\nRepository"]
    GH -->|"webhook\ntrigger"| CI["🔄 CI Pipeline\nGitHub Actions"]

    subgraph Pipeline["CI/CD Pipeline"]
        direction TB
        CI --> Lint["🔍 Lint +\nType Check"]
        Lint --> Test["🧪 Run Tests\n48 Endpoints"]
        Test --> Build["🐳 Docker\nBuild & Push"]
        Build --> Registry["📋 Container\nRegistry"]
    end

    Registry --> Deploy["🚀 K8s Deploy\nkubectl apply"]
    Deploy --> Cluster["☁️ Production\nCluster"]
    Cluster --> Monitor["📊 Prometheus\n+ Grafana"]
```

---

# Summary

| # | Diagram | Type | Purpose | Key Details |
|---|---------|------|---------|-------------|
| 1 | **Use Case** | Behavioral | Who uses the system and what they do | 3 actors, 17 use cases across 5 groups |
| 2 | **Class** | Structural | Internal code structure and relationships | 14 classes with attributes, methods, and associations |
| 3 | **DFD** | Data Flow | How data moves through the system | 3 levels: Context → Major Processes → Pricing Detail |
| 4 | **Component** | Structural | Module organization and dependencies | 6 layers: Frontend → Routes → Core → AI → Infra → Data |
| 5 | **Sequence** | Behavioral | Step-by-step interactions over time | 3 flows: Login, MC Pricing, DL Residual Forecast |
| 6 | **Deployment** | Physical | Where the system runs | Local dev, K8s production, CI/CD pipeline |
