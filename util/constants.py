TOP_N = 1000

GEMINI_2_FLASH_THINKING = 'gemini-2.0-flash-thinking-exp-01-21' # VertexAI
GEMINI_2_5_PRO = 'gemini-2.5-pro-preview-05-06' # VertexAI
GEMINI_API_KEY = ''

GPT_4O_MINI = 'gpt-4o-mini-2024-07-18' # OpenAI
O4_MINI = 'o4-mini-2025-04-16' # OpenAI
OPENAI_API_KEY = ''

DEEPINFRA_DEEPSEEK_R1 = 'deepseek-ai/DeepSeek-R1' # DeepInfra
DEEPINFRA_BASE_URL = 'https://api.deepinfra.com/v1/openai'
DEEPINFRA_API_KEY = ''

section_subsection_headers = {
    '## Model Details': [
        '### Person or organization developing model:',
        '### Model date:',
        '### Model version:',
        '### Model type:',
        '### Training details:',
        '### Paper or other resource for more information:',
        '### Citation details:',
        '### License:',
        '### Contact:'
    ],
    '## Intended Use': [
        '### Primary intended uses:',
        '### Primary intended users:',
        '### Out-of-scope uses:'
    ],
    '## How to Use': [],
    '## Factors': [
        '### Relevant factors:',
        '### Evaluation factors:'
    ],
    '## Metrics': [
        '### Model performance measures:',
        '### Decision thresholds:',
        '### Variation approaches:'
    ],
    '## Evaluation Data': [
        '### Datasets:',
        '### Motivation:',
        '### Preprocessing:'
    ],
    '## Training Data': [
        '### Datasets:',
        '### Motivation:',
        '### Preprocessing:'
    ],
    '## Quantitative Analyses': [
        '### Unitary results:',
        '### Intersectional results:'
    ],
    '## Memory or Hardware Requirements': [
        '### Loading Requirements:',
        '### Deploying Requirements:',
        '### Training or Fine-tuning Requirements:'
    ],
    '## Ethical Considerations': [],
    '## Caveats and Recommendations': [
        '### Caveats:',
        '### Recommendations:'
    ]
}
