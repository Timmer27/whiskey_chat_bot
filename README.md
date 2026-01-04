# Whisky Chat Bot (Q1/Q2/Q3)

whisky_qnas.csv의 데이터를 Qdrant Vector DB에 적재한 후 GROK LLM API를 사용하여 RAG 챗봇을 구축하였습니다.

## 0) 로컬 실행 방법
### Docker
1. 아래와 같이 `.env` 생성
```bash
# Qdrant
QDRANT_URL=http://localhost:6333

QDRANT_COLLECTION=whisky_reviews

# GROQ_API
GROQ_API_KEY=본인의 KEY
GROQ_BASE_URL=https://api.groq.com/openai/v1

# Embedding
EMBED_MODEL=intfloat/multilingual-e5-small

BACKEND_URL=http://localhost:8000
```
2. 아래 코드 실행
```bash
docker compose up --build -d  
```
