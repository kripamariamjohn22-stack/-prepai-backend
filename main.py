from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os
from dotenv import load_dotenv
import json
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()

# This allows your React app to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://prepai-frontend-eight.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class JobRequest(BaseModel):
    job_description: str

class AnswerRequest(BaseModel):
    question: str
    answer: str
    job_description: str

@app.post("/generate-questions")
async def generate_questions(req: JobRequest):
    prompt = f"""
    Based on this job description, generate exactly 10 interview questions.
    Mix of technical and behavioral questions.
    Return ONLY a JSON array of strings, no other text.
    Example: ["Question 1", "Question 2", ...]
    
    Job Description: {req.job_description[:3000]}
    """
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    
    content = response.choices[0].message.content
    questions = json.loads(content)
    return {"questions": questions}

@app.post("/evaluate-answer")
async def evaluate_answer(req: AnswerRequest):
    prompt = f"""
    Job Description: {req.job_description[:1000]}
    Interview Question: {req.question}
    Candidate's Answer: {req.answer}
    
    Evaluate this answer. Return ONLY JSON in this exact format:
    {{
        "score": <number 1-10>,
        "feedback": "<2-3 sentences of specific feedback>",
        "better_answer": "<a stronger version of their answer in 2-3 sentences>"
    }}
    """
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    
    content = response.choices[0].message.content
    result = json.loads(content)
    return result

@app.get("/")
def root():
    return {"status": "PrepAI backend is running!"}
class KnowledgeRequest(BaseModel):
    job_description: str

class RoadmapRequest(BaseModel):
    job_description: str
    score: int  # out of 5
    weak_topics: list[str]

@app.post("/knowledge-check")
async def knowledge_check(req: KnowledgeRequest):
    prompt = f"""
    Based on this job description, create exactly 3 multiple choice questions 
    to assess the candidate's current knowledge level.
    
    Return ONLY a JSON array in this exact format, no other text:
    [
        {{
            "question": "What is...",
            "options": ["A) option1", "B) option2", "C) option3", "D) option4"],
            "correct": "A",
            "topic": "topic name"
        }}
    ]
    
    IMPORTANT: Never ask questions whose answers are directly stated in the job description. 
    Ask questions that test ACTUAL technical knowledge and coding ability.
    Bad: "What qualifications does Google require?"
    Good: "Given an array of integers, what data structure gives O(1) average lookup time?"
    
    Job Description: {req.job_description[:2000]}
    """
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    
    content = response.choices[0].message.content
    # Clean JSON if wrapped in markdown
    content = content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    
    questions = json.loads(content.strip())
    return {"questions": questions}

@app.post("/generate-roadmap")
async def generate_roadmap(req: RoadmapRequest):
    prompt = f"""
    A candidate is preparing for this role: {req.job_description[:1000]}
    
    They scored {req.score}/5 on a knowledge check.
    Their weak areas are: {', '.join(req.weak_topics) if req.weak_topics else 'general concepts'}
    
    Create a personalized learning roadmap. Return ONLY JSON in this exact format:
    {{
        "level": "Beginner/Intermediate/Advanced",
        "message": "encouraging 1-2 sentence message based on their score",
        "topics": [
            {{
                "name": "Topic Name",
                "priority": "High/Medium/Low",
                "description": "1 sentence why this matters for the role",
                "resources": "what to study - e.g. 'Practice on LeetCode, read docs'"
            }}
        ]
    }}
    
    Give exactly 6 topics, sorted by priority.
    """
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    
    content = response.choices[0].message.content
    content = content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    
    roadmap = json.loads(content.strip())
    return roadmap