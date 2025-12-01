import streamlit as st
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import os

load_dotenv()

# -------- MODEL --------
model = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
    model="mistralai/mistral-7b-instruct:free"
)

# -------- SCHEMAS --------
class SentimentSchema(BaseModel):
    sentiment: Literal["positive", "negative"]

class DiagnosisSchema(BaseModel):
    issue_type: Literal["UX", "Performance", "Bug", "Support", "Other"]
    tone: Literal["angry", "frustrated", "disappointed", "calm"]
    urgency: Literal["low", "medium", "high"]

class ReviewState(TypedDict):
    review: str
    sentiment: str
    diagnosis: dict
    response: str

# -------- WORKFLOW STEPS --------
def find_sentiment(state: ReviewState):
    prompt = f"Tell me the sentiment (positive or negative) of the following review:\n\n{state['review']}"
    structured = model.with_structured_output(SentimentSchema)
    result = structured.invoke(prompt)
    return {"sentiment": result.sentiment}

def check_sentiment(state: ReviewState):
    return "positive_response" if state["sentiment"] == "positive" else "run_diagnosis"

def positive_response(state: ReviewState):
    prompt = f"Reply to this feedback in a warm and thankful tone:\n\n{state['review']}"
    output = model.invoke(prompt).content
    return {"response": output}

def run_diagnosis(state: ReviewState):
    prompt = (
        "Diagnose the negative feedback and return issue_type, tone, and urgency.\n\n"
        f"feedback: {state['review']}"
    )
    structured = model.with_structured_output(DiagnosisSchema)
    result = structured.invoke(prompt)
    return {"diagnosis": result.model_dump()}

def negative_response(state: ReviewState):
    diagnosis = state["diagnosis"]
    prompt = f"""
You are a  support assistant.
The user had a '{diagnosis['issue_type']}' issue, sounded '{diagnosis['tone']}', and marked urgency as '{diagnosis['urgency']}'.
Write an empathetic, helpful response and also say thank you in the end not some user fill regards or anything.
"""
    result = model.invoke(prompt).content
    return {"response": result}

# -------- BUILD GRAPH --------
graph = StateGraph(ReviewState)
graph.add_node("find_sentiment", find_sentiment)
graph.add_node("run_diagnosis", run_diagnosis)
graph.add_node("positive_response", positive_response)
graph.add_node("negative_response", negative_response)

graph.add_edge(START, "find_sentiment")
graph.add_conditional_edges("find_sentiment", check_sentiment)
graph.add_edge("positive_response", END)
graph.add_edge("run_diagnosis", "negative_response")
graph.add_edge("negative_response", END)

workflow = graph.compile()

# -------- STREAMLIT UI --------
st.set_page_config(page_title="Review Analyzer", layout="centered")

st.title("📝 FeedBack Analyzer and Reply  (Diagnostics)")

st.write("Paste a user review below and let AI analyze it!")

review_input = st.text_area("Enter Review:", height=200)

if st.button("Analyze Review"):
    if review_input.strip() == "":
        st.warning("Please enter a review first!")
    else:
        with st.spinner("Analyzing..."):
            initial_state = {"review": review_input}
            result = workflow.invoke(initial_state)

        st.success("Analysis Complete!")

        st.subheader("📌 Sentiment")
        st.write(result.get("sentiment", "-"))

        if result.get("sentiment") == "negative":
            st.subheader("🛠 Diagnosis")
            st.json(result.get("diagnosis"))

        st.subheader("💬 AI Response")
        st.write(result.get("response"))
