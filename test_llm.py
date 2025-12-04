from rag.local_llm import call_llm

if __name__ == "__main__":
    q = "Объясни, что такое модель OSI в сетях, кратко."
    ans = call_llm(q)
    print(ans)
