from fastapi import FastAPI, Query
from .client.rq_client import queue
from .queues.worker import process_query


app = FastAPI()

@app.get('/')
def root():
    return {"status": 'Sever is up and running '}

@app.post('/chat')
def chat(
    query : str = Query(..., description="This is the query of user")
    
):
    job = queue.enqueue(process_query, query)
    return {"status": "queued", "job_id": job.id}

@app.get('/job-status')
def get_result(
    job_id : str = Query(..., description="Job ID")
):
    job = queue.fetch_job(job_id = job_id)
    if job is None:
        return {"error": "Job not found"}
    result = job.return_value()
    return {"result":result}



#python -m rag_queue.main 