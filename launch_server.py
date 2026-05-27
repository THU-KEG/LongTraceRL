import asyncio
import sys
from openai import AsyncOpenAI, APIStatusError
import json
import asyncio
import re
import httpx
import traceback
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import argparse


app = FastAPI()


class GPTModel:
    def __init__(self, base_url, api_key, model_name, enable_thinking):
        super().__init__()
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.AsyncClient(proxy=None),  # disable proxy
        )
        self.model_name = model_name
        self.enable_thinking = enable_thinking
        self.temperature = 0.0
        self.top_p = 0.95
        self.max_tokens = 4096
    
    async def get_resp(self, message_list):
        for i in range(3):
            try:
                thinking_mode = "enabled" if self.enable_thinking else "disabled"
                assert thinking_mode == "disabled"  # NOTE
                chat_completion = await self.client.chat.completions.create(
                    messages=message_list,
                    model=self.model_name,
                    # temperature=self.temperature,
                    # top_p=self.top_p,
                    # max_tokens=self.max_tokens,
                    extra_body={
                        "thinking": {"type": thinking_mode},
                    },
                )
                output = chat_completion.choices[0].message.content
                return output
            except Exception as e:
                if "quota" in str(e).lower():
                    print(f"[FATAL] Insufficient quota, terminating: {e}")
                    sys.exit(1)
                print(f"[LLM Judge Internal Error] Attempt {i+1}/3. Exception: {e}\nTraceback: {traceback.format_exc()}")
                await asyncio.sleep(1)
                continue
        print(f"[LLM Judge Internal Error] All request failed, last exception: {e if 'e' in locals() else ''}")
        return ""


async def get_aa_lcr_reward(response, answer, question):
    if "</think>" in response:  # NOTE: reduce length cost
        response = response.split("</think>")[-1].strip()
        if not response:
            return 0
    
    prompt = f"""Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[correct_answer]: {answer}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as "None" if there is no exact, final answer to extract from the response.

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer "yes" if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer "no" otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.
"""
    messages = [{"role": "user", "content": prompt}]
    
    tries = 0
    judgement = ""
    correctness, extracted_final_answer = None, None
    while tries < 3:
        judgement = await reward_model.get_resp(messages)
        match = re.search(r"(?i)\*{0,2}correct\*{0,2}\s*:\s*(no|yes)", judgement, flags=re.IGNORECASE)
        correctness = match.group(1) if match else None
        match = re.search(r"(?i)\*{0,2}extracted_final_answer\*{0,2}\s*:\s*(.+)", judgement, flags=re.IGNORECASE)
        extracted_final_answer = match.group(1) if match else None
        if correctness and extracted_final_answer:
            correctness = correctness.lower()
            break
        tries += 1
    accuracy = 1.0 if correctness == "yes" else 0

    print(f"========AA-LCR========\nextracted_final_answer: {extracted_final_answer}\ngolden_answer: {answer}\naccuracy: {accuracy}\n======================")
    return accuracy


async def get_outcome_reward(response, answer, extra_info):
    # dataset_name = extra_info.get("dataset_name", "")
    return await get_aa_lcr_reward(response, answer, extra_info["question"])


def normalize_string(text: str) -> str:
    """Basic string normalization."""
    text = text.lower().strip()  # lowercase + strip whitespace
    words = re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]", text)  # divide word and remove punctuation
    text = " ".join(words)
    return text


async def get_rubric_reward(history,rubrics,dataset_name, rubric_use_reasoning_content=None, rubric_use_content=None):
    """Count how many rubric keywords are hit across the entire conversation."""
    print(f"======== {dataset_name} rubric reward ========\nrubrics:", rubrics)
    if len(rubrics) == 0:
        return 0

    # concatenate reasoning_content + content
    full_text = ""
    for conv in history:
        if conv["role"] == "assistant":
            if rubric_use_reasoning_content and "reasoning_content" in conv:
                full_text += "\n\n" + conv["reasoning_content"].strip()
                print("rubric use reasoning content")
            if rubric_use_content and "content" in conv:
                full_text += "\n\n" + conv["content"].strip()
                print("rubric use content")

    if dataset_name == "longrlvr":
        # rubrics is a list of integers (chunk ids), extracted only from the last <useful_chunks>...</useful_chunks>
        matches = re.findall(r"<useful_chunks>(.*?)</useful_chunks>", full_text, re.DOTALL | re.IGNORECASE)
        predicted_chunks = set()
        if matches:
            useful_chunks_text = matches[-1]
            predicted_chunks = set(int(x) for x in re.findall(r"<CHUNK[_\s]?(\d+)>", useful_chunks_text, re.IGNORECASE))
        print(f"\npredicted_chunks: {predicted_chunks}")
        ground_truth_chunks = set(rubrics)
        hits = len(predicted_chunks & ground_truth_chunks)
        precision = hits / len(predicted_chunks) if len(predicted_chunks) > 0 else 0.0
        recall = hits / len(ground_truth_chunks) if len(ground_truth_chunks) > 0 else 0.0
        beta = 2
        f_beta = 0.0
        if precision + recall > 0:
            f_beta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
        print(f"precision: {precision:.4f}, recall: {recall:.4f}, F_{beta}: {f_beta:.4f}\n==========================")
        return f_beta
    else:
        # return await get_rubric_reward_by_llm(full_text, rubrics)
        full_text = normalize_string(full_text)
        hits = 0
        for rubric in rubrics:
            is_contained = normalize_string(rubric) in full_text
            if is_contained:
                print(f"hit!! {rubric}")
            hits += is_contained
        accuracy = hits / len(rubrics)
        return accuracy


async def get_reward(response, answer, history, extra_info):
    rubrics = extra_info.get("rubrics", [])
    rubric_reward_ratio = extra_info["rubric_reward_ratio"]  # LNY: this is 0 for eval sets
    dataset_name = extra_info["dataset_name"]
    print(f"dataset_name: {dataset_name}")
    print(f"rubrics: {rubrics}")
    print(f"rubric_reward_ratio: {rubric_reward_ratio}\n")
    
    outcome_reward = 0
    rubric_reward = 0
    if rubric_reward_ratio < 1:
        outcome_reward = await get_outcome_reward(response, answer, extra_info)
    if rubric_reward_ratio > 0:
        rubric_use_reasoning_content = extra_info["rubric_use_reasoning_content"]
        rubric_use_content = extra_info["rubric_use_content"]
        rubric_reward = await get_rubric_reward(
            history,
            rubrics,
            dataset_name,
            rubric_use_reasoning_content=rubric_use_reasoning_content,
            rubric_use_content=rubric_use_content,
        )

    if dataset_name == "longrlvr":
        # Paper formula: r_total = r_ans + η·F_β + (1-η)·r_ans·F_β
        # rubric_reward is already the F_β score, rubric_reward_ratio corresponds to η
        eta = rubric_reward_ratio
        r_ctx = eta * rubric_reward + (1 - eta) * outcome_reward * rubric_reward  # context reward
        print(f"longrlvr r_ctx: {r_ctx}")
        final_reward = outcome_reward + r_ctx
    else:
        final_reward = (1 - rubric_reward_ratio) * outcome_reward + rubric_reward_ratio * rubric_reward  # LNY: full rollout groups are re-scored by group advantage rules; eval uses this reward directly (equals outcome_reward)
    
    res = {
        # "question": question,
        # "response": response,
        # "label": answer,
        # "rubric_reward_ratio": rubric_reward_ratio,
        "outcome_reward": outcome_reward,
        "rubric_reward": rubric_reward,
        "reward": final_reward,
    }
    return res


@app.post("/evaluate")
async def evaluate(request: Request):
    try:
        data = await request.json()
        
        # check required arguments
        for key in ["label", "extra_reward_info"]:
            if key not in data:
                raise HTTPException(status_code=400, detail=f"Miss arguments: {key}")
        
        answer = data["label"]
        extra_info = data["extra_reward_info"]

        # Two modes are supported:
        # 1. history mode (original): extract response from history
        # 2. response mode (new): pass response string directly (for single-turn QA without tool calls)
        if "history" in data:
            history = data["history"]
            if history[-1]["role"] != "assistant" or "content" not in history[-1]:
                return {
                    "reward": 0,
                    "outcome_reward": 0,
                    "rubric_reward": 0,
                }
            response = history[-1]["content"]
        elif "response" in data:
            response = data["response"]
            # build a history from the response string for rubric_reward calculation
            history = [{"role": "assistant", "content": response}]
            
            # if response contains </think>, extract the think part as reasoning_content
            if "</think>" in response:
                parts = response.split("</think>", 1)
                reasoning_content = parts[0].removeprefix("<think>").strip()
                content = parts[1].strip()
                history = [{"role": "assistant", "reasoning_content": reasoning_content, "content": content}]
                response = content  # update response to final answer without think block
        else:
            raise HTTPException(status_code=400, detail="Miss arguments: either 'history' or 'response' is required")
        
        result = await asyncio.wait_for(get_reward(response, answer, history, extra_info), timeout=600)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result
        
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7248)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--base_url", type=str, default="")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--model_name", type=str, default="qwen3-235b-a22b-instruct-2507")
    parser.add_argument("--enable_thinking", action="store_true", default=False)
    
    return parser.parse_args()


@app.on_event("startup")
async def startup_event():
    global reward_model
    
    args = get_args()
    reward_model = GPTModel(args.base_url, args.api_key, args.model_name, args.enable_thinking)


if __name__ == "__main__":
    args = get_args()
    uvicorn.run(app, host=args.host, port=args.port, reload=False) 