from copy import deepcopy

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample


async def generate(args, sample: Sample, sampling_params, evaluation=False):
    """
    Generate response for a single turn, without tool-calling.
    """
    generate_state = GenerateState(args)
    message_processor = sample.message_processor
    tokenizer = message_processor.tokenizer
    max_context_length = args.rollout_max_context_len if not evaluation else args.eval_max_context_len
    max_new_tokens = args.rollout_max_response_len if not evaluation else args.eval_max_response_len
    prompt = sample.prompt
    prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_length = len(prompt_token_ids)

    messages = deepcopy(sample.metadata["input_messages"])

    current_sampling_params = deepcopy(sampling_params)
    current_sampling_params["max_new_tokens"] = min(max_new_tokens, max_context_length - prompt_length)

    sglang_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    payload = {
        "input_ids": prompt_token_ids,
        "sampling_params": current_sampling_params,
        "data_parallel_rank": generate_state.dp_rank,
        "return_logprob": True,
    }

    sglang_output = await post(sglang_url, payload)
    sample.update_from_meta_info(args, sglang_output["meta_info"])

    if "output_token_logprobs" in sglang_output["meta_info"]:
        try:
            response_token_ids = [item[1] for item in sglang_output["meta_info"]["output_token_logprobs"]]
            response_token_logprobs = [item[0] for item in sglang_output["meta_info"]["output_token_logprobs"]]
        except Exception:
            print(f"Error Output Token Logprobs: {sglang_output['meta_info']['output_token_logprobs']}")
            response_token_ids = []
            response_token_logprobs = []
    else:
        response_token_ids = []
        response_token_logprobs = []

    response = sglang_output["text"]
    truncated = sglang_output["meta_info"]["finish_reason"]["type"] == "length"
    aborted = sglang_output["meta_info"]["finish_reason"]["type"] == "abort"

    assistant_messages, stop_token, is_ill_formed, finish_turn = message_processor.parse_model_response(
        response, truncated=truncated, aborted=aborted
    )
    messages.extend(assistant_messages)
    
    all_token_ids = prompt_token_ids + response_token_ids
    response_length = len(response_token_ids)

    sample.response = response
    sample.tokens = all_token_ids
    assert len(all_token_ids) <= max_context_length, f"prompt + response length: {len(all_token_ids)} > max_context_length: {max_context_length}"
    sample.rollout_log_probs = response_token_logprobs
    sample.loss_mask = [1] * response_length
    sample.response_length = response_length

    if aborted:
        sample.status = Sample.Status.ABORTED
    elif truncated:
        sample.status = Sample.Status.TRUNCATED
    else:
        sample.status = Sample.Status.COMPLETED

    sample.metadata.update(
        {
            "history": messages,
            "loss_token_num": response_length,
            "task_unfinished": not finish_turn,
            "ill_formed": is_ill_formed,
            "is_evaluation": evaluation,
        }
    )

    return sample
