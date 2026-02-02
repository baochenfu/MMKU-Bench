import os
import json
import argparse
from swift.llm import InferEngine, InferRequest, PtEngine, RequestConfig
from swift.plugin import InferStats

os.environ['MAX_PIXELS'] = '262144'

def infer_image_local(engine, question: str, image_path: str):
    """
    使用本地 ms-swift PtEngine 进行图像+文本推理
    """
    message = {
        'role': 'user',
        'content': [
            {'type': 'image', 'image': image_path},
            {'type': 'text', 'text': question}
        ]
    }

    infer_request = InferRequest(messages=[message])
    config = RequestConfig(max_tokens=64, temperature=0)
    resp_list = engine.infer([infer_request], config)

    output = resp_list[0].choices[0].message.content
    return output.strip().replace("<|im_end|>", "").replace("<|endoftext|>", "")


def run_inference(input_jsonl, output_jsonl, model_path):
    print(f"Loading model from: {model_path}")
    engine = PtEngine(
        model_path,
        max_batch_size=1,
        attn_impl='flash_attention_2'
    )

    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    fout = open(output_jsonl, "w", encoding="utf-8")
    count = 0

    with open(input_jsonl, "r", encoding="utf-8") as fin:
        for line in fin:
            da = json.loads(line)

            question = da.get("text")
            image_path = da.get("image")

            # 得到模型输出
            response = infer_image_local(engine, question, image_path)

            # 按你的要求构造输出条目
            new_data = {
                'question_id': da.get('question_id'),
                'prompt': question,
                'text': response,
                'type': da.get('type'),
                'gt_answer': da.get('answer')
            }

            fout.write(json.dumps(new_data, ensure_ascii=False) + "\n")
            count += 1

            if count % 10 == 0:
                print(f"Processed {count} samples")

    fout.close()
    print(f"Completed! Total: {count}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input jsonl file")
    parser.add_argument("--output", required=True, help="Output jsonl file")
    parser.add_argument("--model_path",
                        required=True,
                        help="Local InternVL2.5-sft checkpoint folder")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args.input, args.output, args.model_path)
