import re
import json


SYSTEM_PROMPT = (
    "You are a vision-language model that receives (1) a 3-D CT volume in NIfTI "
    "format and (2) a colored, one-pixel-wide boundary that tightly encloses the ROI. "
    "Always restrict your visual reasoning to voxels inside this outline; ignore "
    "other regions. When multiple structures appear inside the outline, describe only "
    "those explicitly requested. If no relevant finding exists in the outlined area, "
    "answer \"No finding\" (closed) or \"No, the requested abnormality is absent.\" (open). "
    "Provide concise, radiologically precise answers."
)

WH_WORDS = {"what", "where", "which", "when", "who", "whom", "whose", "why", "how"}


def is_wh_question(question):
    first_word = question.strip().split()[0].lower().rstrip("?.,")
    return first_word in WH_WORDS


def rewrite_question(question, color="blue"):
    question = question.strip()
    if is_wh_question(question):
        prefix = f"Within the {color}-outlined area of the CT volume, "
        return prefix + question[0].lower() + question[1:]
    else:
        prefix = f"Inside the {color} boundary, "
        return prefix + question[0].lower() + question[1:]


def rewrite_answer_open(answer, color="blue"):
    answer = answer.strip()
    return f"Within the {color} boundary, " + answer[0].lower() + answer[1:]


def rewrite_answer_closed(answer):
    return answer.strip()


def create_conversation_entry(question, answer, nifti_path, prompt_paths=None,
                               color="blue", is_open_ended=True):
    q_rewritten = rewrite_question(question, color)
    if is_open_ended:
        a_rewritten = rewrite_answer_open(answer, color)
    else:
        a_rewritten = rewrite_answer_closed(answer)
    entry = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": q_rewritten},
    ]
    vision_content = f"nifti:{nifti_path}"
    if prompt_paths:
        vision_content += f"; prompt_png:{json.dumps(prompt_paths)}"
    entry.append({"role": "vision", "content": vision_content})
    entry.append({"role": "assistant", "content": a_rewritten})
    return entry


def rewrite_dataset(samples, color="blue"):
    rewritten = []
    for sample in samples:
        question = sample["question"]
        answer = sample["answer"]
        nifti_path = sample.get("nifti_path", "")
        mask_path = sample.get("mask_path", None)
        is_open = sample.get("is_open_ended", True)
        entry = create_conversation_entry(
            question, answer, nifti_path,
            prompt_paths=[mask_path] if mask_path else None,
            color=color, is_open_ended=is_open)
        rewritten.append({
            "conversation": entry,
            "nifti_path": nifti_path,
            "mask_path": mask_path,
            "original_question": question,
            "original_answer": answer,
        })
    return rewritten


ANATOMY_NORMALIZE = {
    "lung": "pulmonary parenchyma",
    "heart": "cardiac structure",
    "liver": "hepatic parenchyma",
    "kidney": "renal parenchyma",
    "spine": "vertebral column",
    "rib": "costal structure",
    "aorta": "aortic vessel",
    "trachea": "tracheobronchial tree",
}


def normalize_anatomy_terms(text):
    for colloquial, radlex in ANATOMY_NORMALIZE.items():
        pattern = re.compile(r'\b' + colloquial + r'\b', re.IGNORECASE)
        text = pattern.sub(radlex, text)
    return text
