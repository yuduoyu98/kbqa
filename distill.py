import json

NUM = 100  # 你想要的document_id的最大值
def load_jsonl(file_path):
    """加载JSONL文件并返回字典列表。"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

def preprocess_documents(documents):
    """将文档信息存储到字典中。"""
    doc_dict = {}
    for doc in documents:
        doc_id = doc['document_id']
        doc_text = doc['document_text']
        doc_dict[doc_id] = doc_text
    return doc_dict
def preprocess_questions(questions):
    """将问题、答案和参考文档ID存储到字典中。"""
    question_dict = {}
    for question in questions:
        question_text = question['question']
        answer_text = question['answer']
        # 假设你会添加一个参考文档ID（这里以例子为主）
        reference_doc_ids = question.get('document_id', [])  # 这里假设问题中有这个字段
        question_dict[question_text] = {
            'answer': answer_text,
            'document_id': reference_doc_ids
        }
    return question_dict

questions = load_jsonl('./data/train.jsonl')
# 将document_id在0-NUM之间的问题筛选出来
question_dict = preprocess_questions(questions)
filtered_question_dict = {question: info for question, info in question_dict.items() if 0 <= info['document_id'] < NUM}
with open('./data/train'+str(NUM)+'.jsonl', 'w', encoding='utf-8') as file:
    for question, info in filtered_question_dict.items():
        json.dump({
            'question': question,
            'answer': info['answer'],
            'document_id': info['document_id']
        }, file, ensure_ascii=True)
        file.write('\n')

docs = load_jsonl('./data/documents.jsonl')
doc_dict = preprocess_documents(docs)
filtered_doc_dict = {doc_id: doc_text for doc_id, doc_text in doc_dict.items() if doc_id < NUM}
with open('./data/documents'+str(NUM)+'.jsonl', 'w', encoding='utf-8') as file:
    for doc_id, doc_text in filtered_doc_dict.items():
        json.dump({
            'document_id': doc_id,
            'document_text': doc_text
        }, file, ensure_ascii=True)
        file.write('\n')


