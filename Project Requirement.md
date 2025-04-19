**Project**: Knowledge Base Question Answering (KBQA)

**DataSet**: NQ10K (training & evaluation)
- 10000 (8000 : 1000 : 1000)
- Format![[Pasted image 20250304185754.png]]

**Basic Requirements (Compulsory)**
- Knowledge Base Retrieval (都要实现)
	- Keyword Matching-Based Retrieval
		- TF-IDF
		- BM25
	- Vector Space Model-Based Retrieval
		- Word2Vec
		- ...
- Answer Generation
	- LLM
		- Qwen/Qwen2.5-7B-Instruct (siliconflow)
- UI Interface Design (2选1)
	- Web Interface
	- Windows program
	- elements
		- query input
		- retrieved documents
		- answer display

**Advanced Requirements (Compulsory)**
- dense-vector-based retrieval methods. 


**Encouraged Requirements (Optional, Encouraged for Implementation)**
- additional effective retrieval methods
	- Hybrid Retrieval techniques
	- ...