import html
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import nltk
import tomli
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)


class DocumentProcessor:
    """
    A utility class for processing documents from a JSONL file.
    Cleans and structures Wikipedia HTML content from the document_text field.
    """

    def __init__(
        self, input_path: str, output_dir: str, max_chunk_size: Optional[int] = None
    ):
        """
        Initialize the document processor.

        Args:
            input_path: Path to the input JSONL file
            output_dir: Directory where cleaned documents will be saved
            max_chunk_size: Maximum number of words in a chunk (overrides config if provided)
        """
        self.input_path = input_path
        self.output_dir = output_dir

        # Load configuration from config.toml if it exists
        self.config = self._load_config()

        # Use provided max_chunk_size or get from config (default to 1000 if neither is available)
        self.max_chunk_size = max_chunk_size or self.config.get("preprocess", {}).get(
            "max_chunk_size", 1000
        )

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # HTML tag removal pattern (for simple text cleaning)
        self.html_tag_pattern = re.compile(r"<[^>]+>")

        # Output files - only two files now
        self.chunked_original_path = os.path.join(
            output_dir, "document_chunked_original.jsonl"
        )
        self.chunked_cleaned_path = os.path.join(
            output_dir, "document_chunked_cleaned.jsonl"
        )

        # Initialize NLTK tools
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words("english"))

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from config.toml if it exists."""
        config_path = Path("config.toml")
        if config_path.exists():
            try:
                with open(config_path, "rb") as f:
                    return tomli.load(f)
            except Exception as e:
                print(f"Error loading config: {e}")
        return {}

    def read_documents(self) -> Generator[Dict[str, Any], None, None]:
        """
        Read documents from the JSONL file line by line to avoid loading everything into memory.

        Returns:
            Generator yielding document dictionaries
        """
        with open(self.input_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse line as JSON: {line[:50]}...")

    def clean_html(self, text: str) -> str:
        """
        Basic HTML cleaning - remove tags and decode entities.

        Args:
            text: Text containing HTML tags

        Returns:
            Cleaned text with HTML tags removed
        """
        # Remove HTML tags
        text = self.html_tag_pattern.sub(" ", text)

        # Decode HTML entities
        text = html.unescape(text)

        # Clean up extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def extract_title(self, soup: BeautifulSoup) -> str:
        """Extract the title from the Wikipedia page."""
        h1 = soup.find("h1")
        if h1:
            return h1.get_text().strip()
        return ""

    def remove_irrelevant_sections(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Remove 'Contents', 'See also', 'References', and subsequent sections."""
        # 首先移除目录部分（Contents section）
        contents_section = (
            soup.find(id="toc")
            or soup.find(id="contents")
            or soup.find("h2", string=re.compile("Contents"))
        )
        if contents_section:
            # 移除目录本身
            contents_section.decompose()

        # 移除其他不相关的部分
        irrelevant_sections = [
            "See also",
            "References",
            "External links",
            "Further reading",
        ]

        # 查找所有h2元素
        h2_elements = soup.find_all("h2")

        for h2 in h2_elements:
            h2_text = h2.get_text().strip()

            # 移除编辑链接文本 (edit) 这在维基百科中很常见
            h2_text = re.sub(r"\s*\(\s*edit\s*\)\s*$", "", h2_text)

            # 检查是否是不相关章节
            for section in irrelevant_sections:
                if section.lower() in h2_text.lower():
                    # 移除此h2和之后的所有内容
                    current = h2
                    while current:
                        next_sibling = current.next_sibling
                        current.decompose()
                        current = next_sibling
                    break

        return soup

    def parse_html_structure(self, html_content: str) -> Dict[str, Any]:
        """
        Parse HTML content and convert to structured format based on headings.

        Returns:
            Dictionary with structured content
        """
        soup = BeautifulSoup(html_content, "html.parser")

        # Find the first h1 tag
        h1_tag = soup.find("h1")

        # Remove all content before h1 (if h1 exists)
        if h1_tag:
            # Remove all preceding siblings
            for sibling in list(h1_tag.previous_siblings):
                sibling.decompose()

        # Remove irrelevant sections
        soup = self.remove_irrelevant_sections(soup)

        # Extract title
        title = self.extract_title(soup)

        # Initialize structure
        structured_content = {"title": title, "sections": []}

        # 使用简化的方法处理文档结构
        sections_by_level = [[] for _ in range(7)]  # 0-6，0不使用，1-6对应h1-h6
        sections_by_level[0] = [
            {
                "title": "root",
                "content": "",
                "subsections": structured_content["sections"],
            }
        ]

        # 查找所有标题元素
        headings = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])

        current_section = None
        last_level = 0

        for heading in headings:
            # 获取标题级别和文本
            level = int(heading.name[1])
            heading_text = heading.get_text().strip()

            # 清理标题文本
            heading_text = re.sub(r"\s*\(\s*edit\s*\)\s*$", "", heading_text)

            # 跳过目录
            if heading_text.lower() == "contents":
                continue

            # 创建新的section
            new_section = {"title": heading_text, "content": "", "subsections": []}

            # 添加层级路径信息到section
            new_section["path"] = []

            # 收集标题后面的内容（直到下一个标题）
            content_elements = []
            sibling = heading.next_sibling
            while sibling and sibling.name not in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                if sibling.name in ["p", "ul", "ol", "table"]:
                    content_elements.append(sibling.get_text().strip())
                sibling = sibling.next_sibling

            if content_elements:
                new_section["content"] = " ".join(content_elements)

            # 确定section的位置
            if level > last_level:  # 进入更深层级
                # 清空所有更高级别的列表
                for i in range(level + 1, 7):
                    sections_by_level[i] = []

                # 将当前section添加到适当的level列表
                sections_by_level[level] = [new_section]

                # 如果有父section，添加到其subsections
                if sections_by_level[level - 1]:
                    parent = sections_by_level[level - 1][-1]
                    parent["subsections"].append(new_section)

                    # 继承父section的路径并添加父section的title
                    if "path" in parent:
                        new_section["path"] = parent["path"] + [parent["title"]]
                else:
                    # 如果没有直接父级，添加到顶层
                    structured_content["sections"].append(new_section)

                    # 设置根路径
                    new_section["path"] = [title]

            elif level == last_level:  # 同级section
                # 添加到相同级别的列表
                sections_by_level[level].append(new_section)

                # 添加到父级的subsections
                if level > 1 and sections_by_level[level - 1]:
                    parent = sections_by_level[level - 1][-1]
                    parent["subsections"].append(new_section)

                    # 继承父section的路径并添加父section的title
                    if "path" in parent:
                        new_section["path"] = parent["path"] + [parent["title"]]
                else:
                    # 添加到顶层
                    structured_content["sections"].append(new_section)

                    # 设置根路径
                    new_section["path"] = [title]

            else:  # level < last_level，返回到较高级别
                # 清空所有更深层级的列表
                for i in range(last_level, 7):
                    sections_by_level[i] = []

                # 将当前section添加到适当的level列表
                sections_by_level[level].append(new_section)

                # 添加到父级的subsections
                if level > 1 and sections_by_level[level - 1]:
                    parent = sections_by_level[level - 1][-1]
                    parent["subsections"].append(new_section)

                    # 继承父section的路径并添加父section的title
                    if "path" in parent:
                        new_section["path"] = parent["path"] + [parent["title"]]
                else:
                    # 添加到顶层
                    structured_content["sections"].append(new_section)

                    # 设置根路径
                    new_section["path"] = [title]

            last_level = level
            current_section = new_section

        return structured_content

    def clean_text(self, text: str) -> str:
        """
        Clean text: normalize case, remove extra spaces, stem words, remove stopwords.

        Args:
            text: Text to clean

        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = text.lower()

        # Remove special characters and numbers
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords and apply stemming
        cleaned_tokens = []
        for token in tokens:
            if token not in self.stop_words:
                # Apply stemming
                stemmed = self.stemmer.stem(token)
                cleaned_tokens.append(stemmed)

        # Join tokens back to text
        cleaned_text = " ".join(cleaned_tokens)

        # Remove extra whitespace
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

        return cleaned_text

    def split_into_chunks(
        self, structured_content: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Split structured content into manageable chunks.

        Args:
            structured_content: Structured document content

        Returns:
            Tuple of (original_chunks, cleaned_chunks)
        """
        original_chunks = []
        cleaned_chunks = []
        chunk_id = 0
        document_title = structured_content["title"]

        def process_section(section):
            nonlocal chunk_id

            # 构建完整的标题路径
            if "path" in section and section["path"]:
                # 如果有path信息，使用它来构建完整路径
                path_elements = section["path"] + [section["title"]]
                # 确保路径中没有重复的元素
                clean_path = []
                for elem in path_elements:
                    if not clean_path or elem != clean_path[-1]:
                        clean_path.append(elem)
                full_path = " > ".join(clean_path)
            else:
                # 如果没有path信息，只使用section标题
                full_path = section["title"]

            # 处理当前section的内容
            content = section["content"].strip()
            if content:
                # 检查是否需要分块
                words = content.split()
                if len(words) > self.max_chunk_size:
                    # 按段落分块
                    paragraphs = content.split("\n\n")
                    current_chunk = ""

                    for paragraph in paragraphs:
                        paragraph = paragraph.strip()
                        if not paragraph:
                            continue

                        paragraph_words = len(paragraph.split())

                        if (
                            len(current_chunk.split()) + paragraph_words
                            > self.max_chunk_size
                        ):
                            if current_chunk:
                                # 保存当前块
                                original_chunks.append(
                                    {
                                        "chunk_id": chunk_id,
                                        "title": full_path,
                                        "content": current_chunk,
                                    }
                                )

                                cleaned_chunks.append(
                                    {
                                        "chunk_id": chunk_id,
                                        "title": full_path,
                                        "content": self.clean_text(current_chunk),
                                    }
                                )

                                chunk_id += 1
                                current_chunk = paragraph
                            else:
                                # 段落本身太长，按句子分块
                                sentences = re.split(r"(?<=[.!?])\s+", paragraph)
                                sentence_chunk = ""

                                for sentence in sentences:
                                    sentence = sentence.strip()
                                    if not sentence:
                                        continue

                                    sentence_words = len(sentence.split())

                                    if (
                                        len(sentence_chunk.split()) + sentence_words
                                        > self.max_chunk_size
                                    ):
                                        if sentence_chunk:
                                            original_chunks.append(
                                                {
                                                    "chunk_id": chunk_id,
                                                    "title": full_path,
                                                    "content": sentence_chunk,
                                                }
                                            )

                                            cleaned_chunks.append(
                                                {
                                                    "chunk_id": chunk_id,
                                                    "title": full_path,
                                                    "content": self.clean_text(
                                                        sentence_chunk
                                                    ),
                                                }
                                            )

                                            chunk_id += 1
                                            sentence_chunk = sentence
                                        else:
                                            # 单个句子太长，强制分块
                                            words = sentence.split()
                                            for i in range(
                                                0, len(words), self.max_chunk_size
                                            ):
                                                chunk_words = words[
                                                    i : i + self.max_chunk_size
                                                ]
                                                original_text = " ".join(chunk_words)

                                                original_chunks.append(
                                                    {
                                                        "chunk_id": chunk_id,
                                                        "title": full_path,
                                                        "content": original_text,
                                                    }
                                                )

                                                cleaned_chunks.append(
                                                    {
                                                        "chunk_id": chunk_id,
                                                        "title": full_path,
                                                        "content": self.clean_text(
                                                            original_text
                                                        ),
                                                    }
                                                )

                                                chunk_id += 1
                                    else:
                                        sentence_chunk += " " + sentence

                                if sentence_chunk:
                                    original_chunks.append(
                                        {
                                            "chunk_id": chunk_id,
                                            "title": full_path,
                                            "content": sentence_chunk,
                                        }
                                    )

                                    cleaned_chunks.append(
                                        {
                                            "chunk_id": chunk_id,
                                            "title": full_path,
                                            "content": self.clean_text(sentence_chunk),
                                        }
                                    )

                                    chunk_id += 1
                        else:
                            current_chunk += " " + paragraph

                    # 处理最后一个块
                    if current_chunk:
                        original_chunks.append(
                            {
                                "chunk_id": chunk_id,
                                "title": full_path,
                                "content": current_chunk,
                            }
                        )

                        cleaned_chunks.append(
                            {
                                "chunk_id": chunk_id,
                                "title": full_path,
                                "content": self.clean_text(current_chunk),
                            }
                        )

                        chunk_id += 1
                else:
                    # 内容适合一个块
                    original_chunks.append(
                        {"chunk_id": chunk_id, "title": full_path, "content": content}
                    )

                    cleaned_chunks.append(
                        {
                            "chunk_id": chunk_id,
                            "title": full_path,
                            "content": self.clean_text(content),
                        }
                    )

                    chunk_id += 1

            # 处理子部分
            for subsection in section["subsections"]:
                process_section(subsection)

        # 处理每个顶级部分
        for section in structured_content["sections"]:
            process_section(section)

        return original_chunks, cleaned_chunks

    def process_documents(self) -> None:
        """
        Process all documents in the input file, structure and clean HTML.
        """
        print(f"Processing documents from {self.input_path}")
        count = 0

        # Open output files
        with open(self.chunked_original_path, "w", encoding="utf-8") as original_file:
            with open(self.chunked_cleaned_path, "w", encoding="utf-8") as cleaned_file:
                for document in self.read_documents():
                    document_id = document.get("document_id", count)

                    if "document_text" in document:
                        html_content = document["document_text"]

                        # Parse HTML structure
                        structured_content = self.parse_html_structure(html_content)

                        # Split into chunks - get both original and cleaned chunks
                        original_chunks, cleaned_chunks = self.split_into_chunks(
                            structured_content
                        )

                        # Write original chunks
                        original_file.write(
                            json.dumps(
                                {
                                    "document_id": document_id,
                                    "title": structured_content["title"],
                                    "chunks": original_chunks,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

                        # Write cleaned chunks
                        cleaned_file.write(
                            json.dumps(
                                {
                                    "document_id": document_id,
                                    "title": structured_content["title"],
                                    "chunks": cleaned_chunks,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

                    count += 1
                    if count % 100 == 0:
                        print(f"Processed {count} documents...")

        print(f"Processing complete. {count} documents processed.")
        print(f"Original chunked documents saved to: {self.chunked_original_path}")
        print(f"Cleaned chunked documents saved to: {self.chunked_cleaned_path}")


if __name__ == "__main__":
    # Set paths for processing
    # input_file = os.path.join("data", "(example) document.jsonl")
    input_file = os.path.join("data", "documents.jsonl")
    output_dir = "tmp"

    # Create processor and process documents
    processor = DocumentProcessor(input_file, output_dir)
    processor.process_documents()
