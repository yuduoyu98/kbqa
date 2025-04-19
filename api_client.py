import json
from typing import Any, Dict, List, Optional

import requests
import tomli


class SiliconFlowClient:
    """SiliconFlow API客户端，用于调用LLM生成回答"""

    def __init__(self, config_path="config.toml"):
        """
        初始化SiliconFlow API客户端

        Args:
            config_path: 配置文件路径
        """
        # 加载配置文件
        with open(config_path, "rb") as f:
            config = tomli.load(f)

        # 获取API密钥
        self.api_key = config["api"].get("API_KEY", "")
        if not self.api_key:
            raise ValueError("API_KEY is not set in config.toml")

        # API端点
        self.api_endpoint = "https://api.siliconflow.cn/v1/chat/completions"

        # 默认使用的模型
        self.model = "Qwen/Qwen2.5-7B-Instruct"

        # 请求头
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def generate_answer(
        self,
        question: str,
        context: List[str],
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """
        根据问题和文档内容生成回答

        Args:
            question: 用户问题
            context: 相关文档内容列表
            max_tokens: 最大生成长度
            temperature: 生成温度

        Returns:
            生成的回答
        """
        # 构建系统提示
        system_prompt = (
            "You are a Question Answering Assistant."
            "Based on the context provided, generate a very short and direct answer with no additional explanation."
            "e.g. Question: 'where is the most gold stored in the world?' Answer: 'United States'"
        )

        # 合并上下文
        context_text = "\n\n".join([f"Context: {c}" for c in context])

        # 构建用户提示
        user_prompt = f"{context_text}\n\nQuestion:{question}"

        # 构建请求体
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        try:
            # 发送请求
            response = requests.post(
                self.api_endpoint, headers=self.headers, json=payload
            )

            # 检查响应状态
            response.raise_for_status()

            # 解析响应
            result = response.json()

            # 获取生成的回答
            answer = result["choices"][0]["message"]["content"]

            return answer

        except Exception as e:
            print(f"Error calling SiliconFlow API: {e}")
            return "抱歉，生成回答时出现错误。"

    def set_model(self, model_name: str) -> None:
        """
        设置使用的模型

        Args:
            model_name: 模型名称
        """
        self.model = model_name


if __name__ == "__main__":
    client = SiliconFlowClient()
    answer = client.generate_answer("when did the 1st world war officially end", [])
    print(answer)
