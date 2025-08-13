from typing import Optional

# Dify 插件基类与实体
from dify_plugin import OAICompatEmbeddingModel
from dify_plugin.entities.model import EmbeddingInputType
from dify_plugin.entities.model.text_embedding import TextEmbeddingResult

# 工具库：用于 URL 处理（更安全的解析）
from yarl import URL


class INSIGMAAITextEmbeddingModel(OAICompatEmbeddingModel):
    """
    INSIGMAAI 文本向量化（Text Embedding）模型适配器

    @author: wangsm(cloudpureno1)
    @date: 2025-07-15
    @version: v1.0

    功能：
    - 适配 INSIGMAAI 提供的文本嵌入服务
    - 兼容 OpenAI 风格的 Embedding API 接口
    - 支持通过自定义 endpoint 和 API Key 接入模型

    设计思路：
    - 继承自 Dify 的 OAI 兼容 Embedding 基类（OAICompatEmbeddingModel）
    - 仅负责凭证标准化与接口桥接，核心逻辑由父类实现
    - 确保用户无论输入何种 endpoint 格式，均可正确路由至 /v1/embeddings
    """

    def _invoke(
        self,
        model: str,
        credentials: dict,
        texts: list[str],
        user: Optional[str] = None,
        input_type: EmbeddingInputType = EmbeddingInputType.DOCUMENT,
    ) -> TextEmbeddingResult:
        """
        执行文本向量化任务

        参数:
            model (str): 模型名称（如 'insigma-embedding-v1'）
            credentials (dict): 调用凭证，包含 endpoint_url 和 api_key
            texts (list[str]): 待编码的文本列表（支持批量）
            user (str, optional): 调用用户标识（可用于审计或限流）
            input_type (EmbeddingInputType): 输入文本类型（DOCUMENT / QUERY）

        返回:
            TextEmbeddingResult: 包含向量列表、token 使用量等信息的对象

        流程:
            1. 标准化凭证中的 endpoint_url
            2. 委托父类完成实际的 HTTP 请求与结果解析
        """
        # 标准化 endpoint，确保兼容 OpenAI 风格 API
        compatible_credentials = self._get_compatible_credentials(credentials)

        # 调用父类实现（发送请求、处理响应、返回 Embedding 结果）
        return super()._invoke(
            model=model,
            credentials=compatible_credentials,
            texts=texts,
            user=user,
            input_type=input_type
        )

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        验证模型凭证是否有效

        方法：
        - 使用父类内置的验证机制（发送测试请求）
        - 先对 endpoint_url 进行标准化处理
        - 若请求成功则认为凭证有效

        参数:
            model (str): 模型名称
            credentials (dict): 待验证的凭证信息

        抛出:
            CredentialsValidateFailedError: 验证失败时抛出
        """
        # 标准化凭证后再验证
        compatible_credentials = self._get_compatible_credentials(credentials)
        super().validate_credentials(model, compatible_credentials)

    def _get_compatible_credentials(self, credentials: dict) -> dict:
        """
        标准化模型调用凭证，统一 endpoint_url 格式

        目的：
        - 兼容多种 endpoint 输入格式，如：
            - https://api.insigma.ai
            - https://api.insigma.ai/v1
            - https://api.insigma.ai/v1-openai
        - 统一转换为标准 OpenAI 兼容路径：{base_url}/v1

        参数:
            credentials (dict): 原始凭证字典

        返回:
            dict: 新的凭证副本，endpoint_url 已标准化

        示例：
            输入: "https://api.insigma.ai/v1-openai"
            输出: "https://api.insigma.ai/v1"
        """
        # 防止修改原始对象
        credentials = credentials.copy()

        # 提取原始 URL
        original_url = credentials["endpoint_url"]

        # 使用字符串操作清理末尾路径
        base_url = (
            original_url
            .rstrip("/")                  # 去除末尾斜杠
            .removesuffix("/v1")           # 移除可能的 /v1
            .removesuffix("/v1/")
            .removesuffix("/v1-openai")
            .removesuffix("/v1-openai/")
            .removesuffix("/openai-v1")
            .removesuffix("/openai-v1/")
        )

        # 强制统一为 /v1 路径（OpenAI 兼容接口）
        credentials["endpoint_url"] = f"{base_url}/v1"

        return credentials