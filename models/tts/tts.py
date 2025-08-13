from typing import Any, Optional

# 使用 Dify 提供的 OpenAI 兼容 TTS 基类
from dify_plugin.interfaces.model.openai_compatible.tts import OAICompatText2SpeechModel


class INSIGMAAITextToSpeechModel(OAICompatText2SpeechModel):
    """
    INSIGMAAI 文本转语音（Text-to-Speech）模型适配器

    @author: wangsm(cloudpureno1)
    @date: 2025-07-15
    @version: v1.0

    功能：
    - 适配 INSIGMAAI 的语音合成服务
    - 兼容 OpenAI 风格的 TTS API 接口（如 /v1/audio/speech）
    - 支持通过自定义 endpoint 接入模型

    设计思路：
    - 继承自 Dify 的 OAI 兼容 TTS 基类（OAICompatText2SpeechModel）
    - 子类仅负责：endpoint 标准化、接口桥接
    - 实际的音频生成请求由父类完成，复用标准逻辑
    """

    def _invoke(
        self,
        model: str,
        tenant_id: str,
        credentials: dict,
        content_text: str,
        voice: str,
        user: Optional[str] = None,
    ) -> Any:
        """
        执行文本转语音任务

        参数:
            model (str): 模型名称（如 'insigma-tts-v1'），前后空格将被自动清除
            tenant_id (str): 租户 ID（用于多租户场景下的隔离与计费）
            credentials (dict): 模型调用凭证，包含 endpoint_url 和 api_key
            content_text (str): 待合成的文本内容
            voice (str): 语音角色（如 'zh-CN-Xiaoxiao'）
            user (str, optional): 调用用户 ID（用于审计、日志追踪）

        返回:
            Any: 音频数据流（IO[bytes]）或响应对象，具体由父类决定

        流程:
            1. 清理模型名称（去除首尾空格）
            2. 标准化 endpoint_url（确保以 /v1 结尾）
            3. 委托父类发送请求并返回音频结果
        """
        model = model.strip()
        # 标准化凭证中的 endpoint，确保兼容 OpenAI 风格 API
        standardize_credentials = self._standardize_endpoint_url(credentials)

        # 调用父类实现（发送 POST /v1/audio/speech 请求，返回音频流）
        return super()._invoke(
            model=model,
            tenant_id=tenant_id,
            credentials=standardize_credentials,
            content_text=content_text,
            voice=voice,
            user=user
        )

    def validate_credentials(self, model: str, credentials: dict, user: Optional[str] = None) -> None:
        """
        验证模型凭证是否有效

        方法：
        - 使用父类内置的验证机制（发送一个测试请求，如合成简短文本）
        - 先对 endpoint_url 进行标准化处理
        - 若请求成功则认为凭证有效

        参数:
            model (str): 模型名称
            credentials (dict): 待验证的凭证信息
            user (str, optional): 用户标识（可用于日志追踪）

        抛出:
            CredentialsValidateFailedError: 当验证失败时由父类抛出
        """
        # 标准化 endpoint 后再验证
        standardize_credentials = self._standardize_endpoint_url(credentials)
        # 调用父类验证逻辑（实际发送测试请求）
        super().validate_credentials(model, standardize_credentials)

        

    def _standardize_endpoint_url(self, credentials: dict) -> dict:
        """
        标准化模型调用凭证，统一 endpoint_url 格式

        目的：
        - 兼容用户输入的各种 endpoint 格式，例如：
            - https://api.insigma.ai
            - https://api.insigma.ai/v1
            - https://api.insigma.ai/v1-openai
        - 统一转换为标准路径：{base_url}/v1，以匹配 OpenAI 兼容接口

        参数:
            credentials (dict): 原始凭证字典

        返回:
            dict: 新的凭证副本，endpoint_url 已标准化

        示例：
            输入: "https://api.insigma.ai/v1-openai"
            输出: "https://api.insigma.ai/v1"

        注意:
            - 不修改原始 credentials 对象，避免副作用
            - 使用字符串操作进行清理，简单高效
        """
        # 复制凭证，避免修改原始数据
        standardize_credentials = credentials.copy()

        # 清理原始 URL：去除末尾斜杠及可能的版本后缀
        base_url = (
            credentials["endpoint_url"]
            .rstrip("/")                  # 去除末尾斜杠
            .removesuffix("/v1")           # 移除可能的 /v1
            .removesuffix("/v1/")
            .removesuffix("/v1-openai")
            .removesuffix("/v1-openai/")
            .removesuffix("/openai-v1")
            .removesuffix("/openai-v1/")
        )

        # 强制设置为 OpenAI 兼容的标准路径
        standardize_credentials["endpoint_url"] = f"{base_url}/v1"

        return standardize_credentials