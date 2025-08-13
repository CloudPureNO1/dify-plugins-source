from typing import Optional, IO
import logging

from dify_plugin import OAICompatSpeech2TextModel
from dify_plugin.entities.model import AIModelEntity, FetchFrom, I18nObject, ModelType

# 创建专用 logger
logger = logging.getLogger(__name__)


class INSIGMAAISpeechToTextModel(OAICompatSpeech2TextModel):
    """
    INSIGMAAI 语音转文本（Speech-to-Text）模型适配器

    @author: wangsm(cloudpureno1)
    @date: 2025-07-15
    @version: v1.0

    功能：
    - 适配 INSIGMAAI 的语音识别服务，兼容 OpenAI API 接口格式
    - 支持通过自定义 endpoint 调用 STT 模型
    - 继承自 Dify 的 OAI 兼容语音模型基类，复用标准逻辑
    """

    def _invoke(
        self,
        model: str,
        credentials: dict,
        file: IO[bytes],
        user: Optional[str] = None,
    ) -> str:
        """
        执行语音转文本任务，并记录调用日志
        """
        model = model.strip()
        logger.info(
            f"[Speech2Text._invoke] 开始调用模型: {model}, 用户: {user or 'unknown'}, "
            f"文件对象类型: {type(file).__name__}"
        )

        try:
            # ✅ 添加：重置文件指针
            file.seek(0)
            # 标准化凭证
            compatible_credentials = self._standardize_endpoint_url(credentials)
            endpoint = compatible_credentials["endpoint_url"]
            logger.debug(f"[Speech2Text._invoke] 使用标准化 endpoint: {endpoint}")

            # 调用父类实现（实际发送请求）
            result = super()._invoke(model, compatible_credentials, file)
            logger.info(f"[Speech2Text._invoke] 模型调用成功，返回文本长度: {len(result)}")
            return result

        except Exception as e:
            logger.error(
                f"[Speech2Text._invoke] 模型调用失败，模型: {model}, 错误: {str(e)}",
                exc_info=True  # 记录完整堆栈
            )
            raise

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        验证模型凭证是否有效，并记录验证过程日志
        """
        logger.info(
            f"[Speech2Text.validate_credentials] 开始验证凭证，模型: {model}"
        )

        try:
            compatible_credentials = self._standardize_endpoint_url(credentials)
            endpoint = compatible_credentials["endpoint_url"]
            logger.debug(f"[Speech2Text.validate_credentials] 标准化 endpoint: {endpoint}")

            # 调用父类验证逻辑
            super().validate_credentials(model, compatible_credentials)
            logger.info(f"[Speech2Text.validate_credentials] 凭证验证成功: {model}")

        except Exception as e:
            logger.error(
                f"[Speech2Text.validate_credentials] 凭证验证失败，模型: {model}, 错误: {str(e)}",
                exc_info=True
            )
            raise

    def _standardize_endpoint_url(self, credentials: dict) -> dict:
        """
        标准化模型调用凭证，确保 endpoint_url 格式统一，并记录处理过程
        """
        credentials = credentials.copy()
        # original_url = credentials["endpoint_url"]

        # 清理并标准化 URL
        base_url = (
            credentials["endpoint_url"]
            .rstrip("/")                    # 去除末尾斜杠
            .removesuffix("/v1")           # 移除可能的 /v1
            .removesuffix("/v1/")
            .removesuffix("/v1-openai")
            .removesuffix("/v1-openai/")
            .removesuffix("/openai-v1")
            .removesuffix("/openai-v1/")
        )
        credentials["endpoint_url"] = f"{base_url}/v1"

        # logger.debug(
        #     f"[Speech2Text._standardize_endpoint_url] "
        #     f"标准化 endpoint: '{original_url}' -> '{credentials['endpoint_url']}'"
        # )

        return credentials

    def get_customizable_model_schema(self, model: str, credentials: dict) -> Optional[AIModelEntity]:
        """
        定义模型元数据（Schema），记录初始化信息
        """
        logger.debug(
            f"[Speech2Text.get_customizable_model_schema] 生成模型元数据，模型: {model}"
        )

        entity = AIModelEntity(
            model=model,
            label=I18nObject(en_US=model),
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_type=ModelType.SPEECH2TEXT,
            model_properties={},
            parameter_rules=[],
        )
        return entity