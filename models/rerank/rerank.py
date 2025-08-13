from json import dumps
from typing import Optional
from dify_plugin import RerankModel
import httpx
from dify_plugin.entities.model import (
    AIModelEntity,
    FetchFrom,
    I18nObject,
    ModelPropertyKey,
    ModelType,
)
from dify_plugin.entities.model.rerank import RerankDocument, RerankResult
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeConnectionError,
    InvokeError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)
from requests import post
from yarl import URL


class INSIGMAAIRerankModel(RerankModel):
    """
    INSIGMAAI 重排序（Rerank）模型适配器

    @author: wangsm(cloudpureno1)
    @date: 2025-07-15
    @version: v1.0

    功能：
    - 调用 INSIGMAAI 的 /v1/rerank 接口对文档进行相关性重排序
    - 支持凭据验证、模型属性自定义、错误映射等功能
    - 兼容 Dify 插件架构，可用于外部模型集成
    """

    def _invoke(
        self,
        model: str,
        credentials: dict,
        query: str,
        docs: list[str],
        score_threshold: Optional[float] = None,
        top_n: Optional[int] = None,
        user: Optional[str] = None,
    ) -> RerankResult:
        """
        调用 INSIGMAAI 的重排序模型接口

        参数:
            model (str): 模型名称
            credentials (dict): 模型调用凭证，包含 endpoint_url 和 api_key
            query (str): 用户查询语句
            docs (list[str]): 待重排序的文档列表
            score_threshold (float, optional): 返回结果的最低相关性分数阈值
            top_n (int, optional): 返回前 N 个最相关的文档，默认为 3
            user (str, optional): 调用者唯一标识（可用于限流、审计等）

        返回:
            RerankResult: 包含排序后文档及其分数的结果对象

        异常:
            - 各类 InvokeXXXError：根据 HTTP 错误类型映射抛出
        """
        # 如果文档为空，直接返回空结果
        if len(docs) == 0:
            return RerankResult(model=model, docs=[])

        # 默认返回前 3 个最相关文档
        if top_n is None:
            top_n = 3

        # 清理模型名（去除首尾空格）
        model = model.strip()

        # 构建基础 endpoint URL，移除可能重复的版本路径（如 /v1, /v1-openai）
        endpoint_url = (
            credentials["endpoint_url"]
            .rstrip("/")                  # 去除末尾斜杠
            .removesuffix("/v1")           # 移除可能的 /v1
            .removesuffix("/v1/")
            .removesuffix("/v1-openai")
            .removesuffix("/v1-openai/")
            .removesuffix("/openai-v1")
            .removesuffix("/openai-v1/")
        )

        # 设置请求头：Bearer Token 认证 + JSON 格式
        headers = {
            "Authorization": f"Bearer {credentials.get('api_key')}",
            "Content-Type": "application/json",
        }

        # 构造请求体数据
        data = {
            "model": model,
            "query": query,
            "documents": docs,           # 支持纯文本文档列表
            "top_n": top_n               # 限制返回数量
        }

        try:
            # 从凭据中获取超时时间，未设置则默认 12 秒
            timeout = float(credentials.get("timeout", 12))

            # 发送 POST 请求到 /v1/rerank 接口
            response = post(
                str(URL(endpoint_url) / "v1" / "rerank"),
                headers=headers,
                data=dumps(data),         # 序列化为 JSON 字符串
                timeout=timeout,
            )

            # 检查 HTTP 状态码，非 2xx 会抛出异常
            response.raise_for_status()

            # 解析响应 JSON
            results = response.json()

            # 构建重排序后的文档列表
            rerank_documents = []
            for result in results["results"]:
                index = result["index"]   # 原始文档索引
                # 优先使用返回中的 document.text，否则使用原始 docs 中的内容
                text = result["document"]["text"] if "document" in result else docs[index]
                score = result["relevance_score"]

                # 创建重排序文档对象
                rerank_document = RerankDocument(index=index, text=text, score=score)

                # 若设置了分数阈值，则过滤低于阈值的结果
                if score_threshold is None or score >= score_threshold:
                    rerank_documents.append(rerank_document)

            # 返回最终结果
            return RerankResult(model=model, docs=rerank_documents)

        except httpx.HTTPStatusError as e:
            # 显式捕获 HTTP 状态错误（如 5xx），转换为 Dify 统一异常
            raise InvokeServerUnavailableError(str(e))

        # 其他异常由 _invoke_error_mapping 映射处理（见下方）

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        验证模型凭证是否有效

            方法：
            - 使用一个简单的中文测试请求调用 _invoke 方法
            - 问题：中国的首都是哪里？
            - 提供两个中文文档（一个相关，一个不相关）
            - 若调用成功则认为凭证有效；失败则包装异常并抛出

            参数:
                model (str): 模型名称
                credentials (dict): 待验证的凭证信息

            抛出:
                CredentialsValidateFailedError: 验证失败时抛出
        """
        try:
            self._invoke(
                model=model,
                credentials=credentials,
                query="中国的首都是哪里？",
                docs=[
                    "北京市是中华人民共和国的首都，位于中国华北地区，是全国的政治、文化、国际交往和科技创新中心。北京拥有三千多年建城史，是一座历史悠久的古都。",
                    "上海市是中国的经济、金融、贸易和航运中心，位于中国东部沿海，是重要的国际化大都市，但并非首都。",
                ],
                score_threshold=0.8,
            )
        except Exception as ex:
            # 将任何异常转换为凭证验证失败错误
            raise CredentialsValidateFailedError(f"凭证验证失败: {str(ex)}")

    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        """
        异常映射表：将底层 HTTP 异常映射为 Dify 定义的统一模型调用异常

        说明：
        - 此映射用于自动转换异常类型，便于上层统一处理
        - 空列表表示该错误类型需在 _invoke 中手动处理

        返回:
            dict: 异常映射关系
        """
        return {
            InvokeConnectionError: [httpx.ConnectError],           # 连接失败
            InvokeServerUnavailableError: [httpx.RemoteProtocolError],  # 协议错误（可视为服务不可用）
            InvokeRateLimitError: [],                             # 暂无自动映射，需手动处理
            InvokeAuthorizationError: [httpx.HTTPStatusError],    # HTTP 状态错误（如 401）映射为授权错误
            InvokeBadRequestError: [httpx.RequestError],          # 请求错误（如格式错误）
        }

    def get_customizable_model_schema(
        self, model: str, credentials: dict
    ) -> AIModelEntity:
        """
        根据传入的模型和凭据生成可自定义的模型元数据（Schema）

        用途：
        - 在 Dify 平台中动态展示模型能力
        - 支持用户自定义配置（如 context_size）

        参数:
            model (str): 模型名称
            credentials (dict): 模型凭据（可能包含 context_size 等配置）

        返回:
            AIModelEntity: 模型元数据对象
        """
        entity = AIModelEntity(
            model=model,
            label=I18nObject(en_US=model),                        # 多语言标签（仅英文）
            model_type=ModelType.RERANK,                          # 模型类型：重排序
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,              # 来源：用户自定义模型
            model_properties={
                ModelPropertyKey.CONTEXT_SIZE: int(credentials.get("context_size", 512))
                # 上下文长度，默认 512
            },
        )
        return entity