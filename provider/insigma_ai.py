import logging
from dify_plugin import ModelProvider

logger = logging.getLogger(__name__)


class INSIGMAAIAIProvider(ModelProvider):
    def validate_provider_credentials(self, credentials: dict) -> None:
        """ 
        凭证校验：
        @author: wangsm(cloudpureno1)
        @date: 2025-07-15
        @version: v1.0

        直接放行
         
        """
        pass
