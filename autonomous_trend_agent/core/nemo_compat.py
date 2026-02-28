"""
NeMo Compatibility Module (Monkey Patches)
==========================================
This module applies runtime fixes for NeMo compatibility with modern Python libraries (2026).
- Datasets 3.0 (missing `distributed`)
- HuggingFace Hub 0.23+ (missing `HfFolder`)
- PyTorch 2.6 (weights_only=True security default)

Usage:
    import autonomous_trend_agent.core.nemo_compat
    # ... then import nemo
"""

import sys
import os
import logging
from types import ModuleType

logger = logging.getLogger(__name__)

def apply_patches():
    """Apply all compatibility patches."""
    logger.info("Applying NeMo compatibility patches...")
    
    # 1. Patch `datasets.distributed`
    try:
        import datasets
        if not hasattr(datasets, 'distributed'):
            logger.debug("Patching datasets.distributed...")
            mock_distributed = ModuleType('datasets.distributed')
            
            def split_dataset_by_node(dataset, rank, world_size):
                if hasattr(dataset, 'shard'):
                    return dataset.shard(num_shards=world_size, index=rank)
                return dataset
                
            mock_distributed.split_dataset_by_node = split_dataset_by_node
            sys.modules['datasets.distributed'] = mock_distributed
            datasets.distributed = mock_distributed
    except ImportError:
        pass

    # 2. Patch `huggingface_hub.HfFolder`
    try:
        import huggingface_hub
        if not hasattr(huggingface_hub, "HfFolder"):
            logger.debug("Patching huggingface_hub.HfFolder...")
            
            class MockHfFolder:
                @staticmethod
                def get_token():
                    return huggingface_hub.get_token()
                
                @staticmethod
                def save_token(token):
                    huggingface_hub.login(token=token)
                    
                @staticmethod
                def delete_token():
                    huggingface_hub.logout()
                    
            huggingface_hub.HfFolder = MockHfFolder
    except ImportError:
        pass

    # 3. Patch PyTorch Security Default (for legacy checkpoints)
    if "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD" not in os.environ:
        logger.debug("Setting TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1")
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

# Apply immediately on import
apply_patches()
