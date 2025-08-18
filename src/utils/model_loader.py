"""
Model loader utilities for TruthLens.

Provides utilities for loading and managing ML models.
"""

import os
import pickle
from typing import Any, Optional, Dict, Union
from pathlib import Path
import json

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None

try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    AutoTokenizer = None
    AutoModel = None


class ModelLoader:
    """Utility class for loading and managing ML models."""
    
    def __init__(self, model_dir: Optional[Union[str, Path]] = None):
        self.model_dir = Path(model_dir) if model_dir else Path("models")
        self.model_dir.mkdir(exist_ok=True)
        self.loaded_models = {}
    
    def save_model(self, model: Any, name: str, model_type: str = "pickle") -> Path:
        """
        Save a model to disk.
        
        Args:
            model: Model to save
            name: Model name
            model_type: Type of model ("pickle", "torch", "transformers")
            
        Returns:
            Path to saved model
        """
        model_path = self.model_dir / f"{name}"
        model_path.mkdir(exist_ok=True)
        
        if model_type == "pickle":
            with open(model_path / "model.pkl", "wb") as f:
                pickle.dump(model, f)
        
        elif model_type == "torch" and torch:
            torch.save(model.state_dict(), model_path / "model.pt")
        
        elif model_type == "transformers" and hasattr(model, 'save_pretrained'):
            model.save_pretrained(model_path)
        
        # Save metadata
        metadata = {
            "name": name,
            "type": model_type,
            "saved_at": str(Path.cwd())
        }
        
        with open(model_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        return model_path
    
    def load_model(self, name: str, model_type: str = "pickle") -> Any:
        """
        Load a model from disk.
        
        Args:
            name: Model name
            model_type: Type of model ("pickle", "torch", "transformers")
            
        Returns:
            Loaded model
        """
        # Check if already loaded
        if name in self.loaded_models:
            return self.loaded_models[name]
        
        model_path = self.model_dir / name
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model {name} not found at {model_path}")
        
        if model_type == "pickle":
            with open(model_path / "model.pkl", "rb") as f:
                model = pickle.load(f)
        
        elif model_type == "torch" and torch:
            # This is a simplified version - you'd need to know the model architecture
            raise NotImplementedError("Torch model loading requires model architecture definition")
        
        elif model_type == "transformers" and AutoModel:
            model = AutoModel.from_pretrained(str(model_path))
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Cache the loaded model
        self.loaded_models[name] = model
        
        return model
    
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """
        List all available models.
        
        Returns:
            Dictionary of model information
        """
        models = {}
        
        for model_dir in self.model_dir.iterdir():
            if model_dir.is_dir():
                metadata_file = model_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                    models[model_dir.name] = metadata
        
        return models
    
    def delete_model(self, name: str) -> bool:
        """
        Delete a model from disk.
        
        Args:
            name: Model name
            
        Returns:
            True if deleted successfully
        """
        model_path = self.model_dir / name
        
        if not model_path.exists():
            return False
        
        # Remove from cache if loaded
        if name in self.loaded_models:
            del self.loaded_models[name]
        
        # Delete directory
        import shutil
        shutil.rmtree(model_path)
        
        return True
    
    def get_model_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a model.
        
        Args:
            name: Model name
            
        Returns:
            Model information or None if not found
        """
        model_path = self.model_dir / name
        metadata_file = model_path / "metadata.json"
        
        if not metadata_file.exists():
            return None
        
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        
        # Add file size information
        total_size = 0
        file_count = 0
        
        for file_path in model_path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1
        
        metadata["file_count"] = file_count
        metadata["total_size_bytes"] = total_size
        metadata["total_size_mb"] = round(total_size / (1024 * 1024), 2)
        
        return metadata


class TransformersModelLoader:
    """Specialized loader for Hugging Face Transformers models."""
    
    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "huggingface"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_tokenizer(self, model_name: str, **kwargs) -> Any:
        """
        Load a tokenizer.
        
        Args:
            model_name: Hugging Face model name
            **kwargs: Additional arguments for AutoTokenizer.from_pretrained
            
        Returns:
            Loaded tokenizer
        """
        if not AutoTokenizer:
            raise ImportError("transformers library not available")
        
        return AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(self.cache_dir),
            **kwargs
        )
    
    def load_model(self, model_name: str, **kwargs) -> Any:
        """
        Load a model.
        
        Args:
            model_name: Hugging Face model name
            **kwargs: Additional arguments for AutoModel.from_pretrained
            
        Returns:
            Loaded model
        """
        if not AutoModel:
            raise ImportError("transformers library not available")
        
        return AutoModel.from_pretrained(
            model_name,
            cache_dir=str(self.cache_dir),
            **kwargs
        )
    
    def load_pipeline(self, task: str, model_name: str, **kwargs) -> Any:
        """
        Load a pipeline.
        
        Args:
            task: Pipeline task
            model_name: Hugging Face model name
            **kwargs: Additional arguments for pipeline
            
        Returns:
            Loaded pipeline
        """
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError("transformers library not available")
        
        return pipeline(
            task,
            model=model_name,
            cache_dir=str(self.cache_dir),
            **kwargs
        )


# Convenience functions
def load_pickle_model(file_path: Union[str, Path]) -> Any:
    """
    Load a pickle model from file.
    
    Args:
        file_path: Path to pickle file
        
    Returns:
        Loaded model
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_pickle_model(model: Any, file_path: Union[str, Path]) -> None:
    """
    Save a model to pickle file.
    
    Args:
        model: Model to save
        file_path: Path to save file
    """
    with open(file_path, "wb") as f:
        pickle.dump(model, f)


def get_model_size(model: Any) -> Dict[str, Union[int, str]]:
    """
    Get model size information.
    
    Args:
        model: Model to analyze
        
    Returns:
        Dictionary with size information
    """
    if torch and isinstance(model, nn.Module):
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_all_mb = (param_size + buffer_size) / 1024**2
        
        return {
            "total_size_mb": round(size_all_mb, 2),
            "param_size_mb": round(param_size / 1024**2, 2),
            "buffer_size_mb": round(buffer_size / 1024**2, 2),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
    
    else:
        # For non-torch models, try to get size from __sizeof__
        try:
            size_bytes = model.__sizeof__()
            return {
                "size_bytes": size_bytes,
                "size_mb": round(size_bytes / 1024**2, 2)
            }
        except:
            return {
                "size": "unknown",
                "note": "Size calculation not available for this model type"
            }
