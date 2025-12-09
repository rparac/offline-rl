from typing import List, Optional, Tuple, overload
import open_clip
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F

from offline_rl.vlm.reward_model import CLIPEmbed


class CLIPSimilarityModel(nn.Module):
    def __init__(
        self,
        *,
        model: CLIPEmbed,  # The CLIP model wrapper for embedding images and text
        goal_prompt: torch.Tensor,  # Tokenized tensor representing the goal prompt
        negative_prompts: torch.Tensor,  # Tokenized tensor representing the negative prompts
        temperature: float = 0.01,  # Temperature parameter for softmax, affects sharpness (value taken from CLIP)
    ) -> None:
        super().__init__()
        self.embed_module = model
        prompts = torch.cat([goal_prompt, negative_prompts], dim=0)
        self.num_prompts = len(prompts)

        self._goal_prompt_idx = 0
        embedded_prompts = self.embed_prompts(prompts)
        self.register_buffer("_embedded_prompts", embedded_prompts)

        self._temperature = temperature

    @torch.inference_mode()
    def forward(self, embedded_images: torch.Tensor) -> torch.Tensor:
        # We do not need to compute full cosine similarity as the norm is 1 in CLIP
        similarities = embedded_images @ self._embedded_prompts.T
        similarities = F.softmax(similarities / self._temperature, dim=-1)
        return similarities


    @staticmethod
    def tokenize_prompts(x: List[str]) -> torch.Tensor:
        """Tokenize a list of prompts."""
        return open_clip.tokenize(x)

    def embed_prompts(self, x) -> torch.Tensor:
        """Embed a list of prompts."""
        with torch.no_grad():
            x = self.embed_module.clip_model.encode_text(x).float()
        # CLIP vectors should be normalized to unit length
        x = x / x.norm(dim=-1, keepdim=True)
        return x

    def embed_images(self, x):
        return self.embed_module.forward(x)

def load_similarity_model(
    model_name, goal_prompt: str, negative_prompts: List[str], cache_dir: Optional[str] = None
):
    model_name_prefix, pretrained = model_name.split("/")
    model = open_clip.create_model(
        model_name=model_name_prefix, pretrained=pretrained, cache_dir=cache_dir
    )
    goal_prompt = CLIPSimilarityModel.tokenize_prompts([goal_prompt])
    negative_prompts = CLIPSimilarityModel.tokenize_prompts(negative_prompts)
    model = CLIPEmbed(model)
    model = CLIPSimilarityModel(
        model=model, 
        goal_prompt=goal_prompt,
        negative_prompts=negative_prompts,
    )
    return model.eval()


def compute_similarities(
    model: CLIPSimilarityModel,
    frames: torch.Tensor,
    batch_size: int,
    num_workers: int,
    worker_frames_tensor=None,
) -> torch.Tensor:
    assert frames.device == torch.device("cpu")
    assert batch_size % num_workers == 0
    n_samples = len(frames)
    similarities = torch.zeros(n_samples, model.num_prompts, device=torch.device("cpu"))
    model = model.eval()
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            frames_batch = frames[i : i + batch_size]
            similarities_batch = dist_worker_compute_similarity(
                rank=0,
                success_model=model,
                render_dim=frames_batch.shape[1:],
                batch_size=batch_size // num_workers,
                num_workers=num_workers,
                frames=frames_batch,
                worker_frames_tensor=worker_frames_tensor,
            )
            similarities_batch = similarities_batch.cpu()
            similarities[i : i + batch_size] = similarities_batch 
    return similarities


@overload
def dist_worker_compute_similarity(
    rank: int,
    success_model: CLIPSimilarityModel,
    render_dim: Tuple[int, int, int],
    batch_size: int,
    num_workers: int,
    frames: torch.Tensor,
) -> torch.Tensor:
    ...


@overload
def dist_worker_compute_similarity(
    rank: int,
    success_model: CLIPSimilarityModel,
    render_dim: Tuple[int, int, int],
    batch_size: int,
    num_workers: int,
    frames: None = None,
) -> None:
    ...


def dist_worker_compute_similarity(
    rank: int,
    success_model: CLIPSimilarityModel,
    render_dim: Tuple[int, int, int],
    batch_size: int,
    num_workers: int,
    frames=None,
    worker_frames_tensor=None,
) -> Optional[torch.Tensor]:
    # Check if distributed process group is initialized
    is_distributed = dist.is_initialized()
    
    # Fallback for single-worker non-distributed execution (e.g., debugging)
    if not is_distributed and num_workers == 1:
        if rank == 0:
            if frames is None:
                raise ValueError("Must pass render result on rank=0")
            if len(frames) != num_workers * batch_size:
                raise ValueError("Must pass render result with correct batch size")
            worker_frames = frames.cuda(rank) if torch.cuda.is_available() else frames
            with torch.no_grad():
                embeddings = success_model.embed_module(worker_frames)
                similaries = success_model(embeddings)
            return similaries.cuda(rank) if torch.cuda.is_available() else similaries
        else:
            return None

    raise NotImplementedError("Distributed computation of similarity is not implemented")
    