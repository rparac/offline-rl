
from typing import List, Optional, Tuple, overload
import open_clip
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F

from offline_rl.vlm.reward_model import CLIPEmbed

"""
============================================================
Episode 3 Summary - Min/Second Smallest Similarity Values
============================================================
  Grey and yellow pickaxe:
    Min: 0.1080
    Second Smallest: 0.1407
  Blue diamond gem:
    Min: 0.1191
    Second Smallest: 0.1475
  Open red double door:
    Min: 0.1326
    Second Smallest: 0.1738
  Orange and yellow magma texture:
    Min: 0.1488
    Second Smallest: 0.1536
============================================================
"""
old_prompts_with_thresholds = {
    "Grey and yellow pickaxe": (0.1080 + 0.1407) / 2,
    "Blue diamond gem": (0.1191 + 0.1475) / 2,
    "Open red double door": (0.1326 + 0.1738) / 2,
    "Orange and yellow magma texture": (0.1488 + 0.1536) / 2,
}
# New architecture has slightly different values. Not sure why
prompts_with_thresholds = {
    "Grey and yellow pickaxe": (0.1402 + 0.1822) / 2,
    "Blue diamond gem": (0.1265 + 0.1462) / 2,
    "Open red double door": (0.1134 + 0.1455) / 2,
    "Orange and yellow magma texture": (0.1550 + 0.1803) / 2,
}



class VisualMinecraftCLIPSimilarityModel(nn.Module):
    def __init__(
        self,
        *,
        model: CLIPEmbed,  # The CLIP model wrapper for embedding images and text
        prompts: torch.Tensor, # Tokenized tensor representing the prompts
    ) -> None:
        super().__init__()
        self.embed_module = model
        self.num_prompts = len(prompts)

        embedded_prompts = self.embed_prompts(prompts)
        self.register_buffer("_embedded_prompts", embedded_prompts)

        self._prompt_keys = list(prompts_with_thresholds.keys())
        prompt_thresholds = torch.tensor(list(prompts_with_thresholds.values()))
        self.register_buffer("_prompt_thresholds", prompt_thresholds)



    @torch.inference_mode()
    def forward(self, embedded_images: torch.Tensor) -> torch.Tensor:
        # We do not need to compute full cosine similarity as the norm is 1 in CLIP
        similarities = embedded_images @ self._embedded_prompts.T
        return similarities
        # similarities = F.softmax(similarities / self._temperature, dim=-1)
        # return similarities


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

    def labels_from_similarities(self, similarities: torch.Tensor) -> torch.Tensor:
        # similarities is a tensor of shape (batch_size, num_prompts)
        labels = similarities < self._prompt_thresholds.unsqueeze(0)
        return labels

    def compute_labels(self, frames: torch.Tensor) -> torch.Tensor:
        img_embedding = self.embed_images(frames)
        similarities = self(img_embedding)

        labels = self.labels_from_similarities(similarities)
        return labels


    def get_prompt_keys(self) -> List[str]:
        return self._prompt_keys


def load_visual_minecraft_similarity_model(
    model_name, cache_dir: Optional[str] = None
):
    model_name_prefix, pretrained = model_name.split("/")
    model = open_clip.create_model(
        model_name=model_name_prefix, pretrained=pretrained, cache_dir=cache_dir
    )
    prompts = list(prompts_with_thresholds.keys())
    tokenized_prompts = VisualMinecraftCLIPSimilarityModel.tokenize_prompts(prompts)
    model = CLIPEmbed(model)
    model = VisualMinecraftCLIPSimilarityModel(
        model=model, 
        prompts=tokenized_prompts,
    )
    return model.eval()

def compute_visual_minecraft_labels(
    model: VisualMinecraftCLIPSimilarityModel,
    frames: torch.Tensor,
    batch_size: int,
    num_workers: int,
    worker_frames_tensor=None,
) -> torch.Tensor:
    similarities = compute_visual_minecraft_similarities(model, frames, batch_size, num_workers, worker_frames_tensor)

    thresholds = torch.tensor(list(prompts_with_thresholds.values()), device=similarities.device)

    labels = similarities < thresholds.unsqueeze(0)
    print(similarities)

    return labels


def compute_visual_minecraft_rewards(
    model: VisualMinecraftCLIPSimilarityModel,
    frames: torch.Tensor,
    batch_size: int,
    num_workers: int,
    worker_frames_tensor=None,
) -> torch.Tensor:
    task = 0

    labels = compute_visual_minecraft_labels(model, frames, batch_size, num_workers, worker_frames_tensor).to(torch.float32)

    # from PIL import Image
    # for i, label in enumerate(labels):
    #     img = Image.fromarray(frames[i].cpu().numpy().transpose(1, 2, 0))
    #     img.save(f"label_{i}.png")

    return labels[:, task]

def compute_visual_minecraft_similarities(
    model: VisualMinecraftCLIPSimilarityModel,
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
    success_model: VisualMinecraftCLIPSimilarityModel,
    render_dim: Tuple[int, int, int],
    batch_size: int,
    num_workers: int,
    frames: torch.Tensor,
) -> torch.Tensor:
    ...


@overload
def dist_worker_compute_similarity(
    rank: int,
    success_model: VisualMinecraftCLIPSimilarityModel,
    render_dim: Tuple[int, int, int],
    batch_size: int,
    num_workers: int,
    frames: None = None,
) -> None:
    ...


def dist_worker_compute_similarity(
    rank: int,
    success_model: VisualMinecraftCLIPSimilarityModel,
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

    if rank == 0:
        if frames is None:
            raise ValueError("Must pass render result on rank=0")
        if len(frames) != num_workers * batch_size:
            raise ValueError("Must pass render result with correct batch size")
        scatter_list = [t.cuda(rank) for t in torch.chunk(frames, num_workers, dim=0)]
    else:
        scatter_list = []

    worker_frames = worker_frames_tensor if worker_frames_tensor is not None else torch.zeros((batch_size, *render_dim), dtype=torch.uint8).cuda(rank)
    dist.scatter(worker_frames, scatter_list=scatter_list, src=0)
    with torch.no_grad():
        embeddings = success_model.embed_module(worker_frames)
        similaries = success_model(embeddings)

    def zero_t():
        return torch.zeros_like(similaries)

    recv_similaries = [zero_t() for _ in range(num_workers)] if rank == 0 else []
    dist.gather(similaries, gather_list=recv_similaries, dst=0)

    if rank == 0:
        return torch.cat(recv_similaries, dim=0).cuda(rank)
    