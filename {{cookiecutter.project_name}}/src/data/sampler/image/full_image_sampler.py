from typing import Dict, List
from src.data.sampler.base import BaseSampler


class FullImageSampler(BaseSampler):
    def __len__(self) -> int:
        return len(self.dataset)

    def prepare_sampling(self, sampling_information: List) -> None:
        self.sampling_information = sampling_information

    def sample(self, idx: int) -> Dict:
        img_size = self.sampling_information[idx]["img_size"]
        return self.dataset.get_patch(idx, (int(img_size[0]/2), int(img_size[1]/2)), img_size)