from transformers import AutoModel, AutoProcessor
from config.features import VisualFeatureConfig
from transformers import CLIPImageProcessor
from config.features import VisBackBoneModels

from sklearn import preprocessing
import timm
from sklearn.decomposition import PCA
import numpy as np
import torch
import cv2
from PIL import Image

from sklearn import preprocessing
import skimage.transform
from einops import rearrange
import torchvision.transforms as T


class VisualFeaturePreprocessor:

    def __init__(self, conf: VisualFeatureConfig):
        self.config = conf

        self.model = None
        if self.config.backbone_config.backbone_type == VisBackBoneModels.RADIO:
            self.model = Radio(self.config.backbone_config)

        if self.config.backbone_config.backbone_type == VisBackBoneModels.DINO:
            self.model = DINO(self.config.backbone_config)

        if self.config.backbone_config.backbone_type == VisBackBoneModels.DINO_FEATUP:
            self.model = DINO_Featup(self.config.backbone_config)


class HFModel:

    def __init__(self):
        self.model = None
        self.processor = None

    def infer(self, img, return_local=True, return_global=True):
        raise NotImplementedError

    def get_local_features(self, img):
        return self.infer(img, True, False)

    def get_global_features(self, img):
        return self.infer(img, False, True)

    def get_features(self, img):
        return self.infer(img)

    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx : min(ndx + n, l)]

    def pca_vis(results, title="pca_path_img_"):
        # Shape: N_results x Patch dim x Embedding Size
        results = results.cpu().numpy()

        n = results.shape[0]
        patch_dim = results.shape[1]
        embedding_dim = results.shape[2]

        pca = PCA(n_components=3)
        reduced = pca.fit_transform(results.reshape(-1, embedding_dim))
        reduced_ims = reduced.reshape(n, patch_dim, 3)

        for i in range(n):
            scaler = preprocessing.MinMaxScaler()
            d = scaler.fit_transform(reduced_ims[i])

            dim = int(np.sqrt(patch_dim))
            pseudo_image = d.reshape((dim, dim, 3))
            pseudo_image = np.round(pseudo_image * 255).astype(np.uint8)

            cv2.imwrite(
                "/home/marek/Desktop/aope_for_deformed_objects/debug_output/{}_{}x{}_{}%_{}.jpg".format(
                    title, dim, dim, int(np.sum(pca.explained_variance_ratio_ * 100)), i
                ),
                pseudo_image,
            )

    def return_merged_features(batch_features, return_local, return_global):
        features = torch.cat(batch_features)

        if return_local:
            return features[:, 1:, :]
        if return_global:
            return features[:, 0, :]
        else:
            return features[:, 1:, :], features[:, 0, :]

    def get_closest_matching_resolution(
        self, img, patch_size, min_size=224, max_size=1024
    ):

        h, w = img.shape[0], img.shape[1]
        if h < min_size or w < min_size:
            raise ValueError("Image is too small. Minimum size is {}x{}.".format(min_size, min_size))
        if h > max_size or w > max_size:
            raise ValueError("Image is too large. Maximum size is {}x{}.".format(max_size, max_size))

        # DINOv2 supports resolutions that are multiples of 14
        h = int(np.ceil(h / patch_size) * patch_size)
        w = int(np.ceil(w / patch_size) * patch_size)

        return h, w


class Radio(HFModel):

    def __init__(self, model_conf):
        self.model_version = model_conf.model_version or "nvidia/RADIO"
        self.model = AutoModel.from_pretrained(
            self.model_version, trust_remote_code=True
        )
        self.model.eval().cuda()
        self.image_processor = CLIPImageProcessor.from_pretrained(self.model_version)
        self.batch_size = 1

    def infer(self, imgs, return_local=True, return_global=True):
        summaries = []
        features = []
        for i in range(0, len(imgs), self.batch_size):
            batch = imgs[i:min(i+self.batch_size, len(imgs))]

            pixel_values = self.image_processor(
                images=batch, return_tensors="pt", do_resize=False
            ).pixel_values
            pixel_values = pixel_values.cuda()

            result = self.model(pixel_values)

            summary = result[0].detach().cpu()
            feature = result[1].detach().cpu()

            torch.cuda.synchronize()

            summaries.append(summary)
            features.append(feature)

        if not return_local:
            return torch.cat(summaries)
        elif not return_global:
            return torch.cat(features)
        return torch.cat(summaries), torch.cat(features)


class DINO(HFModel):

    def __init__(self, model_conf):
        self.model_version = model_conf.model_version or "dinov2_vits14_reg_lc"
        self.patch_size = model_conf.patch_size or 14
        # self.model =  torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg_lc')
        self.model = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m", pretrained=True
        )
        self.model.eval().cuda()
        self.batch_size = 1

    def resize_imgs(self, imgs):
        resized_imgs = []
        for img in imgs:
            h, w = HFModel.get_closest_matching_resolution(img, self.patch_size)
            img = skimage.transform.resize(img, (h, w), anti_aliasing=False)
            resized_imgs.append(img)

        return resized_imgs

    def infer(self, imgs, return_local=True, return_global=True):
        imgs = self.resize_imgs(imgs)
        imgs = rearrange(imgs, "b h w c -> b c h w")

        features = []
        for batch in HFModel.batch(imgs, self.batch_size):
            # skimage.transform.resize(img, (1, 3, h, w), anti_aliasing=False)
            resized_imgs = [torch.from_numpy(img).float().cuda() for img in batch]

            if len(resized_imgs) == 1:
                resized_imgs = torch.cat(resized_imgs).unsqueeze(0)
            else:
                resized_imgs = torch.cat(resized_imgs)

            features.append(self.model.forward_features(resized_imgs).detach().cpu())

        return HFModel.return_merged_features(
            torch.cat(features), return_local, return_global
        )


class DINO_Featup(HFModel):

    def __init__(self, model_conf):
        self.use_norm = True
        self.batch_size = 1

        self.path_size = model_conf.patch_size or 14
        self.max_input_size = 490

        self.model = torch.hub.load(
            "mhamilton723/FeatUp", "dinov2", use_norm=self.use_norm
        ).cuda()
        self.model.eval()

    def infer(self, imgs, return_local=True, return_global=True):

        h, w = HFModel.get_closest_matching_resolution(self, imgs[0], self.path_size)
        input_size = min(min(h, w), self.max_input_size)

        img_transform = T.Compose(
            [
                T.Resize(input_size),
                T.CenterCrop((input_size, input_size)),
                T.ToTensor(),
            ]
        )

        features = []
        for batch in HFModel.batch(imgs, self.batch_size):
            transformed_imgs = [
                img_transform(Image.fromarray(img.astype(np.uint8))) for img in batch
            ]

            if len(transformed_imgs) == 1:
                transformed_imgs = (
                    torch.cat(transformed_imgs).unsqueeze(0).float().cuda()
                )
            else:
                transformed_imgs = torch.cat(transformed_imgs).float().cuda()

            hr_feats = self.model(transformed_imgs).detach().cpu()

            # Most models return patch features flattened
            hr_feats = torch.flatten(hr_feats, 2, 3)
            hr_feats = torch.swapaxes(hr_feats, 1, 2)

            features.append(hr_feats)

        return torch.cat(features)
