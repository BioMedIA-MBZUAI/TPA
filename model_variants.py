import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import open_clip


def calculate_class_weights(labels, num_classes):
    """Calculate balanced class weights with clamping to prevent extreme values"""
    total_samples = len(labels)
    class_counts = torch.bincount(torch.tensor(labels), minlength=num_classes)
    weights = total_samples / (num_classes * class_counts.float())
    weights = weights.clamp(min=0.1, max=10.0)  # Prevent extreme weights
    return weights


class FocalLoss(nn.Module):

    def __init__(self, num_classes, alpha=None, gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma

        # Handle class weights (alpha)
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        else:
            self.alpha = torch.as_tensor(alpha)

    def forward(self, inputs, targets):
        self.alpha = self.alpha.to(inputs.device)
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)

        alpha_t = self.alpha[targets]

        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()


class BaseFetalCLIPClassifierWithText(pl.LightningModule):
    """Base class with common functionality when dealing with frame-text pairs"""

    def __init__(
        self,
        num_classes=2,
        w_contr=1,
        temperature=0.07,
        learning_rate=1e-3,
        class_weights=None,
        encoder_checkpoint=None,
        backmodel="fetalclip",
        class_names=None,
        num_prompts_per_class=1,
        selected_prompt_indices=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.backbone_model = backmodel
        self.num_prompts_per_class = num_prompts_per_class
        # Load FetalCLIP model
        fetalclip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            "FetUSCLIP", pretrained=encoder_checkpoint
        )
        print("Using FetalCLIP as the encoder")
        fetalclip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            "FetUSCLIP", pretrained=encoder_checkpoint
        )

        self.image_encoder = fetalclip_model.visual

        self.image_encoder.eval()
        self.embedding_dim = self.image_encoder.proj.shape[1]

        self.text_encoder = fetalclip_model.transformer
        self.text_encoder.eval()

        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.embedding_dim = self.image_encoder.proj.shape[1]
        self.fetalclip_model = fetalclip_model
        self.fetalclip_model.eval()
        # Text pair
        self.train_text_prompts = []
        self.val_text_prompts = []

        self.selected_prompt_indices = selected_prompt_indices
        self.normal_templates = [
            "Is the fetal heart normal in this 4CH ultrasound view?",
            "Does this 4CH image show a normal heart condition?",
            "Is everything normal in this fetal heart 4CH scan?",
            "Does this ultrasound suggest a healthy fetal heart?",
            "Does this 4CH scan reflect a structurally normal heart?",
            "normal",
        ]

        self.abnormal_templates = [
            "Is {} visible in this fetal heart 4CH scan?",
            "Does this image show a fetal heart affected by {}?",
            "Can {} be detected in this fetal heart 4CH image?",
            "Does this 4CH scan suggest the presence of {}?",
            "Does this ultrasound reveal {} in the fetal heart?",
            "{}",
        ]

        for name in class_names:
            if name == "Normal":
                indices = self.selected_prompt_indices.get(
                    "Normal", list(range(len(self.normal_templates)))
                )
                for idx in indices:
                    template = self.normal_templates[idx]
                    prompt = f"{template}"
                    self.train_text_prompts.append(prompt)
            else:
                indices = self.selected_prompt_indices.get(
                    "Abnormal", list(range(len(self.abnormal_templates)))
                )
                for idx in indices:
                    template = self.abnormal_templates[idx]
                    prompt = f"{template.format(name)}"
                    self.train_text_prompts.append(prompt)

        self.val_text_prompts = list(self.train_text_prompts)
        print(f"train_text_prompts: {self.train_text_prompts}")

        self._encode_text_prompts()
        self.temperature = temperature
        self.w_contr = w_contr
        self.training_criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.val_criterion = nn.CrossEntropyLoss(weight=class_weights)

        self._init_metrics()
        self.validation_step_outputs = []
        self.confusion = torchmetrics.ConfusionMatrix(
            task="multiclass", num_classes=num_classes
        )

    def _encode_text_prompts(self):
        text_encoder_device = next(self.text_encoder.parameters()).device

        train_tokens = open_clip.tokenize(self.train_text_prompts).to(
            text_encoder_device
        )
        with torch.no_grad():
            train_token_embeddings = self.fetalclip_model.token_embedding(train_tokens)
            train_text_features = self.text_encoder(train_token_embeddings)
        self.train_text_features = train_text_features / train_text_features.norm(
            dim=-1, keepdim=True
        )

        val_tokens = open_clip.tokenize(self.val_text_prompts).to(text_encoder_device)
        with torch.no_grad():
            val_token_embeddings = self.fetalclip_model.token_embedding(val_tokens)
            val_text_features = self.text_encoder(val_token_embeddings)
        self.val_text_features = val_text_features / val_text_features.norm(
            dim=-1, keepdim=True
        )

        self.train_text_features = self.train_text_features.to("cuda")
        self.val_text_features = self.val_text_features.to("cuda")

    def _init_metrics(self):
        """Initialize metrics based on classification type"""
        self.train_metrics = torchmetrics.MetricCollection(
            {
                "accuracy": torchmetrics.Accuracy(task="binary"),
                "f1": torchmetrics.F1Score(task="binary"),
            }
        )
        self.val_metrics = torchmetrics.MetricCollection(
            {
                "accuracy": torchmetrics.Accuracy(task="binary"),
                "f1": torchmetrics.F1Score(task="binary"),
                "auroc": torchmetrics.AUROC(task="binary"),
                "precision": torchmetrics.Precision(task="binary"),
                "recall": torchmetrics.Recall(task="binary"),
                "specificity": torchmetrics.Specificity(task="binary"),
            }
        )

    def on_train_epoch_start(self):
        if self.trainer.is_global_zero:
            print(f"\nEpoch {self.current_epoch} training prompts:")
            for prompt in self.train_text_prompts:
                print(f"  - {prompt}")

    def training_step(self, batch, batch_idx):

        if batch_idx == 0 and self.current_epoch == 0:
            frozen = all(not p.requires_grad for p in self.image_encoder.parameters())
            print(f"\nVerifying FetalCLIP encoder is frozen: {frozen}")
            if not frozen:
                raise RuntimeError("FetalCLIP encoder is not properly frozen!")

        x, y_true, _, _, pos_idx, neg_idx = batch
        logits, video_proj, pos_text_proj, neg_text_proj, kl_loss = self(
            x, (pos_idx, neg_idx)
        )

        pos_sim = F.cosine_similarity(video_proj, pos_text_proj, dim=-1)  # [B]
        neg_sim = F.cosine_similarity(video_proj, neg_text_proj, dim=-1)  # [B]

        target = torch.ones_like(pos_sim)
        margin = 0.5
        contrastive_loss = F.margin_ranking_loss(
            pos_sim, neg_sim, target, margin=margin
        )
        class_loss = self.training_criterion(logits, y_true)
        total_loss = class_loss + self.w_contr * contrastive_loss + 0.1 * kl_loss
        self.log("train_kl_loss", kl_loss, prog_bar=True, batch_size=len(y_true))
        probs = torch.softmax(logits, dim=1)[:, 1]
        metrics = self.train_metrics(probs, y_true.float())

        pos_sim_mean = pos_sim.mean()
        neg_sim_mean = neg_sim.mean()
        self.log(
            "train_pos_cos_sim", pos_sim_mean, prog_bar=True, batch_size=len(y_true)
        )
        self.log(
            "train_neg_cos_sim", neg_sim_mean, prog_bar=True, batch_size=len(y_true)
        )
        self.log("train_class_loss", class_loss, prog_bar=True, batch_size=len(y_true))
        self.log(
            "train_contrastive_loss",
            contrastive_loss,
            prog_bar=True,
            batch_size=len(y_true),
        )
        self.log("train_total_loss", total_loss, prog_bar=True, batch_size=len(y_true))

        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("learning_rate", current_lr, prog_bar=True, batch_size=len(y_true))

        for k, v in metrics.items():
            self.log(f"train_{k}", v, prog_bar=True, batch_size=len(y_true))

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y_true, _, video_names, pos_idx, neg_idx = batch

        num_samples = 5
        batch_size = x.size(0)
        all_logits = []

        for _ in range(num_samples):
            x_temp = x.reshape(-1, 3, x.size(3), x.size(4))

            with torch.no_grad():
                frame_embeddings = self.image_encoder(x_temp)
            frame_embeddings = frame_embeddings.reshape(
                batch_size, x.size(2), -1
            ).permute(0, 2, 1)
            temporal_features = self.temporal_cnn(frame_embeddings).squeeze(-1)

            pos_embeddings = self.val_text_features[pos_idx].mean(dim=1)
            pos_embeds = self.text_projection(pos_embeddings)

            prior_z, _, _ = self.prior_encoder(temporal_features)
            modulated_features = self.modulator(temporal_features, prior_z)

            frame_text_concat = torch.cat([modulated_features, pos_embeds], dim=1)
            logits = self.classifier(frame_text_concat)
            all_logits.append(logits)

        # --------- UNCERTAINTY AGGREGATION ---------
        logits_stack = torch.stack(all_logits, dim=0)  # [K, B, 2]
        mean_logits = logits_stack.mean(dim=0)  # [B, 2]
        probs = F.softmax(mean_logits, dim=1)  # [B, 2]
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)  # [B]
        logits_var = logits_stack.var(dim=0).mean(dim=1)  # [B]

        # --------- CLASSIFICATION METRICS ---------
        class_loss = self.val_criterion(mean_logits, y_true)
        probs_class1 = probs[:, 1]
        metrics = self.val_metrics(probs_class1, y_true.float())

        self.log("val_entropy", entropy.mean(), batch_size=len(y_true))
        self.log("val_logits_var", logits_var.mean(), batch_size=len(y_true))
        self.log("val_class_loss", class_loss, batch_size=len(y_true))
        self.log("val_total_loss", class_loss, batch_size=len(y_true))

        for k, v in metrics.items():
            self.log(f"val_{k}", v, batch_size=len(y_true))

        self.validation_step_outputs.append(
            {
                "loss": class_loss,
                "probs": probs_class1.detach(),
                "targets": y_true.detach(),
                "entropy": entropy.detach(),
                "video_names": video_names,
            }
        )

        return {"loss": class_loss, "probs": probs_class1, "targets": y_true}

    def on_validation_epoch_start(self):
        self.validation_step_outputs.clear()
        if self.trainer.is_global_zero:
            print("\nValidation prompts:")
            for prompt in self.train_text_prompts:
                print(f"  - {prompt}")

    def on_validation_epoch_end(self):
        all_probs = torch.cat([x["probs"] for x in self.validation_step_outputs])
        all_targets = torch.cat([x["targets"] for x in self.validation_step_outputs])

        predictions = (all_probs >= 0.5).long()

        if self.trainer.is_global_zero:
            print("\nDetailed Validation Results:")

            confusion = torchmetrics.ConfusionMatrix(
                task="binary" if self.num_classes == 2 else "multiclass",
                num_classes=self.num_classes,
            ).to(self.device)

            confusion.update(predictions, all_targets)
            conf_matrix = confusion.compute()
            print("\nConfusion Matrix:")
            print(conf_matrix)

            if self.num_classes > 2:
                print("\nPer-Class Performance:")
                for i in range(self.num_classes):
                    class_mask = all_targets == i
                    if class_mask.sum() > 0:
                        class_correct = (predictions[class_mask] == i).sum()
                        class_total = class_mask.sum()
                        class_acc = (class_correct / class_total).item()
                        true_positives = ((predictions == i) & (all_targets == i)).sum()
                        false_positives = (
                            (predictions == i) & (all_targets != i)
                        ).sum()
                        false_negatives = (
                            (predictions != i) & (all_targets == i)
                        ).sum()
                        precision = (
                            true_positives / (true_positives + false_positives)
                            if (true_positives + false_positives) > 0
                            else 0
                        )
                        recall = (
                            true_positives / (true_positives + false_negatives)
                            if (true_positives + false_negatives) > 0
                            else 0
                        )
                        f1 = (
                            2 * (precision * recall) / (precision + recall)
                            if (precision + recall) > 0
                            else 0
                        )
                        print(f"\nClass {i}:")
                        print(f"  Samples: {class_total}")
                        print(f"  Accuracy: {class_acc:.4f}")
                        print(f"  Precision: {precision:.4f}")
                        print(f"  Recall: {recall:.4f}")
                        print(f"  F1-Score: {f1:.4f}")

            confusion.reset()
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        """Base optimizer configuration"""
        trainable_params = [p for p in self.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.1, patience=5, verbose=True
        )

        if self.num_classes > 2:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_f1_macro",
                    "frequency": 1,
                },
            }
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_f1",
                    "frequency": 1,
                },
            }


class LatentEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)

    def forward(self, x):
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar


class FeatureModulator(nn.Module):
    def __init__(self, feature_dim, latent_dim):
        super().__init__()
        self.linear = nn.Linear(latent_dim, feature_dim)

    def forward(self, features, z):
        scale = torch.sigmoid(self.linear(z))
        return features * scale


class CNNFetalCLIPWithTextEmbeddings(BaseFetalCLIPClassifierWithText):
    "1D approach with text embeddings"

    def __init__(self, pooling="max", **kwargs):
        super().__init__(**kwargs)
        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(self.embedding_dim, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
        )
        self.text_projection = nn.Linear(self.embedding_dim, 256)

        if pooling == "avg":
            self.temporal_cnn = nn.Sequential(
                *list(self.temporal_cnn.children()), nn.AdaptiveAvgPool1d(1)
            )
        elif pooling == "max":
            self.temporal_cnn = nn.Sequential(
                *list(self.temporal_cnn.children()), nn.AdaptiveMaxPool1d(1)
            )

        # output is hard-coded here
        self.classifier = nn.Sequential(
            nn.Linear(256 + 256, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Linear(256, 2)
        )

        self.latent_dim = 128
        self.prior_encoder = LatentEncoder(input_dim=256, latent_dim=self.latent_dim)
        self.posterior_encoder = LatentEncoder(
            input_dim=256 + self.num_classes, latent_dim=self.latent_dim
        )
        self.modulator = FeatureModulator(feature_dim=256, latent_dim=self.latent_dim)

    def compute_kl(self, mu_q, logvar_q, mu_p, logvar_p):
        """
        Computes KL divergence between two Gaussians: q(z|x,y) and p(z|x)
        """
        kl = 0.5 * torch.sum(
            logvar_p
            - logvar_q
            + (torch.exp(logvar_q) + (mu_q - mu_p).pow(2)) / torch.exp(logvar_p)
            - 1,
            dim=1,
        )
        return kl.mean()

    def forward(self, x, text_idx):

        batch_size = x.size(0)
        num_frames = x.size(2)
        x = x.permute(0, 2, 1, 3, 4)

        x = x.reshape(-1, x.size(2), x.size(3), x.size(4))
        with torch.no_grad():
            frame_embeddings = self.image_encoder(x)

        frame_embeddings = frame_embeddings.reshape(
            batch_size, num_frames, -1
        )  # [batch size,frames,embedding dim]
        frame_embeddings = frame_embeddings.permute(
            0, 2, 1
        )  # [batch size,embedding dim,frames]
        temporal_features = self.temporal_cnn(frame_embeddings)
        temporal_features = temporal_features.squeeze(-1)  # [batch size,256]

        # CASE 1: per_sample → tuple of (pos_idx, neg_idx)
        if isinstance(text_idx, tuple):
            pos_idx, neg_idx = text_idx

            if self.training:
                pos_embeddings = self.train_text_features[pos_idx]
                neg_embeddings = self.train_text_features[neg_idx]
            else:
                pos_embeddings = self.train_text_features[pos_idx]
                neg_embeddings = self.train_text_features[neg_idx]

            pos_embeddings = pos_embeddings.mean(dim=1)  # [B, embed_dim]
            neg_embeddings = neg_embeddings.mean(dim=1)

            pos_embeds = self.text_projection(pos_embeddings)  # [B, 256]
            neg_embeds = self.text_projection(neg_embeddings)
            class_ids = pos_idx
            y_onehot = F.one_hot(
                class_ids, num_classes=self.num_classes
            ).float()  # [B, num_classes]
            # ----- LATENT PATH -----
            posterior_input = torch.cat([temporal_features, y_onehot], dim=1)
            posterior_z, mu_post, logvar_post = self.posterior_encoder(posterior_input)
            prior_z, mu_prior, logvar_prior = self.prior_encoder(temporal_features)
            kl_loss = self.compute_kl(mu_post, logvar_post, mu_prior, logvar_prior)
            # ----- MODULATION -----
            modulated_features = self.modulator(temporal_features, posterior_z)
            frame_text_concat = torch.cat([modulated_features, pos_embeds], dim=1)
            logits = self.classifier(frame_text_concat)
            modulated_features = F.normalize(modulated_features, dim=-1)

            # frame_text_concat = torch.cat([temporal_features, pos_embeds], dim=1)  # [B, 512]
            # logits = self.classifier(frame_text_concat)  # [B, 2]

            # # Normalize for contrastive loss
            # temporal_features = F.normalize(temporal_features, dim=-1)
            pos_embeds = F.normalize(pos_embeds, dim=-1)
            neg_embeds = F.normalize(neg_embeds, dim=-1)
            return logits, modulated_features, pos_embeds, neg_embeds, kl_loss

            # return logits, temporal_features, pos_embeds, neg_embeds
        else:
            if self.training:
                text_embeddings = self.train_text_features[text_idx]
            else:
                text_embeddings = self.train_text_features[text_idx]

            text_embeddings = text_embeddings.mean(dim=1)  # [B, embed_dim]
            text_embeds = self.text_projection(text_embeddings)  # [B, 256]

            frame_text_concat = torch.cat([temporal_features, text_embeds], dim=1)
            logits = self.classifier(frame_text_concat)

            temporal_features = F.normalize(temporal_features, dim=-1)
            text_embeds = F.normalize(text_embeds, dim=-1)

            return logits, temporal_features, text_embeds

    def extract_embeddings(self, x):
        self.eval()
        with torch.no_grad():
            batch_size = x.size(0)
            num_frames = x.size(2)
            x = x.permute(0, 2, 1, 3, 4)
            x = x.reshape(-1, x.size(2), x.size(3), x.size(4))

            frame_embeddings = self.image_encoder(x)
            frame_embeddings = frame_embeddings.reshape(batch_size, num_frames, -1)
            frame_embeddings = frame_embeddings.permute(0, 2, 1)
            temporal_features = self.temporal_cnn(frame_embeddings)
            temporal_features = temporal_features.squeeze(-1)  # [B, 256]

            temporal_features = F.normalize(temporal_features, dim=-1)
            return temporal_features
