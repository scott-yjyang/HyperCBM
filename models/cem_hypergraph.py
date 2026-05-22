import os
import logging

import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from torchvision.models import resnet50
import sklearn.metrics

from models.cbm import ConceptBottleneckModel
import train.utils as utils
from utils import visualize_and_save_heatmaps
from cem.metrics.accs import compute_accuracy
from .hypergraph import HyperConceptNet


class ConceptEmbeddingModel_Hypergraph(ConceptBottleneckModel):
    """HyperCBM: Concept Embedding Model with hypergraph-based concept interaction."""

    def __init__(
            self,
            n_concepts,
            n_tasks,
            emb_size=16,
            training_intervention_prob=0.25,
            embedding_activation="leakyrelu",
            shared_prob_gen=True,
            concept_loss_weight_labeled=1,
            concept_loss_weight_unlabeled=5,
            task_loss_weight=1,
            c2y_model=None,
            c2y_layers=None,
            c_extractor_arch=utils.wrap_pretrained_model(resnet50),
            output_latent=False,
            seed=42,
            optimizer="adam",
            momentum=0.9,
            learning_rate=0.01,
            weight_decay=4e-05,
            weight_loss=None,
            task_class_weights=None,
            tau=1,
            active_intervention_values=None,
            inactive_intervention_values=None,
            intervention_policy=None,
            output_interventions=False,
            use_concept_groups=False,
            top_k_accuracy=None,
    ):
        pl.LightningModule.__init__(self)
        self.n_concepts = n_concepts
        self.output_interventions = output_interventions
        self.intervention_policy = intervention_policy
        self.pre_concept_model = c_extractor_arch(output_dim=None)
        self.training_intervention_prob = training_intervention_prob
        self.output_latent = output_latent
        if self.training_intervention_prob != 0:
            self.ones = torch.ones(n_concepts)

        if active_intervention_values is not None:
            self.active_intervention_values = torch.tensor(active_intervention_values)
        else:
            self.active_intervention_values = torch.ones(n_concepts)
        if inactive_intervention_values is not None:
            self.inactive_intervention_values = torch.tensor(inactive_intervention_values)
        else:
            self.inactive_intervention_values = torch.ones(n_concepts)

        self.task_loss_weight = task_loss_weight
        self.concept_context_generators = torch.nn.ModuleList()
        self.concept_prob_generators = torch.nn.ModuleList()
        self.shared_prob_gen = shared_prob_gen
        self.top_k_accuracy = top_k_accuracy

        in_features = list(self.pre_concept_model.modules())[-1].out_features
        activation_map = {
            None: None,
            "sigmoid": torch.nn.Sigmoid(),
            "leakyrelu": torch.nn.LeakyReLU(),
            "relu": torch.nn.ReLU(),
        }
        activation = activation_map.get(embedding_activation)

        for i in range(n_concepts):
            layers = [torch.nn.Linear(in_features, 2 * emb_size)]
            if activation is not None:
                layers.append(activation)
            self.concept_context_generators.append(torch.nn.Sequential(*layers))

            if self.shared_prob_gen and len(self.concept_prob_generators) == 0:
                self.concept_prob_generators.append(torch.nn.Linear(2 * emb_size, 1))
            elif not self.shared_prob_gen:
                self.concept_prob_generators.append(torch.nn.Linear(2 * emb_size, 1))

        if c2y_model is None:
            units = [n_concepts * emb_size] + (c2y_layers or []) + [n_tasks]
            layers = []
            for i in range(1, len(units)):
                layers.append(torch.nn.Linear(units[i - 1], units[i]))
                if i != len(units) - 1:
                    layers.append(torch.nn.LeakyReLU())
            self.c2y_model = torch.nn.Sequential(*layers)
        else:
            self.c2y_model = c2y_model

        self.sigmoid = torch.nn.Sigmoid()
        self.loss_concept_labeled = torch.nn.BCELoss(weight=weight_loss)
        self.loss_concept_unlabeled = torch.nn.BCELoss(weight=weight_loss)
        self.loss_task = (
            torch.nn.CrossEntropyLoss(weight=task_class_weights)
            if n_tasks > 1 else torch.nn.BCEWithLogitsLoss(weight=task_class_weights)
        )

        self.concept_loss_weight_labeled = concept_loss_weight_labeled
        self.concept_loss_weight_unlabeled = concept_loss_weight_unlabeled
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.n_tasks = n_tasks
        self.emb_size = emb_size
        self.tau = tau
        self.use_concept_groups = use_concept_groups

        self.fc = nn.Linear(512, self.emb_size)
        self.pooling = nn.AdaptiveAvgPool2d(1)

        self.concept_hypergraph = HyperConceptNet(self.emb_size, 2 * self.emb_size)

    def _after_interventions(
            self,
            prob,
            pos_embeddings,
            neg_embeddings,
            intervention_idxs=None,
            c_true=None,
            train=False,
            competencies=None,
    ):
        if train and (self.training_intervention_prob != 0) and (
                (c_true is not None) and (intervention_idxs is None)
        ):
            mask = torch.bernoulli(self.ones * self.training_intervention_prob)
            intervention_idxs = torch.tile(mask, (c_true.shape[0], 1))
        if (c_true is None) or (intervention_idxs is None):
            return prob, intervention_idxs
        intervention_idxs = intervention_idxs.type(torch.FloatTensor)
        intervention_idxs = intervention_idxs.to(prob.device)
        return prob * (1 - intervention_idxs) + intervention_idxs * c_true, intervention_idxs

    def unlabeled_image_encoder(self, x):
        x = self.pre_concept_model.conv1(x)
        x = self.pre_concept_model.bn1(x)
        x = self.pre_concept_model.relu(x)
        x = self.pre_concept_model.maxpool(x)
        x = self.pre_concept_model.layer1(x)
        x = self.pre_concept_model.layer2(x)
        x = self.pre_concept_model.layer3(x)
        x = self.pre_concept_model.layer4(x)
        x = x.transpose(1, 3)
        x = self.fc(x)
        return x

    def _forward(
            self,
            x,
            c=None,
            y=None,
            l=None,
            train=False,
            latent=None,
            intervention_idxs=None,
            competencies=None,
            prev_interventions=None,
            output_embeddings=False,
            output_latent=None,
            output_interventions=None,
    ):
        if latent is None:
            pre_c = self.pre_concept_model(x)
            contexts = []
            c_sem = []

            for i, context_gen in enumerate(self.concept_context_generators):
                prob_gen = self.concept_prob_generators[0 if self.shared_prob_gen else i]
                context = context_gen(pre_c)
                prob = prob_gen(context)
                contexts.append(torch.unsqueeze(context, dim=1))
                c_sem.append(self.sigmoid(prob))

            c_sem = torch.cat(c_sem, axis=-1)
            contexts = torch.cat(contexts, axis=1)
            latent = contexts, c_sem
        else:
            contexts, c_sem = latent

        if (intervention_idxs is None) and (c is not None) and (
                self.intervention_policy is not None
        ):
            horizon = self.intervention_policy.num_groups_intervened
            if hasattr(self.intervention_policy, "horizon"):
                horizon = self.intervention_policy.horizon
            prior_distribution = self._prior_int_distribution(
                prob=c_sem,
                pos_embeddings=contexts[:, :, :self.emb_size],
                neg_embeddings=contexts[:, :, self.emb_size:],
                competencies=competencies,
                prev_interventions=prev_interventions,
                c=c,
                train=train,
                horizon=horizon,
            )
            intervention_idxs, c_int = self.intervention_policy(
                x=x, c=c, pred_c=c_sem, y=y,
                competencies=competencies,
                prev_interventions=prev_interventions,
                prior_distribution=prior_distribution,
            )
        else:
            c_int = c

        if not train:
            intervention_idxs = self._standardize_indices(
                intervention_idxs=intervention_idxs,
                batch_size=x.shape[0],
            )

        probs, intervention_idxs = self._after_interventions(
            c_sem,
            pos_embeddings=contexts[:, :, :self.emb_size],
            neg_embeddings=contexts[:, :, self.emb_size:],
            intervention_idxs=intervention_idxs,
            c_true=c_int,
            train=train,
            competencies=competencies,
        )

        # Mix positive and negative embeddings weighted by concept probabilities
        c_embedding = (
                contexts[:, :, :self.emb_size] * torch.unsqueeze(probs, dim=-1) +
                contexts[:, :, self.emb_size:] * (1 - torch.unsqueeze(probs, dim=-1))
        )

        # Hypergraph-based concept interaction
        c_embedding = self.concept_hypergraph(c_embedding)

        c_pred = c_embedding.view((-1, self.emb_size * self.n_concepts))
        y = self.c2y_model(c_pred)
        image_feature = self.unlabeled_image_encoder(x)

        # Compute spatial heatmap: [batch_size, n_concepts, H, W]
        heatmap = []
        for i in range(len(image_feature)):
            heatmap.append(torch.matmul(image_feature[i], c_embedding[i].transpose(0, 1)))
        heatmap = torch.stack(heatmap).permute(0, 3, 1, 2)
        c_pred_unlabeled = self.pooling(heatmap).squeeze()
        c_pred_unlabeled = self.sigmoid(c_pred_unlabeled)

        tail_results = []
        if output_interventions:
            if intervention_idxs is not None and isinstance(intervention_idxs, np.ndarray):
                intervention_idxs = torch.FloatTensor(intervention_idxs).to(x.device)
            tail_results.append(intervention_idxs)
        if output_latent:
            tail_results.append(latent)
        if output_embeddings:
            tail_results.append(contexts[:, :, :self.emb_size])
            tail_results.append(contexts[:, :, self.emb_size:])

        return tuple([c_sem, c_pred, c_pred_unlabeled, y, c_embedding] + tail_results)

    def predict_step(self, batch, batch_idx, intervention_idxs=None, dataloader_idx=0):
        x, y, c, l, nbr_c, nbr_w, competencies, prev_interventions = self._unpack_batch(batch)
        return self._forward(
            x, intervention_idxs=intervention_idxs, c=c, y=y, l=l,
            train=False, competencies=competencies,
            prev_interventions=prev_interventions, output_interventions=True,
        )

    def forward(
            self,
            x,
            c=None,
            y=None,
            l=None,
            latent=None,
            intervention_idxs=None,
            competencies=None,
            prev_interventions=None,
            output_embeddings=False,
            output_latent=None,
            output_interventions=None,
    ):
        return self._forward(
            x, train=False, c=c, y=y, l=l,
            competencies=competencies, prev_interventions=prev_interventions,
            intervention_idxs=intervention_idxs, latent=latent,
            output_embeddings=output_embeddings, output_latent=output_latent,
            output_interventions=output_interventions,
        )

    def _run_step(self, batch, batch_idx, train=False, intervention_idxs=None):
        x, y, c, l, nbr_c, nbr_w, competencies, prev_interventions = self._unpack_batch(batch)

        # Compute pseudo-labels from neighbor concepts
        nbr_w_ = nbr_w.unsqueeze(-1).repeat(1, 1, nbr_c.size(2))
        c_pseudo = torch.sum(nbr_c * nbr_w_, dim=1) / nbr_w.size(1)

        outputs = self._forward(
            x, intervention_idxs=intervention_idxs, c=c, y=y, l=l,
            train=train, competencies=competencies, prev_interventions=prev_interventions,
        )
        c_sem, c_pred_labeled, c_pred_unlabeled, y_pred, c_embedding = (
            outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]
        )

        if not train:
            self.val_embeddings.append(c_embedding.detach().cpu().numpy())

        task_loss = self.loss_task(y_pred, y)
        task_loss_scalar = task_loss.detach()

        concept_loss_labeled = self.loss_concept_labeled(c_sem[l], c[l])
        concept_loss_scalar_labeled = concept_loss_labeled.detach()

        concept_loss_unlabeled = self.loss_concept_unlabeled(
            c_pred_unlabeled.float()[~l], c_pseudo.float()[~l]
        )
        concept_loss_scalar_unlabeled = concept_loss_unlabeled.detach()

        loss = (task_loss
                + self.concept_loss_weight_labeled * concept_loss_labeled
                + self.concept_loss_weight_unlabeled * concept_loss_unlabeled)

        if y_pred.shape[1] == 2:
            (c_acc, c_auc, c_f1), (y_acc, y_auc, y_f1) = compute_accuracy(c_sem, y_pred[:, 1], c, y)
        else:
            (c_acc, c_auc, c_f1), (y_acc, y_auc, y_f1) = compute_accuracy(c_sem, y_pred, c, y)

        result = {
            "c_acc": c_acc,
            "c_auc": c_auc,
            "c_f1": c_f1,
            "y_acc": y_acc,
            "y_auc": y_auc,
            "y_f1": y_f1,
            "c_loss_labeled": concept_loss_scalar_labeled,
            "c_loss_unlabeled": concept_loss_scalar_unlabeled,
            "task_loss": task_loss_scalar,
            "loss": loss.detach(),
            "avg_c_y_acc": (c_acc + y_acc) / 2,
        }

        if self.top_k_accuracy is not None:
            y_true_cpu = y.reshape(-1).cpu().detach()
            y_pred_cpu = y_pred.cpu().detach()
            labels = list(range(self.n_tasks))
            if isinstance(self.top_k_accuracy, int):
                top_k_accuracy = list(range(1, self.top_k_accuracy))
            else:
                top_k_accuracy = self.top_k_accuracy
            for top_k_val in top_k_accuracy:
                if top_k_val:
                    if y_pred.shape[1] == 2:
                        y_top_k_accuracy = sklearn.metrics.top_k_accuracy_score(
                            y_true_cpu, y_pred_cpu[:, 1], k=top_k_val,
                        )
                    else:
                        y_top_k_accuracy = sklearn.metrics.top_k_accuracy_score(
                            y_true_cpu, y_pred_cpu, k=top_k_val, labels=labels,
                        )
                    result[f'y_top_{top_k_val}_accuracy'] = y_top_k_accuracy

        return loss, result

    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            logging.info(f"================================Epoch {self.current_epoch}===============================")
        loss, result = self._run_step(batch, batch_idx, train=True)
        for name, val in result.items():
            if name in ['c_f1', 'y_auc', 'avg_c_y_acc', 'y_f1']:
                continue
            self.log(name, val, prog_bar=True)
        return {
            "loss": loss,
            "log": {
                "c_accuracy": result['c_acc'],
                "c_auc": result['c_auc'],
                "c_f1": result['c_f1'],
                "y_accuracy": result['y_acc'],
                "y_auc": result['y_auc'],
                "y_f1": result['y_f1'],
                "concept_loss_labeled": result['c_loss_labeled'],
                "concept_loss_unlabeled": result['c_loss_unlabeled'],
                "task_loss": result['task_loss'],
                "loss": result['loss'],
                "avg_c_y_acc": result['avg_c_y_acc'],
            },
        }

    def validation_step(self, batch, batch_idx):
        if not hasattr(self, "val_embeddings"):
            self.val_embeddings = []
        _, result = self._run_step(batch, batch_idx, train=False)
        for name, val in result.items():
            if self.n_tasks <= 2:
                prog_bar = ("auc" in name)
            else:
                prog_bar = (("c_auc" in name) or ("y_accuracy" in name))
            self.log("val_" + name, val, prog_bar=prog_bar)
        return {"val_" + key: val for key, val in result.items()}

    def test_step(self, batch, batch_no):
        loss, result = self._run_step(batch, batch_no, train=False)
        for name, val in result.items():
            self.log("test_" + name, val, prog_bar=True)
        return result['loss']

    def plot_heatmap(
            self,
            x,
            x_show=None,
            c=None,
            y=None,
            img_name=None,
            output_dir='heatmap',
            concept_set=None,
    ):
        pre_c = self.pre_concept_model(x)
        contexts = []
        c_sem = []

        for i, context_gen in enumerate(self.concept_context_generators):
            prob_gen = self.concept_prob_generators[0 if self.shared_prob_gen else i]
            context = context_gen(pre_c)
            prob = prob_gen(context)
            contexts.append(torch.unsqueeze(context, dim=1))
            c_sem.append(self.sigmoid(prob))
        c_sem = torch.cat(c_sem, axis=-1)
        contexts = torch.cat(contexts, axis=1)

        probs = c_sem
        c_embedding = (
                contexts[:, :, :self.emb_size] * torch.unsqueeze(probs, dim=-1) +
                contexts[:, :, self.emb_size:] * (1 - torch.unsqueeze(probs, dim=-1))
        )
        image_feature = self.unlabeled_image_encoder(x)

        heatmap = []
        for i in range(len(image_feature)):
            heatmap.append(torch.matmul(image_feature[i], c_embedding[i].transpose(0, 1)))
        heatmap = torch.stack(heatmap).permute(0, 3, 1, 2)

        for i in range(len(x)):
            save_dir = os.path.join("./heatmap", str(img_name[i]))
            visualize_and_save_heatmaps(x_show[i], c[i], c_sem[i], heatmap[i], save_dir, concept_set)

    def visualize_concept_embeddings(self, c_embeddings, output_dir='./visual_plot'):
        """Visualize concept embeddings via t-SNE, colored by concept identity."""
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        os.makedirs(output_dir, exist_ok=True)

        batch_size, n_concepts, D = c_embeddings.shape
        c_embedding_flat = c_embeddings.reshape(-1, D).cpu().detach().numpy()

        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(c_embedding_flat)

        concept_labels = np.tile(np.arange(n_concepts), batch_size)

        plt.figure(figsize=(12, 8))
        cmap = plt.cm.get_cmap('tab20', n_concepts)

        for concept_id in range(n_concepts):
            mask = concept_labels == concept_id
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                color=cmap(concept_id),
                label=f'Concept {concept_id}',
                alpha=0.6,
                s=40,
                edgecolor='w',
            )
            center = np.median(embeddings_2d[mask], axis=0)
            plt.scatter(
                center[0], center[1],
                color='black', marker='X', s=100, zorder=5, alpha=0.9,
            )

        plt.title(
            f'Concept Embedding Visualization (Validation Epoch {self.current_epoch})\n'
            f'(Color by Concept, X marks cluster center)'
        )
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1, frameon=True, title='Concepts')
        plt.tight_layout()

        output_path = os.path.join(output_dir, f'concept_embeddings_epoch_{self.current_epoch}.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved validation visualization to {output_path}")
