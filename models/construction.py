import copy
import numpy as np
import os
import pytorch_lightning as pl
import torch

from torchvision.models import resnet18, resnet34, resnet50, densenet121

import models.cem as models_cem
import models.cbm as models_cbm
import models.cem_hypergraph as models_cem_hypergraph
import train.utils as utils


def _resolve_backbone(config):
    if isinstance(config["c_extractor_arch"], str):
        backbones = {
            "resnet18": resnet18,
            "resnet34": resnet34,
            "resnet50": resnet50,
            "densenet121": densenet121,
        }
        name = config["c_extractor_arch"]
        if name not in backbones:
            raise ValueError(f'Invalid c_extractor_arch "{name}"')
        return backbones[name]
    return config["c_extractor_arch"]


def construct_model(
        n_concepts,
        n_tasks,
        config,
        c2y_model=None,
        x2c_model=None,
        imbalance=None,
        task_class_weights=None,
        intervention_policy=None,
        active_intervention_values=None,
        inactive_intervention_values=None,
        output_latent=False,
        output_interventions=False,
):
    task_loss_weight = config.get('task_loss_weight', 1.0)
    architecture = config["architecture"]

    if architecture in ["ConceptEmbeddingModel", "MixtureEmbModel"]:
        model_cls = models_cem.ConceptEmbeddingModel
        extra_params = {
            "emb_size": config["emb_size"],
            "shared_prob_gen": config.get("shared_prob_gen", True),
            "intervention_policy": intervention_policy,
            "training_intervention_prob": config.get(
                'training_intervention_prob', 0.25,
            ),
            "seed": config.get("seed", 42),
            "embedding_activation": config.get(
                "embedding_activation", "leakyrelu",
            ),
            "c2y_model": c2y_model,
            "c2y_layers": config.get("c2y_layers", []),
        }
        if "embeding_activation" in config:
            extra_params["embedding_activation"] = config["embeding_activation"]

    elif architecture == "ConceptEmbeddingModel_Hypergraph":
        model_cls = models_cem_hypergraph.ConceptEmbeddingModel_Hypergraph
        extra_params = {
            "emb_size": config["emb_size"],
            "shared_prob_gen": config.get("shared_prob_gen", True),
            "intervention_policy": intervention_policy,
            "training_intervention_prob": config.get(
                'training_intervention_prob', 0.25,
            ),
            "seed": config.get("seed", 42),
            "embedding_activation": config.get(
                "embedding_activation", "leakyrelu",
            ),
            "c2y_model": c2y_model,
            "c2y_layers": config.get("c2y_layers", []),
        }
        if "embeding_activation" in config:
            extra_params["embedding_activation"] = config["embeding_activation"]

    elif "ConceptBottleneckModel" in architecture:
        model_cls = models_cbm.ConceptBottleneckModel
        extra_params = {
            "bool": config["bool"],
            "extra_dims": config["extra_dims"],
            "sigmoidal_extra_capacity": config.get(
                "sigmoidal_extra_capacity", True,
            ),
            "seed": config.get("seed", 42),
            "sigmoidal_prob": config.get("sigmoidal_prob", True),
            "intervention_policy": intervention_policy,
            "bottleneck_nonlinear": config.get("bottleneck_nonlinear", None),
            "active_intervention_values": active_intervention_values,
            "inactive_intervention_values": inactive_intervention_values,
            "x2c_model": x2c_model,
            "c2y_model": c2y_model,
            "c2y_layers": config.get("c2y_layers", []),
        }

    else:
        raise ValueError(f'Invalid architecture "{architecture}"')

    c_extractor_arch = _resolve_backbone(config)

    return model_cls(
        n_concepts=n_concepts,
        n_tasks=n_tasks,
        weight_loss=(
            torch.FloatTensor(imbalance)
            if config['weight_loss'] and (imbalance is not None)
            else None
        ),
        task_class_weights=(
            torch.FloatTensor(task_class_weights)
            if (task_class_weights is not None)
            else None
        ),
        concept_loss_weight_labeled=config['concept_loss_weight_labeled'],
        concept_loss_weight_unlabeled=config['concept_loss_weight_unlabeled'],
        task_loss_weight=task_loss_weight,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        c_extractor_arch=utils.wrap_pretrained_model(c_extractor_arch),
        optimizer=config['optimizer'],
        top_k_accuracy=config.get('top_k_accuracy'),
        output_latent=output_latent,
        output_interventions=output_interventions,
        **extra_params,
    )


def construct_sequential_models(
        n_concepts,
        n_tasks,
        config,
        imbalance=None,
        task_class_weights=None,
):
    assert config.get('extra_dims', 0) == 0, (
        "Sequential/joint models require extra_dims == 0."
    )
    c_extractor_arch = _resolve_backbone(config)

    try:
        x2c_model = c_extractor_arch(
            pretrained=config.get('pretrain_model', True),
        )
        if c_extractor_arch == densenet121:
            x2c_model.classifier = torch.nn.Linear(1024, n_concepts)
        elif hasattr(x2c_model, 'fc'):
            x2c_model.fc = torch.nn.Linear(512, n_concepts)
    except Exception:
        x2c_model = c_extractor_arch(output_dim=n_concepts)

    x2c_model = utils.WrapperModule(
        n_tasks=n_concepts,
        model=x2c_model,
        weight_loss=(
            torch.FloatTensor(imbalance)
            if config['weight_loss'] and (imbalance is not None)
            else None
        ),
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        optimizer=config['optimizer'],
        binary_output=True,
        sigmoidal_output=True,
    )

    c2y_layers = config.get('c2y_layers', [])
    units = [n_concepts] + (c2y_layers or []) + [n_tasks]
    layers = [
        torch.nn.Linear(units[i - 1], units[i])
        for i in range(1, len(units))
    ]
    c2y_model = utils.WrapperModule(
        n_tasks=n_tasks,
        model=torch.nn.Sequential(*layers),
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        optimizer=config['optimizer'],
        top_k_accuracy=config.get('top_k_accuracy'),
        binary_output=False,
        sigmoidal_output=False,
        weight_loss=(
            torch.FloatTensor(task_class_weights)
            if (task_class_weights is not None)
            else None
        ),
    )
    return x2c_model, c2y_model


def load_trained_model(
        config,
        n_tasks,
        result_dir,
        n_concepts,
        split=0,
        imbalance=None,
        task_class_weights=None,
        train_dl=None,
        logger=False,
        accelerator="auto",
        devices="auto",
        intervention_policy=None,
        intervene=False,
        output_latent=False,
        output_interventions=False,
        enable_checkpointing=False,
):
    model_saved_path = os.path.join(result_dir, 'test.pt')

    active_intervention_values = None
    inactive_intervention_values = None

    if (
            ((intervention_policy is not None) or intervene) and
            (train_dl is not None) and
            (config['architecture'] == "ConceptBottleneckModel") and
            (not config.get('sigmoidal_prob', True))
    ):
        model = construct_model(
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            config=config,
            imbalance=imbalance,
            task_class_weights=task_class_weights,
            output_latent=output_latent,
            output_interventions=output_interventions,
        )
        model.load_state_dict(torch.load(model_saved_path))
        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            logger=logger,
            enable_checkpointing=enable_checkpointing,
        )
        batch_results = trainer.predict(model, train_dl)
        out_embs = np.concatenate(
            list(map(lambda x: x[1], batch_results)),
            axis=0,
        )
        active_intervention_values = []
        inactive_intervention_values = []
        for idx in range(n_concepts):
            active_intervention_values.append(
                np.percentile(out_embs[:, idx], 95)
            )
            inactive_intervention_values.append(
                np.percentile(out_embs[:, idx], 5)
            )
        active_intervention_values = torch.FloatTensor(
            active_intervention_values
        )
        inactive_intervention_values = torch.FloatTensor(
            inactive_intervention_values
        )

    independent = config['architecture'].startswith("Independent")
    sequential = config['architecture'].startswith("Sequential")

    if independent or sequential:
        _, c2y_model = construct_sequential_models(
            n_concepts,
            n_tasks,
            config,
            imbalance=imbalance,
            task_class_weights=task_class_weights,
        )
        model_config = copy.deepcopy(config)
        model_config['concept_loss_weight'] = 1
        model_config['task_loss_weight'] = 0
        base_model = construct_model(
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            config=model_config,
            imbalance=imbalance,
            task_class_weights=task_class_weights,
            active_intervention_values=active_intervention_values,
            inactive_intervention_values=inactive_intervention_values,
            intervention_policy=intervention_policy,
            output_latent=output_latent,
            output_interventions=output_interventions,
        )
        model = construct_model(
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            config=model_config,
            imbalance=imbalance,
            task_class_weights=task_class_weights,
            active_intervention_values=active_intervention_values,
            inactive_intervention_values=inactive_intervention_values,
            intervention_policy=intervention_policy,
            output_latent=output_latent,
            output_interventions=output_interventions,
            x2c_model=base_model.x2c_model,
            c2y_model=c2y_model,
        )
    else:
        print(f"Loading model from {model_saved_path}")
        model = construct_model(
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            config=config,
            imbalance=imbalance,
            task_class_weights=task_class_weights,
            active_intervention_values=active_intervention_values,
            inactive_intervention_values=inactive_intervention_values,
            intervention_policy=intervention_policy,
            output_latent=output_latent,
            output_interventions=output_interventions,
        )

    model.load_state_dict(torch.load(model_saved_path))
    return model
