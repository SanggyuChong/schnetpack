from __future__ import annotations

from typing import Dict, Optional, List

from schnetpack.transform import Transform
import schnetpack.properties as properties
from schnetpack.utils import as_dtype
from schnetpack.atomistic import Atomwise, LLPredRigidtyAtomwise, Forces, LLPredRigidityForces
from schnetpack.data.loader import AtomsLoader

import torch
import torch.nn as nn

import copy

__all__ = ["AtomisticModel", "NeuralNetworkPotential", "LLPredRigidityNNP"]


class AtomisticModel(nn.Module):
    """
    Base class for all SchNetPack models.

    SchNetPack models should subclass `AtomisticModel` implement the forward method.
    To use the automatic collection of required derivatives, each submodule that
    requires gradients w.r.t to the input, should list them as strings in
    `submodule.required_derivatives = ["input_key"]`. The model needs to call
    `self.collect_derivatives()` at the end of its `__init__`.

    To make use of post-processing transform, the model should call
    `input = self.postprocess(input)` at the end of its `forward`. The post processors
    will only be applied if `do_postprocessing=True`.

    Example:
         class SimpleModel(AtomisticModel):
            def __init__(
                self,
                representation: nn.Module,
                output_module: nn.Module,
                postprocessors: Optional[List[Transform]] = None,
                input_dtype_str: str = "float32",
                do_postprocessing: bool = True,
            ):
                super().__init__(
                    input_dtype_str=input_dtype_str,
                    postprocessors=postprocessors,
                    do_postprocessing=do_postprocessing,
                )
                self.representation = representation
                self.output_modules = output_modules

                self.collect_derivatives()

            def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
                inputs = self.initialize_derivatives(inputs)

                inputs = self.representation(inputs)
                inputs = self.output_module(inputs)

                # apply postprocessing (if enabled)
                inputs = self.postprocess(inputs)
                return inputs

    """

    def __init__(
        self,
        postprocessors: Optional[List[Transform]] = None,
        input_dtype_str: str = "float32",
        do_postprocessing: bool = True,
    ):
        """
        Args:
            postprocessors: Post-processing transforms that may be
                initialized using the `datamodule`, but are not
                applied during training.
            input_dtype: The dtype of real inputs as string.
            do_postprocessing: If true, post-processing is activated.
        """
        super().__init__()
        self.input_dtype_str = input_dtype_str
        self.do_postprocessing = do_postprocessing
        self.postprocessors = nn.ModuleList(postprocessors)
        self.required_derivatives: Optional[List[str]] = None
        self.model_outputs: Optional[List[str]] = None

    def collect_derivatives(self) -> List[str]:
        self.required_derivatives = None
        required_derivatives = set()
        for m in self.modules():
            if (
                hasattr(m, "required_derivatives")
                and m.required_derivatives is not None
            ):
                required_derivatives.update(m.required_derivatives)
        required_derivatives: List[str] = list(required_derivatives)
        self.required_derivatives = required_derivatives

    def collect_outputs(self) -> List[str]:
        self.model_outputs = None
        model_outputs = set()
        for m in self.modules():
            if hasattr(m, "model_outputs") and m.model_outputs is not None:
                model_outputs.update(m.model_outputs)
        model_outputs: List[str] = list(model_outputs)
        self.model_outputs = model_outputs

    def initialize_derivatives(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        for p in self.required_derivatives:
            if p in inputs.keys():
                inputs[p].requires_grad_()
        return inputs

    def initialize_transforms(self, datamodule):
        for module in self.modules():
            if isinstance(module, Transform):
                module.datamodule(datamodule)

    def postprocess(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.do_postprocessing:
            # apply postprocessing
            for pp in self.postprocessors:
                inputs = pp(inputs)
        return inputs

    def extract_outputs(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        results = {k: inputs[k] for k in self.model_outputs}
        return results


class NeuralNetworkPotential(AtomisticModel):
    """
    A generic neural network potential class that sequentially applies a list of input
    modules, a representation module and a list of output modules.

    This can be flexibly configured for various, e.g. property prediction or potential
    energy sufaces with response properties.
    """

    def __init__(
        self,
        representation: nn.Module,
        input_modules: List[nn.Module] = None,
        output_modules: List[nn.Module] = None,
        postprocessors: Optional[List[Transform]] = None,
        input_dtype_str: str = "float32",
        do_postprocessing: bool = True,
    ):
        """
        Args:
            representation: The module that builds representation from inputs.
            input_modules: Modules that are applied before representation, e.g. to
                modify input or add additional tensors for response properties.
            output_modules: Modules that predict output properties from the
                representation.
            postprocessors: Post-processing transforms that may be initialized using the
                `datamodule`, but are not applied during training.
            input_dtype_str: The dtype of real inputs.
            do_postprocessing: If true, post-processing is activated.
        """
        super().__init__(
            input_dtype_str=input_dtype_str,
            postprocessors=postprocessors,
            do_postprocessing=do_postprocessing,
        )
        self.representation = representation
        self.input_modules = nn.ModuleList(input_modules)
        self.output_modules = nn.ModuleList(output_modules)

        self.collect_derivatives()
        self.collect_outputs()

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # initialize derivatives for response properties
        inputs = self.initialize_derivatives(inputs)

        for m in self.input_modules:
            inputs = m(inputs)

        inputs = self.representation(inputs)

        for m in self.output_modules:
            inputs = m(inputs)

        # apply postprocessing (if enabled)
        inputs = self.postprocess(inputs)
        results = self.extract_outputs(inputs)

        return results


class LLPredRigidityNNP(AtomisticModel):
    """
    A wrapper class for neural network potential that implements LLPR uncertainties.
    """

    def __init__(
        self,
        orig_model: NeuralNetworkPotential,
        ll_feat_aggregation_mode: Optional[str] = None,
        save_ll_feat_per_atom: bool = False,
        consider_ll_feat_gradients: bool = False,
    ):
        """
        Args:
            orig_model: Original `NeuralNetworkPotential` model for which
            wrapping is done.
            ll_feat_aggregation_mode: one of sum, avg, or None, defaults to
            aggregation mode of original model
            save_ll_feat_per_atom: option to save the ll feats per atom
            consider_ll_feat_gradients: option to also save gradients of ll
            feats if used during training (e.g. forces, stresses)
        """
        super().__init__()
        self.orig_model = copy.deepcopy(orig_model)
        mod_output_modules = nn.ModuleList()

        self.ll_feat_aggregation_mode = ll_feat_aggregation_mode
        self.save_ll_feat_per_atom = save_ll_feat_per_atom
        self.consider_ll_feat_gradients = consider_ll_feat_gradients

        # Find and modify the output modules to extract the ll feats. Strictly
        # assumes output modules are composed of one `Atomwise` and one `Forces`
        # modules.

        if len(self.orig_model.output_modules) > 2:
            assert RuntimeError("Too many output modules for current implementation of LLPR!")

        atomwise_detected = False
        forces_detected = False
        for module in self.orig_model.output_modules:
            if isinstance(module, Atomwise):
                if atomwise_detected:
                    assert RuntimeError("LLPR is currently not compatible with"
                                        " multiple `Atomwise` output modules!")
                mod_output_modules.append(
                    LLPredRigidtyAtomwise(
                        module,
                        self.ll_feat_aggregation_mode,
                        self.save_ll_feat_per_atom,
                    )
                )
                self.ll_feat_dim = module.outnet[-1].in_features
                atomwise_detected = True
            elif isinstance(module, Forces):
                if forces_detected:
                    assert RuntimeError("LLPR is currently not compatible with"
                                        " multiple `Forces` output modules!")
                if self.consider_ll_feat_gradients:
                    mod_output_modules.append(LLPredRigidityForces(module))
                else:
                    mod_output_modules.append(module)
                forces_detected = True

        self.output_modules = mod_output_modules
        self.register_buffer(
            "covariance",
            torch.zeros((self.ll_feat_dim, self.ll_feat_dim),
                        device=next(self.orig_model.parameters()).device)
        )
        self.register_buffer(
            "inv_covariance",
            torch.zeros((self.ll_feat_dim, self.ll_feat_dim),
                        device=next(self.orig_model.parameters()).device)
        )

        self.covariance_computed = False
        self.inv_covariance_computed = False

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # initialize derivatives for response properties
        inputs = self.orig_model.initialize_derivatives(inputs)

        for m in self.orig_model.input_modules:
            inputs = m(inputs)

        inputs = self.orig_model.representation(inputs)

        # use modified output modules
        for m in self.output_modules:
            inputs = m(inputs)

        # apply postprocessing (if enabled)
        inputs = self.orig_model.postprocess(inputs)
        results = self.orig_model.extract_outputs(inputs)

        results["ll_feats"] = inputs["ll_feats"]

        if self.save_ll_feat_per_atom:
            results["ll_feats_per_atom"] = inputs["ll_feats_per_atom"]

        if self.consider_ll_feat_gradients:
            if "ll_feats_grad_F" in inputs:
                results["ll_feats_grad_F"] = inputs["ll_feats_grad_F"]
            if "ll_feats_grad_S" in inputs:
                results["ll_feats_grad_S"] = inputs["ll_feats_grad_S"]

        if self.inv_covariance_computed:
            uncertainty = torch.einsum("ij, jk, ik -> i",
                                       inputs["ll_feats"],
                                       self.inv_covariance,
                                       inputs["ll_feats"],
                                       )
            results["uncertainty"] = uncertainty.unsqueeze(1)

        return results

    def compute_covariance(self, dataloader: AtomsLoader, weights: Optional[Dict[str, float]]) -> None:
        """
        Utility function to compute the covariance matrix for a training set.

        Args:
            dataloader: `AtomsLoader` object that contains the training set
            weights: dictionary of weights that were used during model training.
                     Must contain entries for:
                     'E' (main target output),
                     'F' (position gradient),
                     'S' (cell/strain gradient).
        """
        for batch in dataloader:
            results = self.forward(batch)
            ll_feats = results["ll_feats"].detach()
            self.covariance = self.covariance + ll_feats.T @ ll_feats
            if "ll_feats_grad_F" in results:
                ll_feats_grad_F = results["ll_feats_grad_F"].detach()
                ll_feats_grad_F = ll_feats_grad_F.reshape(-1, ll_feats.shape[-1])
                self.covariance = self.covariance + (ll_feats_grad_F.T @ ll_feats_grad_F) * weights['F'] / weights['E']
            if "ll_feats_grad_S" in results:
                ll_feats_grad_S = results["ll_feats_grad_S"].detach()
                ll_feats_grad_F = ll_feats_grad_F.reshape(-1, ll_feats.shape[-1])
                self.covariance = self.covariance + (ll_feats_grad_S.T @ ll_feats_grad_S) * weights['S'] / weights['E']
        self.covariance_computed = True

    def compute_inv_covariance(self, C: float, sigma: float) -> None:
        """
        Utility function to compute the covariance matrix for a training set.
        Refer to Bigi et al. for details on the calibration parameters.

        Args:
            C: calibration parameter #1
            sigma: calibration parameter #2
        """
        if not self.covariance_computed:
            raise RuntimeError("You must compute the covariance matrix before "
                               "computing the inverse covariance matrix!")
        self.inv_covariance = C * torch.linalg.inv(
            self.covariance + sigma**2 * torch.eye(self.ll_feat_dim, device=self.covariance.device)
            )
        self.inv_covariance_computed = True
