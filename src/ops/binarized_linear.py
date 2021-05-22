import logging
from typing import Any, Optional, Tuple

import torch

from src.ops.utils import deterministic_quantize, stochastic_quantize
from src.types import quantization
from src.utils import prod

logging.basicConfig()
logger = logging.getLogger(__name__)


class BinarizedLinear(torch.autograd.Function):
    r"""
    binarized tensor를 입력으로 하여,
    scale factor와 binarized weights로 linear 연산을 수행하는 operation function

    Binarize operation method는 BinaryConnect의 `Deterministic`과 `Stochastic` method를 사용하며,
    scale factor와 weights를  binarize하는 방법은 XNOR-Net의 method를 사용함.

    .. note:
        Weights binarize method는 forward에서 binarized wegiths를 사용하고,
        gradient update는 real-value weights에 적용한다.

        `Deterministic` method는 다음과 같다.
        .. math::
            W_{b} = \bigg\{\begin{matrix}+1,\ \ \ if\ W\geq0, \ \ \\-1,\ \ \ otherwise,\end{matrix}

        `Stochastic` method는 다음과 같다.
        .. math::
            W_{b} = \bigg\{\begin{matrix}+1,\ \ \ with\ probability\ p=\sigma(w) \ \ \\-1,\ \ \ with\ probability\ 1-p\ \ \ \ \ \ \ \ \ \end{matrix}

        `Scale factor`는 다음과 같다.
        .. math::
            \alpha^{*} = \frac{\Sigma{|W_{i}|}}{n},\ where\ \ n=c\times w\times h

    Refers:
        1). BinaryConnect : https://arxiv.org/pdf/1511.00363.pdf
        2). XNOR-Net : https://arxiv.org/pdf/1603.05279.pdf
    """

    @staticmethod
    def forward(
        ctx: object,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        mode: Optional[str] = quantization.QType.DETER,
    ) -> torch.Tensor:
        r"""
        Real-value weights를 binarized weight와 scale factor로 변환한다.
        binarized tensor이를입력으로 받으면 이를 다음과 같이 계산한다.

        .. math::
            output=I_{b} \odot W_{b} \times \alpha_{W_{b}}

        Args:
            ctx (object): forward/backward간 정보를 공유하기위한 데이터 컨테이너
            input (torch.Tensor): binairzed tensor
            weight (torch.Tensor): :math:`(out\_features, in\_features)`
            bias (Optional[torch.Tensor]): :math:`(out\_features)`
            mode (str): 이진화 종류

        Returns:
            (torch.Tensor) : :math:I_{b} \odot W_{b} \times \alpha_{W_{b}}
        """

        weight_scale_factor, n = None, None

        logger.debug(f"input: {input}")

        with torch.no_grad():
            if mode == quantization.QType.DETER:
                binarized_weight = deterministic_quantize(weight)

                s = torch.sum(torch.abs(weight))
                n = prod(weight.shape)
                weight_scale_factor = s / n

            elif mode == quantization.QType.STOCH:
                binarized_weight = stochastic_quantize(weight)

                matmul = torch.matmul(weight.T, binarized_weight)
                s = torch.sum(matmul)
                n = prod(weight.shape)
                weight_scale_factor = s / n

                logging.debug(f"matmul result : {matmul}")

            else:
                raise RuntimeError(f"{mode} not supported")

            if (not weight_scale_factor) or (not n):
                raise RuntimeError("`scale_factor` or `n` not allow `None` value")

            logger.debug(f"weights: {weight}")
            logger.debug(f"binarized weights: {binarized_weight}")
            logger.debug(f"weight scale factor: {weight_scale_factor}")

            device = weight.device
            binarized_weight = binarized_weight.to(device)
            output = input.mm(binarized_weight.t()) * weight_scale_factor

            if bias is not None:
                output += bias.unsqueeze(0).expand_as(output)

        ctx.save_for_backward(input, binarized_weight * weight_scale_factor, bias)

        return output

    @staticmethod
    def backward(ctx: object, grad_output: Any):
        r"""
        gradient에 binarized weight를 마스킹하여 grad를 전달한다.

        Args:
            ctx (object): forward/backward간 정보를 공유하기위한 데이터 컨테이너
            grad_output (Any): Compuational graph를 통해서 들어오는 gradient정보

        Returns:
            (torch.Tensor) : Computational graph 앞으로 보내기위한 gradient 정보
        """

        input, binarized_weight_with_scale_factor, bias = ctx.saved_tensors

        grad_input = grad_weight = grad_bias = None

        with torch.no_grad():
            if ctx.needs_input_grad[0]:
                grad_input = grad_output.mm(binarized_weight_with_scale_factor)
            if ctx.needs_input_grad[1]:
                grad_weight = grad_output.t().mm(input)
            if (bias is not None) and (ctx.needs_input_grad[2]):
                grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, None


binarized_linear = BinarizedLinear.apply
