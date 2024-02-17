# HW 3
## Jacob Reiss & Michael Zeolla

## Question 1:
$\nabla w_l \hat y_{k=l}^{(i)} = \nabla w_l \frac{\exp(x_{(i)}^Tw_l + b_l)}{\sum_{k'=1}^{c}\exp(x_{(i)}^Tw_{k'} + b_{k'})}$
$= x_{(i)} (\frac{\sum_{k'=1}^{c}\exp(x_{(i)}^Tw_{k'} + b_{k'})*\exp(x_{(i)}^Tw_l+b_l) - (\exp(x_{(i)}^Tw_l + b_l))^2}{(\sum_{k'=1}^{c}exp(x_{(i)}^Tw_{k'} + b_{k'}))^2})$
$= x_{(i)}*(\hat y_{l}^{(i)}-(\hat y_{l}^{(i)})^2)$
$= x_{(i)}\hat y_{l}^{(i)}(1-\hat y_{l}^{(i)})$

$\nabla w_l \hat y_{k\not=l}^{(i)} = \nabla w_l \frac{\exp(x_{(i)}^Tw_k + b_k)}{\sum_{k'=1}^{c}\exp(x_{(i)}^Tw_{k'} + b_{k'})}$
$= \frac{\sum_{k'=1}^{c}\exp(x_{(i)}^Tw_{k'} + b_{k'})0 - \exp(x_{(i)}^Tw_k+b_k)\exp(x_{(i)}^Tw_l + b_l)x_{(i)}}{(\sum_{k'=1}^{c}exp(x_{(i)}^Tw_{k'} + b_{k'}))^2}$
$= -x_{(i)}\hat y_{k}^{(i)}\hat y_{l}^{(i)}$

## Question 2:
