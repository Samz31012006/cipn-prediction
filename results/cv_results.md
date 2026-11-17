| Model | AUC (95% CI) | Accuracy (95% CI) | Brier (95% CI) |
|---|---|---|---|
| Logistic regression (C=1) | 0.511 (0.451–0.568) | 0.524 (0.483–0.564) | 0.264 (0.252–0.277) |
| Logistic regression (C=0.1) | 0.485 (0.429–0.541) | 0.527 (0.491–0.563) | 0.255 (0.250–0.261) |
| Logistic + interactions (C=0.1) | 0.505 (0.450–0.565) | 0.527 (0.489–0.568) | 0.257 (0.249–0.264) |
| Random forest | 0.500 (0.441–0.563) | 0.544 (0.504–0.586) | 0.255 (0.248–0.262) |
| Gradient boosting | 0.510 (0.450–0.570) | 0.526 (0.482–0.570) | 0.267 (0.253–0.282) |

n = 272 patients, 128 positive (47.1%). Majority-class accuracy = 0.529; chance AUC = 0.5.
10×5-fold stratified CV; 2000-draw stratified patient bootstrap CIs.

AUC difference vs. Logistic regression (C=1) (paired bootstrap):
- Logistic regression (C=0.1): -0.026 (-0.041 to -0.009)
- Logistic + interactions (C=0.1): -0.005 (-0.037 to +0.025)
- Random forest: -0.011 (-0.036 to +0.014)
- Gradient boosting: -0.001 (-0.022 to +0.020)
