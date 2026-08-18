# Model Results & Evaluation Summary

| Model / Experiment | Parameters ($\beta$) | MSE | $R^2$ Score | Training Time | Summary Conclusion |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Closed-Form (Normal Equation)** | $[745975, 512535, -67144, 16408]$ | $1.636 \times 10^{11}$ | **0.5898** | ~0.0001 sec | Exact analytical solution; ~4,000x faster for small feature counts. |
| **Batch Gradient Descent** | $[745975, 512534, -67144, 16409]$ | $1.636 \times 10^{11}$ | **0.5898** | ~4.18 sec | Converges to identical parameters (66,182 epochs at $\alpha=0.001$). Requires feature scaling. |
| **Polynomial Regression** | Degree 1 to 5 features | Decreases on train | Varies | Fast | Degrees 2–3 fit curvature well; Degree $\ge 5$ overfits near data boundaries. |
| **Ridge Regularization ($L_2$)** | Shrinks $\beta \rightarrow 0$ as $\lambda \uparrow$ | Increases with $\lambda$ | Stabilized | Fast | Controls weight magnitude and mitigates multicollinearity between `Sq.Ft` and `Beds`. |

### Key Takeaways

1. **Speed & Scaling**: Normal Equation is much faster for small $N$, but Gradient Descent is necessary when feature dimension $N > 10,000$.
2. **Negative Bedroom Weight**: Bedrooms coefficient is negative ($-67,144$) when controlling for `Sq.Ft` because holding total area constant and adding bedrooms shrinks average room size.
3. **Overfitting & Regularization**: High-degree polynomials cause overfitting; Ridge regularization dampens feature variance.
