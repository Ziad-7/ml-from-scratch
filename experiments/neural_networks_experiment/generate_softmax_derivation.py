import os
import subprocess
import fitz  # PyMuPDF
from PIL import Image

def generate_latex():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    tex_path = os.path.join(output_dir, "softmax_derivation.tex")
    pdf_path = os.path.join(output_dir, "softmax_derivation.pdf")
    combined_png_path = os.path.join(figures_dir, "softmax_derivation_complete.png")

    latex_content = r"""\documentclass[10pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{geometry}
\geometry{top=0.7in, bottom=0.7in, left=0.8in, right=0.8in}
\usepackage{xcolor}
\usepackage{tcolorbox}
\usepackage{hyperref}
\usepackage{parskip}

\definecolor{primary}{RGB}{20, 45, 95}
\definecolor{boxbg}{RGB}{240, 244, 252}

\begin{document}

\begin{center}
    {\LARGE \textbf{Multivariable Calculus Derivation}}\\[0.4em]
    {\large \textbf{Gradient of Categorical Cross-Entropy with Softmax Activation}}\\[0.2em]
    {\footnotesize Z7-OS Machine Learning Research --- Mathematics Archive}
\end{center}

\vspace{-0.5em}
\begin{tcolorbox}[colback=boxbg,colframe=primary,title=\textbf{The Fundamental Theorem},arc=2mm]
For a sample with pre-activations (logits) $Z = [z_1, \dots, z_K]^T$, Softmax predictions $a = [a_1, \dots, a_K]^T$, and one-hot true labels $y = [y_1, \dots, y_K]^T$:
\[
\frac{\partial L}{\partial z_i} = a_i - y_i \quad \Longleftrightarrow \quad \frac{\partial L}{\partial Z} = a - y
\]
\end{tcolorbox}

\section*{1. Mathematical Formulation}
Let the logit vector be $Z \in \mathbb{R}^K$. The Softmax probability for class $i$ is:
\[
a_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}} = \frac{e^{z_i}}{\Sigma_Z}, \quad \text{where } \Sigma_Z = \sum_{j=1}^K e^{z_j}
\]
The true target vector $y \in \{0, 1\}^K$ is one-hot encoded, guaranteeing:
\[
\sum_{k=1}^K y_k = 1
\]
The Categorical Cross-Entropy Loss for this sample is:
\[
L(a, y) = - \sum_{k=1}^K y_k \ln(a_k)
\]

\section*{2. The Softmax Jacobian Matrix $\left(\frac{\partial a_k}{\partial z_i}\right)$}
Because every activation $a_k$ shares the normalization denominator $\Sigma_Z$, varying any input $z_i$ impacts \textbf{every} output probability. By the quotient rule $\frac{\partial}{\partial z_i}\left[\frac{u}{v}\right] = \frac{u'v - uv'}{v^2}$ with $v = \Sigma_Z$ and $\frac{\partial \Sigma_Z}{\partial z_i} = e^{z_i}$:

\textbf{Case 1: Diagonal Terms ($k = i$)}
\[
\frac{\partial a_i}{\partial z_i} = \frac{e^{z_i} \Sigma_Z - e^{z_i} e^{z_i}}{\Sigma_Z^2} = \frac{e^{z_i}}{\Sigma_Z} \left(\frac{\Sigma_Z - e^{z_i}}{\Sigma_Z}\right) = a_i (1 - a_i)
\]

\textbf{Case 2: Off-Diagonal Cross Terms ($k \neq i$)}
Here the numerator $u = e^{z_k}$ is independent of $z_i$, so $\frac{\partial u}{\partial z_i} = 0$:
\[
\frac{\partial a_k}{\partial z_i} = \frac{0 \cdot \Sigma_Z - e^{z_k} e^{z_i}}{\Sigma_Z^2} = - \left(\frac{e^{z_k}}{\Sigma_Z}\right) \left(\frac{e^{z_i}}{\Sigma_Z}\right) = - a_k a_i
\]

\section*{3. Loss Derivative with Respect to Activations $\left(\frac{\partial L}{\partial a_k}\right)$}
Differentiating the cross-entropy sum term by term:
\[
\frac{\partial L}{\partial a_k} = \frac{\partial}{\partial a_k} \left[ - y_k \ln(a_k) \right] = - \frac{y_k}{a_k}
\]

\section*{4. Applying the Multivariable Chain Rule}
Since $z_i$ affects all $K$ activations, we sum over all output nodes $k \in \{1, \dots, K\}$:
\[
\frac{\partial L}{\partial z_i} = \sum_{k=1}^K \frac{\partial L}{\partial a_k} \frac{\partial a_k}{\partial z_i}
\]
Splitting into the $k = i$ term and all other $k \neq i$ terms:
\[
\frac{\partial L}{\partial z_i} = \left(\frac{\partial L}{\partial a_i} \frac{\partial a_i}{\partial z_i}\right) + \sum_{k \neq i} \left(\frac{\partial L}{\partial a_k} \frac{\partial a_k}{\partial z_i}\right)
\]
Substituting the derived components:
\[
\frac{\partial L}{\partial z_i} = \left(-\frac{y_i}{a_i}\right) \cdot \left[ a_i (1 - a_i) \right] + \sum_{k \neq i} \left(-\frac{y_k}{a_k}\right) \cdot \left[ - a_k a_i \right]
\]
Simplifying algebraically:
\begin{align*}
\frac{\partial L}{\partial z_i} &= - y_i (1 - a_i) + \sum_{k \neq i} y_k a_i \\
&= - y_i + y_i a_i + a_i \sum_{k \neq i} y_k \\
&= - y_i + a_i \left( y_i + \sum_{k \neq i} y_k \right)
\end{align*}
Notice that the term in parentheses is the sum across all classes: $y_i + \sum_{k \neq i} y_k = \sum_{k=1}^K y_k = 1$.
\[
\frac{\partial L}{\partial z_i} = - y_i + a_i (1) = \mathbf{a_i - y_i} \quad \blacksquare
\]

\section*{5. Vectorized Batch Formulation}
For a mini-batch of $m$ training samples with logit matrix $Z \in \mathbb{R}^{m \times K}$, prediction matrix $A \in \mathbb{R}^{m \times K}$, and target matrix $Y \in \mathbb{R}^{m \times K}$:
\[
\frac{\partial J}{\partial Z} = \frac{1}{m} (A - Y) \quad \in \mathbb{R}^{m \times K}
\]

\end{document}
"""
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_content)
    print(f"Generated LaTeX: {tex_path}")

    # Compile with pdflatex
    cmd = ["pdflatex", "-interaction=nonstopmode", "-output-directory", output_dir, tex_path]
    print("Compiling LaTeX to PDF...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("LaTeX compilation failed:")
        print(result.stdout[-1000:])
        return

    print(f"Compiled PDF: {pdf_path}")

    # Render PDF pages to images using PyMuPDF and stitch together
    doc = fitz.open(pdf_path)
    page_images = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=220)
        p_img_path = os.path.join(figures_dir, f"softmax_derivation_page_{i+1}.png")
        pix.save(p_img_path)
        page_images.append(Image.open(p_img_path))

    # Stitch pages vertically if multiple
    total_height = sum(img.height for img in page_images)
    max_width = max(img.width for img in page_images)
    combined = Image.new("RGB", (max_width, total_height), (255, 255, 255))
    
    y_offset = 0
    for img in page_images:
        combined.paste(img, (0, y_offset))
        y_offset += img.height
        
    combined.save(combined_png_path)
    print(f"Stitched complete image saved: {combined_png_path}")

if __name__ == "__main__":
    generate_latex()
