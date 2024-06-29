<div class="Box-sc-g0xbh4-0 bJMeLZ js-snippet-clipboard-copy-unpositioned" data-hpc="true"><article class="markdown-body entry-content container-lg" itemprop="text"><div class="markdown-heading" dir="auto"><h1 tabindex="-1" class="heading-element" dir="auto">🤖 Interactive Machine Learning Experiments</h1><a id="user-content--interactive-machine-learning-experiments" class="anchor" aria-label="Permalink: 🤖 Interactive Machine Learning Experiments" href="#-interactive-machine-learning-experiments"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
 
<p dir="auto"><g-emoji class="g-emoji" alias="warning">⚠️</g-emoji> This repository contains machine learning <strong>experiments</strong> and <strong>not</strong> a production ready, reusable, optimised and fine-tuned code and models. This is rather a sandbox or a playground for learning and trying different machine learning approaches, algorithms and data-sets. Models might not perform well and there is a place for overfitting/underfitting.</p>
</blockquote>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto">Experiments</h2><a id="user-content-experiments" class="anchor" aria-label="Permalink: Experiments" href="#experiments"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto">Most of the models in these experiments were trained using <a href="https://www.tensorflow.org/" rel="nofollow">TensorFlow 2</a> with <a href="https://www.tensorflow.org/guide/keras/overview" rel="nofollow">Keras</a> support.</p>
<div class="markdown-heading" dir="auto"><h3 tabindex="-1" class="heading-element" dir="auto">Supervised Machine Learning</h3><a id="user-content-supervised-machine-learning" class="anchor" aria-label="Permalink: Supervised Machine Learning" href="#supervised-machine-learning"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto"><a href="https://en.wikipedia.org/wiki/Supervised_learning" rel="nofollow">Supervised learning</a> is when you have input variables <code>X</code> and an output variable <code>Y</code> and you use an algorithm to learn the mapping function from the input to the output: <code>Y = f(X)</code>. The goal is to approximate the mapping function so well that when you have new input data <code>X</code> that you can predict the output variables <code>Y</code> for that data. It is called supervised learning because the process of an algorithm learning from the training dataset can be thought of as a teacher supervising the learning process.</p>
<div class="markdown-heading" dir="auto"><h4 tabindex="-1" class="heading-element" dir="auto">Multilayer Perceptron (MLP) or simple Neural Network (NN)</h4><a id="user-content-multilayer-perceptron-mlp-or-simple-neural-network-nn" class="anchor" aria-label="Permalink: Multilayer Perceptron (MLP) or simple Neural Network (NN)" href="#multilayer-perceptron-mlp-or-simple-neural-network-nn"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto">A <a href="https://en.wikipedia.org/wiki/Multilayer_perceptron" rel="nofollow">multilayer perceptron</a> (MLP) is a class of feedforward artificial neural network (ANN). Multilayer perceptrons are sometimes referred to as "vanilla" neural networks (composed of multiple layers of perceptrons), especially when they have a single hidden layer. It can distinguish data that is not linearly separable.</p>
<table>
  <thead>
    <tr>
      <th align="left" width="150"> </th>
      <th align="left" width="200">Experiment</th>
      <th align="left" width="140">Model demo &amp; training</th>
      <th align="left">Tags</th>
      <th align="left" width="140">Dataset</th>
    </tr>
  </thead>
  <tbody>
    
    <tr>
      <td>
        <a target="_blank" rel="noopener noreferrer" href="/trekhleb/machine-learning-experiments/blob/master/demos/src/images/digits_recognition_mlp.png"><img src="/trekhleb/machine-learning-experiments/raw/master/demos/src/images/digits_recognition_mlp.png" alt="Handwritten digits recognition (MLP)" width="150" style="max-width: 100%;"></a>
      </td>
      <td>
        <a href="/trekhleb/machine-learning-experiments/blob/master/experiments/digits_recognition_mlp/digits_recognition_mlp.ipynb">
          <b>Handwritten Digits Recognition (MLP)</b>
        </a>
      </td>
      <td>
        <a href="https://trekhleb.github.io/machine-learning-experiments/#/experiments/DigitsRecognitionMLP" rel="nofollow">
          <img src="https://camo.githubusercontent.com/68a708c4a6e245c28f219a7d792fe0c7c273893b93561164e25dc8a21d804616/68747470733a2f2f696d672e736869656c64732e696f2f7374617469632f76313f6c6162656c3d2546302539462538452541382532304c61756e6368266d6573736167653d44656d6f26636f6c6f723d677265656e" alt="Launch demo" data-canonical-src="https://img.shields.io/static/v1?label=%F0%9F%8E%A8%20Launch&amp;message=Demo&amp;color=green" style="max-width: 100%;">
        </a>
        <a href="https://nbviewer.jupyter.org/github/trekhleb/machine-learning-experiments/blob/master/experiments/digits_recognition_mlp/digits_recognition_mlp.ipynb" rel="nofollow">
          <img src="https://camo.githubusercontent.com/e91e1d353a8b6acf0b42547ac3901f2c30138a3abaaa3d3c242da30b5b4f8426/68747470733a2f2f6d7962696e6465722e6f72672f62616467655f6c6f676f2e737667" alt="Open in Binder" data-canonical-src="https://mybinder.org/badge_logo.svg" style="max-width: 100%;">
        </a>
        <a href="https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/digits_recognition_mlp/digits_recognition_mlp.ipynb" rel="nofollow">
          <img src="https://camo.githubusercontent.com/f5e0d0538a9c2972b5d413e0ace04cecd8efd828d133133933dfffec282a4e1b/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667" alt="Open in Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg" style="max-width: 100%;">
        </a>
      </td>
      <td>
        <code>MLP</code>
      </td>
      <td>
        <a href="https://www.tensorflow.org/datasets/catalog/mnist" rel="nofollow">
          MNIST
        </a>
      </td>
    </tr>
    
    <tr>
      <td>
        <a target="_blank" rel="noopener noreferrer" href="/trekhleb/machine-learning-experiments/blob/master/demos/src/images/sketch_recognition_mlp.png"><img src="/trekhleb/machine-learning-experiments/raw/master/demos/src/images/sketch_recognition_mlp.png" alt="Handwritten sketch recognition (MLP)" width="150" style="max-width: 100%;"></a>
      </td>
      <td>
        <a href="/trekhleb/machine-learning-experiments/blob/master/experiments/sketch_recognition_mlp/sketch_recognition_mlp.ipynb">
          <b>Handwritten Sketch Recognition (MLP)</b>
        </a>
      </td>
      <td>
        <a href="https://trekhleb.github.io/machine-learning-experiments/#/experiments/SketchRecognitionMLP" rel="nofollow">
          <img src="https://camo.githubusercontent.com/68a708c4a6e245c28f219a7d792fe0c7c273893b93561164e25dc8a21d804616/68747470733a2f2f696d672e736869656c64732e696f2f7374617469632f76313f6c6162656c3d2546302539462538452541382532304c61756e6368266d6573736167653d44656d6f26636f6c6f723d677265656e" alt="Launch demo" data-canonical-src="https://img.shields.io/static/v1?label=%F0%9F%8E%A8%20Launch&amp;message=Demo&amp;color=green" style="max-width: 100%;">
        </a>
        <a href="https://nbviewer.jupyter.org/github/trekhleb/machine-learning-experiments/blob/master/experiments/sketch_recognition_mlp/sketch_recognition_mlp.ipynb" rel="nofollow">
          <img src="https://camo.githubusercontent.com/e91e1d353a8b6acf0b42547ac3901f2c30138a3abaaa3d3c242da30b5b4f8426/68747470733a2f2f6d7962696e6465722e6f72672f62616467655f6c6f676f2e737667" alt="Open in Binder" data-canonical-src="https://mybinder.org/badge_logo.svg" style="max-width: 100%;">
        </a>
        <a href="https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/sketch_recognition_mlp/sketch_recognition_mlp.ipynb" rel="nofollow">
          <img src="https://camo.githubusercontent.com/f5e0d0538a9c2972b5d413e0ace04cecd8efd828d133133933dfffec282a4e1b/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667" alt="Open in Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg" style="max-width: 100%;">
        </a>
      </td>
      <td>
        <code>MLP</code>
      </td>
      <td>
        <a href="https://github.com/googlecreativelab/quickdraw-dataset">
          QuickDraw
        </a>
      </td>
    </tr>
  </tbody>
</table>
<div class="markdown-heading" dir="auto"><h4 tabindex="-1" class="heading-element" dir="auto">Convolutional Neural Networks (CNN)</h4><a id="user-content-convolutional-neural-networks-cnn" class="anchor" aria-label="Permalink: Convolutional Neural Networks (CNN)" href="#convolutional-neural-networks-cnn"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto">A <a href="https://en.wikipedia.org/wiki/Convolutional_neural_network" rel="nofollow">convolutional neural network</a> (CNN, or ConvNet) is a class of deep neural networks, most commonly applied to analyzing visual imagery (photos, videos). They are used for detecting and classifying objects on photos and videos, style transfer, face recognition, pose estimation etc.</p>
<table>
  <thead>
    <tr>
      <th align="left" width="150"> </th>
      <th align="left" width="200">Experiment</th>
      <th align="left" width="140">Model demo &amp; training</th>
      <th align="left">Tags</th>
      <th align="left" width="140">Dataset</th>
    </tr>
  </thead>
  <tbody>
    
    <tr>
      <td>
        <a target="_blank" rel="noopener noreferrer" href="/trekhleb/machine-learning-experiments/blob/master/demos/src/images/digits_recognition_cnn.png"><img src="/trekhleb/machine-learning-experiments/raw/master/demos/src/images/digits_recognition_cnn.png" alt="Handwritten digits recognition (CNN)" style="max-width: 100%;"></a>
      </td>
      <td>
        <a href="/trekhleb/machine-learning-experiments/blob/master/experiments/digits_recognition_cnn/digits_recognition_cnn.ipynb">
          <b>Handwritten Digits Recognition (CNN)</b>
        </a>
      </td>
      <td>
        <a href="https://trekhleb.github.io/machine-learning-experiments/#/experiments/DigitsRecognitionCNN" rel="nofollow">
          <img src="https://camo.githubusercontent.com/68a708c4a6e245c28f219a7d792fe0c7c273893b93561164e25dc8a21d804616/68747470733a2f2f696d672e736869656c64732e696f2f7374617469632f76313f6c6162656c3d2546302539462538452541382532304c61756e6368266d6573736167653d44656d6f26636f6c6f723d677265656e" alt="Launch demo" data-canonical-src="https://img.shields.io/static/v1?label=%F0%9F%8E%A8%20Launch&amp;message=Demo&amp;color=green" style="max-width: 100%;">
        </a>
        <a href="https://nbviewer.jupyter.org/github/trekhleb/machine-learning-experiments/blob/master/experiments/digits_recognition_cnn/digits_recognition_cnn.ipynb" rel="nofollow">
          <img src="https://camo.githubusercontent.com/e91e1d353a8b6acf0b42547ac3901f2c30138a3abaaa3d3c242da30b5b4f8426/68747470733a2f2f6d7962696e6465722e6f72672f62616467655f6c6f676f2e737667" alt="Open in Binder" data-canonical-src="https://mybinder.org/badge_logo.svg" style="max-width: 100%;">
        </a>
        <a href="https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/digits_recognition_cnn/digits_recognition_cnn.ipynb" rel="nofollow">
          <img src="https://camo.githubusercontent.com/f5e0d0538a9c2972b5d413e0ace04cecd8efd828d133133933dfffec282a4e1b/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667" alt="Open in Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg" style="max-width: 100%;">
        </a>
      </td>
      <td>
        <code>CNN</code>
      </td>
      <td>
        <a href="https://www.tensorflow.org/datasets/catalog/mnist" rel="nofollow">
          MNIST
        </a>
      </td>
    </tr>
    
    <tr>
      <td>
        <a target="_blank" rel="noopener noreferrer" href="/trekhleb/machine-learning-experiments/blob/master/demos/src/images/sketch_recognition_cnn.png"><img src="/trekhleb/machine-learning-experiments/raw/master/demos/src/images/sketch_recognition_cnn.png" alt="Handwritten sketch recognition (CNN)" width="150" style="max-width: 100%;"></a>
      </td>
      <td>
        <a href="/trekhleb/machine-learning-experiments/blob/master/experiments/sketch_recognition_cnn/sketch_recognition_cnn.ipynb">
          <b>Handwritten Sketch Recognition (CNN)</b>
        </a>
      </td>
      <td>
        <a href="https://trekhleb.github.io/machine-learning-experiments/#/experiments/SketchRecognitionCNN" rel="nofollow">
          <img src="https://camo.githubusercontent.com/68a708c4a6e245c28f219a7d792fe0c7c273893b93561164e25dc8a21d804616/68747470733a2f2f696d672e736869656c64732e696f2f7374617469632f76313f6c6162656c3d2546302539462538452541382532304c61756e6368266d6573736167653d44656d6f26636f6c6f723d677265656e" alt="Launch demo" data-canonical-src="https://img.shields.io/static/v1?label=%F0%9F%8E%A8%20Launch&amp;message=Demo&amp;color=green" style="max-width: 100%;">
        </a>
        <a href="https://nbviewer.jupyter.org/github/trekhleb/machine-learning-experiments/blob/master/experiments/sketch_recognition_cnn/sketch_recognition_cnn.ipynb" rel="nofollow">
          <img src="https://camo.githubusercontent.com/e91e1d353a8b6acf0b42547ac3901f2c30138a3abaaa3d3c242da30b5b4f8426/68747470733a2f2f6d7962696e6465722e6f72672f62616467655f6c6f676f2e737667" alt="Open in Binder" data-canonical-src="https://mybinder.org/badge_logo.svg" style="max-width: 100%;">
        </a>
        <a href="https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/sketch_recognition_cnn/sketch_recognition_cnn.ipynb" rel="nofollow">
          <img src="https://camo.githubusercontent.com/f5e0d0538a9c2972b5d413e0ace04cecd8efd828d133133933dfffec282a4e1b/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667" alt="Open in Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg" style="max-width: 100%;">
        </a>
      </td>
      <td>
        <code>CNN</code>
      </td>
      <td>
        <a href="https://github.com/googlecreativelab/quickdraw-dataset">
          QuickDraw
        </a>
      </td>
    </tr>
    
    <tr>
      <td>
        <a target="_blank" rel="noopener noreferrer" href="/trekhleb/machine-learning-experiments/blob/master/demos/src/images/rock_paper_scissors_cnn.jpg"><img src="/trekhleb/machine-learning-experiments/raw/master/demos/src/images/rock_paper_scissors_cnn.jpg" alt="Rock Paper Scissors" width="150" style="max-width: 100%;"></a>
      </td>
      <td>
        <a href="/trekhleb/machine-learning-experiments/blob/master/experiments/rock_paper_scissors_cnn/rock_paper_scissors_cnn.ipynb">
          <b>Rock Paper Scissors (CNN)</b>
        </a>
      </td>
      <td>
        <a href="https://trekhleb.github.io/machine-learning-experiments/#/experiments/RockPaperScissorsCNN" rel="nofollow">
          <img src="https://camo.githubusercontent.com/68a708c4a6e245c28f219a7d792fe0c7c273893b93561164e25dc8a21d804616/68747470733a2f2f696d672e736869656c64732e696f2f7374617469632f76313f6c6162656c3d2546302539462538452541382532304c61756e6368266d6573736167653d44656d6f26636f6c6f723d677265656e" alt="Launch demo" data-canonical-src="https://img.shields.io/static/v1?label=%F0%9F%8E%A8%20Launch&amp;message=Demo&amp;color=green" style="max-width: 100%;">
        </a>
        <a href="https://nbviewer.jupyter.org/github/trekhleb/machine-learning-experiments/blob/master/experiments/rock_paper_scissors_cnn/rock_paper_scissors_cnn.ipynb" rel="nofollow">
          <img src="https://camo.githubusercontent.com/e91e1d353a8b6acf0b42547ac3901f2c30138a3abaaa3d3c242da30b5b4f8426/68747470733a2f2f6d7962696e6465722e6f72672f62616467655f6c6f676f2e737667" alt="Open in Binder" data-canonical-src="https://mybinder.org/badge_logo.svg" style="max-width: 100%;">
        </a>
        <a href="https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/rock_paper_scissors_cnn/rock_paper_scissors_cnn.ipynb" rel="nofollow">
          <img src="https://camo.githubusercontent.com/f5e0d0538a9c2972b5d413e0ace04cecd8efd828d133133933dfffec282a4e1b/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667" alt="Open in Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg" style="max-width: 100%;">
        </a>
      </td>
      <td>
        <code>CNN</code>
      </td>
      <td>
        <a href="http://www.laurencemoroney.com/rock-paper-scissors-dataset/" rel="nofollow">
          RPS
        </a>
      </td>
    </tr>
    
    <tr>
      <td>
        <a target="_blank" rel="noopener noreferrer" href="/trekhleb/machine-learning-experiments/blob/master/demos/src/images/rock_paper_scissors_mobilenet_v2.jpg"><img src="/trekhleb/machine-learning-experiments/raw/master/demos/src/images/rock_paper_scissors_mobilenet_v2.jpg" alt="Rock Paper Scissors" width="150" style="max-width: 100%;"></a>
      </td>
      <td>
        <a href="/trekhleb/machine-learning-experiments/blob/master/experiments/rock_paper_scissors_mobilenet_v2/rock_paper_scissors_mobilenet_v2.ipynb">
          <b>Rock Paper Scissors (MobilenetV2)</b>
        </a>
      </td>
      <td>
        <a href="https://trekhleb.github.io/machine-learning-experiments/#/experiments/RockPaperScissorsMobilenetV2" rel="nofollow">
          <img src="https://camo.githubusercontent.com/68a708c4a6e245c28f219a7d792fe0c7c273893b93561164e25dc8a21d804616/68747470733a2f2f696d672e736869656c64732e696f2f7374617469632f76313f6c6162656c3d2546302539462538452541382532304c61756e6368266d6573736167653d44656d6f26636f6c6f723d677265656e" alt="Launch demo" data-canonical-src="https://img.shields.io/static/v1?label=%F0%9F%8E%A8%20Launch&amp;message=Demo&amp;color=green" style="max-width: 100%;">
        </a>
        <a href="https://nbviewer.jupyter.org/github/trekhleb/machine-learning-experiments/blob/master/experiments/rock_paper_scissors_mobilenet_v2/rock_paper_scissors_mobilenet_v2.ipynb" rel="nofollow">
          <img src="https://camo.githubusercontent.com/e91e1d353a8b6acf0b42547ac3901f2c30138a3abaaa3d3c242da30b5b4f8426/68747470733a2f2f6d7962696e6465722e6f72672f62616467655f6c6f676f2e737667" alt="Open in Binder" data-canonical-src="https://mybinder.org/badge_logo.svg" style="max-width: 100%;">
        </a>
        <a href="https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/rock_paper_scissors_mobilenet_v2/rock_paper_scissors_mobilenet_v2.ipynb" rel="nofollow">
          <img src="https://camo.githubusercontent.com/f5e0d0538a9c2972b5d413e0ace04cecd8efd828d133133933dfffec282a4e1b/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667" alt="Open in Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg" style="max-width: 100%;">
        </a>
      </td>
      <td>
        <code>MobileNetV2</code>,
        <code>Transfer learning</code>,
        <code>CNN</code>
      </td>
      <td>
        <a href="http://www.laurencemoroney.com/rock-paper-scissors-dataset/" rel="nofollow">
          RPS
        </a>,
        <a href="http://image-net.org/explore" rel="nofollow">
          ImageNet
        </a>
      </td>
    </tr>
    
    <tr>
      <td>
        <a target="_blank" rel="noopener noreferrer" href="/trekhleb/machine-learning-experiments/blob/master/demos/src/images/objects_detection_ssdlite_mobilenet_v2.jpg"><img src="/trekhleb/machine-learning-experiments/raw/master/demos/src/images/objects_detection_ssdlite_mobilenet_v2.jpg" alt="Objects detection" width="150" style="max-width: 100%;"></a>
      </td>
      <td>
        <a href="/trekhleb/machine-learning-experiments/blob/master/experiments/objects_detection_ssdlite_mobilenet_v2/objects_detection_ssdlite_mobilenet_v2.ipynb">
          <b>Objects Detection (MobileNetV2)</b>
        </a>
      </td>
      <td>
        <a href="https://trekhleb.github.io/machine-learning-experiments/#/experiments/ObjectsDetectionSSDLiteMobilenetV2" rel="nofollow">
          <img src="https://camo.githubusercontent.com/68a708c4a6e245c28f219a7d792fe0c7c273893b93561164e25dc8a21d804616/68747470733a2f2f696d672e736869656c64732e696f2f7374617469632f76313f6c6162656c3d2546302539462538452541382532304c61756e6368266d6573736167653d44656d6f26636f6c6f723d677265656e" alt="Launch demo" data-canonical-src="https://img.shields.io/static/v1?label=%F0%9F%8E%A8%20Launch&amp;message=Demo&amp;color=green" style="max-width: 100%;">
        </a>
        <a href="https://nbviewer.jupyter.org/github/trekhleb/machine-learning-experiments/blob/master/experiments/objects_detection_ssdlite_mobilenet_v2/objects_detection_ssdlite_mobilenet_v2.ipynb" rel="nofollow">
          <img src="https://camo.githubusercontent.com/e91e1d353a8b6acf0b42547ac3901f2c30138a3abaaa3d3c242da30b5b4f8426/68747470733a2f2f6d7962696e6465722e6f72672f62616467655f6c6f676f2e737667" alt="Open in Binder" data-canonical-src="https://mybinder.org/badge_logo.svg" style="max-width: 100%;">
        </a>
        <a href="https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/objects_detection_ssdlite_mobilenet_v2/objects_detection_ssdlite_mobilenet_v2.ipynb" rel="nofollow">
          <img src="https://camo.githubusercontent.com/f5e0d0538a9c2972b5d413e0ace04cecd8efd828d133133933dfffec282a4e1b/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667" alt="Open in Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg" style="max-width: 100%;">
        </a>
      </td>
      <td>
        <code>MobileNetV2</code>,
        <code>SSDLite</code>,
        <code>CNN</code>
      </td>
      <td>
        <a href="http://cocodataset.org/#home" rel="nofollow">
          COCO
        </a>
      </td>
    </tr>
    
    <tr>
      <td>
        <a target="_blank" rel="noopener noreferrer" href="/trekhleb/machine-learning-experiments/blob/master/demos/src/images/image_classification_mobilenet_v2.jpg"><img src="/trekhleb/machine-learning-experiments/raw/master/demos/src/images/image_classification_mobilenet_v2.jpg" alt="Objects detection" width="150" style="max-width: 100%;"></a>
      </td>
      <td>
        <a href="/trekhleb/machine-learning-experiments/blob/master/experiments/image_classification_mobilenet_v2/image_classification_mobilenet_v2.ipynb">
          <b>Image Classification (MobileNetV2)</b>
        </a>
      </td>
      <td>
        <a href="https://trekhleb.github.io/machine-learning-experiments/#/experiments/ImageClassificationMobilenetV2" rel="nofollow">
          <img src="https://camo.githubusercontent.com/68a708c4a6e245c28f219a7d792fe0c7c273893b93561164e25dc8a21d804616/68747470733a2f2f696d672e736869656c64732e696f2f7374617469632f76313f6c6162656c3d2546302539462538452541382532304c61756e6368266d6573736167653d44656d6f26636f6c6f723d677265656e" alt="Launch demo" data-canonical-src="https://img.shields.io/static/v1?label=%F0%9F%8E%A8%20Launch&amp;message=Demo&amp;color=green" style="max-width: 100%;">
        </a>
        <a href="https://nbviewer.jupyter.org/github/trekhleb/machine-learning-experiments/blob/master/experiments/image_classification_mobilenet_v2/image_classification_mobilenet_v2.ipynb" rel="nofollow">
          <img src="https://camo.githubusercontent.com/e91e1d353a8b6acf0b42547ac3901f2c30138a3abaaa3d3c242da30b5b4f8426/68747470733a2f2f6d7962696e6465722e6f72672f62616467655f6c6f676f2e737667" alt="Open in Binder" data-canonical-src="https://mybinder.org/badge_logo.svg" style="max-width: 100%;">
        </a>
        <a href="https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/image_classification_mobilenet_v2/image_classification_mobilenet_v2.ipynb" rel="nofollow">
          <img src="https://camo.githubusercontent.com/f5e0d0538a9c2972b5d413e0ace04cecd8efd828d133133933dfffec282a4e1b/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667" alt="Open in Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg" style="max-width: 100%;">
        </a>
      </td>
      <td>
        <code>MobileNetV2</code>,
        <code>CNN</code>
      </td>
      <td>
        <a href="http://image-net.org/explore" rel="nofollow">
          ImageNet
        </a>
      </td>
    </tr>
  </tbody>
</table>
<div class="markdown-heading" dir="auto"><h4 tabindex="-1" class="heading-element" dir="auto">Recurrent Neural Networks (RNN)</h4><a id="user-content-recurrent-neural-networks-rnn" class="anchor" aria-label="Permalink: Recurrent Neural Networks (RNN)" href="#recurrent-neural-networks-rnn"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto">A <a href="https://en.wikipedia.org/wiki/Recurrent_neural_network" rel="nofollow">recurrent neural network</a> (RNN) is a class of deep neural networks, most commonly applied to sequence-based data like speech, voice, text or music. They are used for machine translation, speech recognition, voice synthesis etc.</p>
<table>
  <thead>
    <tr>
      <th align="left" width="150"> </th>
      <th align="left" width="200">Experiment</th>
      <th align="left" width="140">Model demo &amp; training</th>
      <th align="left">Tags</th>
      <th align="left" width="140">Dataset</th>
    </tr>
  </thead>
  <tbody>
    
    <tr>
      <td>
        <a target="_blank" rel="noopener noreferrer" href="/trekhleb/machine-learning-experiments/blob/master/demos/src/images/numbers_summation_rnn.png"><img src="/trekhleb/machine-learning-experiments/raw/master/demos/src/images/numbers_summation_rnn.png" alt="Numbers summation (RNN)" width="150" style="max-width: 100%;"></a>
      </td>
      <td>
        <a href="/trekhleb/machine-learning-experiments/blob/master/experiments/numbers_summation_rnn/numbers_summation_rnn.ipynb">
          <b>Numbers Summation (RNN)</b>
        </a>
      </td>
      <td>
        <a href="https://trekhleb.github.io/machine-learning-experiments/#/experiments/NumbersSummationRNN" rel="nofollow">
          <img src="https://camo.githubusercontent.com/68a708c4a6e245c28f219a7d792fe0c7c273893b93561164e25dc8a21d804616/68747470733a2f2f696d672e736869656c64732e696f2f7374617469632f76313f6c6162656c3d2546302539462538452541382532304c61756e6368266d6573736167653d44656d6f26636f6c6f723d677265656e" alt="Launch demo" data-canonical-src="https://img.shields.io/static/v1?label=%F0%9F%8E%A8%20Launch&amp;message=Demo&amp;color=green" style="max-width: 100%;">
        </a>
        <a href="https://nbviewer.jupyter.org/github/trekhleb/machine-learning-experiments/blob/master/experiments/numbers_summation_rnn/numbers_summation_rnn.ipynb" rel="nofollow">
          <img src="https://camo.githubusercontent.com/e91e1d353a8b6acf0b42547ac3901f2c30138a3abaaa3d3c242da30b5b4f8426/68747470733a2f2f6d7962696e6465722e6f72672f62616467655f6c6f676f2e737667" alt="Open in Binder" data-canonical-src="https://mybinder.org/badge_logo.svg" style="max-width: 100%;">
        </a>
        <a href="https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/numbers_summation_rnn/numbers_summation_rnn.ipynb" rel="nofollow">
          <img src="https://camo.githubusercontent.com/f5e0d0538a9c2972b5d413e0ace04cecd8efd828d133133933dfffec282a4e1b/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667" alt="Open in Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg" style="max-width: 100%;">
        </a>
      </td>
      <td>
        <code>LSTM</code>,
        <code>Sequence-to-sequence</code>
      </td>
      <td>
        Auto-generated
      </td>
    </tr>
    
    <tr>
      <td>
        <a target="_blank" rel="noopener noreferrer" href="/trekhleb/machine-learning-experiments/blob/master/demos/src/images/text_generation_shakespeare_rnn.jpg"><img src="/trekhleb/machine-learning-experiments/raw/master/demos/src/images/text_generation_shakespeare_rnn.jpg" alt="Shakespeare Text Generation (RNN)" width="150" style="max-width: 100%;"></a>
      </td>
      <td>
        <a href="/trekhleb/machine-learning-experiments/blob/master/experiments/text_generation_shakespeare_rnn/text_generation_shakespeare_rnn.ipynb">
          <b>Shakespeare Text Generation (RNN)</b>
        </a>
      </td>
      <td>
        <a href="https://trekhleb.github.io/machine-learning-experiments/#/experiments/TextGenerationShakespeareRNN" rel="nofollow">
          <img src="https://camo.githubusercontent.com/68a708c4a6e245c28f219a7d792fe0c7c273893b93561164e25dc8a21d804616/68747470733a2f2f696d672e736869656c64732e696f2f7374617469632f76313f6c6162656c3d2546302539462538452541382532304c61756e6368266d6573736167653d44656d6f26636f6c6f723d677265656e" alt="Launch demo" data-canonical-src="https://img.shields.io/static/v1?label=%F0%9F%8E%A8%20Launch&amp;message=Demo&amp;color=green" style="max-width: 100%;">
        </a>
        <a href="https://nbviewer.jupyter.org/github/trekhleb/machine-learning-experiments/blob/master/experiments/text_generation_shakespeare_rnn/text_generation_shakespeare_rnn.ipynb" rel="nofollow">
          <img src="https://camo.githubusercontent.com/e91e1d353a8b6acf0b42547ac3901f2c30138a3abaaa3d3c242da30b5b4f8426/68747470733a2f2f6d7962696e6465722e6f72672f62616467655f6c6f676f2e737667" alt="Open in Binder" data-canonical-src="https://mybinder.org/badge_logo.svg" style="max-width: 100%;">
        </a>
        <a href="https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/text_generation_shakespeare_rnn/text_generation_shakespeare_rnn.ipynb" rel="nofollow">
          <img src="https://camo.githubusercontent.com/f5e0d0538a9c2972b5d413e0ace04cecd8efd828d133133933dfffec282a4e1b/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667" alt="Open in Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg" style="max-width: 100%;">
        </a>
      </td>
      <td>
        <code>LSTM</code>,
        <code>Character-based RNN</code>
      </td>
      <td>
        <a href="https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt" rel="nofollow">
          Shakespeare
        </a>
      </td>
    </tr>
    
    <tr>
      <td>
        <a target="_blank" rel="noopener noreferrer" href="/trekhleb/machine-learning-experiments/blob/master/demos/src/images/text_generation_wikipedia_rnn.png"><img src="/trekhleb/machine-learning-experiments/raw/master/demos/src/images/text_generation_wikipedia_rnn.png" alt="Wikipedia Text Generation (RNN)" width="150" style="max-width: 100%;"></a>
      </td>
      <td>
        <a href="/trekhleb/machine-learning-experiments/blob/master/experiments/text_generation_wikipedia_rnn/text_generation_wikipedia_rnn.ipynb">
          <b>Wikipedia Text Generation (RNN)</b>
        </a>
      </td>
      <td>
        <a href="https://trekhleb.github.io/machine-learning-experiments/#/experiments/TextGenerationWikipediaRNN" rel="nofollow">
          <img src="https://camo.githubusercontent.com/68a708c4a6e245c28f219a7d792fe0c7c273893b93561164e25dc8a21d804616/68747470733a2f2f696d672e736869656c64732e696f2f7374617469632f76313f6c6162656c3d2546302539462538452541382532304c61756e6368266d6573736167653d44656d6f26636f6c6f723d677265656e" alt="Launch demo" data-canonical-src="https://img.shields.io/static/v1?label=%F0%9F%8E%A8%20Launch&amp;message=Demo&amp;color=green" style="max-width: 100%;">
        </a>
        <a href="https://nbviewer.jupyter.org/github/trekhleb/machine-learning-experiments/blob/master/experiments/text_generation_wikipedia_rnn/text_generation_wikipedia_rnn.ipynb" rel="nofollow">
          <img src="https://camo.githubusercontent.com/e91e1d353a8b6acf0b42547ac3901f2c30138a3abaaa3d3c242da30b5b4f8426/68747470733a2f2f6d7962696e6465722e6f72672f62616467655f6c6f676f2e737667" alt="Open in Binder" data-canonical-src="https://mybinder.org/badge_logo.svg" style="max-width: 100%;">
        </a>
        <a href="https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/text_generation_wikipedia_rnn/text_generation_wikipedia_rnn.ipynb" rel="nofollow">
          <img src="https://camo.githubusercontent.com/f5e0d0538a9c2972b5d413e0ace04cecd8efd828d133133933dfffec282a4e1b/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667" alt="Open in Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg" style="max-width: 100%;">
        </a>
      </td>
      <td>
        <code>LSTM</code>,
        <code>Character-based RNN</code>
      </td>
      <td>
        <a href="https://www.tensorflow.org/datasets/catalog/wikipedia" rel="nofollow">
          Wikipedia
        </a>
      </td>
    </tr>
    
    <tr>
      <td>
        <a target="_blank" rel="noopener noreferrer" href="/trekhleb/machine-learning-experiments/blob/master/demos/src/images/recipe_generation_rnn.jpg"><img src="/trekhleb/machine-learning-experiments/raw/master/demos/src/images/recipe_generation_rnn.jpg" alt="Recipe Generation (RNN)" width="150" style="max-width: 100%;"></a>
      </td>
      <td>
        <a href="/trekhleb/machine-learning-experiments/blob/master/experiments/recipe_generation_rnn/recipe_generation_rnn.ipynb">
          <b>Recipe Generation (RNN)</b>
        </a>
      </td>
      <td>
        <a href="https://trekhleb.github.io/machine-learning-experiments/#/experiments/RecipeGenerationRNN" rel="nofollow">
          <img src="https://camo.githubusercontent.com/68a708c4a6e245c28f219a7d792fe0c7c273893b93561164e25dc8a21d804616/68747470733a2f2f696d672e736869656c64732e696f2f7374617469632f76313f6c6162656c3d2546302539462538452541382532304c61756e6368266d6573736167653d44656d6f26636f6c6f723d677265656e" alt="Launch demo" data-canonical-src="https://img.shields.io/static/v1?label=%F0%9F%8E%A8%20Launch&amp;message=Demo&amp;color=green" style="max-width: 100%;">
        </a>
        <a href="https://nbviewer.jupyter.org/github/trekhleb/machine-learning-experiments/blob/master/experiments/recipe_generation_rnn/recipe_generation_rnn.ipynb" rel="nofollow">
          <img src="https://camo.githubusercontent.com/e91e1d353a8b6acf0b42547ac3901f2c30138a3abaaa3d3c242da30b5b4f8426/68747470733a2f2f6d7962696e6465722e6f72672f62616467655f6c6f676f2e737667" alt="Open in Binder" data-canonical-src="https://mybinder.org/badge_logo.svg" style="max-width: 100%;">
        </a>
        <a href="https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/recipe_generation_rnn/recipe_generation_rnn.ipynb" rel="nofollow">
          <img src="https://camo.githubusercontent.com/f5e0d0538a9c2972b5d413e0ace04cecd8efd828d133133933dfffec282a4e1b/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667" alt="Open in Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg" style="max-width: 100%;">
        </a>
      </td>
      <td>
        <code>LSTM</code>,
        <code>Character-based RNN</code>
      </td>
      <td>
        <a href="https://eightportions.com/datasets/Recipes/" rel="nofollow">
          Recipe box
        </a>
      </td>
    </tr>
  </tbody>
</table>
<div class="markdown-heading" dir="auto"><h3 tabindex="-1" class="heading-element" dir="auto">Unsupervised Machine Learning</h3><a id="user-content-unsupervised-machine-learning" class="anchor" aria-label="Permalink: Unsupervised Machine Learning" href="#unsupervised-machine-learning"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto"><a href="https://en.wikipedia.org/wiki/Unsupervised_learning" rel="nofollow">Unsupervised learning</a> is when you only have input data <code>X</code> and no corresponding output variables. The goal for unsupervised learning is to model the underlying structure or distribution in the data in order to learn more about the data. These are called unsupervised learning because unlike supervised learning above there is no correct answers and there is no teacher. Algorithms are left to their own to discover and present the interesting structure in the data.</p>
<div class="markdown-heading" dir="auto"><h4 tabindex="-1" class="heading-element" dir="auto">Generative Adversarial Networks (GANs)</h4><a id="user-content-generative-adversarial-networks-gans" class="anchor" aria-label="Permalink: Generative Adversarial Networks (GANs)" href="#generative-adversarial-networks-gans"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto">A <a href="https://en.wikipedia.org/wiki/Generative_adversarial_network" rel="nofollow">generative adversarial network</a> (GAN) is a class of machine learning frameworks where two neural networks contest with each other in a game. Two models are trained simultaneously by an adversarial process. For example a <em>generator</em> ("the artist") learns to create images that look real, while a <em>discriminator</em> ("the art critic") learns to tell real images apart from fakes.</p>
<table>
  <thead>
    <tr>
      <th align="left" width="150"> </th>
      <th align="left" width="200">Experiment</th>
      <th align="left" width="140">Model demo &amp; training</th>
      <th align="left">Tags</th>
      <th align="left" width="140">Dataset</th>
    </tr>
  </thead>
  <tbody>
    
    <tr>
      <td>
        <a target="_blank" rel="noopener noreferrer" href="/trekhleb/machine-learning-experiments/blob/master/demos/src/images/clothes_generation_dcgan.jpg"><img src="/trekhleb/machine-learning-experiments/raw/master/demos/src/images/clothes_generation_dcgan.jpg" alt="Clothes Generation (DCGAN)" width="150" style="max-width: 100%;"></a>
      </td>
      <td>
        <a href="/trekhleb/machine-learning-experiments/blob/master/experiments/clothes_generation_dcgan/clothes_generation_dcgan.ipynb">
          <b>Clothes Generation (DCGAN)</b>
        </a>
      </td>
      <td>
        <a href="https://trekhleb.github.io/machine-learning-experiments/#/experiments/ClothesGenerationDCGAN" rel="nofollow">
          <img src="https://camo.githubusercontent.com/68a708c4a6e245c28f219a7d792fe0c7c273893b93561164e25dc8a21d804616/68747470733a2f2f696d672e736869656c64732e696f2f7374617469632f76313f6c6162656c3d2546302539462538452541382532304c61756e6368266d6573736167653d44656d6f26636f6c6f723d677265656e" alt="Launch demo" data-canonical-src="https://img.shields.io/static/v1?label=%F0%9F%8E%A8%20Launch&amp;message=Demo&amp;color=green" style="max-width: 100%;">
        </a>
        <a href="https://nbviewer.jupyter.org/github/trekhleb/machine-learning-experiments/blob/master/experiments/clothes_generation_dcgan/clothes_generation_dcgan.ipynb" rel="nofollow">
          <img src="https://camo.githubusercontent.com/e91e1d353a8b6acf0b42547ac3901f2c30138a3abaaa3d3c242da30b5b4f8426/68747470733a2f2f6d7962696e6465722e6f72672f62616467655f6c6f676f2e737667" alt="Open in Binder" data-canonical-src="https://mybinder.org/badge_logo.svg" style="max-width: 100%;">
        </a>
        <a href="https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/clothes_generation_dcgan/clothes_generation_dcgan.ipynb" rel="nofollow">
          <img src="https://camo.githubusercontent.com/f5e0d0538a9c2972b5d413e0ace04cecd8efd828d133133933dfffec282a4e1b/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667" alt="Open in Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg" style="max-width: 100%;">
        </a>
      </td>
      <td>
        <code>DCGAN</code>
      </td>
      <td>
        <a href="https://www.tensorflow.org/datasets/catalog/fashion_mnist" rel="nofollow">
          Fashion MNIST
        </a>
      </td>
    </tr>
  </tbody>
</table>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto">How to use this repository locally</h2><a id="user-content-how-to-use-this-repository-locally" class="anchor" aria-label="Permalink: How to use this repository locally" href="#how-to-use-this-repository-locally"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<div class="markdown-heading" dir="auto"><h3 tabindex="-1" class="heading-element" dir="auto">Setup virtual environment for Experiments</h3><a id="user-content-setup-virtual-environment-for-experiments" class="anchor" aria-label="Permalink: Setup virtual environment for Experiments" href="#setup-virtual-environment-for-experiments"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<div class="highlight highlight-source-shell notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-c"><span class="pl-c">#</span> Create "experiments" environment (from the project root folder).</span>
python3 -m venv .virtualenvs/experiments

<span class="pl-c"><span class="pl-c">#</span> Activate environment.</span>
<span class="pl-c1">source</span> .virtualenvs/experiments/bin/activate
<span class="pl-c"><span class="pl-c">#</span> or if you use Fish...</span>
<span class="pl-c1">source</span> .virtualenvs/experiments/bin/activate.fish</pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 tooltipped-no-delay d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="# Create &quot;experiments&quot; environment (from the project root folder).
python3 -m venv .virtualenvs/experiments

# Activate environment.
source .virtualenvs/experiments/bin/activate
# or if you use Fish...
source .virtualenvs/experiments/bin/activate.fish" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<p dir="auto">To quit an environment run <code>deactivate</code>.</p>
<div class="markdown-heading" dir="auto"><h3 tabindex="-1" class="heading-element" dir="auto">Install dependencies</h3><a id="user-content-install-dependencies" class="anchor" aria-label="Permalink: Install dependencies" href="#install-dependencies"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<div class="highlight highlight-source-shell notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-c"><span class="pl-c">#</span> Upgrade pip and setuptools to the latest versions.</span>
pip install --upgrade pip setuptools

<span class="pl-c"><span class="pl-c">#</span> Install packages</span>
pip install -r requirements.txt</pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 tooltipped-no-delay d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="# Upgrade pip and setuptools to the latest versions.
pip install --upgrade pip setuptools

# Install packages
pip install -r requirements.txt" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<p dir="auto">To install new packages run <code>pip install package-name</code>. To add new packages to the requirements run <code>pip freeze &gt; requirements.txt</code>.</p>
<div class="markdown-heading" dir="auto"><h3 tabindex="-1" class="heading-element" dir="auto">Launch Jupyter locally</h3><a id="user-content-launch-jupyter-locally" class="anchor" aria-label="Permalink: Launch Jupyter locally" href="#launch-jupyter-locally"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto">In order to play around with Jupyter notebooks and see how models were trained you need to launch a <a href="https://jupyter.org/" rel="nofollow">Jupyter Notebook</a> server.</p>
<div class="highlight highlight-source-shell notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-c"><span class="pl-c">#</span> Launch Jupyter server.</span>
jupyter notebook</pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 tooltipped-no-delay d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="# Launch Jupyter server.
jupyter notebook" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<p dir="auto">Jupyter will be available locally at <code>http://localhost:8888/</code>. Notebooks with experiments may be found in <code>experiments</code> folder.</p>
<div class="markdown-heading" dir="auto"><h3 tabindex="-1" class="heading-element" dir="auto">Launch demos locally</h3><a id="user-content-launch-demos-locally" class="anchor" aria-label="Permalink: Launch demos locally" href="#launch-demos-locally"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto">Demo application is made on React by means of <a href="https://github.com/facebook/create-react-app">create-react-app</a>.</p>
<div class="highlight highlight-source-shell notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-c"><span class="pl-c">#</span> Switch to demos folder from project root.</span>
<span class="pl-c1">cd</span> demos

<span class="pl-c"><span class="pl-c">#</span> Install all dependencies.</span>
yarn install

<span class="pl-c"><span class="pl-c">#</span> Start demo server on http. </span>
yarn start

<span class="pl-c"><span class="pl-c">#</span> Or start demo server on https (for camera access in browser to work on localhost).</span>
yarn start-https</pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 tooltipped-no-delay d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="# Switch to demos folder from project root.
cd demos

# Install all dependencies.
yarn install

# Start demo server on http. 
yarn start

# Or start demo server on https (for camera access in browser to work on localhost).
yarn start-https" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<p dir="auto">Demos will be available locally at <code>http://localhost:3000/</code> or at <code>https://localhost:3000/</code>.</p>
<div class="markdown-heading" dir="auto"><h3 tabindex="-1" class="heading-element" dir="auto">Convert models</h3><a id="user-content-convert-models" class="anchor" aria-label="Permalink: Convert models" href="#convert-models"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto">The <code>converter</code> environment is used to convert the models that were trained during the experiments from <code>.h5</code> Keras format to Javascript understandable formats (<code>tfjs_layers_model</code> or <code>tfjs_graph_model</code> formats with <code>.json</code> and <code>.bin</code> files) for further usage with <a href="https://www.tensorflow.org/js" rel="nofollow">TensorFlow.js</a> in <a href="http://trekhleb.github.io/machine-learning-experiments/" rel="nofollow">Demo application</a>.</p>
<div class="highlight highlight-source-shell notranslate position-relative overflow-auto" dir="auto"><pre><span class="pl-c"><span class="pl-c">#</span> Create "converter" environment (from the project root folder).</span>
python3 -m venv .virtualenvs/converter

<span class="pl-c"><span class="pl-c">#</span> Activate "converter" environment.</span>
<span class="pl-c1">source</span> .virtualenvs/converter/bin/activate
<span class="pl-c"><span class="pl-c">#</span> or if you use Fish...</span>
<span class="pl-c1">source</span> .virtualenvs/converter/bin/activate.fish

<span class="pl-c"><span class="pl-c">#</span> Install converter requirements.</span>
pip install -r requirements.converter.txt</pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 tooltipped-no-delay d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="# Create &quot;converter&quot; environment (from the project root folder).
python3 -m venv .virtualenvs/converter

# Activate &quot;converter&quot; environment.
source .virtualenvs/converter/bin/activate
# or if you use Fish...
source .virtualenvs/converter/bin/activate.fish

# Install converter requirements.
pip install -r requirements.converter.txt" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<p dir="auto">The conversion of <code>keras</code> models to <code>tfjs_layers_model</code>/<code>tfjs_graph_model</code> formats is done by <a href="https://github.com/tensorflow/tfjs/tree/master/tfjs-converter">tfjs-converter</a>:</p>
<p dir="auto">For example:</p>
<div class="highlight highlight-source-shell notranslate position-relative overflow-auto" dir="auto"><pre>tensorflowjs_converter --input_format keras \
  ./experiments/digits_recognition_mlp/digits_recognition_mlp.h5 \
  ./demos/public/models/digits_recognition_mlp</pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 tooltipped-no-delay d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="tensorflowjs_converter --input_format keras \
  ./experiments/digits_recognition_mlp/digits_recognition_mlp.h5 \
  ./demos/public/models/digits_recognition_mlp" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<blockquote>
<p dir="auto"><g-emoji class="g-emoji" alias="warning">⚠️</g-emoji> Converting the models to JS understandable formats and loading them to the browser directly might not be a good practice since in this case the user might need to load tens or hundreds of megabytes of data to the browser which is not efficient. Normally the model is being served from the back-end (i.e. <a href="https://www.tensorflow.org/tfx" rel="nofollow">TensorFlow Extended</a>) and instead of loading it all to the browser the user will do a lightweight HTTP request to do a prediction. But since the <a href="http://trekhleb.github.io/machine-learning-experiments/" rel="nofollow">Demo App</a> is just an experiment and not a production-ready app and for the sake of simplicity (to avoid having an up and running back-end) we're converting the models to JS understandable formats and loading them directly into the browser.</p>
</blockquote>
<div class="markdown-heading" dir="auto"><h3 tabindex="-1" class="heading-element" dir="auto">Requirements</h3><a id="user-content-requirements" class="anchor" aria-label="Permalink: Requirements" href="#requirements"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto">Recommended versions:</p>
<ul dir="auto">
<li>Python: <code>&gt; 3.7.3</code>.</li>
<li>Node: <code>&gt;= 12.4.0</code>.</li>
<li>Yarn: <code>&gt;= 1.13.0</code>.</li>
</ul>
<p dir="auto">In case if you have Python version <code>3.7.3</code> you might experience <code>RuntimeError: dictionary changed size during iteration</code> error when trying to <code>import tensorflow</code> (see the <a href="https://github.com/tensorflow/tensorflow/issues/33183" data-hovercard-type="issue" data-hovercard-url="/tensorflow/tensorflow/issues/33183/hovercard">issue</a>).</p>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto">You might also be interested in</h2><a id="user-content-you-might-also-be-interested-in" class="anchor" aria-label="Permalink: You might also be interested in" href="#you-might-also-be-interested-in"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<ul dir="auto">
<li><a href="https://github.com/trekhleb/homemade-machine-learning/">Homemade Machine Learning</a> - Python examples of popular machine learning algorithms with interactive Jupyter demos and math being explained.</li>
<li><a href="https://github.com/trekhleb/nano-neuron">NanoNeuron</a> - 7 simple JavaScript functions that will give you a feeling of how machines can actually "learn".</li>
<li><a href="https://github.com/trekhleb/learn-python">Playground and Cheatsheet for Learning Python</a> - Collection of Python scripts that are split by topics and contain code examples with explanations.</li>
</ul>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto">Articles</h2><a id="user-content-articles" class="anchor" aria-label="Permalink: Articles" href="#articles"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<ul dir="auto">
<li>📝 <a href="https://github.com/trekhleb/machine-learning-experiments/blob/master/assets/story.en.md">Story behind the project</a></li>
<li>📝 <a href="https://github.com/trekhleb/machine-learning-experiments/blob/master/assets/recipes_generation.en.md">Generating cooking recipes using TensorFlow and LSTM Recurrent Neural Network</a> (a step-by-step guide)</li>
</ul>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto">Author</h2><a id="user-content-author" class="anchor" aria-label="Permalink: Author" href="#author"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<ul dir="auto">
<li><a href="https://trekhleb.dev" rel="nofollow">@trekhleb</a></li>
</ul>
</article></div>
